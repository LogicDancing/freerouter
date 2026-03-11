"""
freerouter/core/smart_router.py
Enhanced router with dynamic model discovery and benchmark-based scoring.
"""
from __future__ import annotations
import logging
from typing import Optional
from dataclasses import dataclass

from freerouter.core.config import FreeRouterConfig
from freerouter.core.models import ProbeResult, ProbeStatus, RoutingDecision, TaskType
from freerouter.core.platforms import ALL_PLATFORMS, PLATFORM_MAP, Platform
from freerouter.core.prober import Prober
from freerouter.core.router import Router, detect_task_type, ROUTING_TABLE
from freerouter.discovery import DiscoveryManager, DiscoveryConfig, DiscoveredModel
from freerouter.benchmarks import get_model_capabilities, ModelCapabilities

log = logging.getLogger("freerouter.smart_router")


# Domain mapping for task types
TASK_TYPE_TO_DOMAIN = {
    TaskType.CODE_COMPLETION: "coding",
    TaskType.CODE_GENERATION: "coding",
    TaskType.REASONING: "reasoning",
    TaskType.AGENTIC: "instruction",
    TaskType.LONG_CONTEXT: "long_context",
    TaskType.GENERAL: "general",
}


@dataclass
class ModelChoice:
    """A model choice with scoring information."""
    platform: str
    model_id: str
    score: float
    ttft_ms: float
    capabilities: ModelCapabilities
    discovered: Optional[DiscoveredModel] = None


class SmartRouter(Router):
    """
    Enhanced router that considers:
    1. Dynamic model discovery from platform APIs
    2. Benchmark scores for model quality
    3. Platform health (TTFT) from prober
    4. Rate limit awareness
    
    Selection algorithm:
    1. Detect task type from request
    2. Get candidate models from all available platforms
    3. Score each model based on:
       - Quality score for the task domain (from benchmarks)
       - Platform health (TTFT)
       - Rate limit availability
    4. Select best scoring model
    """
    
    def __init__(
        self,
        config: FreeRouterConfig,
        prober: Prober,
        discovery_manager: Optional[DiscoveryManager] = None,
    ):
        super().__init__(config, prober)
        self.discovery_manager = discovery_manager
        self._model_cache: dict[str, ModelCapabilities] = {}
    
    def decide(
        self,
        messages: list[dict],
        estimated_tokens: int,
        user_override_platform: Optional[str] = None,
        user_override_model: Optional[str] = None,
    ) -> RoutingDecision:
        """Return a routing decision using smart model selection."""
        
        # 1. Hard user override
        if user_override_platform and user_override_model:
            return RoutingDecision(
                task_type=TaskType.GENERAL,
                selected_platform=user_override_platform,
                selected_model=user_override_model,
                reason="user_override",
                fallback_chain=[],
                probe_ttft_ms=0,
            )
        
        # 2. Detect task
        task = detect_task_type(messages, estimated_tokens)
        domain = TASK_TYPE_TO_DOMAIN.get(task, "general")
        log.debug("Detected task type: %s (domain: %s)", task.value, domain)
        
        # 3. Get available platforms
        available_platforms = self.config.api_keys.available_platforms()
        probe_results = self.prober.results
        threshold = self.config.routing.ttft_threshold_ms
        
        # 4. Collect candidate models from all platforms
        candidates = self._collect_candidates(
            task, available_platforms, probe_results, threshold
        )
        
        if not candidates:
            # Fall back to traditional routing
            log.warning("No smart candidates, falling back to traditional routing")
            return super().decide(messages, estimated_tokens, user_override_platform, user_override_model)
        
        # 5. Score and rank candidates
        scored = self._score_candidates(candidates, domain, task)
        
        # 6. Select best candidate
        best = scored[0]
        
        # Build fallback chain
        fallback_chain = [c.platform for c in scored[1:5]]  # Top 5 alternatives
        
        log.info(
            "Selected %s/%s (score=%.1f, ttft=%.0fms) for %s",
            best.platform, best.model_id, best.score, best.ttft_ms, task.value
        )
        
        return RoutingDecision(
            task_type=task,
            selected_platform=best.platform,
            selected_model=best.model_id,
            reason=f"smart_score={best.score:.1f}",
            fallback_chain=fallback_chain,
            probe_ttft_ms=best.ttft_ms,
        )
    
    def _collect_candidates(
        self,
        task: TaskType,
        available_platforms: list[str],
        probe_results: dict[str, ProbeResult],
        threshold: float,
    ) -> list[ModelChoice]:
        """Collect candidate models from all platforms."""
        candidates = []
        
        for platform_name in available_platforms:
            if platform_name not in PLATFORM_MAP:
                continue
            
            platform = PLATFORM_MAP[platform_name]
            probe = probe_results.get(platform_name)
            
            # Skip unavailable platforms
            if probe and not probe.is_available:
                continue
            
            ttft_ms = probe.ttft_ms if probe and probe.is_available else 0
            
            # Get models from platform
            models = self._get_platform_models(platform, task)
            
            for model_id in models:
                # Get capabilities
                caps = self._get_model_capabilities(model_id)
                
                # Calculate base score from capabilities
                domain = TASK_TYPE_TO_DOMAIN.get(task, "general")
                quality_score = caps.get_score_for_domain(domain) or 0
                
                # Adjust for platform health
                if probe and probe.is_available:
                    health_penalty = max(0, (ttft_ms - threshold) / 1000)
                    adjusted_score = quality_score - health_penalty
                else:
                    adjusted_score = quality_score * 0.8  # Penalty for unknown health
                
                candidates.append(ModelChoice(
                    platform=platform_name,
                    model_id=model_id,
                    score=adjusted_score,
                    ttft_ms=ttft_ms,
                    capabilities=caps,
                ))
        
        return candidates
    
    def _get_platform_models(self, platform: Platform, task: TaskType) -> list[str]:
        """Get suitable models for a task from a platform."""
        # First, check if discovery manager has fresh data
        if self.discovery_manager:
            discovered = self.discovery_manager.get_models_for_platform(platform.name)
            if discovered:
                # Filter by task type
                return [m.model_id for m in discovered]
        
        # Fall back to static platform data
        suitable = []
        for model in platform.models:
            if task in model.task_types or task == TaskType.GENERAL:
                suitable.append(model.model_id)
        
        if not suitable:
            suitable = [m.model_id for m in platform.models]
        
        return suitable
    
    def _get_model_capabilities(self, model_id: str) -> ModelCapabilities:
        """Get cached or fetch model capabilities."""
        if model_id in self._model_cache:
            return self._model_cache[model_id]
        
        caps = get_model_capabilities(model_id)
        self._model_cache[model_id] = caps
        return caps
    
    def _score_candidates(
        self,
        candidates: list[ModelChoice],
        domain: str,
        task: TaskType,
    ) -> list[ModelChoice]:
        """Score and rank candidates."""
        scored = []
        
        for candidate in candidates:
            # Base quality score
            quality = candidate.capabilities.get_score_for_domain(domain) or 0
            
            # Task-specific adjustments
            if task == TaskType.LONG_CONTEXT:
                # Heavily favor models with large context windows
                context_bonus = min(20, candidate.capabilities.context_window / 50000)
                quality += context_bonus
            
            elif task == TaskType.CODE_COMPLETION:
                # Favor faster models for code completion
                if candidate.ttft_ms > 0 and candidate.ttft_ms < 500:
                    quality += 10  # Bonus for very fast response
            
            # Platform health adjustment
            if candidate.ttft_ms > 0:
                # Penalize slow responses
                speed_penalty = max(0, candidate.ttft_ms - 1000) / 200
                quality -= speed_penalty
            
            # Store final score
            candidate.score = quality
            scored.append(candidate)
        
        # Sort by score descending
        scored.sort(key=lambda c: c.score, reverse=True)
        
        return scored
    
    def get_model_ranking(self, domain: str = "general") -> list[tuple[str, str, float]]:
        """
        Get current model ranking for a domain.
        
        Returns list of (platform, model_id, score) tuples.
        """
        available = self.config.api_keys.available_platforms()
        all_models = []
        
        for platform_name in available:
            if platform_name not in PLATFORM_MAP:
                continue
            
            platform = PLATFORM_MAP[platform_name]
            for model in platform.models:
                caps = self._get_model_capabilities(model.model_id)
                score = caps.get_score_for_domain(domain) or 0
                all_models.append((platform_name, model.model_id, score))
        
        all_models.sort(key=lambda x: x[2], reverse=True)
        return all_models
