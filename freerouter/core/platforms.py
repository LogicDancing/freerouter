"""
freerouter/core/platforms.py
Registry of all supported free LLM platforms and their model pools.
"""
from freerouter.core.models import Platform, PlatformModel, TaskType

# ─── NVIDIA Build (187 models) ───────────────────────────────────────────────
NVIDIA_BUILD = Platform(
    name="nvidia_build",
    display_name="NVIDIA Build",
    api_base="https://integrate.api.nvidia.com/v1",
    probe_model="meta/llama-3.1-8b-instruct",
    rpm_limit=40,
    tags=["most-models", "flagship-quality", "1M-context"],
    models=[
        # GLM 系列 - 中文优化
        PlatformModel("z-ai/glm5", 200000, 1, [TaskType.GENERAL, TaskType.AGENTIC], "GLM-5 最新版"),
        PlatformModel("z-ai/glm4.7", 131072, 2, [TaskType.GENERAL], "GLM-4.7"),
        PlatformModel("thudm/chatglm3-6b", 8192, 3, [TaskType.GENERAL], "ChatGLM3"),
        # Llama 系列
        PlatformModel("meta/llama-3.1-8b-instruct", 131072, 1, [TaskType.CODE_COMPLETION, TaskType.GENERAL], "Llama 3.1 8B"),
        PlatformModel("meta/llama-3.1-70b-instruct", 131072, 2, [TaskType.CODE_GENERATION, TaskType.GENERAL], "Llama 3.1 70B"),
        PlatformModel("meta/llama-3.1-405b-instruct", 131072, 3, [TaskType.CODE_GENERATION, TaskType.REASONING], "Llama 3.1 405B"),
        PlatformModel("meta/llama-3.2-11b-vision-instruct", 131072, 4, [TaskType.GENERAL], "Llama 3.2 Vision"),
        PlatformModel("meta/llama-3.3-70b-instruct", 131072, 2, [TaskType.CODE_GENERATION, TaskType.GENERAL], "Llama 3.3 70B"),
        # Qwen 系列
        PlatformModel("qwen/qwen3-coder-480b-a35b-instruct", 262144, 1, [TaskType.CODE_GENERATION], "Qwen3 Coder 480B"),
        PlatformModel("qwen/qwen2.5-coder-32b-instruct", 131072, 2, [TaskType.CODE_GENERATION], "Qwen Coder 32B"),
        PlatformModel("qwen/qwen3.5-397b-a17b", 200000, 3, [TaskType.GENERAL, TaskType.REASONING], "Qwen3.5 397B"),
        PlatformModel("qwen/qwq-32b", 131072, 4, [TaskType.REASONING], "QwQ 推理模型"),
        # DeepSeek 系列
        PlatformModel("deepseek-ai/deepseek-v3", 131072, 1, [TaskType.CODE_GENERATION, TaskType.GENERAL], "DeepSeek V3"),
        PlatformModel("deepseek-ai/deepseek-v3.1", 131072, 2, [TaskType.CODE_GENERATION, TaskType.GENERAL], "DeepSeek V3.1"),
        PlatformModel("deepseek-ai/deepseek-v3.2", 131072, 2, [TaskType.CODE_GENERATION], "DeepSeek V3.2"),
        PlatformModel("deepseek-ai/deepseek-r1", 131072, 1, [TaskType.REASONING], "DeepSeek R1"),
        PlatformModel("deepseek-ai/deepseek-r1-distill-qwen-32b", 131072, 3, [TaskType.REASONING], "DeepSeek R1 Distill"),
        # Coder 系列
        PlatformModel("bigcode/starcoder2-15b", 16384, 1, [TaskType.CODE_GENERATION], "StarCoder2 15B"),
        PlatformModel("deepseek-ai/deepseek-coder-6.7b-instruct", 16384, 2, [TaskType.CODE_COMPLETION], "DeepSeek Coder"),
        # 长上下文模型 (1M+ context)
        PlatformModel("nvidia/llama-3.1-nemotron-nano-8b-v1", 1048576, 1, [TaskType.LONG_CONTEXT], "1M 上下文"),
        PlatformModel("nvidia/llama-3.1-nemotron-8b-ultralong-1m-instruct", 1048576, 1, [TaskType.LONG_CONTEXT], "1M UltraLong"),
        PlatformModel("nvidia/nemotron-3-nano-30b-a3b", 1048576, 1, [TaskType.LONG_CONTEXT, TaskType.REASONING], "Nemotron 3 Nano 1M"),
        # Mistral 系列
        PlatformModel("mistralai/mistral-large-3-2512", 131072, 2, [TaskType.GENERAL, TaskType.REASONING], "Mistral Large 3"),
        PlatformModel("mistralai/codestral-2508", 262144, 1, [TaskType.CODE_GENERATION], "Codestral"),
    ]
)

# ─── Groq ────────────────────────────────────────────────────────────────────
GROQ = Platform(
    name="groq",
    display_name="Groq",
    api_base="https://api.groq.com/openai/v1",
    probe_model="llama-3.1-8b-instant",
    rpm_limit=30,
    daily_request_limit=14400,
    tags=["fastest", "lpu-chip", "300t/s", "daily-reset"],
    models=[
        PlatformModel("llama-3.1-8b-instant", 131072, 1, [TaskType.CODE_COMPLETION, TaskType.GENERAL], "14,400 req/day"),
        PlatformModel("llama-3.3-70b-versatile", 131072, 2, [TaskType.CODE_GENERATION, TaskType.GENERAL], "1,000 req/day"),
        PlatformModel("llama-4-scout-17b-16e-instruct", 131072, 3, [TaskType.GENERAL], "Llama 4 Scout"),
        PlatformModel("qwen-qwq-32b", 131072, 4, [TaskType.REASONING], "QwQ 推理"),
        PlatformModel("deepseek-r1-distill-llama-70b", 131072, 5, [TaskType.REASONING], "DeepSeek R1"),
        PlatformModel("moonshotai/kimi-k2-instruct", 131072, 3, [TaskType.REASONING, TaskType.GENERAL], "Kimi K2"),
        PlatformModel("openai/gpt-oss-120b", 131072, 2, [TaskType.CODE_GENERATION, TaskType.GENERAL], "GPT-OSS 120B"),
    ]
)

# ─── Cerebras ────────────────────────────────────────────────────────────────
CEREBRAS = Platform(
    name="cerebras",
    display_name="Cerebras",
    api_base="https://api.cerebras.ai/v1",
    probe_model="llama3.1-8b",
    rpm_limit=30,
    daily_token_limit=1_000_000,
    tags=["1M-tokens/day", "wafer-chip", "2500t/s", "agentic-best"],
    models=[
        PlatformModel("llama3.1-8b", 131072, 1, [TaskType.CODE_COMPLETION, TaskType.GENERAL], "~2200 t/s"),
        PlatformModel("llama3.3-70b", 131072, 2, [TaskType.CODE_GENERATION, TaskType.AGENTIC], "Llama 3.3 70B"),
        PlatformModel("llama-4-scout-17b-16e", 131072, 3, [TaskType.GENERAL, TaskType.AGENTIC], "Llama 4 Scout"),
        PlatformModel("qwen-3-32b", 131072, 4, [TaskType.CODE_GENERATION, TaskType.REASONING], "Qwen3 32B"),
        PlatformModel("qwen-3-235b-a22b-instruct-2507", 131072, 1, [TaskType.GENERAL, TaskType.CODE_GENERATION], "Qwen3 235B"),
        PlatformModel("zai-glm-4.7", 131072, 2, [TaskType.GENERAL], "GLM 4.7"),
        PlatformModel("gpt-oss-120b", 131072, 2, [TaskType.CODE_GENERATION, TaskType.GENERAL], "~3000 t/s"),
    ]
)

# ─── SambaNova ───────────────────────────────────────────────────────────────
SAMBANOVA = Platform(
    name="sambanova",
    display_name="SambaNova",
    api_base="https://api.sambanova.ai/v1",
    probe_model="Meta-Llama-3.1-8B-Instruct",  # High quota (288K/day) - reliable
    rpm_limit=1440,
    tags=["free-tier", "rdu-chip", "288k-requests/day"],
    models=[
        # High quota models (288K req/day)
        PlatformModel("Meta-Llama-3.1-8B-Instruct", 131072, 1, [TaskType.CODE_COMPLETION, TaskType.GENERAL], "288K req/day"),
        # Medium quota models (48K req/day)
        PlatformModel("Meta-Llama-3.3-70B-Instruct", 131072, 2, [TaskType.GENERAL, TaskType.CODE_GENERATION], "48K req/day"),
        PlatformModel("DeepSeek-R1-Distill-Llama-70B", 131072, 3, [TaskType.REASONING], "48K req/day"),
        # Low quota models (12K req/day)
        PlatformModel("Qwen3-32B", 131072, 4, [TaskType.CODE_GENERATION], "12K req/day"),
        PlatformModel("DeepSeek-V3.1", 131072, 5, [TaskType.CODE_GENERATION, TaskType.GENERAL], "12K req/day"),
        PlatformModel("DeepSeek-R1-0528", 131072, 6, [TaskType.REASONING], "12K req/day"),
        PlatformModel("Llama-4-Maverick-17B-128E-Instruct", 131072, 7, [TaskType.GENERAL], "Llama 4"),
        PlatformModel("Qwen2.5-Coder-32B-Instruct", 16384, 3, [TaskType.CODE_GENERATION], "Qwen2.5 Coder"),
    ]
)

# ─── Mistral AI (Free Experiment Plan) ──────────────────────────────────────
MISTRAL = Platform(
    name="mistral",
    display_name="Mistral AI",
    api_base="https://api.mistral.ai/v1",
    probe_model="mistral-small-3-2-2506",
    rpm_limit=2,  # Free tier: 2 RPM
    daily_token_limit=1_000_000_000,  # Free tier: 1B tokens/month
    tags=["free-tier", "experiment", "codestral", "1B-tokens/month"],
    models=[
        # Code models
        PlatformModel("codestral-2508", 262144, 1, [TaskType.CODE_GENERATION], "Code specialist"),
        PlatformModel("devstral-2-2512", 131072, 2, [TaskType.CODE_GENERATION, TaskType.AGENTIC], "Code agents model"),
        # General models
        PlatformModel("mistral-large-3-2512", 131072, 1, [TaskType.GENERAL, TaskType.REASONING], "Flagship multimodal"),
        PlatformModel("mistral-medium-3-1-2508", 131072, 2, [TaskType.GENERAL], "Frontier multimodal"),
        PlatformModel("mistral-small-3-2-2506", 131072, 3, [TaskType.CODE_COMPLETION, TaskType.GENERAL], "Fast general"),
        # Reasoning models
        PlatformModel("magistral-medium-1-2-2509", 131072, 2, [TaskType.REASONING], "Reasoning model"),
        PlatformModel("magistral-small-1-2-2509", 131072, 3, [TaskType.REASONING], "Open reasoning"),
        # Lightweight models
        PlatformModel("ministral-3-8b-2512", 131072, 4, [TaskType.CODE_COMPLETION, TaskType.GENERAL], "Efficient 8B"),
        PlatformModel("ministral-3-3b-2512", 131072, 5, [TaskType.CODE_COMPLETION, TaskType.GENERAL], "Tiny 3B"),
    ]
)

# ─── All platforms ───────────────────────────────────────────────────────────
ALL_PLATFORMS: list[Platform] = [NVIDIA_BUILD, GROQ, CEREBRAS, SAMBANOVA, MISTRAL]
PLATFORM_MAP: dict[str, Platform] = {p.name: p for p in ALL_PLATFORMS}


def get_platform(name: str) -> Platform:
    if name not in PLATFORM_MAP:
        raise KeyError(f"Unknown platform: {name}. Available: {list(PLATFORM_MAP)}")
    return PLATFORM_MAP[name]
