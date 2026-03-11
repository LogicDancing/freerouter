"""
freerouter/core/platforms.py
Registry of all supported free LLM platforms and their model pools.
Each platform's Top-5 models per task domain.
"""
from freerouter.core.models import Platform, PlatformModel, TaskType

NVIDIA_BUILD = Platform(
    name="nvidia_build",
    display_name="NVIDIA Build",
    api_base="https://integrate.api.nvidia.com/v1",
    probe_model="meta/llama-3.1-8b-instruct",
    rpm_limit=40,
    tags=["most-models", "flagship-quality", "1M-context"],
    models=[
        # Coding
        PlatformModel("meta/llama-3.1-8b-instruct", 131072, 1,
            [TaskType.CODE_COMPLETION, TaskType.GENERAL, TaskType.CODE_GENERATION], "Fast 8B"),
        PlatformModel("qwen/qwen2.5-coder-32b-instruct", 131072, 2,
            [TaskType.CODE_GENERATION], "Qwen Coder 32B"),
        PlatformModel("meta/llama-3.1-70b-instruct", 131072, 3,
            [TaskType.CODE_GENERATION, TaskType.GENERAL], "Llama 70B"),
        # Reasoning
        PlatformModel("qwen/qwq-32b", 131072, 1,
            [TaskType.REASONING], "QwQ reasoning"),
        PlatformModel("nvidia/llama-3.1-nemotron-70b-instruct", 131072, 2,
            [TaskType.REASONING], "Nemotron 70B"),
        # Long context
        PlatformModel("nvidia/llama-3.1-nemotron-nano-8b-v1", 1048576, 1,
            [TaskType.LONG_CONTEXT], "1M context"),
    ]
)

# ─── Groq ────────────────────────────────────────────────────────────────────
GROQ = Platform(
    name="groq",
    display_name="Groq",
    api_base="https://api.groq.com/openai/v1",
    probe_model="llama-3.1-8b-instant",
    rpm_limit=30,
    daily_request_limit=14400,  # for 8B model; 1000 for 70B
    tags=["fastest", "lpu-chip", "300t/s", "daily-reset"],
    models=[
        PlatformModel("llama-3.1-8b-instant", 131072, 1,
                      [TaskType.CODE_COMPLETION, TaskType.GENERAL],
                      "14,400 req/day · ultra-fast"),
        PlatformModel("llama-3.3-70b-versatile", 131072, 2,
                      [TaskType.CODE_GENERATION, TaskType.GENERAL],
                      "1,000 req/day · best quality on Groq"),
        PlatformModel("kimi-k2-instruct", 131072, 3,
                      [TaskType.CODE_GENERATION, TaskType.REASONING],
                      "Kimi K2 on LPU"),
        PlatformModel("qwen-qwq-32b", 131072, 4,
                      [TaskType.REASONING],
                      "QwQ reasoning chain"),
        PlatformModel("deepseek-r1-distill-llama-70b", 131072, 5,
                      [TaskType.REASONING],
                      "DeepSeek R1 distilled"),
        PlatformModel("mixtral-8x7b-32768", 32768, 5,
                      [TaskType.GENERAL],
                      "14,400 req/day · MoE fast"),
    ]
)

# ─── Cerebras ────────────────────────────────────────────────────────────────
CEREBRAS = Platform(
    name="cerebras",
    display_name="Cerebras",
    api_base="https://api.cerebras.ai/v1",
    probe_model="llama3.1-8b",
    rpm_limit=30,
    daily_token_limit=1_000_000,  # 1M tokens/day — most generous
    tags=["1M-tokens/day", "wafer-chip", "2500t/s", "agentic-best"],
    models=[
        PlatformModel("llama3.1-8b", 131072, 1,
                      [TaskType.CODE_COMPLETION, TaskType.GENERAL],
                      "1,800 t/s · ultra-fast"),
        PlatformModel("llama3.3-70b", 131072, 2,
                      [TaskType.CODE_GENERATION, TaskType.AGENTIC],
                      "Latest 70B on WSE-3"),
        PlatformModel("llama3.1-70b", 131072, 3,
                      [TaskType.AGENTIC, TaskType.GENERAL],
                      "Stable 70B"),
        PlatformModel("qwen-3-32b", 131072, 4,
                      [TaskType.CODE_GENERATION, TaskType.REASONING],
                      "Qwen3 32B"),
        PlatformModel("deepseek-r1-distill-llama-70b", 131072, 5,
                      [TaskType.REASONING],
                      "R1 distilled on WSE"),
        PlatformModel("llama-4-scout-17b-16e", 131072, 3,
                      [TaskType.GENERAL, TaskType.AGENTIC],
                      "Llama 4 Scout MoE"),
    ]
)

SAMBANOVA = Platform(
    name="sambanova",
    display_name="SambaNova",
    api_base="https://api.sambanova.ai/v1",
    probe_model="DeepSeek-V3.1",
    rpm_limit=10,
    tags=["deepseek-v3", "rdu-chip", "quality-backup"],
    models=[
        PlatformModel("DeepSeek-V3.1", 131072, 1,
            [TaskType.CODE_GENERATION, TaskType.GENERAL], "DeepSeek V3.1 latest"),
        PlatformModel("DeepSeek-R1-0528", 131072, 2,
            [TaskType.REASONING], "DeepSeek R1 reasoning"),
        PlatformModel("Meta-Llama-3.3-70B-Instruct", 131072, 3,
            [TaskType.GENERAL], "Llama 3.3 70B"),
        PlatformModel("Qwen3-32B", 131072, 4,
            [TaskType.CODE_GENERATION], "Qwen3 32B"),
    ]
)

# ─── All platforms ordered by routing priority ───────────────────────────────
ALL_PLATFORMS: list[Platform] = [NVIDIA_BUILD, GROQ, CEREBRAS, SAMBANOVA]

PLATFORM_MAP: dict[str, Platform] = {p.name: p for p in ALL_PLATFORMS}


def get_platform(name: str) -> Platform:
    if name not in PLATFORM_MAP:
        raise KeyError(f"Unknown platform: {name}. Available: {list(PLATFORM_MAP)}")
    return PLATFORM_MAP[name]
