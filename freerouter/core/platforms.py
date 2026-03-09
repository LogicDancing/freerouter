"""
freerouter/core/platforms.py
Registry of all supported free LLM platforms and their model pools.
Each platform's Top-5 models per task domain.
"""
from freerouter.core.models import Platform, PlatformModel, TaskType

# ─── NVIDIA Build ───────────────────────────────────────────────────────────
NVIDIA_BUILD = Platform(
    name="nvidia_build",
    display_name="NVIDIA Build",
    api_base="https://integrate.api.nvidia.com/v1",
    probe_model="meta/llama-3.1-8b-instruct",
    rpm_limit=40,
    tags=["most-models", "flagship-quality", "1M-context"],
    models=[
        # Coding
        PlatformModel("nvidia/qwen3-coder-480b-a35b-instruct", 262144, 1,
                      [TaskType.CODE_GENERATION], "Global #1 coding model"),
        PlatformModel("nvidia/devstral-2-123b", 262144, 2,
                      [TaskType.CODE_GENERATION], "Best open-source coding"),
        PlatformModel("nvidia/llama-3.1-405b-instruct", 131072, 3,
                      [TaskType.CODE_GENERATION, TaskType.GENERAL], "Flagship base"),
        PlatformModel("nvidia/deepseek-coder-v2-instruct", 131072, 4,
                      [TaskType.CODE_GENERATION], "Algorithm specialist"),
        PlatformModel("nvidia/codellama-70b-instruct", 102400, 5,
                      [TaskType.CODE_COMPLETION], "Fast completion"),
        # Reasoning
        PlatformModel("nvidia/kimi-k2-thinking", 262144, 1,
                      [TaskType.REASONING], "INT4 quantized reasoning"),
        PlatformModel("nvidia/deepseek-r1-32b", 65536, 2,
                      [TaskType.REASONING], "Distilled reasoning"),
        PlatformModel("nvidia/qwen3.5-397b-a17b", 200000, 3,
                      [TaskType.REASONING, TaskType.GENERAL], "Flagship general"),
        # Long context
        PlatformModel("nvidia/llama-3.1-nemotron-nano-8b-v1", 1048576, 1,
                      [TaskType.LONG_CONTEXT], "1M context — world record"),
        # General / Agentic
        PlatformModel("nvidia/glm5", 200000, 2,
                      [TaskType.AGENTIC, TaskType.GENERAL], "GLM-5 flagship"),
        PlatformModel("nvidia/kimi-k2.5", 200000, 3,
                      [TaskType.AGENTIC], "Multimodal reasoning"),
        # Lightweight
        PlatformModel("meta/llama-3.1-8b-instruct", 131072, 5,
                      [TaskType.CODE_COMPLETION, TaskType.GENERAL], "Fast probe model"),
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

# ─── SambaNova ───────────────────────────────────────────────────────────────
SAMBANOVA = Platform(
    name="sambanova",
    display_name="SambaNova",
    api_base="https://api.sambanova.ai/v1",
    probe_model="Meta-Llama-3.2-3B-Instruct",
    rpm_limit=10,
    tags=["405B-fullprecision", "rdu-chip", "quality-backup"],
    models=[
        PlatformModel("Meta-Llama-3.1-405B-Instruct", 131072, 1,
                      [TaskType.CODE_GENERATION, TaskType.REASONING],
                      "Full-precision 405B · 132 t/s"),
        PlatformModel("DeepSeek-V3-0324", 131072, 2,
                      [TaskType.CODE_GENERATION, TaskType.GENERAL],
                      "DeepSeek V3 latest"),
        PlatformModel("Meta-Llama-3.3-70B-Instruct", 131072, 3,
                      [TaskType.GENERAL],
                      "70B quality backup"),
        PlatformModel("Qwen2.5-72B-Instruct", 131072, 4,
                      [TaskType.GENERAL],
                      "Chinese-optimized 72B"),
        PlatformModel("Qwen2.5-Coder-32B-Instruct", 131072, 5,
                      [TaskType.CODE_GENERATION],
                      "Coder 32B"),
        PlatformModel("Meta-Llama-3.2-3B-Instruct", 131072, 6,
                      [TaskType.GENERAL],
                      "Tiny probe model"),
    ]
)

# ─── All platforms ordered by routing priority ───────────────────────────────
ALL_PLATFORMS: list[Platform] = [NVIDIA_BUILD, GROQ, CEREBRAS, SAMBANOVA]

PLATFORM_MAP: dict[str, Platform] = {p.name: p for p in ALL_PLATFORMS}


def get_platform(name: str) -> Platform:
    if name not in PLATFORM_MAP:
        raise KeyError(f"Unknown platform: {name}. Available: {list(PLATFORM_MAP)}")
    return PLATFORM_MAP[name]
