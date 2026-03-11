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
        # Qwen 系列
        PlatformModel("qwen/qwen3-coder-480b-a35b-instruct", 262144, 1, [TaskType.CODE_GENERATION], "Qwen3 Coder 480B"),
        PlatformModel("qwen/qwen2.5-coder-32b-instruct", 131072, 2, [TaskType.CODE_GENERATION], "Qwen Coder 32B"),
        PlatformModel("qwen/qwen3.5-397b-a17b", 200000, 3, [TaskType.GENERAL, TaskType.REASONING], "Qwen3.5 397B"),
        PlatformModel("qwen/qwq-32b", 131072, 4, [TaskType.REASONING], "QwQ 推理模型"),
        # DeepSeek 系列
        PlatformModel("deepseek-ai/deepseek-v3.1", 131072, 1, [TaskType.CODE_GENERATION, TaskType.GENERAL], "DeepSeek V3.1"),
        PlatformModel("deepseek-ai/deepseek-v3.2", 131072, 2, [TaskType.CODE_GENERATION], "DeepSeek V3.2"),
        PlatformModel("deepseek-ai/deepseek-r1-distill-qwen-32b", 131072, 3, [TaskType.REASONING], "DeepSeek R1"),
        # Coder 系列
        PlatformModel("bigcode/starcoder2-15b", 16384, 1, [TaskType.CODE_GENERATION], "StarCoder2 15B"),
        PlatformModel("deepseek-ai/deepseek-coder-6.7b-instruct", 16384, 2, [TaskType.CODE_COMPLETION], "DeepSeek Coder"),
        # 长上下文
        PlatformModel("nvidia/llama-3.1-nemotron-nano-8b-v1", 1048576, 1, [TaskType.LONG_CONTEXT], "1M 上下文"),
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
        PlatformModel("qwen-qwq-32b", 131072, 3, [TaskType.REASONING], "QwQ 推理"),
        PlatformModel("deepseek-r1-distill-llama-70b", 131072, 4, [TaskType.REASONING], "DeepSeek R1"),
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
        PlatformModel("llama3.1-8b", 131072, 1, [TaskType.CODE_COMPLETION, TaskType.GENERAL], "1,800 t/s"),
        PlatformModel("llama3.3-70b", 131072, 2, [TaskType.CODE_GENERATION, TaskType.AGENTIC], "Llama 3.3 70B"),
        PlatformModel("llama-4-scout-17b-16e", 131072, 3, [TaskType.GENERAL, TaskType.AGENTIC], "Llama 4 Scout"),
        PlatformModel("qwen-3-32b", 131072, 4, [TaskType.CODE_GENERATION, TaskType.REASONING], "Qwen3 32B"),
    ]
)

# ─── SambaNova ───────────────────────────────────────────────────────────────
SAMBANOVA = Platform(
    name="sambanova",
    display_name="SambaNova",
    api_base="https://api.sambanova.ai/v1",
    probe_model="Meta-Llama-3.1-8B-Instruct",  # 1440 RPM 免费层
    rpm_limit=1440,
    tags=["free-tier", "rdu-chip", "288k-requests/day"],
    models=[
        # 高配额模型 (1440 RPM / 288000 RPD)
        PlatformModel("Meta-Llama-3.1-8B-Instruct", 131072, 1, [TaskType.CODE_COMPLETION, TaskType.GENERAL], "288K req/day · 极速"),
        # 中配额模型 (240 RPM / 48000 RPD)
        PlatformModel("Meta-Llama-3.3-70B-Instruct", 131072, 2, [TaskType.GENERAL, TaskType.CODE_GENERATION], "48K req/day · 70B"),
        PlatformModel("DeepSeek-R1-Distill-Llama-70B", 131072, 3, [TaskType.REASONING], "48K req/day · R1"),
        # 低配额模型 (60 RPM / 12000 RPD)
        PlatformModel("DeepSeek-V3.1", 131072, 4, [TaskType.CODE_GENERATION, TaskType.GENERAL], "12K req/day · V3.1"),
        PlatformModel("DeepSeek-R1-0528", 131072, 5, [TaskType.REASONING], "12K req/day · R1"),
        PlatformModel("Qwen3-32B", 131072, 6, [TaskType.CODE_GENERATION], "Qwen3 32B"),
        PlatformModel("Llama-4-Maverick-17B-128E-Instruct", 131072, 7, [TaskType.GENERAL], "Llama 4"),
    ]
)

# ─── All platforms ───────────────────────────────────────────────────────────
ALL_PLATFORMS: list[Platform] = [NVIDIA_BUILD, GROQ, CEREBRAS, SAMBANOVA]
PLATFORM_MAP: dict[str, Platform] = {p.name: p for p in ALL_PLATFORMS}


def get_platform(name: str) -> Platform:
    if name not in PLATFORM_MAP:
        raise KeyError(f"Unknown platform: {name}. Available: {list(PLATFORM_MAP)}")
    return PLATFORM_MAP[name]
