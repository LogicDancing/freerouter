# FreeRouter — Free LLM Forever

> Auto-route across NVIDIA Build, Groq, Cerebras & SambaNova.  
> Always find the fastest, least-congested free model. Zero manual switching.

```
pip install freerouter
freerouter init        # enter your free API keys (no credit card needed)
freerouter setup-tools # configure opencode / Cursor / Continue / Cline
freerouter start       # proxy running at localhost:8080/v1
```

That's it. Your tools now automatically use the fastest free LLM available.

---

## Why FreeRouter?

You're using opencode at 2am, rushing a demo. The flagship model is queuing.  
NVIDIA Build has 100+ free models. Groq hits 300 tokens/s. Cerebras gives you 1M tokens/day.  
But your tool only knows about one.

**FreeRouter is the missing glue layer.**

---

## Supported Platforms (all free, no credit card)

| Platform | Speed | Free Quota | Best For |
|----------|-------|------------|----------|
| [NVIDIA Build](https://build.nvidia.com) | ~100 t/s | 40 RPM, monthly | Flagship models (Qwen3 Coder 480B, 1M context) |
| [Groq](https://console.groq.com) | **300+ t/s** | 14,400 req/day | Code completion, real-time interaction |
| [Cerebras](https://cloud.cerebras.ai) | **2,500 t/s** | **1M tokens/day** | Agentic workflows, multi-step tasks |
| [SambaNova](https://cloud.sambanova.ai) | 132 t/s | ~100 req/day | 405B full-precision quality backup |

---

## How It Works

```
Your tool  →  localhost:8080/v1  →  FreeRouter
                                        ↓
                              detect task type
                                        ↓
                         probe all platforms (TTFT)
                                        ↓
                         route to fastest available
                                        ↓
                    NVIDIA Build / Groq / Cerebras / SambaNova
```

FreeRouter probes all platforms every 30 seconds, measuring TTFT (time-to-first-token).  
When your tool makes a request, it instantly routes to the fastest, non-congested option.

---

## Routing Logic

| Task | Primary | Fallback |
|------|---------|----------|
| Code completion | Cerebras → Groq | NVIDIA Build |
| Code generation | NVIDIA Build (Qwen3 Coder 480B) | SambaNova → Groq |
| Agentic/multi-step | Cerebras (1M token budget) | Groq → NVIDIA |
| Reasoning/math | NVIDIA Build (Kimi K2) | Groq QwQ → SambaNova |
| Long context (>128K) | NVIDIA Build (up to 1M) | — |
| General | Groq → Cerebras | NVIDIA Build |

---

## Tool Integration

After `freerouter setup-tools`, each tool is configured automatically:

**opencode** (`~/.config/opencode/opencode.json`):
```json
{ "openai": { "baseURL": "http://localhost:8080/v1", "model": "auto" } }
```

**Cursor**, **Continue**, **Cline** — same pattern, auto-detected paths.

---

## CLI Reference

```bash
freerouter init          # interactive setup, enter API keys
freerouter setup-tools   # configure all AI coding tools
freerouter start         # start proxy + live dashboard
freerouter probe         # one-shot latency check
freerouter probe --watch # live latency monitor
freerouter status        # check if proxy is running
```

---

## License

MIT — Free LLM Forever.
