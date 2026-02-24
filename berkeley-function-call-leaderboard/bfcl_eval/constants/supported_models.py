# -----------------------------------------------------------------------------
# Supported Model Index  •  Convenience helper
#
# The canonical model-config mapping lives in `model_config.py` and is ~2000
# lines long. Navigating that file just to see whether a model key exists was
# getting painful, so this lightweight companion keeps **only** the keys in a
# flat list so you can:
#
#   •  skim the supported models at a glance;
#   •  hit ⌘/Ctrl-F and jump straight to the one you need;
#   •  import the list in quick scripts/tests without hauling in the whole
#      config (e.g. `if model_name in SUPPORTED_MODELS:`).
# -----------------------------------------------------------------------------

SUPPORTED_MODELS = [
    "gorilla-openfunctions-v2",
    "DeepSeek-V3.2-Exp",
    "DeepSeek-V3.2-Exp-FC",
    "DeepSeek-V3.2-Exp-thinking",
    "gpt-5.2-2025-12-11-FC",
    "gpt-5.2-2025-12-11",
    "gpt-5-mini-2025-08-07-FC",
    "gpt-5-mini-2025-08-07",
    "gpt-5-nano-2025-08-07-FC",
    "gpt-5-nano-2025-08-07",
    "gpt-4.1-2025-04-14-FC",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14-FC",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano-2025-04-14-FC",
    "gpt-4.1-nano-2025-04-14",
    "gpt-4o-2024-11-20",
    "gpt-4o-2024-11-20-FC",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-mini-2024-07-18-FC",
    "o3-2025-04-16",
    "o3-2025-04-16-FC",
    "o4-mini-2025-04-16",
    "o4-mini-2025-04-16-FC",
    "claude-opus-4-5-20251101",
    "claude-opus-4-5-20251101-FC",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-5-20250929-FC",
    "claude-haiku-4-5-20251001",
    "claude-haiku-4-5-20251001-FC",
    "nova-pro-v1.0",
    "nova-2-lite-v1.0",
    "nova-micro-v1.0",
    "open-mistral-nemo-2407",
    "open-mistral-nemo-2407-FC",
    "mistral-large-2411",
    "mistral-large-2411-FC",
    "mistral-small-2506",
    "mistral-small-2506-FC",
    "mistral-medium-2505",
    "mistral-medium-2505-FC",
    "firefunction-v2-FC",
    "gemini-2.5-flash-lite-FC",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-FC",
    "gemini-2.5-flash",
    "gemini-3-pro-preview-FC",
    "gemini-3-pro-preview",
    "meetkai/functionary-small-v3.1-FC",
    "meetkai/functionary-medium-v3.1-FC",
    "command-r7b-12-2024-FC",
    "command-a-03-2025-FC",
    "command-a-reasoning-08-2025-FC",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "nvidia/nemotron-4-340b-instruct",
    "BitAgent/GoGoAgent",
    "palmyra-x-004",
    "grok-4-0709-FC",
    "grok-4-0709",
    "grok-4-1-fast-reasoning-FC",
    "grok-4-1-fast-non-reasoning-FC",
    "qwen3-0.6b-FC",
    "qwen3-0.6b",
    "qwen3-1.7b-FC",
    "qwen3-1.7b",
    "qwen3-4b-FC",
    "qwen3-4b",
    "qwen3-8b-FC",
    "qwen3-8b",
    "qwen3-14b-FC",
    "qwen3-14b",
    "qwen3-32b-FC",
    "qwen3-32b",
    "qwen3-30b-a3b-instruct-2507-FC",
    "qwen3-30b-a3b-instruct-2507",
    "qwen3-235b-a22b-instruct-2507-FC",
    "qwen3-235b-a22b-instruct-2507",
    "qwq-32b-FC",
    "qwq-32b",
    "xiaoming-14B",
    "DM-Cito-8B-v3",
    "DM-Cito-32B-v1",
    "Ling/ling-lite-v1.5",
    "glm-4.6-FC",
    "glm-4.5-FC",
    "glm-4.5-air-FC",
    "kimi-k2-0905-preview-FC",
    "kimi-k2-0905-preview",
    "Nanbeige3.5-Pro-Thinking-FC",
    "deepseek-ai/DeepSeek-R1",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    "google/functiongemma-270m-it-FC",
    "meta-llama/Llama-3.1-8B-Instruct-FC",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct-FC",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct-FC",
    "meta-llama/Llama-3.2-3B-Instruct-FC",
    "meta-llama/Llama-3.3-70B-Instruct-FC",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct-FC",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8-FC",
    "Salesforce/Llama-xLAM-2-70b-fc-r",
    "Salesforce/Llama-xLAM-2-8b-fc-r",
    "Salesforce/xLAM-2-32b-fc-r",
    "Salesforce/xLAM-2-3b-fc-r",
    "Salesforce/xLAM-2-1b-fc-r",
    "mistralai/Ministral-8B-Instruct-2410",
    "microsoft/phi-4",
    "microsoft/Phi-4-mini-instruct",
    "microsoft/Phi-4-mini-instruct-FC",
    "ibm-granite/granite-3.2-8b-instruct",
    "ibm-granite/granite-3.1-8b-instruct",
    "ibm-granite/granite-4.0-350m",
    "ibm-granite/granite-20b-functioncalling",
    "MadeAgents/Hammer2.1-7b",
    "MadeAgents/Hammer2.1-3b",
    "MadeAgents/Hammer2.1-1.5b",
    "MadeAgents/Hammer2.1-0.5b",
    "THUDM/glm-4-9b-chat",
    "Qwen/Qwen3-0.6B-FC",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B-FC",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B-Instruct-2507-FC",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-8B-FC",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B-FC",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B-FC",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FC",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-235B-A22B-Instruct-2507-FC",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "Team-ACE/ToolACE-2-8B",
    "openbmb/MiniCPM3-4B",
    "openbmb/MiniCPM3-4B-FC",
    "watt-ai/watt-tool-8B",
    "watt-ai/watt-tool-70B",
    "ZJared/Haha-7B",
    "speakleash/Bielik-11B-v2.3-Instruct",
    "NovaSky-AI/Sky-T1-32B-Preview",
    "tiiuae/Falcon3-10B-Instruct-FC",
    "tiiuae/Falcon3-7B-Instruct-FC",
    "tiiuae/Falcon3-3B-Instruct-FC",
    "tiiuae/Falcon3-1B-Instruct-FC",
    "uiuc-convai/CoALM-8B",
    "uiuc-convai/CoALM-70B",
    "uiuc-convai/CoALM-405B",
    "katanemo/Arch-Agent-1.5B",
    "katanemo/Arch-Agent-3B",
    "katanemo/Arch-Agent-7B",
    "katanemo/Arch-Agent-32B",
    "BitAgent/BitAgent-8B",
    "BitAgent/BitAgent-Bounty-8B",
    "ThinkAgents/ThinkAgent-1B",
    "phronetic-ai/RZN-T",
    "Nanbeige/Nanbeige4-3B-Thinking-2511",
    "meta-llama/llama-4-maverick-17b-128e-instruct-fp8-novita",
    "meta-llama/llama-4-maverick-17b-128e-instruct-fp8-FC-novita",
    "meta-llama/llama-4-scout-17b-16e-instruct-novita",
    "meta-llama/llama-4-scout-17b-16e-instruct-FC-novita",
    "qwen/qwq-32b-FC-novita",
    "qwen/qwq-32b-novita",
    "qwen3-4b-think-FC",
    "qwen3-4b-nothink-FC",
    "DM-Cito-32B-v1",
]

# Llama-xLAM-2-8b-fc-r LoRA finetune checkpoints
# 2batch-256seq
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-299-merged-2batch-256seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-598-merged-2batch-256seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-897-merged-2batch-256seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1196-merged-2batch-256seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1495-merged-2batch-256seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1794-merged-2batch-256seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2093-merged-2batch-256seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2392-merged-2batch-256seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2691-merged-2batch-256seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2990-merged-2batch-256seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3289-merged-2batch-256seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3588-merged-2batch-256seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3887-merged-2batch-256seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-4186-merged-2batch-256seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-4485-merged-2batch-256seq")
# 1batch-768seq
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-598-merged-1batch-768seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1196-merged-1batch-768seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1794-merged-1batch-768seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2392-merged-1batch-768seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2990-merged-1batch-768seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3588-merged-1batch-768seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-4186-merged-1batch-768seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-4784-merged-1batch-768seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-5382-merged-1batch-768seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-5385-merged-1batch-768seq")
# 2batch-1024seq
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-478-merged-2batch-1024seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-956-merged-2batch-1024seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1434-merged-2batch-1024seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1912-merged-2batch-1024seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2390-merged-2batch-1024seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2868-merged-2batch-1024seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3346-merged-2batch-1024seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3824-merged-2batch-1024seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-4302-merged-2batch-1024seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-4305-merged-2batch-1024seq")
# 1batch-2048seq
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-957-merged-1batch-2048seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1914-merged-1batch-2048seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2871-merged-1batch-2048seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3828-merged-1batch-2048seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-4785-merged-1batch-2048seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-5742-merged-1batch-2048seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-6699-merged-1batch-2048seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-7656-merged-1batch-2048seq")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-8613-merged-1batch-2048seq")
# 2batch-1024seq-apigen
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-284-merged-2batch-1024seq-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-568-merged-2batch-1024seq-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-852-merged-2batch-1024seq-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1136-merged-2batch-1024seq-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1420-merged-2batch-1024seq-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1704-merged-2batch-1024seq-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1988-merged-2batch-1024seq-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2272-merged-2batch-1024seq-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2556-merged-2batch-1024seq-apigen")
# 1batch-2048seq-apigen
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-568-merged-1batch-2048seq-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1136-merged-1batch-2048seq-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1704-merged-1batch-2048seq-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2272-merged-1batch-2048seq-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2840-merged-1batch-2048seq-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3408-merged-1batch-2048seq-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3976-merged-1batch-2048seq-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-4544-merged-1batch-2048seq-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-5112-merged-1batch-2048seq-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-5115-merged-1batch-2048seq-apigen")
# 1batch-2560seq-1epoch-apigen
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-568-merged-1batch-2560seq-1epoch-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1136-merged-1batch-2560seq-1epoch-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1704-merged-1batch-2560seq-1epoch-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1705-merged-1batch-2560seq-1epoch-apigen")
# 1batch-2560seq-3epoch-apigen
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-568-merged-1batch-2560seq-3epoch-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1136-merged-1batch-2560seq-3epoch-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1704-merged-1batch-2560seq-3epoch-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2272-merged-1batch-2560seq-3epoch-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2840-merged-1batch-2560seq-3epoch-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3408-merged-1batch-2560seq-3epoch-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3976-merged-1batch-2560seq-3epoch-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-4544-merged-1batch-2560seq-3epoch-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-5112-merged-1batch-2560seq-3epoch-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-5115-merged-1batch-2560seq-3epoch-apigen")
# 2batch-1024seq-3epoch-apigen
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-263-merged-2batch-1024seq-3epoch-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-526-merged-2batch-1024seq-3epoch-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-789-merged-2batch-1024seq-3epoch-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1052-merged-2batch-1024seq-3epoch-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1315-merged-2batch-1024seq-3epoch-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1578-merged-2batch-1024seq-3epoch-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1841-merged-2batch-1024seq-3epoch-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2104-merged-2batch-1024seq-3epoch-apigen")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2364-merged-2batch-1024seq-3epoch-apigen")
# 1batch-2560seq-3epoch-IF
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-415-merged-1batch-2560seq-3epoch-IF")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-830-merged-1batch-2560seq-3epoch-IF")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1245-merged-1batch-2560seq-3epoch-IF")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1660-merged-1batch-2560seq-3epoch-IF")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2075-merged-1batch-2560seq-3epoch-IF")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2490-merged-1batch-2560seq-3epoch-IF")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2905-merged-1batch-2560seq-3epoch-IF")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3320-merged-1batch-2560seq-3epoch-IF")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3735-merged-1batch-2560seq-3epoch-IF")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3738-merged-1batch-2560seq-3epoch-IF")
# 1batch-2560seq-3epoch-FC
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-957-merged-1batch-2560seq-3epoch-FC")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1914-merged-1batch-2560seq-3epoch-FC")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2871-merged-1batch-2560seq-3epoch-FC")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3828-merged-1batch-2560seq-3epoch-FC")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-4785-merged-1batch-2560seq-3epoch-FC")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-5742-merged-1batch-2560seq-3epoch-FC")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-6699-merged-1batch-2560seq-3epoch-FC")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-7656-merged-1batch-2560seq-3epoch-FC")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-8613-merged-1batch-2560seq-3epoch-FC")
# 1batch-2560seq-3epoch-ALL
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-984-merged-1batch-2560seq-3epoch-ALL")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1968-merged-1batch-2560seq-3epoch-ALL")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2952-merged-1batch-2560seq-3epoch-ALL")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3936-merged-1batch-2560seq-3epoch-ALL")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-4920-merged-1batch-2560seq-3epoch-ALL")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-5904-merged-1batch-2560seq-3epoch-ALL")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-6888-merged-1batch-2560seq-3epoch-ALL")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-7872-merged-1batch-2560seq-3epoch-ALL")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-8853-merged-1batch-2560seq-3epoch-ALL")
# 1batch-2560seq-3epoch-apigen-nomlp
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-568-merged-1batch-2560seq-3epoch-apigen-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1136-merged-1batch-2560seq-3epoch-apigen-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1704-merged-1batch-2560seq-3epoch-apigen-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2272-merged-1batch-2560seq-3epoch-apigen-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2840-merged-1batch-2560seq-3epoch-apigen-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3408-merged-1batch-2560seq-3epoch-apigen-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3976-merged-1batch-2560seq-3epoch-apigen-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-4544-merged-1batch-2560seq-3epoch-apigen-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-5112-merged-1batch-2560seq-3epoch-apigen-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-5115-merged-1batch-2560seq-3epoch-apigen-nomlp")
# 2batch-1024seq-3epoch-apigen-nomlp
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-263-merged-2batch-1024seq-3epoch-apigen-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-526-merged-2batch-1024seq-3epoch-apigen-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-789-merged-2batch-1024seq-3epoch-apigen-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1052-merged-2batch-1024seq-3epoch-apigen-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1315-merged-2batch-1024seq-3epoch-apigen-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1578-merged-2batch-1024seq-3epoch-apigen-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1841-merged-2batch-1024seq-3epoch-apigen-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2104-merged-2batch-1024seq-3epoch-apigen-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2364-merged-2batch-1024seq-3epoch-apigen-nomlp")
# 1batch-2560seq-3epoch-multi3000-nomlp
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-693-merged-1batch-2560seq-3epoch-multi3000-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1386-merged-1batch-2560seq-3epoch-multi3000-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2079-merged-1batch-2560seq-3epoch-multi3000-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2772-merged-1batch-2560seq-3epoch-multi3000-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3465-merged-1batch-2560seq-3epoch-multi3000-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-4158-merged-1batch-2560seq-3epoch-multi3000-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-4851-merged-1batch-2560seq-3epoch-multi3000-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-5544-merged-1batch-2560seq-3epoch-multi3000-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-6237-merged-1batch-2560seq-3epoch-multi3000-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-6240-merged-1batch-2560seq-3epoch-multi3000-nomlp")
# 1batch-2560seq-3epoch-multi2000-nomlp
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-652-merged-1batch-2560seq-3epoch-multi2000-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1304-merged-1batch-2560seq-3epoch-multi2000-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1956-merged-1batch-2560seq-3epoch-multi2000-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2608-merged-1batch-2560seq-3epoch-multi2000-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3260-merged-1batch-2560seq-3epoch-multi2000-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3912-merged-1batch-2560seq-3epoch-multi2000-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-4564-merged-1batch-2560seq-3epoch-multi2000-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-5216-merged-1batch-2560seq-3epoch-multi2000-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-5865-merged-1batch-2560seq-3epoch-multi2000-nomlp")
# 1batch-2560seq-3epoch-multi4000-nomlp
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-735-merged-1batch-2560seq-3epoch-multi4000-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1470-merged-1batch-2560seq-3epoch-multi4000-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2205-merged-1batch-2560seq-3epoch-multi4000-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2940-merged-1batch-2560seq-3epoch-multi4000-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3675-merged-1batch-2560seq-3epoch-multi4000-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-4410-merged-1batch-2560seq-3epoch-multi4000-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-5145-merged-1batch-2560seq-3epoch-multi4000-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-5880-merged-1batch-2560seq-3epoch-multi4000-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-6615-merged-1batch-2560seq-3epoch-multi4000-nomlp")
# 1batch-2560seq-3epoch-multi-all-nomlp
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-849-merged-1batch-2560seq-3epoch-multi-all-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1698-merged-1batch-2560seq-3epoch-multi-all-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2547-merged-1batch-2560seq-3epoch-multi-all-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3396-merged-1batch-2560seq-3epoch-multi-all-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-4245-merged-1batch-2560seq-3epoch-multi-all-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-5094-merged-1batch-2560seq-3epoch-multi-all-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-5943-merged-1batch-2560seq-3epoch-multi-all-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-6792-merged-1batch-2560seq-3epoch-multi-all-nomlp")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-7638-merged-1batch-2560seq-3epoch-multi-all-nomlp")
# 1batch-2560seq-3epoch-eng-multi-all-only
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-171-merged-1batch-2560seq-3epoch-eng-multi-all-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-342-merged-1batch-2560seq-3epoch-eng-multi-all-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-513-merged-1batch-2560seq-3epoch-eng-multi-all-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-684-merged-1batch-2560seq-3epoch-eng-multi-all-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-855-merged-1batch-2560seq-3epoch-eng-multi-all-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1026-merged-1batch-2560seq-3epoch-eng-multi-all-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1197-merged-1batch-2560seq-3epoch-eng-multi-all-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1368-merged-1batch-2560seq-3epoch-eng-multi-all-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1539-merged-1batch-2560seq-3epoch-eng-multi-all-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1542-merged-1batch-2560seq-3epoch-eng-multi-all-only")
# 1batch-2560seq-5epoch-zhtw-multi-3000-only
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-62-merged-1batch-2560seq-5epoch-zhtw-multi-3000-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-124-merged-1batch-2560seq-5epoch-zhtw-multi-3000-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-186-merged-1batch-2560seq-5epoch-zhtw-multi-3000-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-248-merged-1batch-2560seq-5epoch-zhtw-multi-3000-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-310-merged-1batch-2560seq-5epoch-zhtw-multi-3000-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-372-merged-1batch-2560seq-5epoch-zhtw-multi-3000-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-434-merged-1batch-2560seq-5epoch-zhtw-multi-3000-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-496-merged-1batch-2560seq-5epoch-zhtw-multi-3000-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-558-merged-1batch-2560seq-5epoch-zhtw-multi-3000-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-620-merged-1batch-2560seq-5epoch-zhtw-multi-3000-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-682-merged-1batch-2560seq-5epoch-zhtw-multi-3000-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-744-merged-1batch-2560seq-5epoch-zhtw-multi-3000-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-806-merged-1batch-2560seq-5epoch-zhtw-multi-3000-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-868-merged-1batch-2560seq-5epoch-zhtw-multi-3000-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-930-merged-1batch-2560seq-5epoch-zhtw-multi-3000-only")
# 1batch-2560seq-3epoch-accumulate32-multi-1500
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-173-merged-1batch-2560seq-3epoch-accumulate32-multi-1500")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-346-merged-1batch-2560seq-3epoch-accumulate32-multi-1500")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-519-merged-1batch-2560seq-3epoch-accumulate32-multi-1500")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-692-merged-1batch-2560seq-3epoch-accumulate32-multi-1500")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-865-merged-1batch-2560seq-3epoch-accumulate32-multi-1500")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1038-merged-1batch-2560seq-3epoch-accumulate32-multi-1500")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1211-merged-1batch-2560seq-3epoch-accumulate32-multi-1500")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1384-merged-1batch-2560seq-3epoch-accumulate32-multi-1500")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1557-merged-1batch-2560seq-3epoch-accumulate32-multi-1500")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1560-merged-1batch-2560seq-3epoch-accumulate32-multi-1500")
# 1batch-2560seq-3epoch-multi-mix
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-205-merged-1batch-2560seq-3epoch-multi-mix")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-410-merged-1batch-2560seq-3epoch-multi-mix")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-615-merged-1batch-2560seq-3epoch-multi-mix")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-820-merged-1batch-2560seq-3epoch-multi-mix")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1025-merged-1batch-2560seq-3epoch-multi-mix")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1230-merged-1batch-2560seq-3epoch-multi-mix")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1435-merged-1batch-2560seq-3epoch-multi-mix")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1640-merged-1batch-2560seq-3epoch-multi-mix")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1842-merged-1batch-2560seq-3epoch-multi-mix")
#mixture-1 的各類型數量統計：
#Simple           :   1290
#Multiple         :   3863
#Parallel         :   1273
#Parallel-Multiple:   4074
#Revalance  :   3500
#以上來自apigen
#---------------------------------
#multi-zhtw        :   3000(目前使用的量) 
#multi-eng        :   3000(目前使用的量) 
#---------------------------------
# 1batch-2560seq-3epoch-accumulate64-mixture-1
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-102-merged-1batch-2560seq-3epoch-accumulate64-mixture-1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-204-merged-1batch-2560seq-3epoch-accumulate64-mixture-1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-306-merged-1batch-2560seq-3epoch-accumulate64-mixture-1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-408-merged-1batch-2560seq-3epoch-accumulate64-mixture-1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-510-merged-1batch-2560seq-3epoch-accumulate64-mixture-1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-612-merged-1batch-2560seq-3epoch-accumulate64-mixture-1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-714-merged-1batch-2560seq-3epoch-accumulate64-mixture-1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-816-merged-1batch-2560seq-3epoch-accumulate64-mixture-1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-918-merged-1batch-2560seq-3epoch-accumulate64-mixture-1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-921-merged-1batch-2560seq-3epoch-accumulate64-mixture-1")
# 1batch-2560seq-3epoch-accumulate128-mixture-1
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-51-merged-1batch-2560seq-4epoch-accumulate128-mixture-1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-102-merged-1batch-2560seq-4epoch-accumulate128-mixture-1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-153-merged-1batch-2560seq-4epoch-accumulate128-mixture-1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-204-merged-1batch-2560seq-4epoch-accumulate128-mixture-1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-255-merged-1batch-2560seq-4epoch-accumulate128-mixture-1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-306-merged-1batch-2560seq-4epoch-accumulate128-mixture-1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-357-merged-1batch-2560seq-4epoch-accumulate128-mixture-1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-408-merged-1batch-2560seq-4epoch-accumulate128-mixture-1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-459-merged-1batch-2560seq-4epoch-accumulate128-mixture-1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-510-merged-1batch-2560seq-4epoch-accumulate128-mixture-1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-561-merged-1batch-2560seq-4epoch-accumulate128-mixture-1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-612-merged-1batch-2560seq-4epoch-accumulate128-mixture-1")
#mixture-2 的各類型數量統計：
#Simple           :   1290
#Multiple         :   3863
#Parallel         :   1273
#Parallel-Multiple:   4074
#Revalance  :   1500
#以上來自apigen
#---------------------------------
#multi-zhtw        :   2500(目前使用的量) 
#multi-eng        :   2500(目前使用的量) 
#---------------------------------
# 1batch-2560seq-3epoch-accumulate32-mixture-2
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-177-merged-1batch-2560seq-3epoch-accumulate32-mixture-2")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-354-merged-1batch-2560seq-3epoch-accumulate32-mixture-2")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-531-merged-1batch-2560seq-3epoch-accumulate32-mixture-2")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-708-merged-1batch-2560seq-3epoch-accumulate32-mixture-2")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-885-merged-1batch-2560seq-3epoch-accumulate32-mixture-2")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1062-merged-1batch-2560seq-3epoch-accumulate32-mixture-2")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1239-merged-1batch-2560seq-3epoch-accumulate32-mixture-2")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1416-merged-1batch-2560seq-3epoch-accumulate32-mixture-2")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1593-merged-1batch-2560seq-3epoch-accumulate32-mixture-2")
# 1batch-2560seq-3epoch-accumulate64-mixture-2
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-88-merged-1batch-2560seq-3epoch-accumulate64-mixture-2")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-176-merged-1batch-2560seq-3epoch-accumulate64-mixture-2")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-264-merged-1batch-2560seq-3epoch-accumulate64-mixture-2")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-352-merged-1batch-2560seq-3epoch-accumulate64-mixture-2")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-440-merged-1batch-2560seq-3epoch-accumulate64-mixture-2")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-528-merged-1batch-2560seq-3epoch-accumulate64-mixture-2")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-616-merged-1batch-2560seq-3epoch-accumulate64-mixture-2")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-704-merged-1batch-2560seq-3epoch-accumulate64-mixture-2")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-792-merged-1batch-2560seq-3epoch-accumulate64-mixture-2")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-795-merged-1batch-2560seq-3epoch-accumulate64-mixture-2")
# 1epoch-accumulate32-new-multiturn
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-168-merged-1epoch-accumulate32-new-multiturn")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-336-merged-1epoch-accumulate32-new-multiturn")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-504-merged-1epoch-accumulate32-new-multiturn")
# 3epoch-accumulate32-new-multiturn
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-168-merged-3epoch-accumulate32-new-multiturn")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-336-merged-3epoch-accumulate32-new-multiturn")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-504-merged-3epoch-accumulate32-new-multiturn")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-672-merged-3epoch-accumulate32-new-multiturn")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-840-merged-3epoch-accumulate32-new-multiturn")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1008-merged-3epoch-accumulate32-new-multiturn")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1176-merged-3epoch-accumulate32-new-multiturn")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1344-merged-3epoch-accumulate32-new-multiturn")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1512-merged-3epoch-accumulate32-new-multiturn")
# 1epoch-accumulate32-toollist-17K
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-168-merged-1epoch-accumulate32-toollist-17K")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-336-merged-1epoch-accumulate32-toollist-17K")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-504-merged-1epoch-accumulate32-toollist-17K")
# 3epoch-accumulate32-toollist-17K
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-168-merged-3epoch-accumulate32-toollist-17K")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-336-merged-3epoch-accumulate32-toollist-17K")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-504-merged-3epoch-accumulate32-toollist-17K")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-672-merged-3epoch-accumulate32-toollist-17K")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-840-merged-3epoch-accumulate32-toollist-17K")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1008-merged-3epoch-accumulate32-toollist-17K")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1176-merged-3epoch-accumulate32-toollist-17K")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1344-merged-3epoch-accumulate32-toollist-17K")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1512-merged-3epoch-accumulate32-toollist-17K")
# 3epoch-accumulate32-toollist-14K-fixed
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-144-merged-3epoch-accumulate32-toollist-14K-fixed")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-288-merged-3epoch-accumulate32-toollist-14K-fixed")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-432-merged-3epoch-accumulate32-toollist-14K-fixed")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-576-merged-3epoch-accumulate32-toollist-14K-fixed")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-720-merged-3epoch-accumulate32-toollist-14K-fixed")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-864-merged-3epoch-accumulate32-toollist-14K-fixed")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1008-merged-3epoch-accumulate32-toollist-14K-fixed")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1152-merged-3epoch-accumulate32-toollist-14K-fixed")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1293-merged-3epoch-accumulate32-toollist-14K-fixed")
# 3epoch-accumulate32-toollist-2K-mix
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-135-merged-3epoch-accumulate32-toollist-2K-mix")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-270-merged-3epoch-accumulate32-toollist-2K-mix")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-405-merged-3epoch-accumulate32-toollist-2K-mix")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-540-merged-3epoch-accumulate32-toollist-2K-mix")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-675-merged-3epoch-accumulate32-toollist-2K-mix")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-810-merged-3epoch-accumulate32-toollist-2K-mix")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-945-merged-3epoch-accumulate32-toollist-2K-mix")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1080-merged-3epoch-accumulate32-toollist-2K-mix")
# 3epoch-accumulate32-toollist-2K-only
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-20-merged-3epoch-accumulate32-toollist-2K-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-40-merged-3epoch-accumulate32-toollist-2K-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-60-merged-3epoch-accumulate32-toollist-2K-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-80-merged-3epoch-accumulate32-toollist-2K-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-100-merged-3epoch-accumulate32-toollist-2K-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-120-merged-3epoch-accumulate32-toollist-2K-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-140-merged-3epoch-accumulate32-toollist-2K-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-160-merged-3epoch-accumulate32-toollist-2K-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-177-merged-3epoch-accumulate32-toollist-2K-only")
# 3epoch-accumulate32-toollist-2K-only-new
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-20-merged-3epoch-accumulate32-toollist-2K-only-new")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-40-merged-3epoch-accumulate32-toollist-2K-only-new")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-60-merged-3epoch-accumulate32-toollist-2K-only-new")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-80-merged-3epoch-accumulate32-toollist-2K-only-new")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-100-merged-3epoch-accumulate32-toollist-2K-only-new")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-120-merged-3epoch-accumulate32-toollist-2K-only-new")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-140-merged-3epoch-accumulate32-toollist-2K-only-new")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-160-merged-3epoch-accumulate32-toollist-2K-only-new")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-177-merged-3epoch-accumulate32-toollist-2K-only-new")
# 3epoch-accumulate32-funreason-8K
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-85-merged-3epoch-accumulate32-funreason-8K")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-170-merged-3epoch-accumulate32-funreason-8K")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-255-merged-3epoch-accumulate32-funreason-8K")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-340-merged-3epoch-accumulate32-funreason-8K")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-425-merged-3epoch-accumulate32-funreason-8K")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-510-merged-3epoch-accumulate32-funreason-8K")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-595-merged-3epoch-accumulate32-funreason-8K")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-680-merged-3epoch-accumulate32-funreason-8K")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-765-merged-3epoch-accumulate32-funreason-8K")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-768-merged-3epoch-accumulate32-funreason-8K")
# 1epoch-funreason-clean-toollist
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-47-merged-1epoch-funreason-clean-toollist")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-94-merged-1epoch-funreason-clean-toollist")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-140-merged-1epoch-funreason-clean-toollist")
# 3epoch-funreason-clean
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-47-merged-3epoch-funreason-clean")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-94-merged-3epoch-funreason-clean")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-141-merged-3epoch-funreason-clean")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-188-merged-3epoch-funreason-clean")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-235-merged-3epoch-funreason-clean")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-282-merged-3epoch-funreason-clean")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-329-merged-3epoch-funreason-clean")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-376-merged-3epoch-funreason-clean")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-420-merged-3epoch-funreason-clean")
# 3epoch-funreason-clean-toollist
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-47-merged-3epoch-funreason-clean-toollist")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-94-merged-3epoch-funreason-clean-toollist")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-141-merged-3epoch-funreason-clean-toollist")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-188-merged-3epoch-funreason-clean-toollist")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-235-merged-3epoch-funreason-clean-toollist")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-282-merged-3epoch-funreason-clean-toollist")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-329-merged-3epoch-funreason-clean-toollist")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-376-merged-3epoch-funreason-clean-toollist")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-420-merged-3epoch-funreason-clean-toollist")
# 3epoch-funreason-pythonic-v1
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-21-merged-3epoch-funreason-pythonic-v1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-42-merged-3epoch-funreason-pythonic-v1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-63-merged-3epoch-funreason-pythonic-v1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-84-merged-3epoch-funreason-pythonic-v1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-105-merged-3epoch-funreason-pythonic-v1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-126-merged-3epoch-funreason-pythonic-v1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-147-merged-3epoch-funreason-pythonic-v1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-168-merged-3epoch-funreason-pythonic-v1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-186-merged-3epoch-funreason-pythonic-v1")
# 3epoch-simple-long-text
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-12-merged-3epoch-simple-long-text")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-60-merged-3epoch-simple-long-text")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-111-merged-3epoch-simple-long-text")
# 3epoch-simple-miss-func
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-12-merged-3epoch-simple-miss-func")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-60-merged-3epoch-simple-miss-func")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-111-merged-3epoch-simple-miss-func")
# 3epoch-simple-miss-para
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-12-merged-3epoch-simple-miss-para")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-60-merged-3epoch-simple-miss-para")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-111-merged-3epoch-simple-miss-para")
# 3epoch-Funreason-BFCL-sys-prompt
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-47-merged-3epoch-Funreason-BFCL-sys-prompt")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-235-merged-3epoch-Funreason-BFCL-sys-prompt")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-420-merged-3epoch-Funreason-BFCL-sys-prompt")
# 3epoch-all-multi-1-1
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-213-merged-3epoch-all-multi-1-1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1065-merged-3epoch-all-multi-1-1")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1917-merged-3epoch-all-multi-1-1")
# 3epoch-all-multi-1-06
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-192-merged-3epoch-all-multi-1-06")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-768-merged-3epoch-all-multi-1-06")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1536-merged-3epoch-all-multi-1-06")
# 3epoch-all-multi-zhtw-only
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-163-merged-3epoch-all-multi-zhtw-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-652-merged-3epoch-all-multi-zhtw-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1304-merged-3epoch-all-multi-zhtw-only")
# 3epoch-zhtw-simple-only
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-47-merged-3epoch-zhtw-simple-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-235-merged-3epoch-zhtw-simple-only")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-420-merged-3epoch-zhtw-simple-only")
# 3epoch-pythonic-v2
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-15-merged-3epoch-pythonic-v2")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-75-merged-3epoch-pythonic-v2")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-138-merged-3epoch-pythonic-v2")
# 3epoch-simple-fix-tool-role
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-49-merged-3epoch-simple-fix-tool-role")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-245-merged-3epoch-simple-fix-tool-role")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-438-merged-3epoch-simple-fix-tool-role")
# 1epoch-apigenmt
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-172-merged-1epoch-apigenmt")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-344-merged-1epoch-apigenmt")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-516-merged-1epoch-apigenmt")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-688-merged-1epoch-apigenmt")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-860-merged-1epoch-apigenmt")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1032-merged-1epoch-apigenmt")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1204-merged-1epoch-apigenmt")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1376-merged-1epoch-apigenmt")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1377-merged-1epoch-apigenmt")
# 1epoch_apigenmt_loss_3e6
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-19-merged-1epoch-apigenmt-loss-3e6")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-95-merged-1epoch-apigenmt-loss-3e6")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-156-merged-1epoch-apigenmt-loss-3e6")
# 1epoch_bfcl_turn2e6_fix_length_fix_t_split
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-8-merged-1epoch-bfcl-turn2e6-fix-length-fix-t-split")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-104-merged-1epoch-bfcl-turn2e6-fix-length-fix-t-split")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-200-merged-1epoch-bfcl-turn2e6-fix-length-fix-t-split")
# 3epoch_bfcl5e7_fix_length_fixt
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3-merged-3epoch-bfcl5e7-fix-length-fixt")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-27-merged-3epoch-bfcl5e7-fix-length-fixt")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-57-merged-3epoch-bfcl5e7-fix-length-fixt")
# 3epoch-2bfcl80-fix-length-fixt5e-7-setweight03
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3-merged-3epoch-2bfcl80-fix-length-fixt5e-7-setweight03")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-27-merged-3epoch-2bfcl80-fix-length-fixt5e-7-setweight03")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-57-merged-3epoch-2bfcl80-fix-length-fixt5e-7-setweight03")
# 3epoch-2bfcl80-fix-length-fixt5e-7-setweight07
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3-merged-3epoch-2bfcl80-fix-length-fixt5e-7-setweight07")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-27-merged-3epoch-2bfcl80-fix-length-fixt5e-7-setweight07")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-57-merged-3epoch-2bfcl80-fix-length-fixt5e-7-setweight07")
# 1epoch-pythonic-non-split
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-7-merged-1epoch-pythonic-non-split")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-35-merged-1epoch-pythonic-non-split")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-60-merged-1epoch-pythonic-non-split")
# 1epoch-apigenmt-label-all-assistant5e-6-setweight07-fix-tool
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-19-merged-1epoch-apigenmt-label-all-assistant5e-6-setweight07-fix-tool")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-95-merged-1epoch-apigenmt-label-all-assistant5e-6-setweight07-fix-tool")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-156-merged-1epoch-apigenmt-label-all-assistant5e-6-setweight07-fix-tool")
# 1epoch-apigenmt-label-all-assistant5e-6-fixtool
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-19-merged-1epoch-apigenmt-label-all-assistant5e-6-fixtool")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-95-merged-1epoch-apigenmt-label-all-assistant5e-6-fixtool")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-156-merged-1epoch-apigenmt-label-all-assistant5e-6-fixtool")
# 1epoch-pythonic-bfcl-like-v1-prototype
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-4-merged-1epoch-pythonic-bfcl-like-v1-prototype")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-28-merged-1epoch-pythonic-bfcl-like-v1-prototype")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-60-merged-1epoch-pythonic-bfcl-like-v1-prototype")
# 1epoch-pythonic-v2-production
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-42-merged-1epoch-pythonic-v2-production")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-210-merged-1epoch-pythonic-v2-production")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-338-merged-1epoch-pythonic-v2-production")
# 3epoch-toollossonly-lr5e-6
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3-merged-3epoch-toollossonly-lr5e-6")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-30-merged-3epoch-toollossonly-lr5e-6")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-57-merged-3epoch-toollossonly-lr5e-6")
# 5epoch-toollossonly-lr1e-5
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-6-merged-5epoch-toollossonly-lr1e-5")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-54-merged-5epoch-toollossonly-lr1e-5")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-95-merged-5epoch-toollossonly-lr1e-5")
# 3epoch-apigen-mt-messages-remove-think-fixtool-lr1e-5
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-208-merged-3epoch-apigen-mt-messages-remove-think-fixtool-lr1e-5")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1040-merged-3epoch-apigen-mt-messages-remove-think-fixtool-lr1e-5")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1875-merged-3epoch-apigen-mt-messages-remove-think-fixtool-lr1e-5")
# 3epoch-tootagonly-lr1e-5 
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-6-merged-3epoch-tootagonly-lr1e-5")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-30-merged-3epoch-tootagonly-lr1e-5")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-57-merged-3epoch-tootagonly-lr1e-5")
# 3epoch-tootagonly-lr2e-6
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-6-merged-3epoch-tootagonly-lr2e-6")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-30-merged-3epoch-tootagonly-lr2e-6")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-57-merged-3epoch-tootagonly-lr2e-6")
# 3epoch-tootagonly-lr5e-6
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-6-merged-3epoch-tootagonly-lr5e-6")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-30-merged-3epoch-tootagonly-lr5e-6")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-57-merged-3epoch-tootagonly-lr5e-6")
# 5epoch-tootagonly-lr5e-6
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-6-merged-5epoch-tootagonly-lr5e-6")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-48-merged-5epoch-tootagonly-lr5e-6")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-95-merged-5epoch-tootagonly-lr5e-6")
# 1epoch-pythonic-production-v2-normal-10K-multirow-4K
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-43-merged-1epoch-pythonic-production-v2-normal-10K-multirow-4K")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-172-merged-1epoch-pythonic-production-v2-normal-10K-multirow-4K")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-339-merged-1epoch-pythonic-production-v2-normal-10K-multirow-4K")
# 3epoch-tooltagonly-lr1e-5-r8a16
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-6-merged-3epoch-tooltagonly-lr1e-5-r8a16")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-30-merged-3epoch-tooltagonly-lr1e-5-r8a16")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-54-merged-3epoch-tooltagonly-lr1e-5-r8a16")
# 3epoch-tooltagonly-lr5e-6-r8a16
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-6-merged-3epoch-tooltagonly-lr5e-6-r8a16")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-30-merged-3epoch-tooltagonly-lr5e-6-r8a16")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-54-merged-3epoch-tooltagonly-lr5e-6-r8a16")
# 6epoch-bfcl-turn5e-6-fix-length-new
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-9-merged-6epoch-bfcl-turn5e-6-fix-length-new")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-126-merged-6epoch-bfcl-turn5e-6-fix-length-new")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-246-merged-6epoch-bfcl-turn5e-6-fix-length-new")
# 3epoch-llama-factory-apigen-mt
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-400-merged-3epoch-llama-factory-apigen-mt")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2000-merged-3epoch-llama-factory-apigen-mt")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3651-merged-3epoch-llama-factory-apigen-mt")
# 15epoch-llama-factory-bfcl-lr1-5e5
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-20-merged-15epoch-llama-factory-bfcl-lr1-5e5")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-160-merged-15epoch-llama-factory-bfcl-lr1-5e5")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-300-merged-15epoch-llama-factory-bfcl-lr1-5e5")
# 3epoch-llama-factory-bfcl-lr8e6-cl
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-20-merged-3epoch-llama-factory-bfcl-lr8e6-cl")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-40-merged-3epoch-llama-factory-bfcl-lr8e6-cl")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-60-merged-3epoch-llama-factory-bfcl-lr8e6-cl")
# 3epoch-llama-factory-bfcl-lr8e6
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-20-merged-3epoch-llama-factory-bfcl-lr8e6")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-40-merged-3epoch-llama-factory-bfcl-lr8e6")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-60-merged-3epoch-llama-factory-bfcl-lr8e6")
# 1epoch-pythonic-zhtw-lr1e-5
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-450-merged-1epoch-pythonic-zhtw-lr1e-5")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-900-merged-1epoch-pythonic-zhtw-lr1e-5")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1448-merged-1epoch-pythonic-zhtw-lr1e-5")
# 1epoch-pythonic-zhtw-lr5e-7
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-450-merged-1epoch-pythonic-zhtw-lr5e-7")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-900-merged-1epoch-pythonic-zhtw-lr5e-7")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1448-merged-1epoch-pythonic-zhtw-lr5e-7")
# 1epoch-all-zhtw-lr5e-7
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1600-merged-1epoch-all-zhtw-lr5e-7")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3200-merged-1epoch-all-zhtw-lr5e-7")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-4787-merged-1epoch-all-zhtw-lr5e-7")
# 3epoch-all-zhtw-lr5e-7
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1550-merged-3epoch-all-zhtw-lr5e-7")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-7750-merged-3epoch-all-zhtw-lr5e-7")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-14361-merged-3epoch-all-zhtw-lr5e-7")
# 1epoch-all-zhtw-split-turn-lr5e-7
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-2275-merged-1epoch-all-zhtw-split-turn-lr5e-7")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-4550-merged-1epoch-all-zhtw-split-turn-lr5e-7")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-6860-merged-1epoch-all-zhtw-split-turn-lr5e-7")
# 1epoch-all-zhtw-lr5e-7-16-32
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-1550-merged-1epoch-all-zhtw-lr1e-5-16-32")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-3100-merged-1epoch-all-zhtw-lr1e-5-16-32")
SUPPORTED_MODELS.append("Salesforce/Llama-xLAM-2-8b-fc-r-checkpoint-4787-merged-1epoch-all-zhtw-lr1e-5-16-32")

# Qwen/Qwen2.5-7B-Instruct 
# Base
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct")
# 3epoch-llama-factory-apigen-mt
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-416-merged-3epoch-llama-factory-apigen-mt")
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-2080-merged-3epoch-llama-factory-apigen-mt")
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-3651-merged-3epoch-llama-factory-apigen-mt")
# 3epoch-llama-factory-apigen-mt
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-75-merged-3epoch-llama-factory-glaive-toolcall-zhtw_1k")
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-375-merged-3epoch-llama-factory-glaive-toolcall-zhtw_1k")
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-720-merged-3epoch-llama-factory-glaive-toolcall-zhtw_1k")
# 3epoch-llama-factory-glaive-toolcall-zhtw-new-1k
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-75-merged-3epoch-llama-factory-glaive-toolcall-zhtw-new-1k")
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-375-merged-3epoch-llama-factory-glaive-toolcall-zhtw-new-1k")
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-750-merged-3epoch-llama-factory-glaive-toolcall-zhtw-new-1k")
# 15epoch-llama-factory-bfcl-lr1-5e5
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-20-merged-15epoch-llama-factory-bfcl-lr1-5e5")
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-160-merged-15epoch-llama-factory-bfcl-lr1-5e5")
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-300-merged-15epoch-llama-factory-bfcl-lr1-5e5")
# 3epoch-llama-factory-bfcl-lr8e6-cl
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-20-merged-3epoch-llama-factory-bfcl-lr8e6-cl")
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-40-merged-3epoch-llama-factory-bfcl-lr8e6-cl")
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-60-merged-3epoch-llama-factory-bfcl-lr8e6-cl")
# 3epoch-llama-factory-bfcl-lr8e6
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-20-merged-3epoch-llama-factory-bfcl-lr8e6")
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-40-merged-3epoch-llama-factory-bfcl-lr8e6")
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-60-merged-3epoch-llama-factory-bfcl-lr8e6")
# 1epoch-pythonic-zhtw-lr1e-5
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-450-merged-1epoch-pythonic-zhtw-lr1e-5")
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-900-merged-1epoch-pythonic-zhtw-lr1e-5")
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-1448-merged-1epoch-pythonic-zhtw-lr1e-5")
# 1epoch-pythonic-zhtw-lr5e-7
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-450-merged-1epoch-pythonic-zhtw-lr5e-7")
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-900-merged-1epoch-pythonic-zhtw-lr5e-7")
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-1448-merged-1epoch-pythonic-zhtw-lr5e-7")
# 1epoch-all-zhtw-lr5e-7
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-1550-merged-1epoch-all-zhtw-lr5e-7")
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-3100-merged-1epoch-all-zhtw-lr5e-7")
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-4787-merged-1epoch-all-zhtw-lr5e-7")
# 1epoch-all-zhtw-lr1e-5
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-1550-merged-1epoch-all-zhtw-lr1e-5")
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-3100-merged-1epoch-all-zhtw-lr1e-5")
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-4787-merged-1epoch-all-zhtw-lr1e-5")
# 1epoch-all-zhtw-split-turn-lr1e-5
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-2275-merged-1epoch-all-zhtw-split-turn-lr1e-5")
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-4550-merged-1epoch-all-zhtw-split-turn-lr1e-5")
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-4787-merged-1epoch-all-zhtw-split-turn-lr1e-5")
# 1epoch-all-zhtw-split-turn-lr5e-7
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-2275-merged-1epoch-all-zhtw-split-turn-lr5e-7")
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-4550-merged-1epoch-all-zhtw-split-turn-lr5e-7")
SUPPORTED_MODELS.append("Qwen/Qwen2.5-7B-Instruct-checkpoint-4787-merged-1epoch-all-zhtw-split-turn-lr5e-7")

# Meta Llama 3.1-8B-Instruct (Prompt)
# 3epoch-llama-factory-apigen-mt
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-400-merged-3epoch-llama-factory-apigen-mt")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-2000-merged-3epoch-llama-factory-apigen-mt")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-3651-merged-3epoch-llama-factory-apigen-mt")
# 3epoch-llama-factory-glaive-toolcall-zhtw_1k
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-75-merged-3epoch-llama-factory-glaive-toolcall-zhtw_1k")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-375-merged-3epoch-llama-factory-glaive-toolcall-zhtw_1k")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-720-merged-3epoch-llama-factory-glaive-toolcall-zhtw_1k")
# 3epoch-llama-factory-glaive-toolcall-zhtw-new-1k
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-75-merged-3epoch-llama-factory-glaive-toolcall-zhtw-new-1k")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-375-merged-3epoch-llama-factory-glaive-toolcall-zhtw-new-1k")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-750-merged-3epoch-llama-factory-glaive-toolcall-zhtw-new-1k")
# 15epoch-llama-factory-bfcl-lr1-5e5
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-20-merged-15epoch-llama-factory-bfcl-lr1-5e5")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-160-merged-15epoch-llama-factory-bfcl-lr1-5e5")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-300-merged-15epoch-llama-factory-bfcl-lr1-5e5")
# 3epoch-llama-factory-bfcl-lr8e6-cl
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-20-merged-3epoch-llama-factory-bfcl-lr8e6-cl")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-40-merged-3epoch-llama-factory-bfcl-lr8e6-cl")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-60-merged-3epoch-llama-factory-bfcl-lr8e6-cl")
# 3epoch-llama-factory-bfcl-lr8e6
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-20-merged-3epoch-llama-factory-bfcl-lr8e6")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-40-merged-3epoch-llama-factory-bfcl-lr8e6")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-60-merged-3epoch-llama-factory-bfcl-lr8e6")
# 1epoch-pythonic-zhtw-lr1e-5
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-450-merged-1epoch-pythonic-zhtw-lr1e-5")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-900-merged-1epoch-pythonic-zhtw-lr1e-5")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-1448-merged-1epoch-pythonic-zhtw-lr1e-5")
# 1epoch-pythonic-zhtw-lr5e-7
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-450-merged-1epoch-pythonic-zhtw-lr5e-7")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-900-merged-1epoch-pythonic-zhtw-lr5e-7")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-1448-merged-1epoch-pythonic-zhtw-lr5e-7")
# 1epoch-all-zhtw-lr5e-7
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-1550-merged-1epoch-all-zhtw-lr5e-7")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-3100-merged-1epoch-all-zhtw-lr5e-7")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-4787-merged-1epoch-all-zhtw-lr5e-7")
# 1epoch-all-zhtw-lr1e-5
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-1550-merged-1epoch-all-zhtw-lr1e-5")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-3100-merged-1epoch-all-zhtw-lr1e-5")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-4787-merged-1epoch-all-zhtw-lr1e-5")
# 1epoch-all-zhtw-split-turn-lr1e-5
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-2275-merged-1epoch-all-zhtw-split-turn-lr1e-5")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-4550-merged-1epoch-all-zhtw-split-turn-lr1e-5")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-6860-merged-1epoch-all-zhtw-split-turn-lr1e-5")
# 1epoch-all-zhtw-split-turn-lr5e-7
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-2275-merged-1epoch-all-zhtw-split-turn-lr5e-7")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-4550-merged-1epoch-all-zhtw-split-turn-lr5e-7")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-6860-merged-1epoch-all-zhtw-split-turn-lr5e-7")
# 1epoch-all-zhtw-lr1e-5-16-32-128
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-50-merged-1epoch-all-zhtw-lr1e-5-16-32-128")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-100-merged-1epoch-all-zhtw-lr1e-5-16-32-128")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-149-merged-1epoch-all-zhtw-lr1e-5-16-32-128")
# 1epoch-all-zhtw-lr1e-5-16-32-128-turn-tokenize
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-50-merged-1epoch-all-zhtw-lr1e-5-16-32-128-turn-tokenize")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-100-merged-1epoch-all-zhtw-lr1e-5-16-32-128-turn-tokenize")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-155-merged-1epoch-all-zhtw-lr1e-5-16-32-128-turn-tokenize")
# 1epoch-all-zhtw-split-turn-lr1e-5-16-32-128
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-70-merged-1epoch-all-zhtw-split-turn-lr1e-5-16-32-128")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-140-merged-1epoch-all-zhtw-split-turn-lr1e-5-16-32-128")
SUPPORTED_MODELS.append("meta-llama/Llama-3.1-8B-Instruct-checkpoint-215-merged-1epoch-all-zhtw-split-turn-lr1e-5-16-32-128")

# openai/gpt-oss-20b
# Base
SUPPORTED_MODELS.append("openai/gpt-oss-20b")
# 1epoch-all-zhtw-lr1e-5
SUPPORTED_MODELS.append("openai/gpt-oss-20b-checkpoint-50-merged-1epoch-all-zhtw-lr1e-5")
SUPPORTED_MODELS.append("openai/gpt-oss-20b-checkpoint-100-merged-1epoch-all-zhtw-lr1e-5")
SUPPORTED_MODELS.append("openai/gpt-oss-20b-checkpoint-149-merged-1epoch-all-zhtw-lr1e-5")
# 1epoch-all-zhtw-lr5e-7
SUPPORTED_MODELS.append("openai/gpt-oss-20b-checkpoint-50-merged-1epoch-all-zhtw-lr5e-7")
SUPPORTED_MODELS.append("openai/gpt-oss-20b-checkpoint-100-merged-1epoch-all-zhtw-lr5e-7")
SUPPORTED_MODELS.append("openai/gpt-oss-20b-checkpoint-149-merged-1epoch-all-zhtw-lr5e-7")
# 1epoch-all-zhtw-split-turn-lr1e-5
SUPPORTED_MODELS.append("openai/gpt-oss-20b-checkpoint-70-merged-1epoch-all-zhtw-split-turn-lr1e-5")
SUPPORTED_MODELS.append("openai/gpt-oss-20b-checkpoint-140-merged-1epoch-all-zhtw-split-turn-lr1e-5")
SUPPORTED_MODELS.append("openai/gpt-oss-20b-checkpoint-215-merged-1epoch-all-zhtw-split-turn-lr1e-5")
# 1epoch-all-zhtw-split-turn-lr5e-7
SUPPORTED_MODELS.append("openai/gpt-oss-20b-checkpoint-70-merged-1epoch-all-zhtw-split-turn-lr5e-7")
SUPPORTED_MODELS.append("openai/gpt-oss-20b-checkpoint-140-merged-1epoch-all-zhtw-split-turn-lr5e-7")
SUPPORTED_MODELS.append("openai/gpt-oss-20b-checkpoint-215-merged-1epoch-all-zhtw-split-turn-lr5e-7")
# 5epoch-all-zhtw-split-turn-lr5e-7-16-64-128
SUPPORTED_MODELS.append("openai/gpt-oss-20b-checkpoint-70-merged-5epoch-all-zhtw-split-turn-lr5e-7-16-64-128")
SUPPORTED_MODELS.append("openai/gpt-oss-20b-checkpoint-140-merged-5epoch-all-zhtw-split-turn-lr5e-7-16-64-128")
SUPPORTED_MODELS.append("openai/gpt-oss-20b-checkpoint-210-merged-5epoch-all-zhtw-split-turn-lr5e-7-16-64-128")

# openai/gpt-oss-120b
# Base
SUPPORTED_MODELS.append("openai/gpt-oss-120b")

# Qwen/Qwen3-8B-FC LoRA finetune checkpoints
# 2epoch-pythonic-production-v2-normal-10K-multirow-4K
SUPPORTED_MODELS.append("Qwen/Qwen3-8B-FC-checkpoint-85-merged-2epoch-pythonic-production-v2-normal-10K-multirow-4K")
SUPPORTED_MODELS.append("Qwen/Qwen3-8B-FC-checkpoint-510-merged-2epoch-pythonic-production-v2-normal-10K-multirow-4K")
SUPPORTED_MODELS.append("Qwen/Qwen3-8B-FC-checkpoint-858-merged-2epoch-pythonic-production-v2-normal-10K-multirow-4K")
# 5epoch-bfcl-lr5e-6-r8a16-turn
SUPPORTED_MODELS.append("Qwen/Qwen3-8B-FC-checkpoint-5-merged-5epoch-bfcl-lr5e-6-r8a16-turn")
SUPPORTED_MODELS.append("Qwen/Qwen3-8B-FC-checkpoint-55-merged-5epoch-bfcl-lr5e-6-r8a16-turn")
SUPPORTED_MODELS.append("Qwen/Qwen3-8B-FC-checkpoint-110-merged-5epoch-bfcl-lr5e-6-r8a16-turn")
# 10epoch-bfcl-delete-long-lr2e-6-r8a16
SUPPORTED_MODELS.append("Qwen/Qwen3-8B-FC-checkpoint-3-merged-10epoch-bfcl-delete-long-lr2e-6-r8a16")
SUPPORTED_MODELS.append("Qwen/Qwen3-8B-FC-checkpoint-51-merged-10epoch-bfcl-delete-long-lr2e-6-r8a16")
SUPPORTED_MODELS.append("Qwen/Qwen3-8B-FC-checkpoint-100-merged-10epoch-bfcl-delete-long-lr2e-6-r8a16")
# 10epoch-bfcl-lr2e-6-r8a16
SUPPORTED_MODELS.append("Qwen/Qwen3-8B-FC-checkpoint-3-merged-10epoch-bfcl-lr2e-6-r8a16")
SUPPORTED_MODELS.append("Qwen/Qwen3-8B-FC-checkpoint-51-merged-10epoch-bfcl-lr2e-6-r8a16")
SUPPORTED_MODELS.append("Qwen/Qwen3-8B-FC-checkpoint-100-merged-10epoch-bfcl-lr2e-6-r8a16")
