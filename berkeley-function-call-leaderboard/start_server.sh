# 重新啟動（GPU）
PYTHONUNBUFFERED=1 python -X faulthandler -u /workspace/serve_openai_gptoss.py \
  --model-id openai/gpt-oss-20b \
  --host 0.0.0.0 --port 8000 \
  --dtype auto \
  --device-map auto \
  --trust-remote-code

# 或 CPU 模式（除錯時用）
# PYTHONUNBUFFERED=1 python -X faulthandler -u /workspace/serve_openai_gptoss.py \
#   --model-id openai/gpt-oss-20b --host 0.0.0.0 --port 8000 \
#   --dtype float32 --device-map cpu --trust-remote-code
