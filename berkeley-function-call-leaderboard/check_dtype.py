from safetensors.torch import safe_open
from pathlib import Path

model_dir = Path("/workspace/gorilla/berkeley-function-call-leaderboard/gpt-oss-20b")
parts = sorted(model_dir.glob("model-*.safetensors"))

dtype_count = {}
total = 0
for p in parts:
    with safe_open(p.as_posix(), framework="pt") as f:
        for k in f.keys():
            dt = str(f.get_tensor(k).dtype)
            dtype_count[dt] = dtype_count.get(dt, 0) + 1
            total += 1

print("Tensors total:", total)
for k, v in sorted(dtype_count.items(), key=lambda x: -x[1]):
    print(f"{k}: {v}")
