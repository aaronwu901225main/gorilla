import argparse
import time
import uuid
from typing import List, Dict, Any, Optional

import torch
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

def extract_final(text: str) -> str:
    """
    gpt-oss 的 harmony 會產生像 'analysis...assistantfinalHello!'。
    我們只要 'assistantfinal' 之後的內容；若沒有，就回原文。
    """
    # 先找常見的 'assistantfinal'
    m = re.search(r'assistantfinal\s*', text, flags=re.IGNORECASE)
    if m:
        return text[m.end():].strip()

    # 退而求其次：抓 'final'（避免把 'analysis' 誤砍）
    m2 = re.search(r'\bfinal\b\s*:?\s*', text, flags=re.IGNORECASE)
    if m2:
        # 確保不是 'analysis' 的子字串
        if not re.search(r'analysi\s*$', text[:m2.start()], flags=re.IGNORECASE):
            return text[m2.end():].strip()

    return text.strip()

app = FastAPI()

# ---------- OpenAI-like request/response schemas ----------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionsRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 1.0
    stop: Optional[List[str]] = None

class ChoiceMessage(BaseModel):
    role: str
    content: str

class Choice(BaseModel):
    index: int
    message: ChoiceMessage
    finish_reason: str = "stop"

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionsResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

# ---------- Globals ----------
tokenizer = None
model = None
model_id = None
device = None

def build_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Use tokenizer.apply_chat_template to format harmony/chat template.
    """
    return tokenizer.apply_chat_template(
        [{"role": m["role"], "content": m["content"]} for m in messages],
        tokenize=False,
        add_generation_prompt=True,
    )

@app.get("/v1/models")
def get_models():
    return {
        "object": "list",
        "data": [{
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }]
    }

@app.post("/v1/chat/completions", response_model=ChatCompletionsResponse)
def chat_completions(req: ChatCompletionsRequest = Body(...)):
    # 1) build prompt via chat template
    prompt = build_prompt([m.dict() for m in req.messages])

    # 2) tokenize
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_tokens = input_ids.shape[-1]

    # 3) generate
    gen_out = model.generate(
        input_ids=input_ids,
        max_new_tokens=req.max_tokens or 512,
        temperature=req.temperature if req.temperature is not None else 0.7,
        top_p=req.top_p if req.top_p is not None else 1.0,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # 4) decode only the generated part
    gen_ids = gen_out[0][prompt_tokens:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # 👉 新增這行：抽出最終答案
    text = extract_final(text)

    # 5) usage estimation（簡易版）
    completion_tokens = len(tokenizer.encode(text))
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )

    resp = ChatCompletionsResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=model_id,
        choices=[Choice(index=0, message=ChoiceMessage(role="assistant", content=text))],
        usage=usage,
    )
    return resp

def main():
    global tokenizer, model, model_id, device

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto","bfloat16","float16","float32"])
    parser.add_argument("--device-map", type=str, default="auto", choices=["auto","cpu"])
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id

    dtype_map = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # device
    if args.device_map == "cpu":
        device_map = {"": "cpu"}
        device_name = "cpu"
    else:
        device_map = "auto"
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    
    # --- workaround for transformers MXFP4 bug (NameError: is_triton_kernels_availalble) ---
    try:
        import transformers.quantizers.quantizer_mxfp4 as qm
        # 某些版本用的是 is_kernels_available，某些叫 is_triton_kernels_available
        if not hasattr(qm, "is_triton_kernels_availalble"):
            if hasattr(qm, "is_kernels_available"):
                qm.is_triton_kernels_availalble = qm.is_kernels_available
            elif hasattr(qm, "is_triton_kernels_available"):
                qm.is_triton_kernels_availalble = qm.is_triton_kernels_available
    except Exception as _e:
        # 若匯入失敗就略過，後續 from_pretrained 也許不會觸發該路徑
        pass
    # --- end workaround ---

    # load
    print(f"Loading {model_id} with device_map={device_map}, dtype={args.dtype}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
    )
    # 若沒有 pad token，設置一個避免警告
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
