from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, load_from_disk
import torch
import re
from datetime import datetime
import json
import os

TEST_SIZE = 'all'

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

REPEAT = 1
CONTEXT_LENGTH = 32000
MAX_TOKENS = 30000
TEMPERATURE = 0.6
TOP_P = 0.95

start_time = datetime.now()
filename = start_time.strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs("./test_cot", exist_ok=True)

llm = LLM(
    model=MODEL_ID,
    dtype=torch.float16,
    trust_remote_code=True,
    quantization='bitsandbytes',
    load_format='bitsandbytes',
    max_model_len=CONTEXT_LENGTH,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

root_dir = "./mmlu_splits"
# subjects = [name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]
subjects = [name for name in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, name)) and name.startswith("college")]
print(subjects)

for subject in subjects:
    print(f"\n🧪 Evaluating subject: {subject}")
    data = load_from_disk(os.path.join(root_dir, subject, "test"))
    
    if TEST_SIZE != 'all':
        data = data.select(range(TEST_SIZE))
    
    def extract_boxed_answer(text: str) -> str:
        matches = re.findall(r'\\boxed\{(.*?)\}', text)
        return matches[-1].strip() if matches else None
    
    params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    
    results = {"correct": 0, "wrong": 0}
    choices = ['A', 'B', 'C', 'D']
    correct_cots = []
    wrong_cots = []
    
    for i, d in enumerate(data):
        user_prompt = (
            f"Solve the following multiple choices question:\n{d['question']}\n"
            f"A) {d['choices'][0]}\n"
            f"B) {d['choices'][1]}\n"
            f"C) {d['choices'][2]}\n"
            f"D) {d['choices'][3]}\n"
            "Please reason step by step, and put your final answer within \\boxed{}.For example if the answer is A ,then output \\boxed{A}"
        )
    
        prompt =f"<｜begin▁of▁sentence｜><｜User｜>{user_prompt}\n<｜Assistant｜><think>"

        output1 = llm.generate(prompt, params)
        
        response_1 = output1[0].outputs[0].text +'\nFinal answer: \\boxed{'

        output2 = llm.generate(prompt + response_1, params)

        response_2 = output2[0].outputs[0].text
        
        response = response_1 + response_2

        extracted_answer = extract_boxed_answer(response)
    
        print(prompt + response)
        print(f'\n{subject} Question {i + 1}/{len(data)}')
        print(f'res toks 1: {len(output1[0].outputs[0].token_ids)}')
        print(f'res toks 2: {len(output2[0].outputs[0].token_ids)}')
        print(f'Extracted answer: {extracted_answer}')
        print(f'True answer: {choices[d["answer"]]}')
    
        cot = {
            'question': user_prompt,
            'CoT': '<think>' + response,
            'extracted_answer': extracted_answer,
            'true_answer': choices[d["answer"]],
            'label': 1 if extracted_answer == choices[d["answer"]] else 0,
        }
    
        if extracted_answer == choices[d["answer"]]:
            results["correct"] += 1
            correct_cots.append(cot)
        else:
            results["wrong"] += 1
            wrong_cots.append(cot)
    
        print(f"Current results: {results}")
        print('Spend time:', datetime.now() - start_time)
        print('=' * 20)
        
    # 儲存所有題目結果（正確與錯誤整合）
    all_cots = correct_cots + wrong_cots
    with open(f"./test_cot/test_cot_{subject}_{filename}_all.json", "w") as f:
        json.dump(all_cots, f, indent=4)

    # 重設資料以進行下一個 subject 的處理
    correct_cots = []
    wrong_cots = []
    results = {"correct": 0, "wrong": 0}
