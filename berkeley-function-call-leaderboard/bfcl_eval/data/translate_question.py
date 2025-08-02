import os
import json
from googletrans import Translator

# 設定來源資料夾
input_folder = "./" 
translator = Translator()

# 遍歷資料夾中所有 .json 檔案
for filename in os.listdir(input_folder):
    if filename.endswith(".json") and not filename.endswith("_translated.json"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(input_folder, filename.replace(".json", "_translated.json"))

        # 讀取 JSON 檔
        with open(input_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"❌ 無法解析 JSON：{filename}")
                continue

        # 翻譯處理
        if isinstance(data, dict):
            # 單筆資料
            if 'question' in data:
                translated = translator.translate(data['question'], src='en', dest='zh-tw')
                data['question'] = translated.text

        elif isinstance(data, list):
            # 多筆資料（list of dicts）
            for item in data:
                if isinstance(item, dict) and 'question' in item:
                    translated = translator.translate(item['question'], src='en', dest='zh-tw')
                    item['question'] = translated.text

        else:
            print(f"⚠️ 格式不支援：{filename}")
            continue

        # 儲存翻譯結果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ 已翻譯：{filename} → {output_path}")

