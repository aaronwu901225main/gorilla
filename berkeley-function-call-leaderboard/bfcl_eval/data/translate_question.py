import os
import time
import jsonlines
from googletrans import Translator

# 設定來源與輸出資料夾
input_folder = "./"
translated_folder = "./translated"
os.makedirs(translated_folder, exist_ok=True)

translator = Translator()

def safe_translate(text, retries=3, delay=1):
    last_error = "unknown"
    for attempt in range(retries):
        try:
            translated = translator.translate(text, src='en', dest='zh-tw').text
            return translated, None
        except Exception as e:
            if "timed out" in str(e).lower():
                print(f"⏰ 第 {attempt+1} 次翻譯超時，稍後再試...")
                last_error = "timeout"
            else:
                print(f"⚠️ 第 {attempt+1} 次翻譯錯誤：{e}")
                last_error = "other"
            time.sleep(delay)
    return None, last_error

# 遍歷資料夾中所有 .json 檔案
for filename in os.listdir(input_folder):
    if filename.endswith(".json") and not filename.endswith("_translated.json"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(translated_folder, filename.replace(".json", "_translated.json"))

        # 統計資料
        stats = {
            "translated": 0,
            "skipped_none": [],
            "skipped_empty": [],
            "skipped_not_str": [],
            "failed_timeout": []
        }

        try:
            with jsonlines.open(input_path, 'r') as reader, jsonlines.open(output_path, 'w') as writer:
                for obj in reader:
                    obj_id = obj.get("id", "unknown")
                    if isinstance(obj, dict) and 'question' in obj:
                        qlist = obj['question']
                        for turn in qlist:
                            if isinstance(turn, list):
                                for msg in turn:
                                    if isinstance(msg, dict) and 'content' in msg:
                                        content = msg['content']
                                        if content is None:
                                            stats['skipped_none'].append(obj_id)
                                        elif isinstance(content, str):
                                            if content.strip() == "":
                                                stats['skipped_empty'].append(obj_id)
                                            else:
                                                translated, error_type = safe_translate(content)
                                                if translated is not None:
                                                    msg['content'] = translated
                                                    stats['translated'] += 1
                                                else:
                                                    if error_type == "timeout":
                                                        stats['failed_timeout'].append(obj_id)
                                                    else:
                                                        print(f"⚠️ 翻譯失敗（非超時）→ id: {obj_id}")
                                        else:
                                            stats['skipped_not_str'].append(obj_id)
                    writer.write(obj)

        except Exception as e:
            print(f"❌ 讀取或翻譯失敗：{filename}，錯誤：{e}")
            continue

        print(f"\n✅ 已翻譯：{filename} → {output_path}")
        print("🔎 翻譯統計：")
        print(f" - 成功翻譯筆數：{stats['translated']}")
        print(f" - 跳過 None：{len(stats['skipped_none'])} 筆 → {stats['skipped_none']}")
        print(f" - 跳過空字串：{len(stats['skipped_empty'])} 筆 → {stats['skipped_empty']}")
        print(f" - 跳過非字串：{len(stats['skipped_not_str'])} 筆 → {stats['skipped_not_str']}")
        print(f" - 翻譯超時失敗：{len(stats['failed_timeout'])} 筆 → {stats['failed_timeout']}")
