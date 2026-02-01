你的任務是生成「無關性」測試資料，用於評估 LLM 的函式呼叫能力。

目標是創建以下情境：
1. 使用者提出一個只需要自然語言就能回答的問題（不需要呼叫函式）
2. 可用的函式與使用者的問題完全無關
3. LLM 的正確行為是不呼叫任何函式，直接用自然語言回答

你需要同時生成：
- 一個可以用常識或簡單推理回答的使用者問題
- 一個或多個與問題無關的函式定義

## 輸入資訊：
<domain>
{{domain}}
</domain>

<subdomain>
{{subdomain}}
</subdomain>

## 要求：

### 使用者問題：
- 應該可以用常識、數學、邏輯推理或一般知識來回答
- 不應該需要任何外部 API、資料庫或工具存取
- 範例：數學問題、常識問題、定義解釋、建議諮詢
- 使用繁體中文撰寫問題

### 無關函式：
- 應該是真實、有完整文件的函式，可能存在於真實系統中
- 應該與使用者的問題完全無關
- 使用 domain/subdomain 作為函式設計的背景
- 每個函式都要有正確的型別標註和 docstring
- 函式名稱保持英文，docstring 使用繁體中文

## 輸出格式：

生成 {{num_samples}} 個樣本。每個樣本使用以下格式：

<sample>
<question>
使用者的問題（可以用自然語言回答，不需要呼叫函式）
</question>
<natural_response>
問題的預期自然語言回答
</natural_response>
<function>
<signature>
```python
def function_name(param1: str, param2: int) -> Dict[str, Any]:
    """函式的簡短描述。
    
    :param param1: param1 的說明。
    :param param2: param2 的說明。
    :return_fields:
      - field1 (type): field1 的說明。
      - field2 (type): field2 的說明。
    """
    pass
```
</signature>
<expected>
{"field1": "example_value", "field2": 123}
</expected>
</function>
</sample>

## 良好的無關性配對範例：

1. 問題：「法國的首都是哪裡？」
   無關函式：`def calculate_mortgage_payment(principal: float, rate: float, years: int) -> Dict[str, float]`

2. 問題：「六邊形有幾個邊？」
   無關函式：`def send_email(recipient: str, subject: str, body: str) -> Dict[str, bool]`

3. 問題：「200 的 15% 是多少？」
   無關函式：`def get_weather_forecast(city: str, days: int) -> Dict[str, Any]`

現在為給定的 domain/subdomain 生成 {{num_samples}} 個多樣化的無關性樣本：
