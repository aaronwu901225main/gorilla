You are tasked with generating "irrelevance" test data for evaluating LLM function-calling capabilities.

The goal is to create scenarios where:
1. A user asks a question that can be answered using ONLY natural language (no function call needed)
2. The available functions are IRRELEVANT to the user's question
3. The correct behavior for an LLM is to NOT call any function and simply answer in natural language

Your task is to generate BOTH:
- A user question that can be answered with common knowledge or simple reasoning
- One or more function definitions that are UNRELATED to the question

## Input Information:
<domain>
{{domain}}
</domain>

<subdomain>
{{subdomain}}
</subdomain>

## Requirements:

### For the User Question:
- Should be answerable using general knowledge, math, common sense, or reasoning
- Should NOT require any external API, database, or tool access
- Examples: math problems, general knowledge questions, definitions, explanations, advice

### For the Irrelevant Functions:
- Should be realistic, well-documented functions that could exist in a real system
- Should be completely UNRELATED to the user's question
- Use the domain/subdomain as context for the function design
- Each function should have proper type annotations and docstrings

## Output Format:

Generate {{num_samples}} samples. For each sample, use this format:

<sample>
<question>
The user's question that can be answered with natural language only
</question>
<natural_response>
The expected natural language response to the question
</natural_response>
<function>
<signature>
```python
def function_name(param1: str, param2: int) -> Dict[str, Any]:
    """Brief description of what the function does.
    
    :param param1: Description of param1.
    :param param2: Description of param2.
    :return_fields:
      - field1 (type): Description of field1.
      - field2 (type): Description of field2.
    """
    pass
```
</signature>
<expected>
{"field1": "example_value", "field2": 123}
</expected>
</function>
</sample>

## Examples of Good Irrelevance Pairs:

1. Question: "What is the capital of France?"
   Irrelevant Function: `def calculate_mortgage_payment(principal: float, rate: float, years: int) -> Dict[str, float]`

2. Question: "How many sides does a hexagon have?"
   Irrelevant Function: `def send_email(recipient: str, subject: str, body: str) -> Dict[str, bool]`

3. Question: "What is 15% of 200?"
   Irrelevant Function: `def get_weather_forecast(city: str, days: int) -> Dict[str, Any]`

Now generate {{num_samples}} diverse irrelevance samples for the given domain/subdomain:
