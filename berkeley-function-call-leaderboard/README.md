# Berkeley Function Calling Leaderboard (BFCL)

## Table of Contents

- [Berkeley Function Calling Leaderboard (BFCL)](#berkeley-function-calling-leaderboard-bfcl)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation \& Setup](#installation--setup)
    - [Basic Installation](#basic-installation)
    - [Installing from PyPI](#installing-from-pypi)
    - [Extra Dependencies for Self-Hosted Models](#extra-dependencies-for-self-hosted-models)
    - [Configuring Project Root Directory](#configuring-project-root-directory)
    - [Setting up Environment Variables](#setting-up-environment-variables)
      - [Configuring SerpAPI for Web Search Category](#configuring-serpapi-for-web-search-category)
  - [Running Evaluations](#running-evaluations)
    - [Generating LLM Responses](#generating-llm-responses)
      - [Selecting Models and Test Categories](#selecting-models-and-test-categories)
      - [Selecting Specific Test Cases with `--run-ids`](#selecting-specific-test-cases-with---run-ids)
      - [Output and Logging](#output-and-logging)
      - [For API-based Models](#for-api-based-models)
      - [For Locally-hosted OSS Models](#for-locally-hosted-oss-models)
        - [For Pre-existing OpenAI-compatible Endpoints](#for-pre-existing-openai-compatible-endpoints)
      - [(Alternate) Script Execution for Generation](#alternate-script-execution-for-generation)
  - [zh\_\* Semantic Evaluation (LLM-based Judge) \[Experimental\]](#zh_-semantic-evaluation-llm-based-judge-experimental)
    - [Traditional Chinese Test Categories](#traditional-chinese-test-categories)
    - [Running the Chinese Evaluation](#running-the-chinese-evaluation)
    - [Enabling the Judge](#enabling-the-judge)
    - [When a Failed Case Is Re-evaluated](#when-a-failed-case-is-re-evaluated)
    - [OpenAI API Keys and Quota Tracking](#openai-api-keys-and-quota-tracking)
    - [Outputs Produced by the Judge](#outputs-produced-by-the-judge)
    - [Notes](#notes)
    - [Evaluating Generated Responses](#evaluating-generated-responses)
      - [Output Structure](#output-structure)
      - [(Optional) WandB Evaluation Logging](#optional-wandb-evaluation-logging)
      - [(Alternate) Script Execution for Evaluation](#alternate-script-execution-for-evaluation)
  - [Evaluating External or Fine-tuned Models](#evaluating-external-or-fine-tuned-models)
    - [Registering a Checkpoint](#registering-a-checkpoint)
    - [Choosing and Editing a Generation Example](#choosing-and-editing-a-generation-example)
    - [Using a Pre-existing vLLM Endpoint](#using-a-pre-existing-vllm-endpoint)
    - [Editing the Evaluation Example](#editing-the-evaluation-example)
    - [Handoff File Map](#handoff-file-map)
  - [Contributing \& How to Add New Models](#contributing--how-to-add-new-models)
  - [Additional Resources](#additional-resources)

---

## Introduction

We introduce the Berkeley Function Calling Leaderboard (BFCL), the **first comprehensive and executable function call evaluation** dedicated to assessing Large Language Models' (LLMs) ability to invoke functions. Unlike previous evaluations, BFCL accounts for various forms of function calls, diverse scenarios, and executability.

💡 Read more in our blog posts:

- [BFCL v1: Simple, Parallel, and Multiple Function Call eval with AST](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)
- [BFCL v2: Enterprise and OSS-contributed Live Data](https://gorilla.cs.berkeley.edu/blogs/12_bfcl_v2_live.html)
- [BFCL v3: Multi-Turn & Multi-Step Function Call Evaluation](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html)
- [BFCL V4 Part 1: Agentic Web Search](https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html)
- [BFCL V4 Part 2: Agentic Memory Management](https://gorilla.cs.berkeley.edu/blogs/16_bfcl_v4_memory.html)
- [BFCL V4 Part 3: Agentic Format Sensitivity](https://gorilla.cs.berkeley.edu/blogs/17_bfcl_v4_prompt_variation.html)

🦍 See the live leaderboard at [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html#leaderboard)

![Architecture Diagram](https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/architecture_diagram.png)

---

## Installation & Setup

### Basic Installation

```bash
# Create a new Conda environment with Python 3.10
conda create -n BFCL python=3.10 -y
conda activate BFCL
conda install cuda -c nvidia

# Clone the Gorilla repository
git clone https://github.com/ShishirPatil/gorilla.git

# Change directory to the `berkeley-function-call-leaderboard`
cd gorilla/berkeley-function-call-leaderboard

# Install the package in editable mode
pip install -e .
```

### Installing from PyPI

If you simply want to run the evaluation without making code changes, you can
install the prebuilt wheel instead. **Be careful not to confuse our package with
the *unrelated* `bfcl` project on PyPI—make sure you install `bfcl-eval`:**

```bash
pip install bfcl-eval  # Be careful not to confuse with the unrelated `bfcl` project on PyPI!
```

### Extra Dependencies for Self-Hosted Models

For locally hosted models, choose one of the following backends, ensuring you have the right GPU and OS setup:

`sglang` is *much faster* than `vllm` in our specific multi-turn use case, but it only supports newer GPUs with SM 80+ (Ampere etc).
If you are using an older GPU (T4/V100), you should use `vllm` instead as it supports a much wider range of GPUs.

**Using `vllm`:**
```bash
pip install -U pip uv
uv pip install -e ".[oss_eval_vllm]" --torch-backend=cu130
```

**Using `sglang`:**
```bash
pip install -e .[oss_eval_sglang]
```

*Optional:* If using `sglang`, we recommend installing `flashinfer` for speedups. Find instructions [here](https://docs.flashinfer.ai/installation.html).

### Configuring Project Root Directory

**Important:** If you installed the package from PyPI (using `pip install bfcl-eval`), you **must** set the `BFCL_PROJECT_ROOT` environment variable to specify where the evaluation results and score files should be stored.
Otherwise, you'll need to navigate deep into the Python package's source code folder to access the evaluation results and configuration files.

For editable installations (using `pip install -e .`), setting `BFCL_PROJECT_ROOT` is *optional*--it defaults to the `berkeley-function-call-leaderboard` directory.

Set `BFCL_PROJECT_ROOT` as an environment variable in your shell environment:

```bash
# In your shell environment
export BFCL_PROJECT_ROOT=/path/to/your/desired/project/directory
```

When `BFCL_PROJECT_ROOT` is set:

- The `result/` folder (containing model responses) will be created at `$BFCL_PROJECT_ROOT/result/`
- The `score/` folder (containing evaluation results) will be created at `$BFCL_PROJECT_ROOT/score/`
- The library will look for the `.env` configuration file at `$BFCL_PROJECT_ROOT/.env` (see [Setting up Environment Variables](#setting-up-environment-variables))

### Setting up Environment Variables

We store API keys and other configuration variables (separate from the `BFCL_PROJECT_ROOT` variable mentioned above) in a `.env` file. A sample `.env.example` file is distributed with the package.

**For editable installations:**

```bash
cp bfcl_eval/.env.example .env
# Fill in necessary values in `.env`
```

**For PyPI installations (using `pip install bfcl-eval`):**

```bash
cp $(python -c "import bfcl_eval; print(bfcl_eval.__path__[0])")/.env.example $BFCL_PROJECT_ROOT/.env
# Fill in necessary values in `.env`
```

If you are running any proprietary models, make sure the model API keys are included in your `.env` file. Models like GPT, Claude, Mistral, Gemini, Nova, will require them.

The library looks for the `.env` file in the project root, i.e. `$BFCL_PROJECT_ROOT/.env`.

#### Configuring SerpAPI for Web Search Category

For the `web_search` test category, we use the [SerpAPI](https://serpapi.com/) service to perform web search. You need to sign up for an API key and add it to your `.env` file. You can also switch to other web search APIs by changing the `search_engine_query` function in `bfcl_eval/eval_checker/multi_turn_eval/func_source_code/web_search.py`.

---

## Running Evaluations

### Generating LLM Responses

#### Selecting Models and Test Categories

- `MODEL_NAME`: For available models, please refer to [SUPPORTED_MODELS.md](./SUPPORTED_MODELS.md). If not specified, the default model `gorilla-openfunctions-v2` is used.
- `TEST_CATEGORY`: For available test categories, please refer to [TEST_CATEGORIES.md](./TEST_CATEGORIES.md). If not specified, all categories are included by default.

You can provide multiple models or test categories by separating them with commas. For example:

```bash
bfcl generate --model claude-3-5-sonnet-20241022-FC,gpt-4o-2024-11-20-FC --test-category simple_python,parallel,live_multiple,multi_turn
```

#### Selecting Specific Test Cases with `--run-ids`

Sometimes you may only need to regenerate a handful of test entries—for instance when iterating on a new model or after fixing an inference bug. Passing the `--run-ids` flag lets you target **exact test IDs** rather than an entire category:

```bash
bfcl generate --model MODEL_NAME --run-ids   # --test-category will be ignored
```

When this flag is set the generation pipeline reads a JSON file named
`test_case_ids_to_generate.json` located in the *project root* (the same
place where `.env` lives). The file should map each test category to a list of
IDs to run:

```json
{
    "simple_python": ["simple_python_102", "simple_python_103"],
    "multi_turn_base": ["multi_turn_base_15"]
}
```

> Note: When using `--run-ids`, the `--test-category` flag is ignored.

A sample file is provided at `bfcl_eval/test_case_ids_to_generate.json.example`; **copy it to your project root** so the CLI can pick it up regardless of your working directory:

**For editable installations:**

```bash
cp bfcl_eval/test_case_ids_to_generate.json.example ./test_case_ids_to_generate.json
```

**For PyPI installations:**

```bash
cp $(python -c "import bfcl_eval, pathlib; print(pathlib.Path(bfcl_eval.__path__[0]) / 'test_case_ids_to_generate.json.example')") $BFCL_PROJECT_ROOT/test_case_ids_to_generate.json
```

Once `--run-ids` is provided only the IDs listed in the JSON will be evaluated.

#### Output and Logging

- By default, generated model responses are stored in a `result/` folder under the project root (which defaults to the package directory): `result/MODEL_NAME/BFCL_v3_TEST_CATEGORY_result.json`.
- You can customise the location by setting the `BFCL_PROJECT_ROOT` environment variable or passing the `--result-dir` option.

An inference log is included with the model responses to help analyze/debug the model's performance, and to better understand the model behavior. For more verbose logging, use the `--include-input-log` flag. Refer to [LOG_GUIDE.md](./LOG_GUIDE.md) for details on how to interpret the inference logs.

#### For API-based Models

```bash
bfcl generate --model MODEL_NAME --test-category TEST_CATEGORY --num-threads 1
```

- Use `--num-threads` to control the level of parallel inference. The default (`1`) means no parallelization.
- The maximum allowable threads depends on your API's rate limits.

#### For Locally-hosted OSS Models

```bash
bfcl generate \
  --model MODEL_NAME \
  --test-category TEST_CATEGORY \
  --backend {sglang|vllm} \
  --num-gpus 1 \
  --gpu-memory-utilization 0.9 \
  --local-model-path /path/to/base/model \
  --enable-lora \
  --max-lora-rank 128 \
  --lora-modules module1="/path/to/lora/adapter1" module2="/path/to/lora/adapter2" # ← optional
```

- Choose your backend using `--backend sglang` or `--backend vllm`. The default backend is `vllm`.
- Control GPU usage by adjusting `--num-gpus` (default `1`, relevant for multi-GPU tensor parallelism) and `--gpu-memory-utilization` (default `0.9`), which can help avoid out-of-memory errors.
- `--local-model-path` (optional): Point this flag at a directory that already contains the model's files (`config.json`, tokenizer, weights, etc.). Use it only when you've pre-downloaded the model and the weights live somewhere other than the default `$HF_HOME` cache.
- `--enable-lora` (optional): Enable LoRA for the vLLM backend. This flag is required to use LoRA modules. This only works when backend is `vllm`.
- `--max-lora-rank` (optional): Specify the maximum LoRA rank for the vLLM backend. This is an integer value. This only works when backend is `vllm` and `--enable-lora` flag is set.
- `--lora-modules` (optional): Specify the path to the LoRA modules for the vLLM backend in `name="path"` format. This allows evaluation of fine-tuned models with LoRA adapters. You can specify multiple LoRA modules by repeating this argument. This only works when backend is `vllm` and `--enable-lora` flag is set.

##### For Pre-existing OpenAI-compatible Endpoints

If you have a server already running (e.g., vLLM in a SLURM cluster), you can bypass the vLLM/sglang setup phase and directly generate responses by using the `--skip-server-setup` flag:

```bash
bfcl generate --model MODEL_NAME --test-category TEST_CATEGORY --skip-server-setup
```

In addition, you should specify the endpoint and port used by the local server. By default, the endpoint is `localhost` and the port is `1053`. These can be overridden by the `LOCAL_SERVER_ENDPOINT` and `LOCAL_SERVER_PORT` environment variables in the `.env` file:

```bash
LOCAL_SERVER_ENDPOINT=localhost
LOCAL_SERVER_PORT=1053
```

For remote deployments (e.g., via RunPod, ngrok, or enterprise gateways) that require custom authentication or use non-standard base URLs, you can specify a full base URL and API key:

```bash
REMOTE_OPENAI_BASE_URL=https://your-vllm-server.com/v1
REMOTE_OPENAI_API_KEY=your-api-key-here
REMOTE_OPENAI_TOKENIZER_PATH=/path/to/local/tokenizer  # Optional: specify local tokenizer for local/remote endpoints
```

#### (Alternate) Script Execution for Generation

For those who prefer using script execution instead of the CLI, you can run the following command:

```bash
python -m bfcl_eval.openfunctions_evaluation --model MODEL_NAME --test-category TEST_CATEGORY
```

When specifying multiple models or test categories, separate them with **spaces**, not commas. All other flags mentioned earlier are compatible with the script execution method as well.

---

## zh_* Semantic Evaluation (LLM-based Judge) [Experimental]

This branch adds Traditional Chinese single-turn and multi-turn test categories. It also provides an optional second-stage LLM judge for Chinese cases that fail the normal BFCL checker because semantically equivalent argument values were translated, transliterated, or formatted differently.

The normal exact/AST evaluation always runs first. The semantic judge can only recover a failed `zh_*` case; it does not modify already-correct results and does not affect English categories.

### Traditional Chinese Test Categories

The Chinese prompts are stored in `bfcl_eval/data/Chinese_dataset_format/` and are registered in `bfcl_eval/constants/category_mapping.py`.

| Collection | Categories |
| --- | --- |
| Chinese single-turn | `zh_simple_python`, `zh_multiple`, `zh_parallel`, `zh_parallel_multiple`, `zh_irrelevance` |
| Chinese multi-turn | `zh_multi_turn_base`, `zh_multi_turn_miss_func`, `zh_multi_turn_miss_param`, `zh_multi_turn_long_context` |

Use `zh_all` to select all nine Chinese categories, or `zh_multi_turn` to select only the four Chinese multi-turn categories. `all_scoring` also includes the Chinese categories. Chinese multi-turn prompts reuse the corresponding English multi-turn possible-answer files, while the Chinese single-turn categories have their own possible answers.

### Running the Chinese Evaluation

Generate responses before evaluation, exactly as with the original BFCL categories:

```bash
bfcl generate \
  --model MODEL_NAME \
  --test-category zh_all \
  --include-input-log

# Run only the original BFCL checker, without semantic recovery.
bfcl evaluate \
  --model MODEL_NAME \
  --test-category zh_all \
  --zhtw-eval original
```

In addition to the normal score files, evaluation writes Chinese summaries to `score/data_chinese.csv`, `score/data_chinese_multi_turn.csv`, and `score/data_chinese_overall.csv`. The corresponding `*_no_parallel.csv` files exclude parallel categories from their summary calculation.

### Enabling the Judge

Use the `--zhtw-eval` option when running evaluation:

- `--zhtw-eval original` – default, no semantic judge.
- `--zhtw-eval openai:MODEL` – use an OpenAI model (e.g., `openai:gpt-4.1-mini`).
- `--zhtw-eval MODEL` – known OpenAI model names such as `gpt-4.1-mini` are detected automatically, so the `openai:` prefix is optional.
- `--zhtw-eval <hf_model_id>` – use a local HuggingFace model (e.g., `meta-llama/Llama-3.1-8B-Instruct`) as the judge. Backend can be `vllm` or `transformers`.

Additional judge options:

- `--zhtw-judge-backend {auto|vllm|transformers}` – select backend for HF mode (default `auto`).
- `--zhtw-vllm-tp INT` – tensor parallelism for vLLM (default `1`).
- `--zhtw-vllm-dtype {auto|float16|bfloat16|float32}` – vLLM dtype (default `auto`).
- `--zhtw-judge-debug` – print judge debug logs.

Recommended API-judge example:

```bash
bfcl evaluate \
  --model MODEL_NAME \
  --test-category zh_all \
  --zhtw-eval gpt-4.1-mini \
  --zhtw-judge-debug
```

The judge is still applied only to `zh_*` categories if `--test-category all` or `all_scoring` is used. Make sure the matching `zh_*` result files were generated first.

### When a Failed Case Is Re-evaluated

The re-evaluation is intentionally narrower than a general LLM-as-a-judge pass:

- Single-turn: the original checker must fail with `Invalid value for parameter`, and the predicted function name must match a reference function name. Only the argument dictionaries are then checked for semantic equivalence. Wrong functions, missing required arguments, type errors, and malformed calls remain failures.
- Multi-turn: after the original multi-turn checker fails, the judge receives the decoded calls and references grouped by turn. Every turn must cover all required operations; one missing critical operation makes the whole case fail.
- The judge must return only `yes` or `no`. If the API or local judge fails, the original BFCL result is retained and `semantic_judge_error` is recorded.

This guardrail is important for translation and transliteration cases: it can recover equivalent names, locations, units, or formats without allowing an unrelated function call to pass.

### OpenAI API Keys and Quota Tracking

OpenAI judge mode accepts either one key or a comma-separated key list:

```bash
export OPENAI_API_KEY="sk-..."

# Or enable automatic rotation across several keys.
export OPENAI_API_KEYS="sk-...,sk-..."
export API_DAILY_LIMIT_TOKENS=2500000
export API_ROTATE_MARGIN=25000
export API_ROTATE_VERBOSE=1
```

Usage is tracked in `.api_usage_daily.json` by default. The active key changes when its recorded usage reaches `API_DAILY_LIMIT_TOKENS - API_ROTATE_MARGIN`; the accounting day changes at 08:00 local time. Override the tracking path with `API_USAGE_FILE`. Never commit real API keys or the local usage file.

### Outputs Produced by the Judge

In addition to normal `score/` outputs, the judge writes under `score/zhtw_semantic_judge/`:

- `<MODEL>__<CATEGORY>__judge_log.jsonl` – one JSONL per model/category with the judge's inputs and decision for each case actually sent to the judge:
  - `id`, `test_category`, `model_name`, `judge_mode` (hf/openai), `judge_backend` (vllm/transformers), `judge_model`
  - question, function schema, raw model result, decoded prediction, references, and semantic scope
  - `decision`: `true|false|null`
- `recovery_rate.csv` – appended per run with columns:
  - `model, category, recovered, judged, recovery_rate`

Only `zh_*` categories are logged, and only when the judge is enabled and actually invoked. The per-model/category JSONL is overwritten on a new run, while `recovery_rate.csv` is append-only.

### Notes

- The judge only attempts to “recover” samples that originally failed; if the original evaluation already passed, the judge is skipped.
- HF judge prefers `vllm` when available; otherwise falls back to `transformers+torch` automatically (or as configured by `--zhtw-judge-backend`).
- `--zhtw-eval original` is the reproducible baseline and should be kept alongside semantic-judge results when comparing models.

### Evaluating Generated Responses

**Important:** You must have generated the model responses before running the evaluation.

Once you have the results, run:

```bash
bfcl evaluate --model MODEL_NAME --test-category TEST_CATEGORY
```

If you **only** generated a subset of benchmark entries (e.g. by using `--run-ids` during the generation step or by manually editing the result files) and you wish to evaluate *just* those entries, add the `--partial-eval` flag:

```bash
bfcl evaluate --model MODEL_NAME --test-category TEST_CATEGORY --partial-eval
```

When `--partial-eval` is set, the evaluator silently skips IDs that are not present in the model result file and computes accuracy on the remaining subset. Please note that the score may differ from a full-set evaluation and therefore might not match the official leaderboard numbers.

The `MODEL_NAME` and `TEST_CATEGORY` options are the same as those used in the [Generating LLM Responses](#generating-llm-responses) section. For details, refer to [SUPPORTED_MODELS.md](./SUPPORTED_MODELS.md) and [TEST_CATEGORIES.md](./TEST_CATEGORIES.md).

If in the previous step you stored the model responses in a custom directory, specify it using the `--result-dir` flag or set `BFCL_PROJECT_ROOT` so the evaluator can locate the files.

> Note: For unevaluated test categories, they will be marked as `N/A` in the evaluation result csv files.
> For summary columns (e.g., `Overall Acc`, `Non_Live Overall Acc`, `Live Overall Acc`, and `Multi Turn Overall Acc`), the score reported will treat all unevaluated categories as 0 during calculation.

#### Output Structure

Evaluation scores are stored in a `score/` directory under the project root (defaults to the package directory), mirroring the structure of `result/`: `score/MODEL_NAME/BFCL_v3_TEST_CATEGORY_score.json`.

- To use a custom directory for the score file, set the `BFCL_PROJECT_ROOT` environment variable or specify `--score-dir`.

Additionally, four CSV files are generated in `./score/`:

- `data_overall.csv` – Overall scores for each model. This is used for updating the leaderboard.
- `data_live.csv` – Detailed breakdown of scores for each Live (single-turn) test category.
- `data_non_live.csv` – Detailed breakdown of scores for each Non-Live (single-turn) test category.
- `data_multi_turn.csv` – Detailed breakdown of scores for each Multi-Turn test category.

#### (Optional) WandB Evaluation Logging

If you'd like to log evaluation results to WandB artifacts:

```bash
pip install -e.[wandb]
```

Mkae sure you also set `WANDB_BFCL_PROJECT=ENTITY:PROJECT` in `.env`.

#### (Alternate) Script Execution for Evaluation

For those who prefer using script execution instead of the CLI, you can run the following command:

```bash
python -m bfcl_eval.eval_checker.eval_runner --model MODEL_NAME --test-category TEST_CATEGORY
```

When specifying multiple models or test categories, separate them with **spaces**, not commas. All other flags mentioned earlier are compatible with the script execution method as well.

## Evaluating External or Fine-tuned Models

This branch has been used to evaluate externally trained and merged checkpoints from several model families, including xLAM, Qwen, Llama, Gemma, Mistral/Ministral, and GPT-OSS, as well as third-party function-calling baselines such as BitAgent, ToolACE, watt-tool, and CoALM. The many experiment-specific checkpoint names are kept in the model registry and score artifacts instead of being duplicated in this README.

### Registering a Checkpoint

Before running a new checkpoint, keep the same model key in all three places:

1. Add the key to `bfcl_eval/constants/supported_models.py`.
2. Add its `ModelConfig` to `bfcl_eval/constants/model_config.py`, selecting the handler and setting `is_fc_model` and `underscore_to_dot` for the model's output format.
3. Use that exact key as both the vLLM `--served-model-name` and BFCL `--model` value.

If a checkpoint uses an already-supported prompt/tool-call format, reuse the existing handler. A new handler is needed only when the model serializes tool calls differently. Gemma 4 FC-specific parsing is implemented in `bfcl_eval/model_handler/local_inference/gemma_4.py`.

LoRA checkpoints can be evaluated either through BFCL's `--enable-lora` support or after merging them into standalone model directories. The merge helpers and current experiment templates are under `merge_lora_model/`.

### Choosing and Editing a Generation Example

Choose the example that matches the model family, copy it to a new `.sh` or `.slurm` file, and edit the copy. Do not place local paths or credentials back into the tracked example.

| Model type | Example | Required changes |
| --- | --- | --- |
| Gemma 4 FC | `bfcl-gen-gemma4-fc-1.sh.example` | In `MODELS`, replace `path/to/your/gemma4/model` with the merged model directory and replace the value after `\|` with its registered BFCL model key. Change `LOCAL_SERVER_PORT`, `--tensor-parallel-size`, memory limits, or context length for the target machine if needed. |
| GPT-OSS | `bfcl-gen-gptoss-merged-1.sh.example` | In `MODELS`, replace `path/to/your/gpt-oss/model` and the served/BFCL model key. Change the port and GPU settings if needed. |
| Older supported models | `bfcl-gen.sh.example(for_other_old_models)` | Replace `--model` with the registered model key and `--local-model-path` with the local base or merged-checkpoint directory. Adjust `--num-gpus`, `--gpu-memory-utilization`, and `--test-category` as needed. |

For example, every `MODELS` entry in the Gemma 4 and GPT-OSS templates uses this format:

```bash
"/absolute/path/to/merged/checkpoint|REGISTERED_MODEL_KEY"
```

The path must contain at least `config.json`; the Gemma 4 template also checks for `tokenizer_config.json`. `REGISTERED_MODEL_KEY` must exactly match the entry in `supported_models.py` and `model_config.py`, because the scripts use it for both vLLM's `--served-model-name` and BFCL's `--model` argument.

Keep the model-specific parser flags from the selected example:

- Gemma 4 FC uses `--tool-call-parser gemma4`. Its example also runs `bfcl_eval/scripts/check_vllm_gemma4_compat.py` before starting the server.
- GPT-OSS uses `--tool-call-parser openai` together with `--reasoning-parser openai_gptoss`.
- Older models started directly by BFCL normally use their registered BFCL handler, so changing `--model` and `--local-model-path` is usually sufficient.

Because the older-model example filename contains parentheses, quote it when copying:

```bash
cp 'bfcl-gen.sh.example(for_other_old_models)' run-old-model.sh
```

### Using a Pre-existing vLLM Endpoint

For merged checkpoints, the most repeatable cluster workflow is to start vLLM separately and let BFCL connect to its OpenAI-compatible endpoint:

```bash
export REMOTE_OPENAI_BASE_URL="http://127.0.0.1:8001/v1"
export REMOTE_OPENAI_API_KEY="fake"
export REMOTE_OPENAI_TOKENIZER_PATH="/path/to/merged/model"

bfcl generate \
  --model REGISTERED_MODEL_KEY \
  --backend vllm \
  --skip-server-setup \
  --test-category all_scoring \
  --include-input-log \
  --num-threads 30

bfcl evaluate \
  --model REGISTERED_MODEL_KEY \
  --test-category all_scoring \
  --zhtw-eval gpt-4.1-mini
```

The vLLM tool-call and reasoning parsers are model-specific. See `bfcl-gen-gemma4-fc-1.sh.example` and `bfcl-gen-gptoss-merged-1.sh.example` for working Gemma 4 FC and GPT-OSS server configurations. `bfcl-eval.sh.example` contains the corresponding evaluation command template. Replace all local paths, ports, model keys, and credentials before use.

### Editing the Evaluation Example

After generation finishes, copy `bfcl-eval.sh.example` and edit the copy:

```bash
cp bfcl-eval.sh.example run-bfcl-eval.sh
```

The following values must be reviewed before running it:

1. Set `OPENAI_API_KEYS` to one or more real OpenAI keys if `--zhtw-eval` uses an API judge. Separate multiple keys with commas. Never commit the edited script.
2. Replace the value of `--model` with the exact model key used during generation. To evaluate several generated models in one command, separate the model keys with commas.
3. Select the required data with `--test-category`: use `zh_all` for only the Chinese set, `all_scoring` for all scored English and Chinese categories, or another registered category/collection.
4. Keep `--zhtw-eval gpt-4.1-mini` to re-evaluate eligible Chinese semantic/transliteration failures through the API. Use `--zhtw-eval original` to disable API re-evaluation and retain only the original BFCL checker.
5. `analyze_multi_turn_errors.py` and `plot_ckpt_lora.py` are optional post-processing steps; they are not required for producing the BFCL score files.

Example with two generated models:

```bash
export OPENAI_API_KEYS="sk-...,sk-..."
export API_DAILY_LIMIT_TOKENS=2500000
export API_ROTATE_MARGIN=25000
export API_ROTATE_VERBOSE=1

bfcl evaluate \
  --zhtw-eval gpt-4.1-mini \
  --test-category all_scoring \
  --model REGISTERED_MODEL_KEY_1,REGISTERED_MODEL_KEY_2
```

The result files must already exist for every selected model/category combination. The `--model` value must match the generation key and its `result/<MODEL>/` directory; otherwise the evaluator cannot locate the generated responses. If quota settings are changed in a shell script, remember to `export` them so the Python evaluation process receives the values.

### Handoff File Map

| Area | Main files |
| --- | --- |
| Chinese category registration | `bfcl_eval/constants/category_mapping.py` |
| Chinese dataset routing and possible answers | `bfcl_eval/utils.py`, `bfcl_eval/data/Chinese_dataset_format/`, `bfcl_eval/data/possible_answer/` |
| API/local semantic judge | `bfcl_eval/eval_checker/zhtw_semantic_judge.py`, `bfcl_eval/eval_checker/openai_utils.py` |
| Judge integration and logs | `bfcl_eval/eval_checker/eval_runner.py`, `bfcl_eval/constants/eval_config.py` |
| Chinese score CSVs | `bfcl_eval/eval_checker/eval_runner_helper.py`, `bfcl_eval/constants/column_headers.py` |
| Custom model registry | `bfcl_eval/constants/supported_models.py`, `bfcl_eval/constants/model_config.py` |
| Local checkpoint examples | `bfcl-gen-gemma4-fc-1.sh.example`, `bfcl-gen-gptoss-merged-1.sh.example`, `bfcl-gen.sh.example(for_other_old_models)`, `bfcl-eval.sh.example`, `merge_lora_model/` |

## Contributing & How to Add New Models

We welcome contributions! To add a new model:

1. Review `bfcl_eval/model_handler/base_handler.py` and/or `bfcl_eval/model_handler/local_inference/base_oss_handler.py` (if your model is hosted locally).
2. Implement a new handler class for your model.
3. Update `bfcl_eval/constants/model_config.py`.
4. Submit a Pull Request.

For detailed steps, please see the [Contributing Guide](./CONTRIBUTING.md).

---

## Additional Resources

- [Discord](https://discord.gg/grXXvj9Whz) (`#leaderboard` channel)
- [Project Website](https://gorilla.cs.berkeley.edu/leaderboard.html#leaderboard)

All the leaderboard statistics, and data used to train the models are released under Apache 2.0.
BFCL is an open source effort from UC Berkeley and we welcome contributors.
For any comments, criticisms, or questions, please feel free to raise an issue or a PR. You can also reach us via [email](mailto:huanzhimao@berkeley.edu).
