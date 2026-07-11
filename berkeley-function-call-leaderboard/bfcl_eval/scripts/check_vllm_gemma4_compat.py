"""Validate that the active vLLM can serve a local Gemma 4 checkpoint."""

from __future__ import annotations

import argparse
import json
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


MIN_GEMMA4_UNIFIED_VLLM = (0, 23, 0)


def _version_tuple(raw_version: str) -> tuple[int, int, int]:
    """Return a comparable three-part tuple for stable and development builds."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", raw_version)
    if match is None:
        raise ValueError(f"Unrecognized vLLM version: {raw_version!r}")
    return tuple(int(part) for part in match.groups())


def _load_config(model_path: Path) -> dict:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json not found: {config_path}")

    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Expected a JSON object in {config_path}")
    return config


def is_gemma4_unified(config: dict) -> bool:
    architectures = config.get("architectures") or []
    return (
        config.get("model_type") == "gemma4_unified"
        or "Gemma4UnifiedForConditionalGeneration" in architectures
    )


def validate(model_path: Path, vllm_version: str) -> tuple[bool, str]:
    config = _load_config(model_path)

    if not is_gemma4_unified(config):
        return True, f"vLLM {vllm_version}: standard Gemma 4 checkpoint"

    if _version_tuple(vllm_version) < MIN_GEMMA4_UNIFIED_VLLM:
        minimum = ".".join(str(part) for part in MIN_GEMMA4_UNIFIED_VLLM)
        architectures = ", ".join(config.get("architectures") or ["unknown"])
        return False, (
            f"Gemma 4 Unified checkpoint detected ({architectures}), but vLLM "
            f"{vllm_version} is too old. Gemma 4 Unified requires vLLM >= {minimum}. "
            "Older vLLM versions fall back to the Transformers backend and can fail "
            "during RMSNorm profiling with a hidden-size mismatch. Upgrade the active "
            "BFCL environment with: python -m pip install -e '.[oss_eval_vllm]'"
        )

    return True, f"vLLM {vllm_version}: Gemma 4 Unified support available"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    args = parser.parse_args()

    try:
        vllm_version = version("vllm")
    except PackageNotFoundError:
        print("ERROR: vLLM is not installed in the active Python environment.", file=sys.stderr)
        return 2

    try:
        compatible, message = validate(args.model_path, vllm_version)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    stream = sys.stdout if compatible else sys.stderr
    prefix = "OK" if compatible else "ERROR"
    print(f"{prefix}: {message}", file=stream)
    return 0 if compatible else 2


if __name__ == "__main__":
    raise SystemExit(main())
