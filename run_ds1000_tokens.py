#!/usr/bin/env python3
"""
Run an OpenAI-compatible model over the DS-1000 benchmark and count tokens.

Install:
    pip install openai datasets tqdm

Example:
    export LLM_API_KEY="your-key"
    python run_ds1000_tokens.py --base-url "https://llm-proxy.t-tech.team/v1"

Notes:
- The script counts tokens reported by the API (usage.prompt_tokens and
  usage.completion_tokens). If the proxy does not return usage, it records
  that fact rather than guessing.
- DS-1000 prompts can be large. Use --limit for a smoke test first.
- This measures model calls/token usage; it does not execute generated code
  or calculate DS-1000 pass@1 accuracy.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm


DEFAULT_BASE_URL = "https://llm-proxy.t-tech.team/v1"
DEFAULT_MODEL = "openai/gpt-5"


def get_field(example: Dict[str, Any], candidates, default=""):
    """Return the first non-empty value found among candidate dataset fields."""
    for key in candidates:
        value = example.get(key)
        if value is not None and value != "":
            return value
    return default


def make_prompt(example: Dict[str, Any]) -> str:
    """
    DS-1000 versions may expose slightly different field names. Prefer the
    benchmark's prompt/instruction field; otherwise build a prompt from the
    available question and code context fields.
    """
    prompt = get_field(example, ("prompt", "instruction", "question", "text"))
    if prompt:
        return str(prompt)

    parts = []
    for label, keys in (
        ("Library", ("library",)),
        ("Question", ("problem", "description")),
        ("Code context", ("reference_code", "code_context", "context")),
    ):
        value = get_field(example, keys)
        if value:
            parts.append(f"{label}:\n{value}")
    if not parts:
        # Keep this failure explicit; silently sending an empty prompt is misleading.
        raise ValueError(
            "Couldn't find a prompt field in this DS-1000 row. "
            f"Available fields: {', '.join(example.keys())}"
        )
    return "\n\n".join(parts)


def extract_usage(response) -> Dict[str, Optional[int]]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}

    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)

    # Some compatible APIs expose the newer input_tokens/output_tokens names.
    if prompt_tokens is None:
        prompt_tokens = getattr(usage, "input_tokens", None)
    if completion_tokens is None:
        completion_tokens = getattr(usage, "output_tokens", None)
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run a model on DS-1000 prompts and collect API token usage."
    )
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY"))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=0, help="Number of examples; 0 means all.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset", default="xlangai/DS-1000")
    parser.add_argument("--output", default="ds1000_token_usage.jsonl")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--pause", type=float, default=0.0, help="Seconds to wait between requests.")
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are a coding assistant. Complete the given DS-1000 programming task. "
            "Return the code answer only, without explanations."
        ),
    )
    args = parser.parse_args()

    if not args.api_key:
        parser.error("Set LLM_API_KEY or pass --api-key. Do not put secrets in the source file.")

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=0,  # retry loop below keeps per-example behavior explicit
    )

    print(f"Loading dataset: {args.dataset} [{args.split}]")
    dataset = load_dataset(args.dataset, split=args.split)
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    totals = {
        "examples_attempted": 0,
        "examples_succeeded": 0,
        "examples_failed": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "examples_with_usage": 0,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for idx, example in enumerate(tqdm(dataset, desc="DS-1000")):
            totals["examples_attempted"] += 1
            record = {
                "index": idx,
                "dataset": args.dataset,
                "model": args.model,
                "status": "error",
            }

            try:
                prompt = make_prompt(example)
                record["prompt_chars"] = len(prompt)
                last_error = None
                response = None

                for attempt in range(args.retries + 1):
                    try:
                        response = client.chat.completions.create(
                            model=args.model,
                            temperature=args.temperature,
                            messages=[
                                {"role": "system", "content": args.system_prompt},
                                {"role": "user", "content": prompt},
                            ],
                        )
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt < args.retries:
                            time.sleep(min(2 ** attempt, 15))

                if response is None:
                    raise RuntimeError(f"API request failed after retries: {last_error}")

                answer = response.choices[0].message.content or ""
                usage = extract_usage(response)
                record.update(
                    {
                        "status": "ok",
                        "answer_chars": len(answer),
                        **usage,
                    }
                )
                totals["examples_succeeded"] += 1

                if usage["total_tokens"] is not None:
                    totals["examples_with_usage"] += 1
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        if usage[key] is not None:
                            totals[key] += usage[key]

            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                totals["examples_failed"] += 1

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            if args.pause:
                time.sleep(args.pause)

    summary = {
        **totals,
        "model": args.model,
        "base_url": args.base_url,
        "dataset": args.dataset,
        "split": args.split,
        "temperature": args.temperature,
        "output_file": str(output_path),
        "note": (
            "Token totals include only API-reported usage. Examples without usage "
            "data are not estimated."
        ),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Token usage summary ===")
    print(f"Model:                 {args.model}")
    print(f"Attempted:             {totals['examples_attempted']}")
    print(f"Succeeded:             {totals['examples_succeeded']}")
    print(f"Failed:                {totals['examples_failed']}")
    print(f"Examples with usage:   {totals['examples_with_usage']}")
    print(f"Prompt tokens:         {totals['prompt_tokens']}")
    print(f"Completion tokens:     {totals['completion_tokens']}")
    print(f"Total tokens:          {totals['total_tokens']}")
    print(f"Per-example results:   {output_path}")
    print(f"Summary JSON:          {summary_path}")


if __name__ == "__main__":
    main()
