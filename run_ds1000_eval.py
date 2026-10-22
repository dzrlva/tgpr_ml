
import argparse
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


BASE_URL = "https://llm-proxy.t-tech.team/v1"
MODEL = "openai/gpt-5"
TEMPERATURE = 0.3


def clean_answer(text):
    text = (text or "").strip()
    text = re.sub(
        r"^\s*```(?:python|py)?\s*\n", "", text, flags=re.I
    )
    text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


def get_usage(response):
    usage = getattr(response, "usage", None)
    if not usage:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }

    prompt = getattr(
        usage, "prompt_tokens",
        getattr(usage, "input_tokens", None),
    )
    completion = getattr(
        usage, "completion_tokens",
        getattr(usage, "output_tokens", None),
    )
    total = getattr(usage, "total_tokens", None)

    if total is None and prompt is not None and completion is not None:
        total = prompt + completion

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def evaluate(row, answer):
    """
    Uses the benchmark's code_context test functions.
    Execute only in an isolated environment.
    """
    context = row.get("code_context")

    if not isinstance(context, str) or not context.strip():
        return {
            "passed": False,
            "eval_error": "Missing code_context",
        }

    try:
        namespace = {"__name__": "__ds1000_eval__"}
        exec(compile(context, "<ds1000_context>", "exec"),
             namespace, namespace)

        test_execution = namespace.get("test_execution")
        test_string = namespace.get("test_string")

        if not callable(test_execution):
            raise ValueError("test_execution(solution) not found")

        execution_passed = bool(test_execution(answer))
        string_passed = (
            bool(test_string(answer))
            if callable(test_string)
            else True
        )

        return {
            "passed": execution_passed and string_passed,
            "execution_passed": execution_passed,
            "string_passed": string_passed,
        }

    except Exception as exc:
        return {
            "passed": False,
            "eval_error": f"{type(exc).__name__}: {exc}",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ds1000.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--pause", type=float, default=0.0)
    parser.add_argument(
        "--output", default="results/ds1000_results.jsonl"
    )
    args = parser.parse_args()

    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        parser.error("Set the LLM_API_KEY environment variable.")

    data_path = Path(args.data)
    if not data_path.exists():
        parser.error(
            f"Dataset not found: {data_path}. "
            "Run download_ds1000.py first."
        )

    with data_path.open(encoding="utf-8") as file:
        rows = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if args.limit > 0:
        rows = rows[:args.limit]

    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=120,
        max_retries=0,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "attempted": 0,
        "api_success": 0,
        "api_errors": 0,
        "evaluated": 0,
        "passed": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "examples_with_usage": 0,
    }

    with output_path.open("w", encoding="utf-8") as output:
        for index, row in enumerate(tqdm(rows, desc="DS-1000")):
            stats["attempted"] += 1
            record = {"index": index, "model": MODEL}

            try:
                prompt = row.get("prompt")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("Missing prompt field")

                response = None
                last_error = None

                for attempt in range(args.retries + 1):
                    try:
                        response = client.chat.completions.create(
                            model=MODEL,
                            temperature=TEMPERATURE,
                            max_tokens=args.max_tokens,
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "Solve the DS-1000 coding task. "
                                        "Return only the code needed to "
                                        "complete the prompt, without "
                                        "Markdown or explanation."
                                    ),
                                },
                                {"role": "user", "content": prompt},
                            ],
                        )
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt < args.retries:
                            time.sleep(min(2 ** attempt, 10))

                if response is None:
                    raise RuntimeError(str(last_error))

                answer = clean_answer(
                    response.choices[0].message.content
                )
                usage = get_usage(response)

                record.update({
                    "status": "generated",
                    "answer": answer,
                    **usage,
                })
                stats["api_success"] += 1

                if usage["total_tokens"] is not None:
                    stats["examples_with_usage"] += 1
                    for key in (
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                    ):
                        if usage[key] is not None:
                            stats[key] += usage[key]

                # WARNING: executes benchmark context and generated code.
                evaluation = evaluate(row, answer)
                record.update(evaluation)
                record["status"] = "evaluated"
                stats["evaluated"] += 1

                if evaluation.get("passed"):
                    stats["passed"] += 1

            except Exception as exc:
                record.update({
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                stats["api_errors"] += 1

            output.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            output.flush()

            if args.pause:
                time.sleep(args.pause)

    n = stats["evaluated"]
    summary = {
        **stats,
        "pass_rate": stats["passed"] / n if n else None,
        "pass_rate_percent": (
            100 * stats["passed"] / n if n else None
        ),
        "model": MODEL,
        "temperature": TEMPERATURE,
        "token_note": (
            "Counts API-reported usage only; missing usage is not estimated."
        ),
        "evaluation_note": (
            "Pass rate is passed/evaluated. Confirm the dataset's "
            "code_context and test functions match the official "
            "DS-1000 evaluation protocol."
        ),
    }

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== DS-1000 summary ===")
    for key in (
        "attempted",
        "api_success",
        "api_errors",
        "evaluated",
        "passed",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ):
        print(f"{key}: {stats[key]}")

    print("Pass rate:", summary["pass_rate_percent"], "%")
    print("Per-example results:", output_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
