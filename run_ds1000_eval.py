
import argparse
import json
import os
import re
import time
import logging
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


BASE_URL = "https://llm-proxy.t-tech.team/v1"
MODEL = "openai/gpt-5"
TEMPERATURE = 0.3

INSTRUCTION = (
    "Solve this Python data-science task. "
    "Return the solution as Python code.\n\n"
)
CONTEXT_MAX_CHARS = 12000


def build_prompt(row):
    prompt = row.get("prompt")

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Missing prompt field")

    context = row.get("code_context") or ""
    if not isinstance(context, str):
        context = str(context)

    if context.strip() and "Notebook context:" not in prompt:
        prompt = (
            "Notebook context:\n"
            + context[-CONTEXT_MAX_CHARS:]
            + "\n\nTask:\n"
            + prompt
        )

    return INSTRUCTION + prompt


def clean_answer(text):
    text = (text or "").strip()

    match = re.search(
        r"```(?:python|py)?[ \t]*\n?(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        text = match.group(1).strip()

    return text.strip()


def get_usage(response):
    usage = getattr(response, "usage", None)

    result = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }

    if usage is None:
        return result

    result["prompt_tokens"] = getattr(
        usage, "prompt_tokens", getattr(usage, "input_tokens", None)
    )
    result["completion_tokens"] = getattr(
        usage, "completion_tokens", getattr(usage, "output_tokens", None)
    )
    result["total_tokens"] = getattr(usage, "total_tokens", None)

    if (
        result["total_tokens"] is None
        and result["prompt_tokens"] is not None
        and result["completion_tokens"] is not None
    ):
        result["total_tokens"] = (
            result["prompt_tokens"] + result["completion_tokens"]
        )

    return result


def evaluate(row, answer):
    result = {
        "passed": False,
        "execution_passed": None,
        "string_passed": None,
        "eval_error": None,
    }

    context = row.get("code_context")

    if not isinstance(context, str) or not context.strip():
        result["eval_error"] = "Missing code_context"
        return result

    namespace = {"__name__": "__ds1000_eval__"}

    try:
        exec(
            compile(context, "<ds1000_context>", "exec"),
            namespace,
        )

        test_execution = namespace.get("test_execution")
        test_string = namespace.get("test_string")

        if not callable(test_execution):
            result["eval_error"] = (
                "test_execution(solution) not found in code_context"
            )
            return result

        # Успех — отсутствие исключения, а не обязательно True.
        test_execution(answer)
        result["execution_passed"] = True

        if callable(test_string):
            test_string(answer)
            result["string_passed"] = True

        result["passed"] = True

    except Exception as exc:
        result["passed"] = False
        result["eval_error"] = f"{type(exc).__name__}: {exc}"

        # Если тесты уже выполнялись, отмечаем их провал.
        if result["execution_passed"] is None:
            result["execution_passed"] = False
        elif result["string_passed"] is None:
            result["string_passed"] = False

    return result


def get_library(row):
    metadata = row.get("metadata") or {}

    if isinstance(metadata, dict):
        for key in ("library", "lib", "library_name"):
            if metadata.get(key):
                return str(metadata[key])

    return "Unknown"


def setup_logger(log_path):
    logger = logging.getLogger("ds1000")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.FileHandler(
        log_path, mode="w", encoding="utf-8"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an API model on DS-1000"
    )
    parser.add_argument("--data", default="data/ds1000.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--pause", type=float, default=0.0)
    parser.add_argument(
        "--output",
        default="results/ds1000_gpt5_results.jsonl",
    )
    args = parser.parse_args()

    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        parser.error("Set the LLM_API_KEY environment variable.")

    data_path = Path(args.data)
    if not data_path.exists():
        parser.error(f"Dataset not found: {data_path}")

    with data_path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

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

    log_path = output_path.with_suffix(".log")
    summary_path = output_path.with_suffix(".summary.json")
    logger = setup_logger(log_path)

    stats = {
        "attempted": 0,
        "api_success": 0,
        "api_errors": 0,
        "evaluated": 0,
        "passed": 0,
        "failed": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "examples_with_usage": 0,
        "generation_seconds": 0.0,
    }

    logger.info("Model: %s", MODEL)
    logger.info("Dataset: %s", data_path)
    logger.info("Examples: %d", len(rows))

    with output_path.open("w", encoding="utf-8") as output:
        for index, row in enumerate(tqdm(rows, desc="DS-1000")):
            stats["attempted"] += 1
            started = time.perf_counter()

            record = {
                "index": index,
                "model": MODEL,
                "library": get_library(row),
            }

            model_prompt = ""
            raw_answer = ""
            answer = ""
            usage = {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            }

            try:
                model_prompt = build_prompt(row)
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
                                        "Solve this Python data-science task. "
                                        "Return the solution as Python code."
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": model_prompt,
                                },
                            ],
                        )
                        break

                    except Exception as exc:
                        last_error = exc
                        if attempt < args.retries:
                            time.sleep(min(2 ** attempt, 10))

                if response is None:
                    raise RuntimeError(str(last_error))

                raw_answer = response.choices[0].message.content or ""
                answer = clean_answer(raw_answer)
                usage = get_usage(response)

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

                evaluation = evaluate(row, answer)
                stats["evaluated"] += 1

                if evaluation["passed"]:
                    stats["passed"] += 1
                else:
                    stats["failed"] += 1

                record.update({
                    "status": "evaluated",
                    "prompt": row.get("prompt"),
                    "input_prompt": model_prompt,
                    "raw_answer": raw_answer,
                    "answer": answer,
                    **usage,
                    **evaluation,
                })

                if evaluation["eval_error"]:
                    logger.warning(
                        "Index %d evaluator error: %s",
                        index,
                        evaluation["eval_error"],
                    )

            except Exception as exc:
                stats["api_errors"] += 1
                record.update({
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "prompt": row.get("prompt"),
                    "input_prompt": model_prompt,
                    "raw_answer": raw_answer,
                    "answer": answer,
                    **usage,
                })

                logger.exception("Index %d failed", index)

            elapsed = time.perf_counter() - started
            record["elapsed_seconds"] = round(elapsed, 4)
            stats["generation_seconds"] += elapsed

            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()

            if args.pause:
                time.sleep(args.pause)

    n = stats["evaluated"]

    summary = {
        **stats,
        "pass_rate": stats["passed"] / n if n else None,
        "pass_rate_percent": 100 * stats["passed"] / n if n else None,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": args.max_tokens,
        "evaluation_note": (
            "Success means test_execution and optional test_string "
            "completed without exceptions."
        ),
    }

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== API evaluation summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nResults:", output_path)
    print("Log:", log_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()










python evaluate_gpt5.py \
  --data data/ds1000.jsonl \
  --limit 5 \
  --max-tokens 1024 \
  --output results/ds1000_gpt5_test.jsonl


python evaluate_gpt5.py \
  --data data/ds1000.jsonl \
  --max-tokens 1024 \
  --output results/ds1000_gpt5_results.jsonl
