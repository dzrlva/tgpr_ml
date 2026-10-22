
import os
from openai import OpenAI

BASE_URL = "https://llm-proxy.t-tech.team/v1"
MODEL = "openai/gpt-5"


def main():
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set the LLM_API_KEY environment variable first."
        )

    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=45,
    )

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.3,
        max_tokens=20,
        messages=[
            {
                "role": "user",
                "content": "Reply exactly with API_OK",
            }
        ],
    )

    print("API request successful")
    print("Model response:", response.choices[0].message.content)

    usage = response.usage
    if usage:
        print("Prompt tokens:", usage.prompt_tokens)
        print("Completion tokens:", usage.completion_tokens)
        print("Total tokens:", usage.total_tokens)
    else:
        print("Token usage was not returned by the API.")


if __name__ == "__main__":
    main()
