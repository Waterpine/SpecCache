import asyncio
import datetime
import os
import random
import sys
import time

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.join(os.path.dirname(__file__), os.path.pardir), os.path.pardir
        )
    )
)

from benchmark.utils_completions import (
    call_deepseek_completions_with_time,
    call_openai_completions_with_time,
    call_togetherai_completions_with_time,
    call_anthropic_completions_with_time,
    call_google_genai_completions_with_time,
)


openai_models = ["gpt-4o-mini", "gpt-4o"]
togetherai_models = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "Qwen/QwQ-32B",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "moonshotai/Kimi-K2-Instruct",
]
deepseek_models = ["deepseek-chat", "deepseek-reasoner"]
anthropic_models = [
    "claude-3-5-sonnet-20240620",
    "claude-3-5-haiku-20241022",
    "claude-3-7-sonnet-20250219",
]
google_genai_models = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

res_dir = "../../results-parallel"


async def call_completions_parallel(model, n_samples, prompt):
    prompts = [prompt for _ in range(n_samples)]
    city = "Mass"
    if model in openai_models:
        t0 = time.perf_counter()
        tasks = [
            asyncio.create_task(
                call_openai_completions_with_time(
                    prompt=input_prompt,
                    model=model,
                )
            )
            for input_prompt in prompts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        total_time = time.perf_counter() - t0
    elif model in togetherai_models:
        t0 = time.perf_counter()
        tasks = [
            asyncio.create_task(
                call_togetherai_completions_with_time(
                    prompt=input_prompt,
                    model=model,
                )
            )
            for input_prompt in prompts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        total_time = time.perf_counter() - t0
    elif model in deepseek_models:
        t0 = time.perf_counter()
        tasks = [
            asyncio.create_task(
                call_deepseek_completions_with_time(
                    prompt=input_prompt,
                    model=model,
                )
            )
            for input_prompt in prompts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        total_time = time.perf_counter() - t0
    elif model in anthropic_models:
        t0 = time.perf_counter()
        tasks = [
            asyncio.create_task(
                call_anthropic_completions_with_time(
                    prompt=input_prompt,
                    model=model,
                )
            )
            for input_prompt in prompts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        total_time = time.perf_counter() - t0
    elif model in google_genai_models:
        t0 = time.perf_counter()
        tasks = [
            asyncio.create_task(
                call_google_genai_completions_with_time(
                    prompt=input_prompt,
                    model=model,
                )
            )
            for input_prompt in prompts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        total_time = time.perf_counter() - t0
    else:
        raise ValueError("the model is not defined")
    print(f"total_time: {total_time}")

    utc_now = datetime.datetime.now(datetime.timezone.utc)
    current_date_str = utc_now.strftime("%Y-%m-%d-%H")
    if model in togetherai_models:
        model_name = model.split("/")[1]
    else:
        model_name = model
    filename = f"{res_dir}/{current_date_str}_{city}_{model_name}_{n_samples}_parallel_call.txt"

    with open(filename, "a") as file:
        file.write(f"total_time: {total_time}\n")

    for res in results:
        print(f"Time Taken: {res['time']:.2f} seconds\n")
        with open(filename, "a") as file:
            file.write(f"start_time: {res['start_time']}\n")
            file.write(f"end_time: {res['end_time']}\n")
            file.write(f"time: {res['time']}\n")
            file.write(f"response: {res['response']}\n")
            file.write(f"prompt_tokens: {res['prompt_tokens']}\n")
            file.write(f"completion_tokens: {res['completion_tokens']}\n")


if __name__ == "__main__":
    os.makedirs(res_dir, exist_ok=True)
    model_list = (
        openai_models
    )
    samples_list = [1]

    while True:
        item = random.choice(
            [
                "Apple",
                "Banana",
                "Blackberry",
                "Carambola",
                "Cherry",
                "Coconut",
                "Damson",
                "Kiwifruit",
                "Lemon",
                "Loquat",
                "Mango",
                "Papaya",
                "Peach",
                "Pear",
                "Pineapple",
                "Watermelon",
            ]
        )
        prompt = (
            f"Tell a story about {item}. "
            f"Make the story detailed, with rich descriptions, character development, and dialogue. "
            f"Aim for a story that would take at least 256 tokens to tell."
        )
        for n_sample in samples_list:
            for model in model_list:
                print(f"model: {model}")
                # python >= 3.11
                # with asyncio.Runner() as runner:
                # python <= 3.10
                asyncio.run(
                    call_completions_parallel(
                        model=model,
                        n_samples=n_sample,
                        prompt=prompt,
                    )
                )
                time.sleep(120)
            print("======start sleep======")
            time.sleep(1500)
