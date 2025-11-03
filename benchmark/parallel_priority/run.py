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

from benchmark.utils_completions_priority import call_openai_completions_with_time


openai_models = ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "o4-mini"]
res_dir = "../../results-parallel-priority"


async def call_completions_parallel(model, n_samples, prompt):
    prompts = [prompt for _ in range(n_samples)]
    city = "Wisc"
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
    else:
        raise ValueError("the model is not defined")
    print(f"total_time: {total_time}")

    utc_now = datetime.datetime.now(datetime.timezone.utc)
    current_date_str = utc_now.strftime("%Y-%m-%d-%H")
    model_name = model
    filename = f"{res_dir}/{current_date_str}_{city}_{model_name}_{n_samples}_priority_call.txt"

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
    model_list = openai_models
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
            time.sleep(3200)
