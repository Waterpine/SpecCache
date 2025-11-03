import time
from openai import AsyncOpenAI


OAI_API_KEY = "OAI_API_KEY"


async def call_openai_completions_with_time(
    prompt, model="gpt-4o", max_tokens=256, tier="default"
):
    client = AsyncOpenAI(
        api_key=OAI_API_KEY,
    )
    t_start = time.perf_counter()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            service_tier=tier,
        )
        t_end = time.perf_counter()
        elapsed_time = t_end - t_start
        await client.close()
        return {
            "prompt": prompt,
            "response": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "time": elapsed_time,
            "start_time": t_start,
            "end_time": t_end,
        }
    except Exception as e:
        t_end = time.perf_counter()
        elapsed_time = t_end - t_start
        await client.close()
        return {
            "prompt": prompt,
            "response": e,
            "prompt_tokens": None,
            "completion_tokens": None,
            "time": elapsed_time,
            "start_time": t_start,
            "end_time": t_end,
        }
