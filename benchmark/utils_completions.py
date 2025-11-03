import anthropic
import time
from openai import AsyncOpenAI
from google import genai
from google.genai import types
from together import AsyncTogether
import aiohttp
import json

OAI_API_KEY = "OAI_API_KEY"
DS_API_KEY = "DS_API_KEY"
TOGETHER_API_KEY = "TOGETHER_API_KEY"
ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
GOOGLE_API_KEY = "GOOGLE_API_KEY"
CENTML_API_KEY = "CENTML_API_KEY"


async def call_deepseek_completions_with_time(
    prompt, model="deepseek-chat", max_tokens=256
):
    client = AsyncOpenAI(
        api_key=DS_API_KEY,
        base_url="https://api.deepseek.com",
    )
    t_start = time.perf_counter()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
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
        t_end = time.time()
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


async def call_openai_completions_with_time(prompt, model="gpt-4o", max_tokens=256):
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


async def call_togetherai_completions_with_time(
    prompt, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", max_tokens=256
):
    client = AsyncOpenAI(
        api_key=TOGETHER_API_KEY,
        base_url="https://api.together.xyz/v1",
    )
    t_start = time.perf_counter()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
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


async def call_anthropic_completions_with_time(
    prompt, model="claude-3-haiku-20240307", max_tokens=256
):
    client = anthropic.AsyncAnthropic(
        api_key=ANTHROPIC_API_KEY,
    )
    t_start = time.perf_counter()
    try:
        response = await client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        t_end = time.perf_counter()
        elapsed_time = t_end - t_start
        await client.close()
        return {
            "prompt": prompt,
            "response": response.content[0].text,
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
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


async def call_google_genai_completions_with_time(
    prompt, model="gemini-1.5-pro", max_tokens=256
):
    client = genai.Client(
        api_key=GOOGLE_API_KEY,
    )
    t_start = time.perf_counter()
    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=[prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
            ),
        )
        t_end = time.perf_counter()
        elapsed_time = t_end - t_start
        return {
            "prompt": prompt,
            "response": response.text,
            "prompt_tokens": response.usage_metadata.prompt_token_count,
            "completion_tokens": response.usage_metadata.candidates_token_count,
            "time": elapsed_time,
            "start_time": t_start,
            "end_time": t_end,
        }
    except Exception as e:
        t_end = time.perf_counter()
        elapsed_time = t_end - t_start
        return {
            "prompt": prompt,
            "response": e,
            "prompt_tokens": None,
            "completion_tokens": None,
            "time": elapsed_time,
            "start_time": t_start,
            "end_time": t_end,
        }


async def call_dedicated_completions_with_time(prompt, max_tokens=256):
    client = AsyncTogether(
        api_key=TOGETHER_API_KEY,
    )
    t_start = time.perf_counter()
    try:
        response = await client.chat.completions.create(
            model="songbian/Qwen/Qwen2.5-72B-Instruct-Turbo-4eb8cc5e",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        t_end = time.perf_counter()
        elapsed_time = t_end - t_start
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
        return {
            "prompt": prompt,
            "response": e,
            "prompt_tokens": None,
            "completion_tokens": None,
            "time": elapsed_time,
            "start_time": t_start,
            "end_time": t_end,
        }


async def call_centml_completions_with_time(
    prompt, model="meta-llama/Llama-4-Scout-17B-16E-Instruct", max_tokens=256
):
    headers = {
        "Authorization": f"Bearer {CENTML_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "n": 1,
        "stream": False,
        "temperature": 0.0,
        "top_p": 1,
        "top_k": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": [],
    }
    json_payload = json.dumps(data)
    url = "https://api.centml.com/openai/v1/chat/completions"

    t_start = time.perf_counter()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, headers=headers, data=json_payload
            ) as response:
                response.raise_for_status()  # Raise an exception for bad status codes
                result = await response.json()
                t_end = time.perf_counter()
                elapsed_time = t_end - t_start

                # Extract relevant information from the CentML response
                if result and "choices" in result and len(result["choices"]) > 0:
                    completion = result["choices"][0]["message"]["content"]
                    prompt_tokens = result.get("usage", {}).get("prompt_tokens")
                    completion_tokens = result.get("usage", {}).get("completion_tokens")
                else:
                    completion = None
                    prompt_tokens = None
                    completion_tokens = None

                return {
                    "prompt": prompt,
                    "response": completion,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "time": elapsed_time,
                    "start_time": t_start,
                    "end_time": t_end,
                }
    except aiohttp.ClientError as e:
        t_end = time.perf_counter()
        elapsed_time = t_end - t_start
        return {
            "prompt": prompt,
            "response": f"AIOHTTP Client Error: {e}",
            "prompt_tokens": None,
            "completion_tokens": None,
            "time": elapsed_time,
            "start_time": t_start,
            "end_time": t_end,
        }
    except json.JSONDecodeError as e:
        t_end = time.perf_counter()
        elapsed_time = t_end - t_start
        return {
            "prompt": prompt,
            "response": f"JSON Decode Error: {e}",
            "prompt_tokens": None,
            "completion_tokens": None,
            "time": elapsed_time,
            "start_time": t_start,
            "end_time": t_end,
        }
    except Exception as e:
        t_end = time.perf_counter()
        elapsed_time = t_end - t_start
        return {
            "prompt": prompt,
            "response": f"An unexpected error occurred: {e}",
            "prompt_tokens": None,
            "completion_tokens": None,
            "time": elapsed_time,
            "start_time": t_start,
            "end_time": t_end,
        }
