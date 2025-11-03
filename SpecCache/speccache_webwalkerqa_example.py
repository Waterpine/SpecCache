from speccache_core import AssistedWebWalkerCore
import os
import datasets
import time
import json
import httpx
import random

def run_walker_with_retry(walker, base_url, query, memory, max_rounds, metrics_file, use_assisted_llm=True, max_retries=3, base_delay=2):
    """
    Run the walker with retry logic for handling connection errors.
    
    Args:
        walker: The WebWalker instance
        base_url: The base URL to start from
        query: The query to process
        memory: Initial memory
        max_rounds: Maximum number of rounds
        metrics_file: File to save metrics
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
    
    Returns:
        The result from the walker, or None if all retries failed
    """
    for attempt in range(max_retries + 1):
        try:
            print(f"Attempt {attempt + 1}/{max_retries + 1} for walker execution...")
            
            result = walker.run(
                base_url=base_url,
                query=query,
                memory=memory,
                max_rounds=max_rounds,
                metrics_file=metrics_file,
                use_assisted_llm=use_assisted_llm
            )

            print(f"Successfully completed walker execution on attempt {attempt + 1}")
            return result
            
        except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout, 
                httpx.WriteTimeout, httpx.PoolTimeout, httpx.ConnectTimeout) as e:
            print(f"Connection error on attempt {attempt + 1}: {type(e).__name__}: {e}")
            
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                print(f"All {max_retries + 1} attempts failed. Skipping this example.")
                return None
                
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}: {type(e).__name__}: {e}")
            
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                print(f"All {max_retries + 1} attempts failed. Skipping this example.")
                return None
    
    return None

def main():

    api_key = "YOUR API KEY"

    test_case = {
        "model_name": "gpt-5-nano", 
        "draft_name_1": "gpt-4.1-mini", 
        "draft_name_2": "gpt-4.1", 
        "use_assisted_llm": True
    }
    
    print(f"\n{'='*80}")
    print(f"TESTING CONTROLLED ASSISTED WALKER")
    print(f"Main Model: {test_case['model_name']}")
    print(f"Draft Model 1: {test_case['draft_name_1']}")
    print(f"Draft Model 2: {test_case['draft_name_2']}")
    print(f"Assisted LLM: {test_case['use_assisted_llm']}")
    print(f"{'='*80}")
    
    walker = AssistedWebWalkerCore(
        api_key=api_key, 
        model=test_case['model_name'], 
        draft_name_1=test_case['draft_name_1'],
        draft_name_2=test_case['draft_name_2'],
        provider="<YOUR PROVIDER NAME>", 
        model_server="<YOUR SERVER URL>"
    )


    metrics_dir = "controlled_metrics_priority"
    os.makedirs(metrics_dir, exist_ok=True)
    
    base_metrics_file = os.path.join(metrics_dir, f"priority_controlled_{test_case['model_name']}_draft1_{test_case['draft_name_1']}_draft2_{test_case['draft_name_2']}")
    
    ds = datasets.load_dataset("callanwu/WebWalkerQA", split="main")
    
    all_items = []
    for item in ds:
        info = item["info"]  
        if info.get("lang") == "en":
            all_items.append(item)
    
    print(f"Loaded {len(all_items)} English queries from dataset")
    
    url_to_queries = {}
    for item in all_items:
        root_url = item["root_url"]
        if root_url not in url_to_queries:
            url_to_queries[root_url] = []
        url_to_queries[root_url].append(item["question"])
    
    sampled_urls = []
    sampled_queries = []
    
    import random
    random.seed(42)

    for root_url, url_queries in url_to_queries.items():
        selected_query = random.choice(url_queries)
        sampled_urls.append(root_url)
        sampled_queries.append(selected_query)
    
    print(f"Loaded {len(sampled_urls)} unique URLs with sampled queries")
    print(f"Total unique URLs in dataset: {len(url_to_queries)}")
    
    urls = sampled_urls
    queries = sampled_queries
    
    REPEAT_TIMES = 5
    successful_examples = 0
    failed_examples = 0
    
    for round_num in range(1, REPEAT_TIMES + 1):   
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}/{REPEAT_TIMES}")
        print(f"{'='*60}")
        
        metrics_file = f"{base_metrics_file}_round_{round_num}.json"
        print(f"Metrics will be saved to: {metrics_file}")
        
        for i, (base_url, query) in enumerate(zip(urls, queries)):
            print(f"\n{'-'*40}")
            print(f"Processing Example {i+1}/{len(urls)} (Round {round_num})")
            print(f"URL: {base_url}")
            print(f"Query: {query}")
            print(f"{'-'*40}")
            
            memory = "No Memory"
            max_rounds = 10
            
            try:
                result = run_walker_with_retry(
                    walker=walker,
                    base_url=base_url,
                    query=query,
                    memory=memory,
                    max_rounds=max_rounds,
                    metrics_file=metrics_file,
                    use_assisted_llm=test_case['use_assisted_llm'],
                    max_retries=3,
                    base_delay=2
                )
                
                if result is None:
                    print(f"Failed to process example {i+1} after all retry attempts")
                    failed_examples += 1
                    continue
                
                successful_examples += 1
                
                print("\nFinal Answer:")
                print(result['answer'])
                
                print("\nPerformance Metrics:")
                print(f"Main model web time: {result['metrics']['main_web_time']:.2f} seconds")
                print(f"Draft 1 web time: {result['metrics']['draft1_web_time']:.2f} seconds")
                print(f"Draft 2 web time: {result['metrics']['draft2_web_time']:.2f} seconds")
                print(f"Total web time: {result['metrics']['web_time']:.2f} seconds")
                print(f"LLM processing time: {result['metrics']['llm_time']:.2f} seconds")
                print(f"Total time: {result['metrics']['total_time']:.2f} seconds")
                
                print("\nModel Information:")
                print(f"Main Model: {result['metrics']['model']}")
                
                print("\nCache Performance:")
                print(f"Total subpage visits: {result['metrics']['total_subpage_visits']}")
                print(f"Cache hits: {result['metrics']['cache_hits']}")
                print(f"Cache hit rate: {result['metrics']['cache_hit_rate']:.2%}")
                print(f"Draft 1 cache misses: {result['metrics']['draft1_cache_misses']}")
                print(f"Draft 2 cache misses: {result['metrics']['draft2_cache_misses']}")
                
                print("\nLength Metrics:")
                print(f"Initial prompt tokens: {result['metrics']['initial_prompt_tokens']} tokens")
                print(f"Final context tokens: {result['metrics']['final_context_tokens']} tokens")
                
            except Exception as e:
                print(f"Error processing example {i+1}: {str(e)}")
                failed_examples += 1
                continue
        
        print(f"\nRound {round_num} Summary:")
        print(f"Successful: {successful_examples}")
        print(f"Failed: {failed_examples}")
    
    print(f"\nFinal Summary:")
    print(f"Total Successful: {successful_examples}")
    print(f"Total Failed: {failed_examples}")
    print(f"Metrics saved to: {base_metrics_file}_round_*.json")
    
    print(f"\n{'='*80}")
    print(f"Completed evaluation of controlled assisted walker")
    print(f"{'='*80}")


if __name__ == "__main__":
    main() 