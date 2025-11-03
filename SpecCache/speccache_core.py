import os
import json
import asyncio
import time
import tiktoken
import re
from typing import Dict, List, Tuple, Optional
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from spec_cache_agent import AssistedWebWalker
from utils import *
import collections
from prompts import *
from tool import VisitPage

try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None


class AssistedWebWalkerCore:
    def __init__(self, api_key: str, model: str = "Qwen/QwQ-32B", model_server: str = "https://api.together.xyz/v1", cache_size: int = 10, provider: str = 'openai', draft_name_1: Optional[str] = None, draft_name_2: Optional[str] = None):
        """Initialize WebWalker core with API configuration.
        
        Args:
            api_key (str): API key for the LLM service
            model (str): Model name to use
            model_server (str): Model server URL
            cache_size (int): Maximum size of the cache
            provider (str): LLM provider ('openai' or 'azure')
            draft_name_1 (str, optional): First draft model name for assisted LLM
            draft_name_2 (str, optional): Second draft model name for assisted LLM
        """
        self.llm_cfg = {
            'provider': provider,
            'model': model,
            'model_server': model_server,
            'api_key': api_key,
            'generate_cfg': {
                'max_tokens': 2048,
                'max_input_tokens': 30721, 
                'temperature': 0.1,
                'top_p': 0.9,
            }
        }
        
        self.web_time_root = []  
        self.web_time_subpages = []  
        self.web_time = 0  
        self.metrics_per_round = []  
        self.main_web_time = 0  
        self.draft1_web_time = 0  
        self.draft2_web_time = 0  
        if "gpt" or "o4" or "o1" or "o3" in model.lower():
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.token_count_accuracy = "exact"  
        elif "qwen" in model.lower() or "together" in model_server.lower():
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.token_count_accuracy = "approximate"  
        else:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.token_count_accuracy = "approximate"  
            print(f"Warning: Using approximate token counting for unknown model {model}. Counts may not match the actual model's tokenization.")
        
        self.visited_links = []
        
        self.cache_size = cache_size
        self.page_cache = collections.OrderedDict()  
        self.root_url_cache = collections.OrderedDict()  
        
        self.button_dict = {}  
        self.root_url = None  
        
        self.draft_name_1 = draft_name_1
        self.draft_name_2 = draft_name_2
    
    def get_web_time_breakdown(self) -> Dict[str, List[float]]:
        return {
            'root': self.web_time_root,
            'subpages': self.web_time_subpages,
            'total': self.web_time,
            'main_web_time': self.main_web_time,
            'draft1_web_time': self.draft1_web_time,
            'draft2_web_time': self.draft2_web_time
        }
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.
        
        Args:
            text (str): Text to count tokens for
            
        Returns:
            int: Number of tokens
        """
        return len(self.tokenizer.encode(text))
    
    def quick_crop_markdown(self, markdown: str, target_tokens: int) -> str:
        """Quickly crop markdown by removing paragraphs from the end.
        
        Args:
            markdown (str): Markdown content to crop
            target_tokens (int): Target number of tokens
            
        Returns:
            str: Cropped markdown content
        """
        current_tokens = self.count_tokens(markdown)
        if current_tokens <= target_tokens:
            return markdown
        
        print(f"Quick cropping markdown from {current_tokens} to {target_tokens} tokens")
        
        paragraphs = markdown.split('\n\n')
        original_paragraphs = len(paragraphs)
        
        while self.count_tokens('\n\n'.join(paragraphs)) > target_tokens and len(paragraphs) > 1:
            remove_count = max(1, len(paragraphs) // 5)
            paragraphs = paragraphs[:-remove_count]
        
        markdown = '\n\n'.join(paragraphs)
        print(f"Removed {original_paragraphs - len(paragraphs)} paragraphs")
        return markdown

    def extract_links_with_text(self, html: str, root_url: str):
        """Extract clickable links from HTML content and return as formatted string of <button> tags."""
        soup = BeautifulSoup(html, 'html.parser')
        links = []

        navigation_buttons = {
            'About Wikipedia', 
        }

        for a_tag in soup.find_all('a', href=True):
            url = a_tag['href']
            text = ''.join(a_tag.stripped_strings)
            if url.startswith('#'):
                continue
            if text in navigation_buttons:
                continue
            if text and "javascript" not in url and not url.endswith(('.jpg', '.png', '.gif', '.jpeg', '.pdf')):
                processed_url = process_url(root_url, url)
                links.append({'url': processed_url, 'text': text})

        for button_tag in soup.find_all('button'):
            text = ''.join(button_tag.stripped_strings)
            if text in navigation_buttons:
                continue
            if text:
                links.append({'url': None, 'text': text})

        seen = set()
        unique_links = []
        for link in links:
            if isinstance(link, dict):
                key = (link.get('url'), link.get('text'))
            else:
                key = link
            if key not in seen:
                seen.add(key)
                unique_links.append(link)

        info = ""
        button_dict = {}
        for i in unique_links:
            info += "<button>" + i["text"] + "</button>" + "\n"
            button_dict[i["text"]] = i["url"] # Store the URL for each button text
        
        print(f"Current page has {len(unique_links)} buttons")
        
        return info, button_dict

    async def process_webpage(self, url: str, is_root: bool = False) -> Tuple[str, str, Optional[str]]:
        """Process a webpage and extract its content.
        
        Args:
            url (str): URL to process
            is_root (bool): Whether this is the root page (for time tracking)
            
        Returns:
            Tuple[str, str, Optional[str]]: HTML content, markdown content, and screenshot (if available)
        """
        web_start_time = time.time()
        html, markdown = await get_info(url)
        call_time = time.time() - web_start_time
        
        if is_root:
            self.web_time_root.append(call_time)
            print(f'Time taken for root page web fetch: {call_time:.2f} seconds')
        else:
            self.web_time_subpages.append(call_time)
            print(f'Time taken for subpage web fetch: {call_time:.2f} seconds')
        
        self.web_time += call_time
        
        self.update_cache(url, html, markdown)
        
        return html, markdown

    def save_metrics(self, metrics: Dict, output_file: str):
        """Save metrics to a JSON file with timestamp.
        
        Args:
            metrics (Dict): Metrics dictionary to save
            output_file (str): Path to save the metrics file
        """
        metrics['timestamp'] = time.strftime('%Y-%m-%d_%H-%M-%S')
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'a') as f:
            json.dump(metrics, f, indent=2)

    def run(self, base_url: str, query: str, memory: str = "No Memory", max_rounds: int = 10, metrics_file: str = None, use_assisted_llm: bool = True) -> Dict:
        """Run the WebWalker agent with the given inputs.
        
        Args:
            base_url (str): Starting URL for the website
            query (str): User's query
            memory (str): Initial memory state
            max_rounds (int): Maximum number of interaction rounds
            metrics_file (str, optional): Path to save metrics JSON file. If None, metrics won't be saved.
            
        Returns:
            Dict: Results including answer, metrics, and interaction history
        """
        self.metrics_per_round = []
        self.web_time = 0
        self.web_time_root = []  
        self.web_time_subpages = []  
        self.page_cache.clear()  
        self.button_dict = {}  
        self.root_url = base_url  
        self.main_web_time = 0
        self.draft1_web_time = 0
        self.draft2_web_time = 0
        
        self.llm_cfg["query"] = query
        self.llm_cfg["action_count"] = max_rounds
        
        html, markdown = asyncio.run(self.process_webpage(base_url, is_root=True))
        buttons, button_dict = self.extract_links_with_text(html, base_url)
        
        self.button_dict.update(button_dict)
        print(f"Populated button_dict with {len(self.button_dict)} buttons from initial page")
        
        start_prompt = WEBWALKER_PROMPT_TEMPLATE.format(
            query=query,
            base_url=base_url,
            markdown=markdown,
            buttons=buttons
        )
        
        max_input_tokens = self.llm_cfg['generate_cfg']['max_input_tokens']
        actual_max_tokens = 30721
        prompt_tokens = self.count_tokens(start_prompt)
        
        if prompt_tokens > actual_max_tokens:
            print(f"Warning: Input prompt has {prompt_tokens} tokens, cropping to {actual_max_tokens}")
            
            excess_tokens = prompt_tokens - actual_max_tokens
            print(f"Need to remove {excess_tokens} tokens")
            
            other_tokens = prompt_tokens - self.count_tokens(markdown)
            target_markdown_tokens = actual_max_tokens - other_tokens - 100  # Much larger buffer
            
            print(f"Target markdown tokens: {target_markdown_tokens}")
            print(f"Current markdown tokens: {self.count_tokens(markdown)}")
            
            markdown = self.quick_crop_markdown(markdown, target_markdown_tokens)
            
            start_prompt = WEBWALKER_PROMPT_TEMPLATE.format(
                query=query,
                base_url=base_url,
                markdown=markdown,
                buttons=buttons
            )
            print(f"After aggressive cropping: {self.count_tokens(start_prompt)} tokens")
            
            if self.count_tokens(start_prompt) > actual_max_tokens:
                print("Still too long, applying emergency cropping...")
                paragraphs = markdown.split('\n\n')
                markdown = '\n\n'.join(paragraphs[:len(paragraphs)//2])
                
                start_prompt = WEBWALKER_PROMPT_TEMPLATE.format(
                    query=query,
                    base_url=base_url,
                    markdown=markdown,
                    buttons=buttons
                )
                print(f"After emergency cropping: {self.count_tokens(start_prompt)} tokens")
            
        
        messages = [{'role': 'user', 'content': start_prompt}]
        initial_prompt_length = len(start_prompt)
        initial_prompt_tokens = self.count_tokens(start_prompt)
        
        round_metrics = []
        current_context_length = initial_prompt_length
        current_context_tokens = initial_prompt_tokens
        llm_input_token_counts = []
        
        all_responses = []
        all_llm_input_token_counts = []
        total_llm_time = 0
        total_web_time = 0
        
        print("=== STEP 1: Running Main Model ===")
        bot_main = AssistedWebWalker(
            llm=self.llm_cfg,
            function_list=["visit_page"],
            use_assisted_llm=use_assisted_llm,  
            draft_name=self.draft_name_1  
        )
        
        if self.draft_name_2:
            bot_main.assist_llm_2 = {
                'model': self.draft_name_2,
                'api_key': self.llm_cfg['api_key'],
                'model_server': self.llm_cfg['model_server'],
                'generate_cfg': {
                    'top_p': 0.8,
                    'max_input_tokens': 30721,
                    'max_tokens': 2048,
                    'max_retries': 20
                },
            }
        
        responses_gen_main = bot_main.run(messages=messages, lang="zh", walker_core=self, markdown=markdown, buttons=buttons)
        responses_main = list(responses_gen_main)
        all_responses.extend(responses_main)
        all_llm_input_token_counts.extend(getattr(bot_main, 'llm_input_token_counts', []))
        total_llm_time += bot_main.llm_time
        self.main_web_time = self.web_time  
        total_web_time += self.web_time
        
        if use_assisted_llm:
            root_time = sum(self.web_time_root) if self.web_time_root else 0
            
            draft1_iteration_web_times = getattr(bot_main, 'draft1_iteration_web_times', [])
            self.draft1_web_time = root_time + sum(draft1_iteration_web_times)
            
            draft2_iteration_web_times = getattr(bot_main, 'draft2_iteration_web_times', [])
            self.draft2_web_time = root_time + sum(draft2_iteration_web_times)
        
        round_metrics.append({
            'input_tokens': initial_prompt_tokens,
            'output_tokens': 0,  
            'total_context_tokens': initial_prompt_tokens,
            'web_time': total_web_time,
            'llm_time': total_llm_time
        })
        
        final_answer = None
        thoughts = []
        memory_updates = []
        
        if all_responses:
            content = all_responses[-1][0]["content"]
            print(f"Final content: {content}")
            output_length = len(content)
            output_tokens = self.count_tokens(content)
            
            round_metrics.append({
                'content': content,
                'input_tokens': sum(all_llm_input_token_counts),
                'output_tokens': output_tokens,
                'total_context_tokens': sum(all_llm_input_token_counts) + output_tokens,
                'web_time': total_web_time,
                'llm_time': total_llm_time,
            })
            
            current_context_length += output_length
            current_context_tokens += output_tokens
            
            if "Final Answer" in content:
                final_answer = content
            elif "Memory" in content:
                memory_updates.append(content)
            elif "Action" in content:
                thoughts.append(content.split("Action")[0])
        
        total_time = total_web_time + total_llm_time
        total_rounds = len(all_responses)
        total_context_length = current_context_length
        total_context_tokens = current_context_tokens
        
        llm_time_breakdown = {
            'main_action': getattr(bot_main, 'llm_time_main_action', []),
            'observation_extraction': getattr(bot_main, 'llm_time_observation_extraction', []),
            'critic': getattr(bot_main, 'llm_time_critic', []),
            'assisted_gen': getattr(bot_main, 'llm_time_assisted_gen', []),
            'total': total_llm_time
        }
        
        web_time_breakdown = {
            'root': self.web_time_root,
            'subpages': self.web_time_subpages,
            'total': total_web_time,
            'main_web_time': self.main_web_time,
            'draft1_web_time': self.draft1_web_time,
            'draft2_web_time': self.draft2_web_time
        }
        
        results = {
            'answer': final_answer,
            'thoughts': thoughts,
            'memory_updates': memory_updates,
            'metrics': {
                'query': query,
                'url': base_url,
                'web_time': total_web_time,
                'llm_time': total_llm_time,
                'total_time': total_time,
                'round_metrics': round_metrics,
                'initial_prompt_tokens': initial_prompt_tokens,
                'final_context_tokens': total_context_tokens,
                'model': self.llm_cfg['model'],
                'trajectory': getattr(bot_main, 'visited_urls', []),  
                'trajectory_steps': getattr(bot_main, 'trajectory_steps', []),  
                'final_answer': final_answer,  
                'total_subpage_visits': getattr(bot_main, 'total_subpage_visits', 0),
                'cache_hits': getattr(bot_main, 'cache_hits', 0),
                'cache_hit_rate': (getattr(bot_main, 'cache_hits', 0) / getattr(bot_main, 'total_subpage_visits', 1)) if getattr(bot_main, 'total_subpage_visits', 0) > 0 else 0.0,
                'root_url_cache_hits': getattr(bot_main, 'root_url_cache_hits', 0),
                'llm_input_token_counts': all_llm_input_token_counts,
                'cache_stats': self.get_cache_stats(),
                'llm_time_breakdown': llm_time_breakdown,
                'llm_time_main_action': llm_time_breakdown['main_action'],
                'llm_time_observation_extraction': llm_time_breakdown['observation_extraction'],
                'llm_time_critic': llm_time_breakdown['critic'],
                'llm_time_assisted_gen': llm_time_breakdown['assisted_gen'],
                'web_time_breakdown': web_time_breakdown,
                'web_time_root': web_time_breakdown['root'],
                'web_time_subpages': web_time_breakdown['subpages'],
                'main_web_time': self.main_web_time,
                'draft1_web_time': self.draft1_web_time,
                'draft2_web_time': self.draft2_web_time,
                'draft1_cache_misses': getattr(bot_main, 'draft1_cache_misses', 0),
                'draft2_cache_misses': getattr(bot_main, 'draft2_cache_misses', 0),
                'draft1_cache_hits': getattr(bot_main, 'draft1_cache_hits', 0),
                'draft2_cache_hits': getattr(bot_main, 'draft2_cache_hits', 0)
            }
        }
        
        if metrics_file:
            self.save_metrics(results['metrics'], metrics_file)
        
        return results


    async def fetch_pages_async(self, urls: List[str]) -> Dict[str, Tuple[str, str]]:
        """Fetch multiple pages asynchronously and return their HTML and markdown.
        Args:
            urls (List[str]): URLs to fetch
        Returns:
            Dict[str, Tuple[html, markdown]]: Mapping from URL to (html, markdown)
        """
        async def fetch(url):
            html, markdown = await self.process_webpage(url, is_root=False)
            return url, (html, markdown)
        tasks = [fetch(url) for url in urls]
        results = await asyncio.gather(*tasks)
        return {url: (html, markdown) for url, (html, markdown) in results}


    def extract_root_url(self, url: str) -> str:
        """Extract the root URL from any given URL.
        
        Args:
            url (str): Any URL
            
        Returns:
            str: The root URL (scheme + netloc)
        """
        parsed = urlparse(url)
        root_url = f"{parsed.scheme}://{parsed.netloc}"
        return root_url

    def get_from_cache(self, url: str) -> Optional[Tuple[str, str, str, str]]:
        """Check if a URL is in cache and return cached content if available.
        
        Args:
            url (str): URL to check in cache
            
        Returns:
            Optional[Tuple[str, str, str, str]]: Cached content (formatted_response, markdown, buttons, cache_type) or None
        """
        if url in self.page_cache:
            print(f"Cache hit for {url}")
            self.page_cache.move_to_end(url)
            return self.page_cache[url] + ("page",)
        
        root_url = self.extract_root_url(url)
        if root_url in self.root_url_cache:
            print(f"Root URL cache hit for {url} (root: {root_url})")
            self.root_url_cache.move_to_end(root_url)
            return self.root_url_cache[root_url] + ("root_url",)
        
        return None

    def update_cache(self, url: str, html: str, markdown: str):
        """Update the cache with new page content.
        
        Args:
            url (str): URL of the page
            html (str): HTML content
            markdown (str): Markdown content
        """
        response_buttons, page_button_dict = self.extract_links_with_text(html, url)
        
        self.button_dict.update(page_button_dict)
        
        formatted_response = "The web information is:\n\n" + (markdown if markdown else "The information of the current page is not accessible") + "\n\n"
        formatted_response += "Clickable buttons are wrapped in <button> tag\n" + response_buttons

        self.page_cache[url] = (formatted_response, markdown, response_buttons)
        if len(self.page_cache) > self.cache_size:
            self.page_cache.popitem(last=False)  
        
        root_url = self.extract_root_url(url)
        if url == root_url:  
            self.root_url_cache[root_url] = (formatted_response, markdown, response_buttons)
            if len(self.root_url_cache) > 10:  
                self.root_url_cache.popitem(last=False)  

    def get_cache_stats(self) -> Dict[str, any]:
        """Get statistics about caches.
        
        Returns:
            Dict containing cache statistics
        """
        return {
            'page_cache_size': len(self.page_cache),
            'page_cache_max_size': self.cache_size,
            'root_url_cache_size': len(self.root_url_cache),
            'root_url_cache_max_size': 10,
            'page_cache_keys': list(self.page_cache.keys()),
            'root_url_cache_keys': list(self.root_url_cache.keys())
        }

    def print_cache_stats(self):
        """Print cache statistics to console."""
        stats = self.get_cache_stats()
        print(f"Page Cache: {stats['page_cache_size']}/{stats['page_cache_max_size']} entries")
        print(f"Root URL Cache: {stats['root_url_cache_size']}/{stats['root_url_cache_max_size']} entries")
        print(f"Page Cache URLs: {stats['page_cache_keys']}")
        print(f"Root URL Cache URLs: {stats['root_url_cache_keys']}")