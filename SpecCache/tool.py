import os
import json5
import json
import asyncio
import time
from typing import Tuple, List
from qwen_agent.tools.base import BaseTool, register_tool
from concurrent.futures import ThreadPoolExecutor
import requests
import re

class ToolRegistry:
    def __init__(self):
        self.visit_page_tool = None
        self.search_tool = None
    
    def register_visit_page(self, tool):
        self.visit_page_tool = tool
    
    def register_search(self, tool):
        self.search_tool = tool
    
    def get_visit_page(self):
        return self.visit_page_tool
    
    def get_search(self):
        return self.search_tool

tool_registry = ToolRegistry()

def get_content_between_a_b(a, b, s):
    try:
        return s.split(a, 1)[1].split(b, 1)[0]
    except Exception:
        return ""

def normalize_button_text(text):
    """Normalize button text for robust matching: lowercase, remove punctuation, collapse whitespace."""
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


@register_tool('visit_page', allow_overwrite=True)
class VisitPage(BaseTool):
    description = 'A tool analyzes the content of a webpage and extracts buttons associated with sublinks.'
    parameters = [{
        'name': 'button',
        'type': 'string',
        'description': 'the button you want to click',
        'required': True
    }]

    def __init__(self, config=None):
        super().__init__()
        self.search_result_buttons = {}  
        tool_registry.register_visit_page(self)

    def call(self, params: str, **kwargs) -> Tuple[str, str, str]:
        if not params.strip().endswith("}"):
            if "}" in params.strip():
                params = "{" + get_content_between_a_b("{","}", params) + "}"
            else:
                if not params.strip().endswith("\""):
                    params = params.strip() + '"}'
                else:
                    params = params.strip() + '}'
        params = "{" + get_content_between_a_b("{","}", params) + "}"

        if 'button' not in json5.loads(params):
            return "Your input is invalid, please output the action input correctly!"

        walker = kwargs['walker']  
        button_url_dict = walker.button_dict

        button_raw = json5.loads(params)['button']
        button_text = button_raw.replace("<button>","").replace("</button>","").strip()
        
        if button_text.startswith('http') or button_text.startswith('Link ') or '://' in button_text:
            return f"Invalid button name: '{button_text}'. You can only click on actual button names, not URLs or links. Please choose a button from the available list."
        
        button_text_norm = normalize_button_text(button_text)
        static_match = None
        for k in button_url_dict:
            k_norm = normalize_button_text(k)
            if button_text_norm == k_norm:
                static_match = k
                break
            if all(word in k_norm.split() for word in button_text_norm.split()):
                static_match = k
                break

        dynamic_match = None
        for k in self.search_result_buttons:
            k_norm = normalize_button_text(k)
            if button_text_norm == k_norm:
                dynamic_match = k
                break
            if all(word in k_norm.split() for word in button_text_norm.split()):
                dynamic_match = k
                break

        index_match = None
        if button_text.isdigit():
            idx = int(button_text) - 1
            search_keys = list(self.search_result_buttons.keys())
            if 0 <= idx < len(search_keys):
                dynamic_match = search_keys[idx]
                index_match = True

        if not static_match and not dynamic_match:
            available = list(button_url_dict.keys()) + list(self.search_result_buttons.keys())
            return f"The button cannot be clicked, please retry a new button! Available: {available}"

        root_url = walker.root_url

        if static_match:
            url = button_url_dict[static_match]
        else:
            url = self.search_result_buttons[dynamic_match]

        walker = kwargs['walker']  

        print(f"Checking cache for {url}")
        print(f"walker cache: {walker.page_cache.keys()}")
        cached_data = walker.get_from_cache(url)
        if cached_data:
            print(f"Cache hit for {url}. Returning cached content.")
            return cached_data

        html, markdown = asyncio.run(walker.process_webpage(url))
        response_buttons, page_button_dict = walker.extract_links_with_text(html, root_url)
        
        walker.button_dict.update(page_button_dict)
        print(f"Updated button_dict with {len(page_button_dict)} buttons from current page")

        response = "The web information is:\n\n" + (markdown if markdown else "The information of the current page is not accessible") + "\n\n"
        response += "Clickable buttons are wrapped in <button> tag\n" + response_buttons
        return response, markdown, response_buttons

    def update_search_buttons(self, search_results: List[dict]):
        """Update the search result button mappings"""
        self.search_result_buttons.clear()
        for result in search_results:
            title = result.get('title', '')
            link = result.get('link', '')
            if title and link:
                self.search_result_buttons[title] = link