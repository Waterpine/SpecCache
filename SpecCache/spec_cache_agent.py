import json
import json5
import os
import re
import asyncio
import ast

from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union

from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
from qwen_agent.utils.utils import format_as_text_message, merge_generate_cfgs
from openai import OpenAI
try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None
import time
from prompts import *
import tiktoken
from tool import normalize_button_text


TOOL_DESC = (
    '{name_for_model}: Call this tool to interact with the {name_for_human} API. '
    'What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} {args_format}')

class AssistedWebWalker(FnCallAgent):
    """This explorer agent use ReAct format to call tools"""

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 files: Optional[List[str]] = None,
                 use_assisted_llm: bool = True,
                 draft_name: Optional[str] = None,
                 **kwargs):
        super().__init__(function_list=function_list,
                         llm=llm,
                         system_message=system_message,
                         name=name,
                         description=description,
                         files=files,
                         **kwargs)
        self.extra_generate_cfg = merge_generate_cfgs(
            base_generate_cfg=self.extra_generate_cfg,
            new_generate_cfg={
                'stop': ['Observation:', 'Observation:\n'],
                'max_tokens': 2048 
            },
        )
        # --- LLM client selection ---
        provider = llm.get('provider', 'openai') if isinstance(llm, dict) else 'openai'
        if provider == 'azure':
            if AzureOpenAI is None:
                raise ImportError('openai package with Azure support is required for Azure backend.')
            self.client = AzureOpenAI(
                api_version="2024-12-01-preview",
                azure_endpoint=llm['model_server'],
                api_key=llm['api_key'],
            )
            self._llm_provider = 'azure'
        else:
            self.client = OpenAI(
                api_key=llm['api_key'], 
                # base_url=llm['model_server'],
            )
            self._llm_provider = 'openai'
        self.llm_cfg = llm
        self.memory = []
        self.llm_time_main_action = [] 
        self.llm_time_observation_extraction = []
        self.llm_time_critic = [] 
        self.llm_time_assisted_gen = [] 
        
        self.llm_time = 0
        self.visited_urls = [] 
        self.trajectory_steps = []  

        if draft_name is not None:
            self.assist_llm_1 = {
                'model': draft_name,
                'api_key': llm['api_key'],
                # 'model_server': llm['model_server'],
                'generate_cfg': {
                    'top_p': 0.8,
                    'max_input_tokens': 30721,  
                    'max_tokens': 2048,         
                    'max_retries': 20
                },
            }
        else:
            self.assist_llm_1 = None
        
        self.assist_llm_2 = None
        
        self.use_assisted_llm = use_assisted_llm
        
        self.web_time = 0  
        self.web_time_subpages = []  

    def _simple_fetch_pages(self, walker_core, urls_to_fetch: List[str]):
        """
        Fetch pages using the walker core's async fetch method.
        
        Args:
            walker_core: The walker core instance
            urls_to_fetch: List of URLs to fetch
        """
        if not urls_to_fetch:
            return
            
        try:
            print(f"Fetching pages: {urls_to_fetch}")
            asyncio.run(walker_core.fetch_pages_async(urls_to_fetch))
            print(f"Successfully fetched pages: {urls_to_fetch}")
        except Exception as e:
            print(f"Error fetching pages: {e}")



    def _call_tool(self, tool_name: str, tool_args: Union[str, dict] = '{}', **kwargs) -> str:
        """Override _call_tool to add coordination for visit_page actions."""
        if tool_name == 'visit_page' and self.use_assisted_llm:
            try:
                if isinstance(tool_args, str):
                    params = json5.loads(tool_args)
                else:
                    params = tool_args
                
                button = params.get('button', '')
                walker = kwargs.get('walker')
                
                if walker and button:
                    button_url_dict = walker.button_dict
                    url = None
                    
                    if button in button_url_dict:
                        url = button_url_dict[button]
                    else:
                        button_norm = normalize_button_text(button)
                        for k in button_url_dict:
                            k_norm = normalize_button_text(k)
                            if button_norm == k_norm:
                                url = button_url_dict[k]
                                break
                    
                    if url:
                        print(f"Main LLM visit_page action for URL: {url}")
                        self._simple_fetch_pages(walker, [url])
                        
            except Exception as e:
                print(f"Error in visit_page coordination: {e}")
        
        result = super()._call_tool(tool_name, tool_args, **kwargs)
        return result

    def _crop_text_messages(self, text_messages: List[Message], walker_core, max_tokens: int, buffer: int = 1000, crop_ratio: float = 0.3) -> List[Message]:
        """Crop text_messages to fit within token limit by removing paragraphs from the last message.
        
        Args:
            text_messages: List of messages to crop
            walker_core: Walker core instance for token counting
            max_tokens: Maximum allowed tokens
            buffer: Buffer tokens to leave
            crop_ratio: Ratio of paragraphs to remove at once (0.3 = 30%)
            
        Returns:
            Cropped text_messages
        """
        total_tokens = sum(walker_core.count_tokens(m.content) for m in text_messages)
        if total_tokens <= max_tokens:
            return text_messages
            
        print(f"Warning: text_messages has {total_tokens} tokens, cropping to {max_tokens}")
        last_message = text_messages[-1]
        available_tokens = max_tokens - sum(walker_core.count_tokens(m.content) for m in text_messages[:-1]) - buffer
        
        if walker_core.count_tokens(last_message.content) > available_tokens:
            print(f"Applying cropping to last message...")
            paragraphs = last_message.content.split('\n\n')
            original_paragraphs = len(paragraphs)
            
            while walker_core.count_tokens('\n\n'.join(paragraphs)) > available_tokens and len(paragraphs) > 1:
                remove_count = max(1, int(len(paragraphs) * crop_ratio))
                paragraphs = paragraphs[:-remove_count]
            
            last_message.content = '\n\n'.join(paragraphs)
            print(f"Removed {original_paragraphs - len(paragraphs)} paragraphs from last message")
        
        final_tokens = sum(walker_core.count_tokens(m.content) for m in text_messages)
        print(f"After cropping text_messages: {final_tokens} tokens")
        return text_messages


    def observation_information_extraction(self, query, observation):
        user_prompt = "- Query: {query}\n- Observation: {observation}".format(query=query, observation=observation)
        messages = [
            {'role': 'system', 'content': STSTEM_CRITIIC_INFORMATION},
            {'role': 'user', 'content':  user_prompt}]
        
        max_input_tokens = self.llm_cfg['generate_cfg']['max_input_tokens']
        actual_max_tokens = 30721
        total_tokens = sum(self.count_tokens(m['content']) for m in messages)
        
        if total_tokens > actual_max_tokens:
            print(f"Warning: observation_information_extraction input has {total_tokens} tokens, cropping to {actual_max_tokens}")
            
            system_tokens = self.count_tokens(messages[0]['content'])
            query_tokens = self.count_tokens(messages[1]['content'])
            available_tokens = actual_max_tokens - system_tokens - query_tokens - 1000  # Buffer
            
            observation_start = user_prompt.find("- Observation: ") + len("- Observation: ")
            observation_content = user_prompt[observation_start:]
            
            if self.count_tokens(observation_content) > available_tokens:
                paragraphs = observation_content.split('\n\n')
                while self.count_tokens('\n\n'.join(paragraphs)) > available_tokens and len(paragraphs) > 1:
                    paragraphs = paragraphs[:-1]  
                
                cropped_observation = '\n\n'.join(paragraphs)
                user_prompt = "- Query: {query}\n- Observation: {observation}".format(
                    query=query, observation=cropped_observation)
                messages[1]['content'] = user_prompt
                print(f"After cropping observation: {sum(self.count_tokens(m['content']) for m in messages)} tokens")
        
        max_retries = 10
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                model_name = self.llm_cfg['model']
                token_param = {}
                if any(x in model_name for x in ["o4-mini", "o1-mini", "o3-mini", "gpt-5"]):
                    token_param["max_completion_tokens"] = self.llm_cfg['generate_cfg']['max_tokens']
                else:
                    token_param["max_tokens"] = self.llm_cfg['generate_cfg']['max_tokens']
                response = self.client.chat.completions.create(
                    model=model_name,
                    response_format={"type": "json_object"},
                    messages=messages,
                    service_tier="priority", 
                    **token_param
                )
                end_time = time.time()
                call_time = end_time - start_time
                self.llm_time_observation_extraction.append(call_time)
                self.llm_time += call_time
                print(f'Time taken for observation extraction LLM: {call_time:.2f} seconds with model: {self.llm_cfg["model"]}')
                try:
                    response_content = json.loads(response.choices[0].message.content)
                    if response_content.get("usefulness", False):
                        return response_content.get("information", "")
                    else:
                        return None
                except:
                    if "true" in response.choices[0].message.content:
                        return response.choices[0].message.content
                    else:
                        return None
            except Exception as e:
                print(e)
                if attempt < max_retries - 1:
                    time.sleep(1 * (2 ** attempt))  
                else:
                    raise e  

    def critic_information(self, query, memory):  
        memory = "-".join(memory)
        user_prompt = "- Query: {query}\n- Accumulated Information: {memory}".format(query = query, memory=memory)
        messages = [
            {'role': 'system', 'content': STSTEM_CRITIIC_ANSWER},
            {'role': 'user', 'content': user_prompt}]
        
        max_input_tokens = self.llm_cfg['generate_cfg']['max_input_tokens']
        actual_max_tokens = 30721
        total_tokens = sum(self.count_tokens(m['content']) for m in messages)
        
        if total_tokens > actual_max_tokens:
            print(f"Warning: critic_information input has {total_tokens} tokens, cropping to {actual_max_tokens}")
            
            system_tokens = self.count_tokens(messages[0]['content'])
            query_tokens = self.count_tokens(messages[1]['content'])
            available_tokens = actual_max_tokens - system_tokens - query_tokens - 1000  # Buffer
            
            memory_start = user_prompt.find("- Accumulated Information: ") + len("- Accumulated Information: ")
            memory_content = user_prompt[memory_start:]
            
            if self.count_tokens(memory_content) > available_tokens:
                while self.count_tokens(memory_content) > available_tokens and len(memory_content) > 100:
                    memory_content = memory_content[:-100]  
                
                user_prompt = "- Query: {query}\n- Accumulated Information: {memory}".format(
                    query=query, memory=memory_content)
                messages[1]['content'] = user_prompt
                print(f"After cropping memory: {sum(self.count_tokens(m['content']) for m in messages)} tokens")
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                model_name = self.llm_cfg['model']
                token_param = {}
                if any(x in model_name for x in ["o4-mini", "o1-mini", "o3-mini", "gpt-5"]):
                    token_param["max_completion_tokens"] = self.llm_cfg['generate_cfg']['max_tokens']
                else:
                    token_param["max_tokens"] = self.llm_cfg['generate_cfg']['max_tokens']
                
                response = self.client.chat.completions.create(
                    model=model_name,
                    response_format={"type": "json_object"},
                    messages=messages,
                    service_tier="priority", 
                    **token_param
                )
                end_time = time.time()
                call_time = end_time - start_time
                self.llm_time_critic.append(call_time)
                self.llm_time += call_time
                print(f'Time taken for critic LLM: {call_time:.2f} seconds with model: {self.llm_cfg["model"]}')
                
                try:
                    response_content = json.loads(response.choices[0].message.content)
                    if response_content.get("judge", False):
                        return response_content.get("answer", "")
                    else:
                        return None
                except:
                    if "true" in response.choices[0].message.content:
                        return response.choices[0].message.content
                    else:
                        return None
            
            except Exception as e:
                print(e)
                if attempt < max_retries - 1:
                    time.sleep(1 * (2 ** attempt))  
                else:
                    raise e  
                
    def assist_llm_gen(self, messages: List[Message], walker_core, text_messages=None, draft_num=1):
        """
        Generate recommendations using assisted LLM.
        
        Args:
            messages: Input messages
            walker_core: Walker core instance
            text_messages: Preprocessed text messages
            draft_num: Which draft model to use (1 or 2)
        """
        if text_messages is None:
            text_messages = self._prepend_react_prompt(messages, lang='en', template=ASSISTANT_SYSTEM_EXPLORER)
        
        if draft_num == 1:
            assist_llm = self.assist_llm_1
            cache_misses_var = 'draft1_cache_misses'
        elif draft_num == 2:
            assist_llm = self.assist_llm_2
            cache_misses_var = 'draft2_cache_misses'
        else:
            raise ValueError(f"Invalid draft_num: {draft_num}. Must be 1 or 2.")
        
        if assist_llm is None:
            print(f"Draft {draft_num} LLM not configured, skipping...")
            return None, text_messages
        
        print(f"Draft {draft_num} LLM is configured with model: {assist_llm['model']}")
        
        total_tokens = sum(walker_core.count_tokens(m.content) for m in text_messages)
        print(f"[assist_llm_gen draft{draft_num}] Total tokens before LLM call: {total_tokens}")
        assert total_tokens <= 30721, f"Token count {total_tokens} exceeds model limit (30721) before LLM call."
        
        model_name = assist_llm['model']
        token_param = {}
        if any(x in model_name for x in ["o4-mini", "o1-mini", "o3-mini", "gpt-5"]):
            token_param["max_completion_tokens"] = assist_llm['generate_cfg']['max_tokens']
        else:
            token_param["max_tokens"] = assist_llm['generate_cfg']['max_tokens']
        
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=model_name,
            messages=text_messages,
            service_tier="priority", 
            **token_param
        )
        end_time = time.time()
        call_time = end_time - start_time
        self.llm_time_assisted_gen.append(call_time)
        self.llm_time += call_time
        print(f'Time taken for assisted LLM generation (draft{draft_num}): {call_time:.2f} seconds with model: {model_name}')
        
        llm_response = response.choices[0].message.content
        has_action, action, action_inputs, thought = self._detect_tool_multiple_action_inputs("\n"+llm_response)
        
        recommended_urls = []
        if has_action:
            try:
                button_url_dict = walker_core.button_dict
                for action_input in action_inputs:
                    if action_input.startswith('{'):
                        params = json5.loads(action_input)
                        btn = params.get('button')
                        if btn and btn in button_url_dict:
                            recommended_urls.append(button_url_dict[btn])
                        else:
                            btn_norm = normalize_button_text(btn)
                            similar_buttons = []
                            
                            for k in button_url_dict.keys():
                                k_norm = normalize_button_text(k)
                                if btn_norm == k_norm:
                                    similar_buttons = [k]  
                                    break
                                elif all(word in k_norm for word in btn_norm):
                                    similar_buttons.append(k)
                            
                            if similar_buttons:
                                recommended_urls.append(button_url_dict[similar_buttons[0]])
            except Exception as e:
                print(f"Error parsing action inputs: {e}")
                pass

        print(f"Recommended URLs (draft{draft_num}): {recommended_urls}")
        if recommended_urls:
            unique_urls = list(dict.fromkeys(recommended_urls))
            print(f"Unique URLs draft {draft_num} would prefetch: {unique_urls}")
            
            if draft_num == 1:
                draft_cache = self.draft1_cache
            elif draft_num == 2:
                draft_cache = self.draft2_cache
            
            print(f"\n=== DRAFT {draft_num} PREFETCHING ===")
            print(f"Adding {len(unique_urls)} URLs to draft {draft_num} cache:")
            for url in unique_urls:
                draft_cache[url] = True  # Mark as cached in simulation
                print(f"  Draft {draft_num}: {url}")
            print(f"=== END DRAFT {draft_num} PREFETCHING ===\n")
            print(f"DEBUG: After draft {draft_num}, cache contents:")
            if draft_num == 1:
                print(f"  Draft 1 cache: {list(self.draft1_cache.keys())}")
            elif draft_num == 2:
                print(f"  Draft 1 cache: {list(self.draft1_cache.keys())}")
                print(f"  Draft 2 cache: {list(self.draft2_cache.keys())}")
        else:
            print(f"No recommended URLs found (draft{draft_num})")
        
        return llm_response, text_messages

    def _parse_tool_output(self, tool_output, action: str) -> str:
        """Helper function to parse tool output and extract observation.
        
        Args:
            tool_output: The raw output from the tool
            action: The action that was performed
            
        Returns:
            The parsed observation string
        """
        if action == 'visit_page':
            try:
                tool_data = ast.literal_eval(str(tool_output))
                observation, _, _ = tool_data
            except (ValueError, SyntaxError):
                observation = str(tool_output)
        else:
            observation = tool_output
        return observation

    def _crop_context_with_observation(self, base_template: str, accumulated_thoughts: str, observation: str, walker_core, max_tokens: int, buffer: int = 1000) -> str:
        """Crop the observation to fit within token limits while preserving base template and thoughts.
        
        Args:
            base_template: The base template content
            accumulated_thoughts: The accumulated thought history
            observation: The observation to crop
            walker_core: Walker core instance for token counting
            max_tokens: Maximum allowed tokens
            buffer: Buffer tokens to leave
            
        Returns:
            The cropped observation
        """
        temp_messages = [
            Message(role='user', content=base_template + accumulated_thoughts + observation)
        ]
        
        cropped_messages = self._crop_text_messages(temp_messages, walker_core, max_tokens, buffer, crop_ratio=0.3)
        
        cropped_content = cropped_messages[0].content
        if cropped_content.startswith(base_template + accumulated_thoughts):
            cropped_observation = cropped_content[len(base_template + accumulated_thoughts):]
        else:
            cropped_observation = observation
            
        return cropped_observation

    def _run(self, messages: List[Message], lang: Literal['en', 'zh'] = 'en', walker_core=None, markdown=None, buttons=None, **kwargs) -> Iterator[List[Message]]:
        url_visited = None
        text_messages = self._prepend_react_prompt(messages, lang=lang)
        MAX_CONTEXT_TOKENS = 24000

        text_messages = self._crop_text_messages(text_messages, walker_core, MAX_CONTEXT_TOKENS, buffer=1000, crop_ratio=0.2)

        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        response: str = 'Thought: '
        query = self.llm_cfg["query"]
        action_count = self.llm_cfg.get("action_count", MAX_LLM_CALL_PER_RUN)
        num_llm_calls_available = action_count
        self.total_subpage_visits = 0
        self.cache_hits = 0
        self.root_url_cache_hits = 0  
        self.llm_input_token_counts = []
        thought_history = []  
        
        self.draft1_cache_misses = 0  
        self.draft2_cache_misses = 0  
        self.draft1_cache_hits = 0    
        self.draft2_cache_hits = 0    
        
        self.draft1_iteration_web_times = []  
        self.draft2_iteration_web_times = []  
        
        self._last_url_visited = None
        
        self.draft1_cache = {}  
        self.draft2_cache = {}  

        # ---
        while num_llm_calls_available > 0:
            print(f"Entering while loop with num_llm_calls_available: {num_llm_calls_available}")
            num_llm_calls_available -= 1
            output = []
            
            current_round_tokens = 0
            for m in text_messages:
                token_count = walker_core.count_tokens(m.content)
                current_round_tokens += token_count
                self.llm_input_token_counts.append(token_count)
            print(f"Current round tokens: {current_round_tokens}")
            
            print(f"=== ASSISTED LLM SECTION - use_assisted_llm: {self.use_assisted_llm} ===")
            if self.use_assisted_llm:
                assist_text_messages = self._prepend_react_prompt(messages, lang='en', template=ASSISTANT_SYSTEM_EXPLORER)
                assist_text_messages = self._crop_text_messages(assist_text_messages, walker_core, MAX_CONTEXT_TOKENS, buffer=1000, crop_ratio=0.2)
                
                print(f"=== DRAFT MODEL CONFIGURATION ===")
                print(f"Draft 1 config: {self.assist_llm_1}")
                print(f"Draft 2 config: {self.assist_llm_2}")
                
                print("=== STARTING DRAFT 1 LLM ===")
                try:
                    self.assist_llm_gen(messages, walker_core, text_messages=assist_text_messages, draft_num=1)
                    print("=== DRAFT 1 LLM COMPLETED ===")
                except Exception as e:
                    print(f"Error in draft 1 LLM generation: {e}")
                
                print(f"=== CHECKING DRAFT 2 CONFIGURATION ===")
                print(f"Draft 2 config: {self.assist_llm_2}")
                if self.assist_llm_2 is not None:
                    print("=== STARTING DRAFT 2 LLM ===")
                    try:
                        self.assist_llm_gen(messages, walker_core, text_messages=assist_text_messages, draft_num=2)
                        print("=== DRAFT 2 LLM COMPLETED ===")
                    except Exception as e:
                        print(f"Error in draft 2 LLM generation: {e}")
                else:
                    print("Draft 2 LLM not configured, skipping...")
            
            start_time = time.time()
            try:
                model_name = self.llm_cfg['model']
                token_param = {}
                if any(x in model_name for x in ["o4-mini", "o1-mini", "o3-mini", "gpt-5"]):
                    token_param["max_completion_tokens"] = self.llm_cfg['generate_cfg']['max_tokens']
                else:
                    token_param["max_tokens"] = self.llm_cfg['generate_cfg']['max_tokens']
                
                start_time = time.time()
                print("Started main LLM generation")
                if self._llm_provider == 'azure':
                    response_obj = self.client.chat.completions.create(
                        model=self.llm_cfg['model'],
                        messages=text_messages,
                        **token_param
                    )
                else:
                    response_obj = self.client.chat.completions.create(
                        model=self.llm_cfg['model'],
                        messages=text_messages,
                        service_tier="priority", 
                        **token_param
                    )
                end_time = time.time()
                call_time = end_time - start_time
                self.llm_time_main_action.append(call_time)
                self.llm_time += call_time
                print(f'Time taken for main action LLM: {call_time:.2f} seconds with model: {model_name}')
                
                output_content = response_obj.choices[0].message.content
                if output_content:
                    yield [Message(role=ASSISTANT, content=output_content)]
                    response += output_content
            except Exception as e:
                print(f"Error during chat completion: {e}")
                raise
            
            has_action, action, action_input, thought = self._detect_tool("\n"+output_content)
            if not has_action:
                self.trajectory_steps.append({
                    'thought': thought,
                    'action': None,
                    'action_input': None,
                    'observation': None,
                    'visited_url': None
                })
                if "Final Answer:" in output_content:
                    break
                else:
                    continue

            query = self.llm_cfg["query"]
            attempted_url = None
            if action_input:
                try:
                    button_url_dict = walker_core.button_dict
                    if action_input.startswith('{'):
                        params = json5.loads(action_input)
                        btn = params.get('button')
                        if btn and btn in button_url_dict:
                            url_visited = button_url_dict[btn]
                            attempted_url = url_visited
                        else:
                            attempted_url = f"Button '{btn}' not available" if btn else "No button specified"
                except Exception:
                    attempted_url = f"Error parsing action input: {action_input}"
                    pass
            if url_visited:
                self.visited_urls.append(url_visited)
                self.total_subpage_visits += 1

            if self.use_assisted_llm:
                print(f"url_visited: {url_visited}")
                cached_result = None
                if url_visited is not None:
                    cached_result = walker_core.get_from_cache(url_visited)
                print(f"cache is: {walker_core.page_cache.keys()}")
                
                self._last_url_visited = url_visited
                
                current_iteration_web_time = 0  
                
                print(f"\n=== DRAFT CACHE DEBUG ===")
                print(f"URL being checked: {url_visited}")
                print(f"Draft 1 cache contents: {list(self.draft1_cache.keys())}")
                print(f"Draft 2 cache contents: {list(self.draft2_cache.keys())}")
                
                if url_visited is not None:
                    if self.assist_llm_1 is not None and url_visited in self.draft1_cache:
                        self.draft1_cache_hits += 1
                        print(f"Main action would be a cache hit for draft 1: {url_visited}")
                    elif self.assist_llm_1 is not None:
                        self.draft1_cache_misses += 1
                        print(f"Main action is a cache miss for draft 1: {url_visited}")
                    
                    if self.assist_llm_2 is not None and url_visited in self.draft2_cache:
                        self.draft2_cache_hits += 1
                        print(f"Main action would be a cache hit for draft 2: {url_visited}")
                    elif self.assist_llm_2 is not None:
                        self.draft2_cache_misses += 1
                        print(f"Main action is a cache miss for draft 2: {url_visited}")
                else:
                    print(f"URL is None, skipping cache check")
                print(f"=== END DRAFT CACHE DEBUG ===\n")
                
                if cached_result is not None:
                    observation, markdown, buttons, cache_type = cached_result
                    self.cache_hits += 1
                    if cache_type == "root_url":
                        self.root_url_cache_hits += 1
                        print(f"Root URL cache hit! Total root URL cache hits: {self.root_url_cache_hits}")
                else:
                    tool_output = self._call_tool(action, action_input, messages=messages, walker=walker_core)
                    observation = self._parse_tool_output(tool_output, action)
                
                self._track_iteration_web_time(walker_core)
            else:
                print(f"Assisted LLM disabled: calling tool directly for {url_visited}")
                tool_output = self._call_tool(action, action_input, messages=messages, walker=walker_core)
                observation = self._parse_tool_output(tool_output, action)

            self.trajectory_steps.append({
                'thought': thought,
                'action': action,
                'action_input': action_input,
                'observation': observation,
                'visited_url': url_visited,
                'attempted_url': attempted_url
            })

            print(f"Entering observation_information_extraction with num_llm_calls_available: {num_llm_calls_available}")
            stage1 = self.observation_information_extraction(query, observation)
            print(f"Stage 1 finished with status: {stage1}")
            if stage1:
                self.memory.append(stage1+"\n")
                if len(self.memory) > 1:
                    yield [Message(role=ASSISTANT, content= "Memory:\n" + "-".join(self.memory)+"\"")]
                else:
                    yield [Message(role=ASSISTANT, content= "Memory:\n" + "-" + self.memory[0]+"\"")]
                print(f"Entering critic_information with num_llm_calls_available: {num_llm_calls_available}")
                stage2 = self.critic_information(query, self.memory)
                print(f"Stage 2 finished with status: {stage2}")
                if stage2:
                    response = f'{stage2}'
                    yield [Message(role=ASSISTANT, content=response)]
                    break
            print(f"Exiting observation_information_extraction with num_llm_calls_available: {num_llm_calls_available}")
            thought_history.append(f"Thought: {thought}\nAction: {action}\nAction Input: {action_input}")
            
            accumulated_thoughts = "\n".join(thought_history)
            
            base_content = text_messages[-1].content
            thought_start = base_content.find("Thought:")
            if thought_start != -1:
                base_template = base_content[:thought_start]
            else:
                base_template = base_content
            
            raw_observation_tokens = walker_core.count_tokens(observation)
            if raw_observation_tokens > 25000:  
                print(f"Raw observation is very large ({raw_observation_tokens} tokens), cropping first...")
                paragraphs = observation.split('\n\n')
                while raw_observation_tokens > 20000 and len(paragraphs) > 1:
                    remove_count = max(1, len(paragraphs) // 3)
                    paragraphs = paragraphs[:-remove_count]
                    observation = '\n\n'.join(paragraphs)
                    raw_observation_tokens = walker_core.count_tokens(observation)
                print(f"After cropping raw observation: {raw_observation_tokens} tokens")
            
            formatted_observation = f'\nObservation: {observation}\nThought: '
            
            current_context = base_template + accumulated_thoughts + formatted_observation
            
            total_tokens = walker_core.count_tokens(current_context)
            
            if total_tokens > MAX_CONTEXT_TOKENS:
                print(f"Context too large ({total_tokens} tokens), applying cropping...")
                cropped_observation = self._crop_context_with_observation(
                    base_template, accumulated_thoughts, formatted_observation, 
                    walker_core, MAX_CONTEXT_TOKENS, buffer=1000
                )
                current_context = base_template + accumulated_thoughts + cropped_observation
                final_tokens = walker_core.count_tokens(current_context)
                print(f"After cropping context: {final_tokens} tokens")
            else:
                cropped_observation = formatted_observation
            
            response += cropped_observation
            if (not text_messages[-1].content.endswith('\nThought: ')) and (not thought.startswith('\n')):
                text_messages[-1].content += '\n'
            if action_input.startswith('```'):
                action_input = '\n' + action_input
            
            text_messages[-1].content += thought + f'\nAction: {action}\nAction Input: {action_input}' + cropped_observation

        print(f"Total subpage visits: {self.total_subpage_visits}")
        print(f"Cache hits: {self.cache_hits}")
        print(f"Root URL cache hits: {self.root_url_cache_hits}")
        
        return response

    def _track_iteration_web_time(self, walker_core):
        current_iteration_web_time = 0
        if walker_core.web_time_subpages:
            current_iteration_web_time = walker_core.web_time_subpages[-1]
        
        if self.assist_llm_1 is not None:
            if hasattr(self, '_last_url_visited') and self._last_url_visited:
                if self._last_url_visited in self.draft1_cache:
                    self.draft1_iteration_web_times.append(0)
                else:
                    self.draft1_iteration_web_times.append(current_iteration_web_time)
        
        if self.assist_llm_2 is not None:
            if hasattr(self, '_last_url_visited') and self._last_url_visited:
                if self._last_url_visited in self.draft2_cache:
                    self.draft2_iteration_web_times.append(0)
                else:
                    self.draft2_iteration_web_times.append(current_iteration_web_time)

    def _prepend_react_prompt(self, messages: List[Message], lang: Literal['en', 'zh'], template: str = SYSTEM_EXPLORER) -> List[Message]:
        tool_descs = []
        for f in self.function_map.values():
            function = f.function
            name = function.get('name', None)
            name_for_human = function.get('name_for_human', name)
            name_for_model = function.get('name_for_model', name)
            assert name_for_human and name_for_model
            args_format = function.get('args_format', '')
            tool_descs.append(
                TOOL_DESC.format(name_for_human=name_for_human,
                                 name_for_model=name_for_model,
                                 description_for_model=function['description'],
                                 parameters=json.dumps(function['parameters'], ensure_ascii=False),
                                 args_format=args_format).rstrip())
        tool_descs = '\n\n'.join(tool_descs)
        tool_names = ','.join(tool.name for tool in self.function_map.values())
        text_messages = [format_as_text_message(m, add_upload_info=True, lang=lang) for m in messages]
        text_messages[-1].content = template.format(
            tool_descs=tool_descs,
            tool_names=tool_names,
            query=text_messages[-1].content,
        )
        return text_messages

    def _detect_tool(self, text: str) -> Tuple[bool, str, str, str]:
        special_func_token = '\nAction:'
        special_args_token = '\nAction Input:'
        special_obs_token = '\nObservation:'
        func_name, func_args = None, None
        i = text.rfind(special_func_token)
        j = text.rfind(special_args_token)
        k = text.rfind(special_obs_token)
        if 0 <= i < j:
            if k < j or k == -1:
                k = len(text)
            func_name = text[i + len(special_func_token):j].strip()
            func_args = text[j + len(special_args_token):k].strip()
            text = text[:i]  
        return (func_name is not None), func_name, func_args, text
    
    def _detect_tool_multiple_action_inputs(self, text: str) -> Tuple[bool, str, List[str], str]:
        """
        Extracts the last Action, all numbered Action Input lines (Action Input 1, 2, 3, ...), and the Thought from the text.
        Returns (has_action, action, [action_inputs], thought)
        """
        action_match = list(re.finditer(r'(?:^|\n)Action:\s*(.*)', text))
        if not action_match:
            return (False, None, [], text)
        last_action = action_match[-1]
        action = last_action.group(1).strip()
        after_action = text[last_action.end():]
        obs_idx = after_action.find('\nObservation:')
        if obs_idx != -1:
            action_input_block = after_action[:obs_idx]
        else:
            action_input_block = after_action
        action_inputs = re.findall(r'Action Input \d+:\s*(.*)', action_input_block)
        thought = text[:last_action.start()].strip()
        return (True, action, action_inputs, thought)

    def get_llm_time_breakdown(self) -> Dict[str, List[float]]:
        """Get detailed breakdown of LLM call times by type.
        
        Returns:
            Dict containing lists of call times for each LLM type
        """
        return {
            'main_action': self.llm_time_main_action,
            'observation_extraction': self.llm_time_observation_extraction,
            'critic': self.llm_time_critic,
            'assisted_gen': self.llm_time_assisted_gen,
            'total': self.llm_time
        }

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        try:
            encoding = tiktoken.encoding_for_model("gpt-4")  
            return len(encoding.encode(text))
        except:
            return len(text.split()) * 1.3
