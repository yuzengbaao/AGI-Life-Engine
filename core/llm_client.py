import os
import logging
import time
import hashlib
import json
from typing import List, Dict, Any, Optional
from functools import wraps
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file (handle GBK/GB2312 encoding)
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
if os.path.exists(env_path):
    # Try different encodings
    content = None
    for encoding in ['utf-8', 'gbk', 'gb2312']:
        try:
            with open(env_path, 'r', encoding=encoding) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue

    if content:
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                if key:
                    os.environ[key] = value

try:
    from openai import OpenAI, OpenAIError
except ImportError:
    OpenAI = None
    OpenAIError = Exception

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    genai = None
    GOOGLE_AVAILABLE = False


def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
):
    """
    Retry decorator with exponential backoff for LLM API calls.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            delay = initial_delay

            while retry_count <= max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retry_count += 1

                    if retry_count > max_retries:
                        # Log final failure
                        logging.getLogger("LLMService").error(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}: {e}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    current_delay = min(delay, max_delay)
                    if jitter:
                        import random
                        current_delay = current_delay * (0.5 + random.random())

                    logging.getLogger("LLMService").warning(
                        f"Attempt {retry_count}/{max_retries} failed for {func.__name__}: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )

                    time.sleep(current_delay)
                    delay *= exponential_base

            return None  # Should never reach here

        return wrapper
    return decorator


class LLMService:
    def __init__(self):
        self.logger = logging.getLogger("LLMService")
        self.client = None
        self.mock_mode = False
        self.active_provider = None
        self.active_model = None
        self.embedding_model = None
        self.vision_model = None # Model for VLM tasks

        # Response caching to avoid redundant API calls
        self.response_cache = {}
        self.cache_enabled = True
        self.cache_max_size = 1000
        self.cache_file = Path("data/llm_cache.json")
        self._load_cache()

        if not OpenAI:
            self.logger.warning("OpenAI package not found. Running in MOCK mode.")
            self.mock_mode = True
            return

        self._init_provider()

    def _load_cache(self):
        """Load response cache from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.response_cache = json.load(f)
                self.logger.debug(f"Loaded {len(self.response_cache)} cached responses")
        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
            self.response_cache = {}

    def _save_cache(self):
        """Save response cache to disk."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.response_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")

    def _generate_cache_key(self, method: str, **kwargs) -> str:
        """Generate a cache key from method name and arguments."""
        # Create a deterministic string from kwargs
        key_dict = {k: str(v)[:200] for k, v in sorted(kwargs.items())}
        key_str = json.dumps(key_dict, sort_keys=True)
        return hashlib.md5(f"{method}:{key_str}".encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if available."""
        if not self.cache_enabled:
            return None
        return self.response_cache.get(cache_key)

    def _cache_response(self, cache_key: str, response: str):
        """Cache a response."""
        if not self.cache_enabled:
            return

        # Implement simple LRU by size limit
        if len(self.response_cache) >= self.cache_max_size:
            # Remove oldest entry (first 10% of entries)
            keys_to_remove = list(self.response_cache.keys())[:self.cache_max_size // 10]
            for key in keys_to_remove:
                del self.response_cache[key]

        self.response_cache[cache_key] = response

        # Periodically save cache (every 50 new entries)
        if len(self.response_cache) % 50 == 0:
            self._save_cache()

    def clear_cache(self):
        """Clear the response cache."""
        self.response_cache.clear()
        self._save_cache()
        self.logger.info("Response cache cleared")

    def _init_provider(self):
        """
        Initialize LLM provider based on priority list in .env
        """
        priority_list = os.environ.get("LLM_PROVIDER_PRIORITY", "deepseek,dashscope,zhipu").split(",")
        
        # 1. Initialize Chat Client
        for provider in priority_list:
            provider = provider.strip().lower()
            if self._try_init_chat_provider(provider):
                self.logger.info(f"Successfully initialized Chat provider: {provider.upper()}")
                self.active_provider = provider
                break
        
        # 2. Initialize Embedding Client (Separate pass to find best embedding support)
        # DeepSeek often doesn't support standard embedding endpoints, so we prefer DashScope/Zhipu
        self.embedding_client = None
        self.embedding_model = None
        
        embedding_priority = ["zhipu", "dashscope", "openai"] # Preferred order for embeddings
        
        for provider in embedding_priority:
            if self._try_init_embedding_provider(provider):
                self.logger.info(f"Successfully initialized Embedding provider: {provider.upper()}")
                break

        # Fallback to generic OPENAI_* if defined and no provider set
        if not self.client and os.environ.get("OPENAI_API_KEY"):
            self.logger.info("Using generic OPENAI_API_KEY.")
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            )
            self.active_model = "gpt-3.5-turbo"
            if not self.embedding_client:
                self.embedding_client = self.client
                self.embedding_model = "text-embedding-3-small"
        elif not self.client:
            self.logger.warning("No valid LLM provider configured. Falling back to MOCK mode.")
            self.mock_mode = True

    def _try_init_chat_provider(self, provider: str) -> bool:
        api_key = None
        base_url = None
        model = None
        
        if provider == "deepseek":
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            base_url = "https://api.deepseek.com"
            model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        elif provider == "dashscope":
            api_key = os.environ.get("DASHSCOPE_API_KEY")
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            model = os.environ.get("DASHSCOPE_MODEL", "qwen-plus")
        elif provider == "zhipu":
            api_key = os.environ.get("ZHIPU_API_KEY")
            base_url = "https://open.bigmodel.cn/api/paas/v4/"
            model = os.environ.get("ZHIPU_MODEL", "glm-4-flash")
        elif provider == "google":
            if not GOOGLE_AVAILABLE:
                self.logger.error("google-generativeai package not installed.")
                return False
            api_key = os.environ.get("GOOGLE_API_KEY")
            model = os.environ.get("GOOGLE_MODEL", "gemini-1.5-pro")
            if api_key:
                genai.configure(api_key=api_key, transport="rest")
                self.client = genai 
                self.active_model = model
                return True
            
        if api_key and base_url:
            try:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
                self.active_model = model
                return True
            except Exception as e:
                self.logger.error(f"Failed to init Chat {provider}: {e}")
                return False
        return False

    def _try_init_embedding_provider(self, provider: str) -> bool:
        api_key = None
        base_url = None
        model = None
        
        if provider == "dashscope":
            api_key = os.environ.get("DASHSCOPE_API_KEY")
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            model = "text-embedding-v1"
        elif provider == "zhipu":
            api_key = os.environ.get("ZHIPU_API_KEY")
            base_url = "https://open.bigmodel.cn/api/paas/v4/"
            model = "embedding-2"
        elif provider == "openai":
             api_key = os.environ.get("OPENAI_API_KEY")
             base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
             model = "text-embedding-3-small"

        if api_key and base_url:
            try:
                self.embedding_client = OpenAI(api_key=api_key, base_url=base_url)
                self.embedding_model = model
                return True
            except Exception as e:
                self.logger.error(f"Failed to init Embedding {provider}: {e}")
                return False
        return False

    @retry_with_exponential_backoff(max_retries=3, initial_delay=1.0, max_delay=30.0)
    def _chat_completion_api_call(self, target_model: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
        """Internal method for actual API call with retry logic."""
        if self.active_provider == "google":
             # Google Gemini Native Implementation
             generation_config = genai.types.GenerationConfig(
                 temperature=temperature,
                 max_output_tokens=4000
             )
             model = self.client.GenerativeModel(target_model)
             # Combining system prompt as Gemini context window is huge
             combined_prompt = f"System Instruction:\n{system_prompt}\n\nUser Query:\n{user_prompt}"
             response = model.generate_content(combined_prompt, generation_config=generation_config)
             return response.text

        response = self.client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=4000,
            timeout=30.0  # Add timeout to prevent hanging
        )
        return response.choices[0].message.content

    def chat_completion(self, system_prompt: str, user_prompt: str, model: str = None, temperature: float = 0.7, use_cache: bool = True) -> str:
        """
        Generate text using LLM with caching and retry logic.

        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for the LLM
            model: Optional model override
            temperature: Sampling temperature (0.0-1.0), lower = more deterministic
            use_cache: Whether to use response caching
        """
        if self.mock_mode:
            return f"[MOCK LLM RESPONSE] Processed: {user_prompt[:50]}..."

        # Check cache first
        if use_cache:
            cache_key = self._generate_cache_key(
                "chat_completion",
                model=model or self.active_model,
                system=system_prompt[:100],
                user=user_prompt[:500],
                temp=temperature
            )
            cached = self._get_cached_response(cache_key)
            if cached:
                self.logger.debug(f"Cache hit for chat_completion")
                return cached

        # 馃敡 [2026-01-15] 闆嗘垚Insight鎻愮ず澧炲己鍣?- 鎻愬崌Insight鍙墽琛屾€?
        try:
            from core.insight_prompt_enhancer import get_insight_prompt_enhancer
            enhancer = get_insight_prompt_enhancer()
            # 澧炲己user_prompt锛圛nsight浠诲姟浼氳嚜鍔ㄦ坊鍔犲嚱鏁颁娇鐢ㄦ寚鍗楋級
            user_prompt = enhancer.enhance_prompt(user_prompt)
        except ImportError:
            # 濡傛灉澧炲己鍣ㄤ笉鍙敤锛岀户缁娇鐢ㄥ師濮嬫彁绀?
            pass
        except Exception as e:
            self.logger.warning(f"[InsightPromptEnhancer] Failed to enhance prompt: {e}")

        target_model = model or self.active_model

        try:
            response = self._chat_completion_api_call(target_model, system_prompt, user_prompt, temperature)

            # Cache the response
            if use_cache:
                self._cache_response(cache_key, response)

            return response
        except Exception as e:
            self.logger.error(f"LLM Chat Error ({self.active_provider}): {e}")

            # Enhanced fallback with context
            fallback_msg = (
                f"[LLM UNAVAILABLE] The LLM service is currently unavailable ({self.active_provider}). "
                f"Error: {str(e)[:100]}. "
                f"Using deterministic fallback response."
            )
            return fallback_msg

    @retry_with_exponential_backoff(max_retries=3, initial_delay=1.0, max_delay=30.0)
    def _chat_with_vision_api_call(self, vision_model: str, messages: list) -> str:
        """Internal method for actual vision API call with retry logic."""
        if self.active_provider == "google":
             # Google Gemini Native Vision Implementation
             try:
                 import base64
                 import io
                 from PIL import Image
                 
                 # Extract components from standard message format
                 system_msg = messages[0]["content"]
                 user_content = messages[1]["content"]
                 user_text = next(x["text"] for x in user_content if x["type"] == "text")
                 image_url = next(x["image_url"]["url"] for x in user_content if x["type"] == "image_url")
                 
                 # Decode Base64
                 image_data = base64.b64decode(image_url.split(",")[1])
                 image = Image.open(io.BytesIO(image_data))
                 
                 model = self.client.GenerativeModel(vision_model)
                 response = model.generate_content([system_msg, user_text, image])
                 return response.text
             except Exception as e:
                 raise Exception(f"Gemini Vision Error: {e}")

        response = self.client.chat.completions.create(
            model=vision_model,
            messages=messages,
            temperature=0.5,
            timeout=30.0  # Add timeout to prevent hanging
        )
        return response.choices[0].message.content

    def chat_with_vision(self, system_prompt: str, user_prompt: str, base64_image: str, use_cache: bool = False) -> str:
        """
        [New Feature] Multimodal Chat with Image with retry logic.

        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for the LLM
            base64_image: Base64-encoded image data
            use_cache: Whether to use response caching (default False for vision due to image data)
        """
        if self.mock_mode:
            return "[MOCK VISION] I see a screen with some text."

        # Check cache first (disabled by default for vision due to large image data)
        if use_cache:
            cache_key = self._generate_cache_key(
                "chat_with_vision",
                model=self.active_provider,
                system=system_prompt[:100],
                user=user_prompt[:200],
                image_hash=hashlib.md5(base64_image[:1000].encode()).hexdigest()[:16]
            )
            cached = self._get_cached_response(cache_key)
            if cached:
                self.logger.debug(f"Cache hit for chat_with_vision")
                return cached

        # Determine Vision Model based on provider
        vision_model = self.active_model
        if self.active_provider == "dashscope":
            vision_model = "qwen-vl-max"
        elif self.active_provider == "zhipu":
            vision_model = "glm-4v"
        elif self.active_provider == "openai":
            vision_model = "gpt-4o"

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        try:
            response = self._chat_with_vision_api_call(vision_model, messages)

            # Cache the response
            if use_cache:
                self._cache_response(cache_key, response)

            return response
        except Exception as e:
            self.logger.error(f"LLM Vision Error ({vision_model}): {e}")

            # Enhanced fallback
            fallback_msg = (
                f"[VISION UNAVAILABLE] The vision service is currently unavailable ({vision_model}). "
                f"Error: {str(e)[:100]}. "
                f"Image analysis failed - using text-based fallback."
            )
            return fallback_msg

    @retry_with_exponential_backoff(max_retries=3, initial_delay=1.0, max_delay=30.0)
    def _get_embedding_api_call(self, client, target_model: str, text: str) -> List[float]:
        """Internal method for actual embedding API call with retry logic."""
        response = client.embeddings.create(
            model=target_model,
            input=text,
            timeout=30.0  # Add timeout to prevent hanging
        )
        return response.data[0].embedding

    def get_embedding(self, text: str, model: str = None, use_cache: bool = True) -> List[float]:
        """
        Generate vector embedding for text with caching and retry logic.

        Args:
            text: Input text to embed
            model: Optional model override
            use_cache: Whether to use response caching
        """
        if self.mock_mode and not self.embedding_client:
             # Return random vector for testing if no provider
             return np.random.rand(1536).tolist()

        target_model = model or self.embedding_model or "text-embedding-3-small"

        # Check cache first
        if use_cache:
            cache_key = self._generate_cache_key(
                "get_embedding",
                model=target_model,
                text=text[:500]  # First 500 chars for cache key
            )
            cached = self._get_cached_response(cache_key)
            if cached:
                self.logger.debug(f"Cache hit for get_embedding")
                try:
                    return json.loads(cached)  # Parse cached JSON list
                except:
                    pass  # Fall through to API call if cache is invalid

        try:
            # Check which client to use
            client = self.embedding_client or self.client
            if not client:
                 return np.random.rand(1536).tolist()

            embedding = self._get_embedding_api_call(client, target_model, text)

            # Cache the response (as JSON string)
            if use_cache:
                self._cache_response(cache_key, json.dumps(embedding))

            return embedding
        except Exception as e:
            self.logger.error(f"LLM Embedding Error: {e}")

            # Enhanced fallback with random vector
            self.logger.warning(f"Using random fallback embedding due to error: {str(e)[:100]}")
            return np.random.rand(1536).tolist()


