import os
import openai
import logging
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger("LLMInterface")

class LLMInterface:
    """
    A robust interface for interacting with the DeepSeek LLM API.
    Handles initialization, chat completion, and graceful fallbacks.
    """
    
    def __init__(self) -> None:
        self.api_key: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
        self.base_url: str = "https://api.deepseek.com"
        self.model: str = "deepseek-chat"
        self.client: Optional[openai.OpenAI] = None

        if not self.api_key:
            logger.warning("⚠️ DEEPSEEK_API_KEY environment variable not found. LLM features will be disabled or mocked.")
        else:
            try:
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
                logger.info("✅ DeepSeek API Client initialized.")
            except Exception as e:
                logger.error(f"❌ Failed to initialize DeepSeek Client: {e}")
                self.client = None

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """
        Perform a chat completion using the DeepSeek API.

        Args:
            system_prompt (str): The system message to set the behavior of the assistant.
            user_prompt (str): The input message from the user.
            temperature (float): Sampling temperature for response generation. Default is 0.7.
            max_tokens (int): Maximum number of tokens to generate. Default is 1024.

        Returns:
            str: The generated response or an error message if the call fails.
        """
        if not self.client:
            return "Error: API Key not configured."

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content
            return content if content else "Error: Empty response received."
        except openai.AuthenticationError:
            logger.error("❌ Authentication failed. Check your API key.")
            return "Error: Authentication failed. Invalid API key."
        except openai.APIConnectionError as e:
            logger.error(f"❌ Network error during API call: {e}")
            return "Error: Failed to connect to the API. Please check your network connection."
        except openai.BadRequestError as e:
            logger.error(f"❌ Bad request sent to API: {e}")
            return "Error: Invalid request sent to the API."
        except openai.RateLimitError as e:
            logger.error(f"❌ Rate limit exceeded: {e}")
            return "Error: Rate limit exceeded. Please try again later."
        except Exception as e:
            logger.error(f"❌ Unexpected error during API call: {e}")
            return f"Error during API call: {str(e)}"

    def is_enabled(self) -> bool:
        """
        Check if the LLM interface is properly configured and enabled.

        Returns:
            bool: True if client is initialized, False otherwise.
        """
        return self.client is not None