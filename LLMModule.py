# "E:/Project Expo 2/Harshgup16/llama-3-8b-Instruct-bnb-4bit-laptop-recommendation/unsloth.Q4_K_M.gguf"
# import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Generator
from dataclasses import dataclass, asdict

@dataclass
class ModelConfig:
    """Configuration for language model parameters"""
    context_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    chat_template: str = (
        "{% set loop_messages = messages %}{% for message in loop_messages %}"
        "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'"
        "+ message['content'] | trim + '<|eot_id|>' %}"
        "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
        "{{ content }}{% endfor %}"
        "{% if add_generation_prompt %}}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
    )

class LLMException(Exception):
    """Custom exception for LLM-related errors"""
    pass

class LLMModule:
    def __init__(
        self,
        model_path: str,
        config: Optional[ModelConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the LLM module with a GGUF model and chat template support.
        
        Args:
            model_path: Path to the GGUF model file
            config: Optional configuration for model parameters
            logger: Optional logger for tracking events and errors
        """
        # Setup logging
        self.logger = logger or logging.getLogger(__name__)
        
        # Validate and set model path
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            self.logger.error(f"Model file not found: {model_path}")
            raise LLMException(f"Model file not found: {model_path}")
            
        # Set configuration
        self.config = config or ModelConfig()
        
        # Model initialization
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the model and required components"""
        try:
            from llama_cpp import Llama
            
            self.model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.config.context_length,
                n_threads=None,
                verbose=False
            )
            
            self.logger.info(f"Model initialized: {self.model_path}")
        
        except ImportError:
            self.logger.error("llama-cpp-python not found.")
            raise LLMException(
                "llama-cpp-python not found. Install with: pip install llama-cpp-python"
            )
        except Exception as e:
            self.logger.error(f"Model initialization failed: {str(e)}")
            raise LLMException(f"Failed to initialize model: {str(e)}")

    def format_chat_messages(
        self, 
        messages: List[Dict[str, str]]
    ) -> str:
        """
        Format chat messages using the configured chat template.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
        
        Returns:
            Formatted prompt string
        """
        try:
            # Import Jinja2 for template rendering
            from jinja2 import Template
            
            # Create Jinja2 template
            template = Template(self.config.chat_template)
            
            # Render template with messages
            # Simulate BOS token and generation prompt behavior
            rendered_prompt = template.render(
                messages=messages, 
                bos_token="<s>",  # Beginning of sequence token
                add_generation_prompt=True
            )
            
            return rendered_prompt
        
        except ImportError:
            self.logger.error("Jinja2 not found. Install with: pip install jinja2")
            raise LLMException("Jinja2 is required for chat template rendering")
        
        except Exception as e:
            self.logger.error(f"Chat message formatting failed: {str(e)}")
            raise LLMException(f"Chat message formatting error: {str(e)}")

    def chat(
        self, 
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> Dict:
        """
        Generate a chat response using the formatted messages.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Optional list of sequences to stop generation
            **kwargs: Additional generation parameters
        
        Returns:
            Dictionary with generated response and metadata
        """
        # Format messages using chat template
        formatted_prompt = self.format_chat_messages(messages)
        
        # Prepare generation configuration
        generation_config = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "repeat_penalty": self.config.repeat_penalty,
            "max_tokens": max_tokens,
            "stop": stop_sequences or []
        }
        generation_config.update(kwargs)
        
        try:
            # Generate response
            response = self.model(
                formatted_prompt,
                **generation_config
            )
            
            return {
                "generated_text": response["choices"][0]["text"],
                "tokens_used": response["usage"]["total_tokens"],
                "finish_reason": response["choices"][0]["finish_reason"]
            }
        
        except Exception as e:
            self.logger.error(f"Chat generation failed: {str(e)}")
            raise LLMException(f"Chat generation failed: {str(e)}")

    def stream_chat(
        self, 
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream chat response token by token.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Optional list of sequences to stop generation
            **kwargs: Additional generation parameters
        
        Yields:
            Generated tokens
        """
        # Format messages using chat template
        formatted_prompt = self.format_chat_messages(messages)
        
        # Prepare generation configuration
        generation_config = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "repeat_penalty": self.config.repeat_penalty,
            "max_tokens": max_tokens,
            "stop": stop_sequences or [],
            "stream": True
        }
        generation_config.update(kwargs)
        
        try:
            # Start generation
            stream = self.model(
                formatted_prompt,
                **generation_config
            )

            generated_text = ""
            for token_data in stream:
                token_text = token_data["choices"][0]["text"]
                generated_text += token_text
                
                # Check stop sequences
                if stop_sequences and any(stop_seq in generated_text for stop_seq in stop_sequences):
                    break
                
                yield token_text
        
        except Exception as e:
            self.logger.error(f"Chat streaming failed: {str(e)}")
            raise LLMException(f"Chat streaming failed: {str(e)}")