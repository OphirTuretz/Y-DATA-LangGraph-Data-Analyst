import os


# Constants for file paths
PROMPTS_DIR = "prompts"


# LLM
LLM_MODEL_NAME = "gpt-4o-mini"  # "gpt-3.5-turbo"  # "gpt-4o-mini"
LLM_TEMPERATURE = 0.0
LLM_TOP_P = 1.0


# Router
ROUTER_SYSTEM_PROMPT_FILE_NAME = "router_system_prompt.txt"
ROUTER_SYSTEM_PROMPT_FILE_PATH = os.path.join(
    PROMPTS_DIR, ROUTER_SYSTEM_PROMPT_FILE_NAME
)
