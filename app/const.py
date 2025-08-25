import os


# Constants for file paths
PROMPTS_DIR = "prompts"

# Dataset
DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
DATASET_SPLIT_NAME = "train"


# LLM
LLM_MODEL_NAME = "gpt-4o-mini"  # "gpt-3.5-turbo"  # "gpt-4o-mini"
LLM_TEMPERATURE = 0.0
LLM_TOP_P = 1.0


# Router
ROUTER_SYSTEM_PROMPT_FILE_NAME = "router_system_prompt.txt"
ROUTER_SYSTEM_PROMPT_FILE_PATH = os.path.join(
    PROMPTS_DIR, ROUTER_SYSTEM_PROMPT_FILE_NAME
)

# Unstructured Query Agent
SUMMARIZE_BATCH_PROMPT_FILE_NAME = "summarize_batch_prompt.txt"
SUMMARIZE_BATCH_PROMPT_FILE_PATH = os.path.join(
    PROMPTS_DIR, SUMMARIZE_BATCH_PROMPT_FILE_NAME
)

SUMMARIZE_ALL_BATCHES_PROMPT_FILE_NAME = "summarize_all_batches_prompt.txt"
SUMMARIZE_ALL_BATCHES_PROMPT_FILE_PATH = os.path.join(
    PROMPTS_DIR, SUMMARIZE_ALL_BATCHES_PROMPT_FILE_NAME
)

SUMMARIZE_DEFAULT_BATCH_SIZE = 20
SUMMARIZE_DEFAULT_N_BATCHES = 5
