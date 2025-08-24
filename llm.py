from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from app.const import LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_TOP_P

load_dotenv()

llm = ChatOpenAI(
    model=LLM_MODEL_NAME,
    temperature=LLM_TEMPERATURE,
    top_p=LLM_TOP_P,
)
