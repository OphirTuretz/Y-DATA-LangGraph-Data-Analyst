import uuid
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_store
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, AIMessage


from graph_state import UserQueryState
from app.const import SAVE_MEMORY_PROMPT_FILE_PATH, READ_MEMORY_PROMPT_FILE_PATH
from prompt import read_prompt_file
from llm import llm


class MemorySummarySave(BaseModel):
    reasoning: str = Field(
        ..., description="Explain briefly why you decided to save or not save"
    )
    should_save: bool = Field(
        ..., description="Whether this memory should be saved or not"
    )
    summary: str = Field(
        ...,
        description="A short, precise memory of the new user information (empty if should_save is false)",
    )


class MemorySummaryRead(BaseModel):
    reasoning: str = Field(
        ...,
        description="Explain briefly how the past memories are relevant to the user query.",
    )
    relevant_memories: str = Field(
        ...,
        description="A short, user-friendly message summarizing the relevant past memories in natural language.",
    )


def save_memory_node(state: UserQueryState, config: RunnableConfig) -> UserQueryState:

    store = get_store()
    user_id = config["configurable"].get("user_id")

    namespace = ("user_memories", user_id)
    memories = store.search(namespace, limit=20)

    if not memories:
        past_memories = "No relevant past memories."
    else:
        past_memories = "\n".join([f"- {m.value['content']}" for m in memories])

    user_query = state["user_query"]
    final_response = state["final_response"]

    system_propmt = read_prompt_file(SAVE_MEMORY_PROMPT_FILE_PATH).format(
        user_query=user_query,
        analyst_response=final_response,
        past_memories=past_memories,
    )

    response = llm.with_structured_output(MemorySummarySave).invoke(system_propmt)

    if response.should_save:
        memory_data = response.summary

        memory_key = str(uuid.uuid4())
        store.put(namespace, memory_key, {"content": memory_data})

        state["memory_saved"] = True

    return state


def read_memory_node(state: UserQueryState, config: RunnableConfig) -> UserQueryState:

    store = get_store()
    user_id = config["configurable"].get("user_id")

    namespace = ("user_memories", user_id)
    memories = store.search(namespace, limit=50)

    if not memories:
        past_memories = "No relevant past memories."
    else:
        past_memories = "\n".join([f"- {m.value['content']}" for m in memories])

    user_query = state["user_query"]

    system_propmt = read_prompt_file(READ_MEMORY_PROMPT_FILE_PATH).format(
        user_query=user_query,
        past_memories=past_memories,
    )

    response = llm.with_structured_output(MemorySummaryRead).invoke(system_propmt)

    state["final_response"] = response.relevant_memories
    state["is_complete"] = True
    state["messages"] = [
        SystemMessage(content=system_propmt),
        AIMessage(content=response.model_dump_json()),
    ]

    return state
