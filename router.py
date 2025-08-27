from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


from graph_state import UserQueryState
from prompt import read_prompt_file
from app.const import ROUTER_SYSTEM_PROMPT_FILE_PATH
from llm import llm


class QueryLabel(str, Enum):
    structured = "structured"
    unstructured = "unstructured"
    out_of_scope = "out-of-scope"
    memory = "memory"


class QueryClassification(BaseModel):
    reasoning: str = Field(
        ...,
        description="Explain how you arrived at the classification decision based on user request",
    )
    label: QueryLabel = Field(
        ..., description="The chosen label for the query classification"
    )


def router_node(state: UserQueryState) -> UserQueryState:

    user_query = state["user_query"]
    system_prompt = read_prompt_file(ROUTER_SYSTEM_PROMPT_FILE_PATH)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_query)]

    response = llm.with_structured_output(QueryClassification).invoke(messages)

    state["messages"] = messages + [AIMessage(content=response.model_dump_json())]
    state["query_classification_result"] = {
        "reason": response.reasoning,
        "label": response.label,
    }

    return state


def get_query_label(state: UserQueryState) -> QueryLabel:
    return QueryLabel(state["query_classification_result"]["label"])
