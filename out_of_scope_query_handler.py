from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, AIMessage

from graph_state import UserQueryState
from app.const import OUT_OF_SCOPE_HANDLER_SYSTEM_PROMPT_FILE_PATH
from prompt import read_prompt_file
from llm import llm


class OutOfScopeResponse(BaseModel):
    response: str = Field(
        ..., description="A polite message indicating the query is out of scope"
    )


def out_of_scope_handler_node(state: UserQueryState) -> UserQueryState:

    system_prompt = read_prompt_file(OUT_OF_SCOPE_HANDLER_SYSTEM_PROMPT_FILE_PATH)

    response = llm.with_structured_output(OutOfScopeResponse).invoke(
        state["messages"] + [system_prompt]
    )

    state["messages"] = [
        SystemMessage(content=system_prompt),
        AIMessage(content=response.model_dump_json()),
    ]
    state["final_response"] = response.response
    state["is_complete"] = True

    return state
