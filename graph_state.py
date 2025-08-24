from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import Annotated, List


class UserQueryState(TypedDict):
    user_query: str
    query_classification_result: dict
    messages: Annotated[list, add_messages]
    # function_results: List[str]
    # final_response: str
