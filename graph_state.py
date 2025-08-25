from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import Annotated, List

from data import Dataset


class UserQueryState(TypedDict):
    user_query: str
    query_classification_result: dict
    messages: Annotated[list, add_messages]
    dataset: Dataset
    is_complete: bool
    final_response: str
    # function_results: List[str]
