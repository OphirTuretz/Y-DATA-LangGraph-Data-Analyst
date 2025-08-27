from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import Annotated
from operator import add

from data import Dataset


class UserQueryState(TypedDict):
    user_query: str
    query_classification_result: dict
    messages: Annotated[list, add_messages]
    concise_history: Annotated[list, add]
    dataset: Dataset
    is_complete: bool
    final_response: str
    iteration_count: int
    memory_saved: bool
