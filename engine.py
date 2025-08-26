from typing import Callable

from graph import workflow
from data import Dataset


def process_user_query(
    user_query: str, dataset: Dataset, log_fn: Callable[[str], None] = print
):
    """Process user query using the LangGraph workflow.

    Args:
        user_query (str): The user's question
        dataset (Dataset): The dataset instance
        log_fn (callable): Logging function

    Returns:
        dict: Contains 'dataset' and 'response'
    """

    initial_state = {
        "user_query": user_query,
        "query_classification_result": {},
        "messages": [],
        "dataset": dataset,
        "is_complete": False,
        "final_response": None,
        "iteration_count": 0,
    }

    log_fn("Processing query through workflow...")
    final_state = workflow.invoke(initial_state, {"recursion_limit": 100})

    for m in final_state["messages"]:
        log_fn(m.pretty_repr())

    log_fn("Workflow processing complete.")

    return {
        "dataset": final_state["dataset"],
        "response": final_state["final_response"],
    }
