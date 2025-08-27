from typing import Callable

from graph import workflow
from data import Dataset


def process_user_query(
    user_query: str,
    log_fn: Callable[[str], None] = print,
    user_id: str = None,
    thread_id: str = None,
    has_history: bool = False,
):
    """Process user query using the LangGraph workflow.

    Args:
        user_query (str): The user's question
        log_fn (callable): Logging function
        user_id (str): The user ID for this conversation
        thread_id (str): The thread ID for this conversation
        has_history (bool): Whether this thread has existing conversation history

    Returns:
        dict: Contains 'dataset' and 'response'
    """

    initial_state = {
        "user_query": user_query,
        "query_classification_result": {},
        "messages": [],
        "is_complete": False,
        "final_response": None,
        "iteration_count": 0,
        "memory_saved": False,
    }

    # Initialize dataset based on whether thread has history
    if not has_history:
        initial_state["dataset"] = Dataset()  # Fresh dataset for new thread
        log_fn(f"Initialized fresh dataset for new thread {thread_id}")

    # Configure with proper user_id and thread_id
    config = {
        "recursion_limit": 100,
        "configurable": {
            "thread_id": user_id + "_" + thread_id,
            "user_id": user_id,
        },
    }

    log_fn(
        f"Processing query through workflow for user {user_id}, thread {thread_id}..."
    )
    final_state = workflow.invoke(initial_state, config)

    for m in final_state["messages"]:
        log_fn(m.pretty_repr())

    log_fn("Workflow processing complete.")

    return {
        "response": final_state["final_response"],
    }
