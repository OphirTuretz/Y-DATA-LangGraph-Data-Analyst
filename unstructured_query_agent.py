from graph_state import UserQueryState


def unstructured_query_agent_node(state: UserQueryState) -> UserQueryState:
    state["messages"] = ["This is a response from the unstructured query agent."]
    return state
