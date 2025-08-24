from graph_state import UserQueryState


def structured_query_agent_node(state: UserQueryState) -> UserQueryState:
    state["messages"] = ["This is a response from the structured query agent."]
    return state
