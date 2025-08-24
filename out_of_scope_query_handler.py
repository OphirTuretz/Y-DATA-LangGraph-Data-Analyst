from graph_state import UserQueryState


def out_of_scope_handler_node(state: UserQueryState) -> UserQueryState:
    state["messages"] = ["This is a response from the out-of-scope query handler."]
    return state
