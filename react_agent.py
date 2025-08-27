from langchain_core.messages import SystemMessage, AIMessage

from graph_state import UserQueryState
from prompt import read_prompt_file
from llm import llm
from app.const import MAX_ITERATIONS, DEFAULT_PARALLEL_TOOL_CALLS


def react_agent_node(
    state: UserQueryState,
    system_prompt_file_path: str,
    agent_tool_list: list,
) -> UserQueryState:

    # Get the current iteration count, defaulting to 0 if not present
    iteration_count = state.get("iteration_count", 0)

    # Check if the maximum number of iterations has been reached
    if iteration_count >= MAX_ITERATIONS:
        # If so, return a message indicating that processing is stopping
        state["messages"] = [
            AIMessage(
                content="Maximum iterations reached. Stopping further processing.",
            )
        ]
        state["final_response"] = (
            "Sorry, the request caused too many internal steps and could not be completed."
        )
        state["is_complete"] = True
        return state

    # If this is the first iteration, add the system prompt to the messages
    new_messages = []
    if iteration_count == 0:

        concise_history = state.get("concise_history", [])

        if not concise_history:
            history = "No relevant conversation history."
        else:
            history = "\n".join(
                [f"{k}: {v}" for m in concise_history for k, v in m.items()]
            )

        system_prompt = read_prompt_file(system_prompt_file_path).format(
            history=history
        )
        new_messages = [SystemMessage(content=system_prompt)]

    # Bind the tools to the LLM and invoke it with the current messages
    llm_with_tools = llm.bind_tools(
        tools=agent_tool_list,
        parallel_tool_calls=DEFAULT_PARALLEL_TOOL_CALLS,
    )
    response = llm_with_tools.invoke(state["messages"] + new_messages)

    # Update the state with the new messages and increment the iteration count
    state["messages"] = new_messages + [response]
    state["iteration_count"] = iteration_count + 1

    return state
