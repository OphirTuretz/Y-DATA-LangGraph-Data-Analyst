import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.store.sqlite import SqliteStore

from graph_state import UserQueryState
from router import router_node, get_query_label, QueryLabel
from structured_query_agent import (
    structured_query_agent_node,
    structured_query_agent_tool_node,
)
from unstructured_query_agent import (
    unstructured_query_agent_node,
    unstructured_query_agent_tool_node,
)
from out_of_scope_query_handler import out_of_scope_handler_node
from summarized_memory import save_memory_node, read_memory_node
from app.const import (
    CHECKPOINTER_DB_FILE_PATH,
    STORE_DB_FILE_PATH,
    GRAPH_VISUALIZATION_FILE_NAME_BASE_FULL_PATH,
)


def save_graph_visualization(
    graph, filename_base_full_path=GRAPH_VISUALIZATION_FILE_NAME_BASE_FULL_PATH
):
    try:
        # Get the graph representation
        graph_image = graph.get_graph(xray=True)

        # Save Mermaid source code
        mermaid_code = graph_image.draw_mermaid()
        with open(f"{filename_base_full_path}.mmd", "w") as f:
            f.write(mermaid_code)
        print(f"Mermaid source saved as {filename_base_full_path}.mmd")

    except Exception as e:
        print(f"Error saving graph: {e}")


def is_complete(state: UserQueryState) -> bool:
    return bool(state["is_complete"])


workflow_builder = StateGraph(UserQueryState)

workflow_builder.add_node("router", router_node)

workflow_builder.add_node("structured_query_agent", structured_query_agent_node)
workflow_builder.add_node(
    "structured_query_agent_tools", structured_query_agent_tool_node
)

workflow_builder.add_node("unstructured_query_agent", unstructured_query_agent_node)
workflow_builder.add_node(
    "unstructured_query_agent_tools", unstructured_query_agent_tool_node
)

workflow_builder.add_node("out_of_scope_handler", out_of_scope_handler_node)

workflow_builder.add_node("save_memory", save_memory_node)
workflow_builder.add_node("read_memory", read_memory_node)

workflow_builder.add_edge(START, "router")

workflow_builder.add_conditional_edges(
    "router",
    get_query_label,
    {
        QueryLabel.structured: "structured_query_agent",
        QueryLabel.unstructured: "unstructured_query_agent",
        QueryLabel.out_of_scope: "out_of_scope_handler",
        QueryLabel.memory: "read_memory",
    },
)

workflow_builder.add_conditional_edges(
    "structured_query_agent",
    is_complete,
    {
        True: "save_memory",
        False: "structured_query_agent_tools",
    },
)
workflow_builder.add_conditional_edges(
    "structured_query_agent_tools",
    is_complete,
    {True: "save_memory", False: "structured_query_agent"},
)

workflow_builder.add_conditional_edges(
    "unstructured_query_agent",
    is_complete,
    {
        True: "save_memory",
        False: "unstructured_query_agent_tools",
    },
)
workflow_builder.add_conditional_edges(
    "unstructured_query_agent_tools",
    is_complete,
    {True: "save_memory", False: "unstructured_query_agent"},
)

workflow_builder.add_edge("out_of_scope_handler", "save_memory")

workflow_builder.add_edge("read_memory", "save_memory")

workflow_builder.add_edge("save_memory", END)

checkpointer_conn = sqlite3.connect(
    CHECKPOINTER_DB_FILE_PATH, check_same_thread=False, isolation_level=None
)
serde = JsonPlusSerializer(pickle_fallback=True)
checkpointer = SqliteSaver(checkpointer_conn, serde=serde)

store_conn = sqlite3.connect(
    STORE_DB_FILE_PATH, check_same_thread=False, isolation_level=None
)
store = SqliteStore(store_conn)

workflow = workflow_builder.compile(checkpointer=checkpointer, store=store)
