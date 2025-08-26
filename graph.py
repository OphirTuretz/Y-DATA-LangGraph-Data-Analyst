from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

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

workflow_builder.add_edge(START, "router")

workflow_builder.add_conditional_edges(
    "router",
    get_query_label,
    {
        QueryLabel.structured: "structured_query_agent",
        QueryLabel.unstructured: "unstructured_query_agent",
        QueryLabel.out_of_scope: "out_of_scope_handler",
    },
)

workflow_builder.add_conditional_edges(
    "structured_query_agent",
    is_complete,
    {
        True: END,
        False: "structured_query_agent_tools",
    },
)
workflow_builder.add_conditional_edges(
    "structured_query_agent_tools",
    is_complete,
    {True: END, False: "structured_query_agent"},
)

workflow_builder.add_conditional_edges(
    "unstructured_query_agent",
    is_complete,
    {
        True: END,
        False: "unstructured_query_agent_tools",
    },
)
workflow_builder.add_conditional_edges(
    "unstructured_query_agent_tools",
    is_complete,
    {True: END, False: "unstructured_query_agent"},
)

workflow_builder.add_edge("out_of_scope_handler", END)

workflow = workflow_builder.compile()
