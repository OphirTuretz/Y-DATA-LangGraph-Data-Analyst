from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from graph_state import UserQueryState
from router import router_node, get_query_label, QueryLabel
from structured_query_agent import structured_query_agent_node
from unstructured_query_agent import unstructured_query_agent_node
from out_of_scope_query_handler import out_of_scope_handler_node

from general_tools import (
    get_possible_intents_tool,
    get_possible_categories_tool,
)


workflow_builder = StateGraph(UserQueryState)

workflow_builder.add_node("router", router_node)
workflow_builder.add_node("structured_query_agent", structured_query_agent_node)
workflow_builder.add_node("unstructured_query_agent", unstructured_query_agent_node)
workflow_builder.add_node("out_of_scope_handler", out_of_scope_handler_node)

tool_node = ToolNode([get_possible_intents_tool, get_possible_categories_tool])
workflow_builder.add_node("act", tool_node)

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

workflow_builder.add_edge("structured_query_agent", "act")
workflow_builder.add_edge("act", END)

workflow_builder.add_edge("unstructured_query_agent", END)
workflow_builder.add_edge("out_of_scope_handler", END)

workflow = workflow_builder.compile()
