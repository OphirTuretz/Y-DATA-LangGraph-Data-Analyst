import json
from typing_extensions import Annotated
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.types import Command


from data import Dataset
from graph_state import UserQueryState
from prompt import read_prompt_file
from llm import llm
from general_tools import (
    get_possible_intents_tool,
    get_possible_categories_tool,
    select_semantic_intent_tool,
    select_semantic_category_tool,
    finish_tool,
)


# Tools


@tool
def sum_tool(
    reasoning: str,
    a: int,
    b: int,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Sum two numbers.
    Args:
        reasoning (str): Reasoning for the function call.
        a (int): First number to sum.
        b (int): Second number to sum.
        tool_call_id (str): The tool call ID for tracking.
    Returns:
        Command: A command containing the sum of a and b.
    """
    result = a + b
    return Command(
        update={
            "messages": [
                ToolMessage(
                    json.dumps({"sum": result}),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def sort_dict_by_values_tool(
    reasoning: str,
    d: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    ascending: bool = False,
) -> Command:
    """
    Sort a dictionary by its values.
    Args:
        reasoning (str): Reasoning for the function call.
        d (str): A JSON data object that can be translated to dictionary by json.loads() to sort by its values.
        tool_call_id (str): The tool call ID for tracking.
        ascending (bool): Whether to sort in ascending order. Defaults to False.
    Returns:
        Command: A command containing the sorted dictionary.
    """
    d_dict = json.loads(d)
    sorted_dict = dict(
        sorted(d_dict.items(), key=lambda item: item[1], reverse=not ascending)
    )
    return Command(
        update={
            "messages": [
                ToolMessage(
                    json.dumps({"sorted_dict": sorted_dict}),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def count_category_tool(
    reasoning: str,
    category: str,
    dataset: Annotated[Dataset, InjectedState("dataset")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Count the number of rows in the DataFrame that match a specific category.
    Args:
        reasoning (str): Reasoning for the function call.
        category (str): Category to count in the DataFrame.
        dataset (Dataset): The dataset to operate on.
        tool_call_id (str): The tool call ID for tracking.
    Returns:
        Command: A command containing the count of rows matching the specified category.
    """
    count = dataset.count_category(category)
    return Command(
        update={
            "messages": [
                ToolMessage(
                    json.dumps({"count": count}),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def count_intent_tool(
    reasoning: str,
    intent: str,
    dataset: Annotated[Dataset, InjectedState("dataset")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Count the number of rows in the DataFrame that match a specific intent.
    Args:
        reasoning (str): Reasoning for the function call.
        intent (str): Intent to count in the DataFrame.
        dataset (Dataset): The dataset to operate on.
        tool_call_id (str): The tool call ID for tracking.
    Returns:
        Command: A command containing the count of rows matching the specified intent.
    """
    count = dataset.count_intent(intent)
    return Command(
        update={
            "messages": [
                ToolMessage(
                    json.dumps({"count": count}),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def count_rows_tool(
    reasoning: str,
    dataset: Annotated[Dataset, InjectedState("dataset")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Get the number of rows in the DataFrame.
    Args:
        reasoning (str): Reasoning for the function call.
        dataset (Dataset): The dataset to operate on.
        tool_call_id (str): The tool call ID for tracking.
    Returns:
        Command: A command containing the number of rows in the DataFrame.
    """
    count = dataset.count_rows()
    return Command(
        update={
            "messages": [
                ToolMessage(
                    json.dumps({"number_of_rows": count}),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def show_examples_tool(
    reasoning: str,
    n: int,
    dataset: Annotated[Dataset, InjectedState("dataset")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Show a sample of n examples from the DataFrame.
    Args:
        reasoning (str): Reasoning for the function call.
        n (int): Number of examples to show from the DataFrame.
        dataset (Dataset): The dataset to operate on.
        tool_call_id (str): The tool call ID for tracking.
    Returns:
        Command: A command containing n random samples from the dataset.
    """
    examples_df = dataset.show_examples(n)
    examples_dict = examples_df.to_dict(orient="records")
    return Command(
        update={
            "messages": [
                ToolMessage(
                    json.dumps({"examples": examples_dict}),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


structured_query_agent_tool_list = [
    get_possible_intents_tool,
    get_possible_categories_tool,
    select_semantic_intent_tool,
    select_semantic_category_tool,
    finish_tool,
    sort_dict_by_values_tool,
    sum_tool,
    count_category_tool,
    count_intent_tool,
    count_rows_tool,
    show_examples_tool,
]


# Nodes


def structured_query_agent_node(state: UserQueryState) -> UserQueryState:

    # TODO: implement MAX_DEPTH

    user_query = state["user_query"]
    system_prompt = "Choose the correct tool to answer the user query."
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_query)]

    llm_with_tools = llm.bind_tools(
        tools=structured_query_agent_tool_list,
        parallel_tool_calls=False,
    )
    response = llm_with_tools.invoke(messages)
    state["messages"] = (
        ["This is a response from the structured query agent."] + messages + [response]
    )

    return state


structured_query_agent_tool_node = ToolNode(structured_query_agent_tool_list)
