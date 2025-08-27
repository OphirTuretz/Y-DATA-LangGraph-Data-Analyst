import json
from typing import List
from typing_extensions import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from data import Dataset


@tool
def get_possible_intents_tool(
    reasoning: str,
    dataset: Annotated[Dataset, InjectedState("dataset")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Get a list of possible intents from the dataset.
    """
    return Command(
        update={
            "messages": [
                ToolMessage(
                    json.dumps({"possible_intents": dataset.get_possible_intents()}),
                    tool_call_id=tool_call_id,
                )
            ]
        }
    )


@tool
def get_possible_categories_tool(
    reasoning: str,
    dataset: Annotated[Dataset, InjectedState("dataset")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Get a list of possible categories from the dataset.
    """
    return Command(
        update={
            "messages": [
                ToolMessage(
                    json.dumps(
                        {"possible_categories": dataset.get_possible_categories()}
                    ),
                    tool_call_id=tool_call_id,
                )
            ]
        }
    )


@tool
def select_semantic_intent_tool(
    reasoning: str,
    intent_names: List[str],
    dataset: Annotated[Dataset, InjectedState("dataset")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Select rows from the dataset where the 'intent' column matches any of the provided intent names.
    """
    dataset = dataset.select_semantic_intent(intent_names)
    return Command(
        update={
            "dataset": dataset,
            "messages": [
                ToolMessage(
                    json.dumps(
                        {
                            "selected_intents": dataset.get_possible_intents(),
                            "number_of_rows": dataset.count_rows(),
                        }
                    ),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def select_semantic_category_tool(
    reasoning: str,
    category_names: List[str],
    dataset: Annotated[Dataset, InjectedState("dataset")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Select rows from the dataset where the 'category' column matches any of the provided category names.
    """
    dataset = dataset.select_semantic_category(category_names)
    return Command(
        update={
            "dataset": dataset,
            "messages": [
                ToolMessage(
                    json.dumps(
                        {
                            "selected_categories": dataset.get_possible_categories(),
                            "number_of_rows": dataset.count_rows(),
                        }
                    ),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def finish_tool(
    reasoning: str,
    final_response: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Finish the conversation with a final answer.
    """
    return Command(
        update={
            "is_complete": True,
            "final_response": final_response,
            "messages": [
                ToolMessage(
                    json.dumps({"final_response": final_response}),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )
