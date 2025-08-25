import json
from typing_extensions import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from data import Dataset


# class GetPossibleIntentsInput(BaseModel):
#     reasoning: str = Field(..., description="Reasoning for the function call.")
#     function_type: Literal["get_possible_intents"]
@tool
def get_possible_intents_tool(
    reasoning: str,
    dataset: Annotated[Dataset, InjectedState("dataset")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Get a list of possible intents from the dataset.
    Args:
        reasoning (str): Reasoning for the function call.
        dataset (Dataset): The dataset to operate on.
        tool_call_id (str): The tool call ID for tracking.
    Returns:
        Command: A command containing the possible intents.
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


# class GetPossibleCategoriesInput(BaseModel):
#     reasoning: str = Field(..., description="Reasoning for the function call.")
#     function_type: Literal["get_possible_categories"]
# def get_possible_categories_tool(
#     reasoning: str,
#     dataset: Annotated[Dataset, InjectedState("dataset")],
#     tool_call_id: Annotated[str, InjectedToolCallId],
# ) -> List[str]:
#     """
#     Get a list of possible categories from the dataset.
#     Args:
#         reasoning (str): Reasoning for the function call.
#         dataset (Dataset): The dataset to operate on.
#         tool_call_id (str): The tool call ID for tracking.
#     Returns:
#         List[str]: A list of possible categories.
#     """
#     return dataset.get_possible_categories()
@tool
def get_possible_categories_tool(
    reasoning: str,
    dataset: Annotated[Dataset, InjectedState("dataset")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Get a list of possible categories from the dataset.
    Args:
        reasoning (str): Reasoning for the function call.
        dataset (Dataset): The dataset to operate on.
        tool_call_id (str): The tool call ID for tracking.
    Returns:
        Command: A command containing the possible categories.
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


# class SelectSemanticIntentInput(BaseModel):
#     reasoning: str = Field(..., description="Reasoning for the function call.")
#     function_type: Literal["select_semantic_intent"]
#     intent_names: List[str] = Field(
#         ..., description="List of intent names to filter by."
#     )
@tool
def select_semantic_intent_tool(
    reasoning: str,
    intent_names: List[str],
    dataset: Annotated[Dataset, InjectedState("dataset")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Select rows from the dataset where the 'intent' column matches any of the provided intent names.
    Args:
        reasoning (str): Reasoning for the function call.
        intent_names (List[str]): List of intent names to filter by.
        dataset (Dataset): The dataset to operate on.
        tool_call_id (str): The tool call ID for tracking.
    Returns:
        Command: A command containing the updated dataset and selected intents.
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


# class SelectSemanticCategoryInput(BaseModel):
#     reasoning: str = Field(..., description="Reasoning for the function call.")
#     function_type: Literal["select_semantic_category"]
#     category_names: List[str] = Field(
#         ..., description="List of category names to filter by."
#     )
@tool
def select_semantic_category_tool(
    reasoning: str,
    category_names: List[str],
    dataset: Annotated[Dataset, InjectedState("dataset")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Select rows from the dataset where the 'category' column matches any of the provided category names.
    Args:
        reasoning (str): Reasoning for the function call.
        category_names (List[str]): List of category names to filter by.
        dataset (Dataset): The dataset to operate on.
        tool_call_id (str): The tool call ID for tracking.
    Returns:
        Command: A command containing the updated dataset and selected categories.
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


# class FinishInput(BaseModel):
#     reasoning: str = Field(..., description="Reasoning for the function call.")
#     function_type: Literal["finish"]
#     final_answer: str = Field(..., description="Final answer to return.")
# def finish(final_answer: str) -> str:
#     """
#     Finish the conversation with a final answer.
#     Args:
#         final_answer (str): The final answer to return.
#     Returns:
#         str: The final answer.
#     """
#     return final_answer
@tool
def finish_tool(
    reasoning: str,
    final_response: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Finish the conversation with a final answer.
    Args:
        reasoning (str): Reasoning for the function call.
        final_response (str): The final answer to return.
        tool_call_id (str): The tool call ID for tracking.
    Returns:
        Command: A command indicating completion and containing the final response.
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


# FunctionType = Union[
#     GetPossibleIntentsInput,
#     GetPossibleCategoriesInput,
#     SelectSemanticIntentInput,
#     SelectSemanticCategoryInput,
#     SumInput,
#     SortDictByValuesInput,
#     CountCategoryInput,
#     CountIntentInput,
#     ShowExamplesInput,
#     SummarizeInput,
#     FinishInput,
#     CountRowsInput,
# ]


# class FunctionInput(BaseModel):
#     function_call: FunctionType = Field(discriminator="function_type")


# def execute_function(function_call: FunctionType, ds: Dataset):
#     """
#     Execute the function call on the dataset.
#     Args:
#         function_call (FunctionType): The function call to execute.
#         ds (Dataset): The dataset to operate on.
#     Returns:
#         dict: A dictionary containing the dataset and the response from the function call.
#     """
#     output = {}
#     output["dataset"] = ds

#     if isinstance(function_call, GetPossibleIntentsInput):
#         output["response"] = {"possible_intents": ds.get_possible_intents()}
#         return output
#     elif isinstance(function_call, GetPossibleCategoriesInput):
#         output["response"] = {"possible_categories": ds.get_possible_categories()}
#         return output
#     elif isinstance(function_call, SelectSemanticIntentInput):
#         ds = ds.select_semantic_intent(function_call.intent_names)
#         output["dataset"] = ds
#         output["response"] = {
#             "selected_intents": ds.get_possible_intents(),
#             "number_of_rows": ds.count_rows(),
#         }
#         return output
#     elif isinstance(function_call, SelectSemanticCategoryInput):
#         ds = ds.select_semantic_category(function_call.category_names)
#         output["dataset"] = ds
#         output["response"] = {
#             "selected_categories": ds.get_possible_categories(),
#             "number_of_rows": ds.count_rows(),
#         }
#         return output
#     elif isinstance(function_call, CountRowsInput):
#         output["response"] = {"number_of_rows": ds.count_rows()}
#         return output
#     elif isinstance(function_call, SumInput):
#         result = sum(function_call.a, function_call.b)
#         output["response"] = {"sum": result}
#         return output
#     elif isinstance(function_call, SortDictByValuesInput):
#         sorted_dict = sort_dict_by_values(
#             json.loads(function_call.d), ascending=function_call.ascending
#         )
#         output["response"] = {"sorted_dict": sorted_dict}
#         return output
#     elif isinstance(function_call, CountCategoryInput):
#         count = ds.count_category(function_call.category)
#         output["response"] = {"count": count}
#         return output
#     elif isinstance(function_call, CountIntentInput):
#         count = ds.count_intent(function_call.intent)
#         output["response"] = {"count": count}
#         return output
#     elif isinstance(function_call, ShowExamplesInput):
#         examples = ds.show_examples(function_call.n)
#         output["response"] = {"examples": examples.to_dict(orient="records")}
#         return output
#     elif isinstance(function_call, SummarizeInput):
#         summary = summarize(function_call.user_request, ds)
#         output["response"] = {"summary": summary}
#         return output
#     elif isinstance(function_call, FinishInput):
#         final_answer = finish(function_call.final_answer)
#         output["response"] = {"final_answer": final_answer}
#         return output
#     else:
#         raise ValueError(f"Unknown function type: {function_call.function_type}")


# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "get_possible_intents",
#             "description": "Get a list of possible intents from the dataset",
#             "parameters": GetPossibleIntentsInput.model_json_schema(),
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "get_possible_categories",
#             "description": "Get a list of possible categories from the dataset",
#             "parameters": GetPossibleCategoriesInput.model_json_schema(),
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "count_intent",
#             "description": "Count the number of rows in the dataset that match a specific intent",
#             "parameters": CountIntentInput.model_json_schema(),
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "count_category",
#             "description": "Count the number of rows in the dataset that match a specific category",
#             "parameters": CountCategoryInput.model_json_schema(),
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "count_rows",
#             "description": "Count the number of rows in the dataset",
#             "parameters": CountRowsInput.model_json_schema(),
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "show_examples",
#             "description": "Show a number of examples from the dataset",
#             "parameters": ShowExamplesInput.model_json_schema(),
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "summarize",
#             "description": (
#                 "Summarize a user request using the dataset. "
#                 "The user request is provided as input."
#             ),
#             "parameters": SummarizeInput.model_json_schema(),
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "select_semantic_intent",
#             "description": "Filter rows from the dataset where the 'intent' column matches any of the provided intent names",
#             "parameters": SelectSemanticIntentInput.model_json_schema(),
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "select_semantic_category",
#             "description": "Filter rows from the dataset where the 'category' column matches any of the provided category names",
#             "parameters": SelectSemanticCategoryInput.model_json_schema(),
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "sum",
#             "description": "Sum two numbers",
#             "parameters": SumInput.model_json_schema(),
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "sort_dict_by_values",
#             "description": (
#                 "Sort a dictionary by its values. "
#                 "The dictionary is provided as input."
#             ),
#             "parameters": SortDictByValuesInput.model_json_schema(),
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "finish",
#             "description": "Finish the conversation with a final answer. Do not include follow-up prompts, only provide the final answer to the user's original question.",
#             "parameters": FinishInput.model_json_schema(),
#         },
#     },
# ]
