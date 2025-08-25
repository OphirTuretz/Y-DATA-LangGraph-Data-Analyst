import json
from typing_extensions import Annotated
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.types import Command
from pydantic import BaseModel, Field

from data import Dataset
from graph_state import UserQueryState
from prompt import read_prompt_file
from llm import llm
from app.const import (
    SUMMARIZE_DEFAULT_BATCH_SIZE,
    SUMMARIZE_DEFAULT_N_BATCHES,
    SUMMARIZE_BATCH_PROMPT_FILE_PATH,
    SUMMARIZE_ALL_BATCHES_PROMPT_FILE_PATH,
)
from general_tools import (
    get_possible_intents_tool,
    get_possible_categories_tool,
    finish_tool,
)


# Tools


class SummaryResponse(BaseModel):
    reasoning: str = Field(
        ...,
        description="Explain how you arrived at the summary based on user request",
    )
    summary: str = Field(
        ...,
        description="Write a short, precise summary answering the user request",
    )


# class SummarizeInput(BaseModel):
#     reasoning: str = Field(..., description="Reasoning for the function call.")
#     function_type: Literal["summarize"]
#     user_request: str = Field(..., description="User request to summarize.")
@tool
def summarize_tool(
    reasoning: str,
    user_request: str,
    dataset: Annotated[Dataset, InjectedState("dataset")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Summarize a user request using the dataset.
    Args:
        reasoning (str): Reasoning for the function call.
        user_request (str): The user request to summarize.
        dataset (Dataset): The dataset to use for summarization.
        tool_call_id (str): The tool call ID for tracking.
    Returns:
        Command: A command containing the summary of the user request.
    """
    n_batches = SUMMARIZE_DEFAULT_N_BATCHES
    batch_size = SUMMARIZE_DEFAULT_BATCH_SIZE

    # Sample rows from the dataset to use for summarization
    n_rows_to_sample = min(dataset.count_rows(), n_batches * batch_size)
    sampled_df = dataset.show_examples(n_rows_to_sample)

    # Read the prompt files for summarization
    summarize_batch_prompt = read_prompt_file(SUMMARIZE_BATCH_PROMPT_FILE_PATH)
    summarize_all_batches_prompt = read_prompt_file(
        SUMMARIZE_ALL_BATCHES_PROMPT_FILE_PATH
    )

    # Initialize a list to hold batch summaries
    batch_summaries = []

    # Iterate over the sampled DataFrame in batches
    for i in range(0, n_rows_to_sample, batch_size):
        batch_df = sampled_df.iloc[i : i + batch_size]

        # Each batch prompt is in independent conversation in order to avoid biasing the LLM
        messages = [
            {
                "role": "system",
                "content": "You are a helpful analyst that summarizes customer support interactions according to user instructions. Respond in structured JSON.",
            },
            {
                "role": "user",
                "content": summarize_batch_prompt.format(
                    user_request=user_request,
                    data=batch_df.to_dict(orient="records"),
                ),
            },
        ]

        # Perform the initial request to the LLM
        response = llm.with_structured_output(SummaryResponse).invoke(messages)

        # Extract the assistant's message from the response
        batch_summaries.append(response.summary)

    # Combine all batch summaries into a final summary
    final_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that summarizes customer support data based on multiple batch summaries, using a structured JSON format.",
        },
        {
            "role": "user",
            "content": summarize_all_batches_prompt.format(
                user_request=user_request,
                summaries=batch_summaries,
                num_batches=str(int(math.ceil(n_rows_to_sample / batch_size))),
                rows_per_batch=str(int(batch_size)),
                n_rows=str(int(n_rows_to_sample)),
            ),
        },
    ]

    # Perform the final request to the LLM
    final_response = llm.with_structured_output(SummaryResponse).invoke(final_messages)
    final_answer = final_response.summary
    return Command(
        update={
            "messages": [
                ToolMessage(
                    json.dumps({"summary": final_answer}),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


# # TO EDIT


# def summarize(
#     user_request: str,
#     ds: Dataset,
#     n_batches: int = SUMMARIZE_DEFAULT_N_BATCHES,
#     batch_size: int = SUMMARIZE_DEFAULT_BATCH_SIZE,
#     log_function: Callable[[str], None] = print,
# ) -> str:
#     """
#     Summarize a user request using the dataset.
#     Args:
#         user_request (str): The user request to summarize.
#         ds (Dataset): The dataset to use for summarization.
#         n_batches (int): Number of batches to process.
#         batch_size (int): Size of each batch.
#         log_function (Callable[[str], None]): Function to log messages.
#     Returns:
#         str: A summary of the user request.
#     """

#     # Sample rows from the dataset to use for summarization
#     n_rows_to_sample = min(ds.count_rows(), n_batches * batch_size)
#     sampled_df = ds.show_examples(n_rows_to_sample)

#     # Read the prompt files for summarization
#     summarize_batch_prompt = read_prompt_file(SUMMARIZE_BATCH_PROMPT_FILE_PATH)
#     summarize_all_batches_prompt = read_prompt_file(
#         SUMMARIZE_ALL_BATCHES_PROMPT_FILE_PATH
#     )

#     # Initialize a list to hold batch summaries
#     batch_summaries = []

#     # Iterate over the sampled DataFrame in batches
#     for i in range(0, n_rows_to_sample, batch_size):
#         batch_df = sampled_df.iloc[i : i + batch_size]

#         # Each batch prompt is in independent conversation in order to avoid biasing the LLM
#         messages = [
#             {
#                 "role": "system",
#                 "content": "You are a helpful analyst that summarizes customer support interactions according to user instructions. Respond in structured JSON.",
#             },
#             {
#                 "role": "user",
#                 "content": summarize_batch_prompt.format(
#                     user_request=user_request,
#                     data=batch_df.to_dict(orient="records"),
#                 ),
#             },
#         ]

#         log_function(
#             f"Processing batch {i // batch_size + 1} of {math.ceil(n_rows_to_sample / batch_size)} with {len(batch_df)} rows."
#         )
#         log_function(f"Batch messages: {json.dumps(messages, indent=2)}")

#         # Perform the initial request to the LLM
#         response = LLM.perform_structured_outputs_request(
#             messages,
#             response_format=SummaryResponse,
#         )

#         log_function(f"Batch response: {json.dumps(response.model_dump(), indent=2)}")

#         # Extract the assistant's message from the response
#         batch_summaries.append(response.choices[0].message.parsed.summary)

#     # Combine all batch summaries into a final summary
#     final_messages = [
#         {
#             "role": "system",
#             "content": "You are a helpful assistant that summarizes customer support data based on multiple batch summaries, using a structured JSON format.",
#         },
#         {
#             "role": "user",
#             "content": summarize_all_batches_prompt.format(
#                 user_request=user_request,
#                 summaries=batch_summaries,
#                 num_batches=str(int(math.ceil(n_rows_to_sample / batch_size))),
#                 rows_per_batch=str(int(batch_size)),
#                 n_rows=str(int(n_rows_to_sample)),
#             ),
#         },
#     ]

#     log_function(f"Final messages: {json.dumps(final_messages, indent=2)}")

#     # Perform the final request to the LLM
#     final_response = LLM.perform_structured_outputs_request(
#         final_messages,
#         response_format=SummaryResponse,
#     )

#     log_function(f"Final response: {json.dumps(final_response.model_dump(), indent=2)}")

#     # Extract the final summary from the response
#     final_answer = final_response.choices[0].message.parsed.summary

#     return final_answer


unstructured_query_agent_tool_list = [
    get_possible_intents_tool,
    get_possible_categories_tool,
    summarize_tool,
    finish_tool,
]

# Nodes


def unstructured_query_agent_node(state: UserQueryState) -> UserQueryState:
    state["messages"] = ["This is a response from the unstructured query agent."]
    return state


unstructured_query_agent_tool_node = ToolNode(unstructured_query_agent_tool_list)
