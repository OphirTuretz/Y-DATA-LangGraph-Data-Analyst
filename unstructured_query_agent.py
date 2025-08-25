import json
import math
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
    select_semantic_intent_tool,
    select_semantic_category_tool,
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
            SystemMessage(
                content="You are a helpful analyst that summarizes customer support interactions according to user instructions. Respond in structured JSON."
            ),
            HumanMessage(
                content=summarize_batch_prompt.format(
                    user_request=user_request, data=batch_df.to_dict(orient="records")
                )
            ),
        ]

        # Perform the initial request to the LLM
        response = llm.with_structured_output(SummaryResponse).invoke(messages)

        # Extract the assistant's message from the response
        batch_summaries.append(response.summary)

    # Combine all batch summaries into a final summary
    final_messages = [
        SystemMessage(
            "You are a helpful assistant that summarizes customer support data based on multiple batch summaries, using a structured JSON format."
        ),
        HumanMessage(
            summarize_all_batches_prompt.format(
                user_request=user_request,
                summaries=batch_summaries,
                num_batches=str(int(math.ceil(n_rows_to_sample / batch_size))),
                rows_per_batch=str(int(batch_size)),
                n_rows=str(int(n_rows_to_sample)),
            )
        ),
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


unstructured_query_agent_tool_list = [
    get_possible_intents_tool,
    get_possible_categories_tool,
    select_semantic_intent_tool,
    select_semantic_category_tool,
    finish_tool,
    summarize_tool,
]

# Nodes


def unstructured_query_agent_node(state: UserQueryState) -> UserQueryState:
    state["messages"] = ["This is a response from the unstructured query agent."]
    return state


unstructured_query_agent_tool_node = ToolNode(unstructured_query_agent_tool_list)
