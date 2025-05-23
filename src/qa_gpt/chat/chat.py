import os

import openai
from pydantic import BaseModel

from src.qa_gpt.chat.private_keys import openapi_key
from src.qa_gpt.chat.rate_limit_decorator import handle_openai_errors

os.environ["OPENAI_API_KEY"] = openapi_key
sync_client = openai.OpenAI()
async_client = openai.AsyncOpenAI()


def check_api():
    if len(os.environ["OPENAI_API_KEY"]) == 0:
        raise ValueError("OPENAI_API_KEY is not set properly, got empty")


@handle_openai_errors()
async def get_chat_gpt_response_async(messages):
    check_api()
    MODEL = "gpt-4o-mini"
    chat_completion = await async_client.chat.completions.create(
        messages=messages,
        model=MODEL,
    )
    return chat_completion.choices[0].message


@handle_openai_errors()
async def get_chat_gpt_response_structure_async(messages: list, res_obj: BaseModel):
    check_api()
    MODEL = "gpt-4o-mini"

    response = await async_client.beta.chat.completions.parse(
        model=MODEL,
        messages=messages,
        response_format=res_obj,
    )

    return response.choices[0].message.parsed


@handle_openai_errors()
def get_chat_gpt_response(messages):
    check_api()
    MODEL = "gpt-4o-mini"
    chat_completion = sync_client.chat.completions.create(
        messages=messages,
        model=MODEL,
    )
    return chat_completion.choices[0].message


@handle_openai_errors()
def get_chat_gpt_response_structure(messages: list, res_obj: BaseModel):
    check_api()
    MODEL = "gpt-4o-mini"

    response = sync_client.beta.chat.completions.parse(
        model=MODEL,
        messages=messages,
        response_format=res_obj,
    )

    return response.choices[0].message.parsed
