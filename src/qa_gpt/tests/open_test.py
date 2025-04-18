from pydantic import BaseModel

from src.qa_gpt.chat.chat import get_chat_gpt_response_structure


class Step(BaseModel):
    action: str
    explanation: str
    money: int
    human_power: int


if __name__ == "__main__":
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "Can you tell me a joke?"},
    #     {"role": "assistant", "content": "Sure, here's one for you:\n\nWhy don't scientists trust atoms?\n\nBecause they make up everything!"},
    #     {"role": "user", "content": "What's the key point of this joke?"}
    # ]
    # print(get_chat_gpt_response(messages))

    messages = [
        {
            "role": "system",
            "content": "You are a helpful math tutor. Guide the user through the solution step by step.",
        },
        {"role": "user", "content": "how can I solve 8x + 7 = -23"},
    ]

    print(get_chat_gpt_response_structure(messages, res_obj=Step))
