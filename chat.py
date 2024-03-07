from ctransformers import AutoModelForCausalLM

llama_llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-Chat-GGUF", model_file="llama-2-7b-chat.Q3_K_M.gguf"
)


def get_prompt(instruction: str, history: list | None = None) -> str:
    system = "You are a AI assistant that gives helpful answers. You answer the questions in short and concise manner."
    if len(history) > 0:
        prompt = f"### System:\n{system}\n\n### Input:\n{''.join(history)}\n\n### User:\n{instruction}\n\n### Response:\n"
    else:
        prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
    print(prompt)
    return prompt


def get_llama_prompt(instruction: str, history: list | None = None) -> str:
    system = "You are a AI assistant that gives helpful answers. You answer the questions in short and concise manner."
    if history is not None:
        instruction = (
            "This is the context: "
            + "".join(history)
            + ".. Now answer the Question: "
            + instruction
        )
    prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]"
    return prompt


import chainlit as cl


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    msg = cl.Message(content="")
    await msg.send()

    prompt = get_prompt(message.content, message_history)
    response = ""
    for word in llm(prompt, stream=True):
        await msg.stream_token(word)
        response += word
    await msg.update()
    message_history.append(response)


@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("message_history", [])
    global llm
    llm = AutoModelForCausalLM.from_pretrained(
        "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
    )


# question = "Which city is the capital of India? "
# answer = ""

# for word in llm(get_prompt(question), stream=True):
#     print(word, end="", flush=True)
#     answer += word
# print()

# history = []
# history.append(answer)
# question = "And of the United States?"

# for word in llm(get_prompt(question, history), stream=True):
#     print(word, end="", flush=True)
# print()
