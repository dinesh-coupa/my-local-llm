from ctransformers import AutoModelForCausalLM

llama_llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-Chat-GGUF", model_file="llama-2-7b-chat.Q3_K_M.gguf"
)


def get_prompt(
    instruction: str, history: list | None = None, llm_type: str | None = None
) -> str:
    system = "You are a AI assistant that gives helpful answers. You answer the questions in short and concise manner."
    if llm_type == "orca":
        if len(history) > 0:
            prompt = f"### System:\n{system}\n\n### Input:\n{''.join(history)}\n\n### User:\n{instruction}\n\n### Response:\n"
        else:
            prompt = (
                f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
            )
    else:
        if len(history) > 0:
            prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\nInput:\n{''.join(history)}\n\nUser:\n{instruction} [/INST]"
        else:
            prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]"
    return prompt


import chainlit as cl


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")

    if message.content == "use llama2":
        cl.user_session.set("llm_type", "llama2")
        msg = cl.Message("ok, using llama2")
        await msg.send()
    elif message.content == "use orca":
        cl.user_session.set("llm_type", "orca")
        msg = cl.Message("ok, using orca")
        await msg.send()
    elif message.content == "forget memory":
        cl.user_session.set("message_history", [])
        msg = cl.Message("ok, erased memory")
        await msg.send()
    else:

        llm_type = cl.user_session.get("llm_type")
        if llm_type == "orca":
            current_llm = ocra_llm
        else:
            current_llm = llama_llm

        prompt = get_prompt(message.content, message_history, llm_type)

        # debug prompt
        debug_msg = cl.Message(llm_type + "\n" + prompt)
        await debug_msg.send()

        response = ""
        msg = cl.Message(content="")
        await msg.send()
        for word in current_llm(prompt, stream=True):
            await msg.stream_token(word)
            response += word
        await msg.update()
        message_history.append(response)


@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("message_history", [])
    cl.user_session.set("llm_type", "orca")
    global ocra_llm
    global llama_llm
    ocra_llm = AutoModelForCausalLM.from_pretrained(
        "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
    )
    llama_llm = AutoModelForCausalLM.from_pretrained(
        "TheBloke/Llama-2-7B-Chat-GGUF", model_file="llama-2-7b-chat.Q3_K_M.gguf"
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
