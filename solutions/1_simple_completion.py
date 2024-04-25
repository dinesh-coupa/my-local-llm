from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
)

prompt = "Hi! Good Morning! "

print(llm(prompt))


# prompt_2 = "Hi! The capital city of Tamil Nadu is called as "
