from unsloth import FastLanguageModel

# Load the goods
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model", # Your saved folder
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

print("""
Model has been loaded.
You may end the conversation by sending an empty string.
Happy chatting!
""")

messages = [
    {"role": "user", "content": input("User: ")} 
]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

outputs = model.generate(input_ids, max_new_tokens=128, use_cache=True, temperature=0.9)
print(f"Bongwater: {tokenizer.decode(outputs[0], skip_special_tokens=True).splitlines()[-1]}")

a = True
s = ""

while a:
    s = input("User: ")
    if s == "":
        a = False
        break

    message = [
        {"role": "user", "content": s}
    ]

    input_ids = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        use_cache=True,
        temperature = 0.9,        # Higher = more chaotic/creative
        top_k = 50,               # Limits choices to top 50 tokens
        top_p = 0.95,             # Nucleus sampling
        do_sample = True,         # REQUIRED for variety. False = boring robot.
        repetition_penalty = 1.2, # Stop it from repeating "furry furry furry"
        pad_token_id = tokenizer.eos_token_id
    )
    print(f"Bongwater: {tokenizer.decode(outputs[0], skip_special_tokens=True).splitlines()[-1]}")
