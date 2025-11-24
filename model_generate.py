from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model", # Point this to your saved folder
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": ""}], # Dummy to get start tokens if needed, or just raw BOS
    tokenize=True,
    add_generation_prompt=True
) if hasattr(tokenizer, "chat_template") and tokenizer.chat_template else [tokenizer.bos_token_id]

def generate():
    s = input("Initial string: ")
    n = input("Number of lines to generate: ")

    if s == "":
        inputs = torch.tensor([[tokenizer.eos_token_id]]).to("cuda")

    else:
        inputs = tokenizer([s], return_tensors = "pt").to("cuda").input_ids

    for i in range(int(n or 1)):
        # 🎲 Generation Settings (The Chaos Dial)
        outputs = model.generate(
            inputs,
            max_new_tokens = 128,     # How much it yaps
            use_cache = True,
            temperature = 0.9,        # Higher = more chaotic/creative
            top_k = 64,               # Limits choices to top 64 tokens
            top_p = 0.95,             # Nucleus sampling
            do_sample = True,         # REQUIRED for variety. False = boring robot.
            repetition_penalty = 1.2, # Stop it from repeating "furry furry furry"
        )

        print("")
        print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])

    print("")
    if input("Generate more? ").strip().lower() == "y":
        generate()
    else:
        print("ok bye bye")

generate()
