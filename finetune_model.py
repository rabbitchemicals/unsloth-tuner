import torch
import torchvision
torch.hub.list('pytorch/vision', force_reload=True)

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 🐾 Configuration (Setting up the Den)
# Qwen3-0.6B is a smol bean, so we can give it a long leash (context length).
# Akkoma limit is ~3000 chars -> ~750-1000 tokens.
# 2048 is a comfy bed size to fit 2-3 posts per batch item with packing.
max_seq_length = 1024
dtype = None # None = auto sniff. Float16 for T4, Bfloat16 for the fancy Ampere+ kits.
load_in_4bit = True # 4bit quantization prevents your VRAM from choking on a hairball.

# 🐾 Waking the Beast (Load Model)
# Unsloth patches the model on the fly to make it zoomie-fast (2x speed).
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen3-0.6B", # Unsloth will fetch the pup and patch it
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 🐾 Grooming the Coat (LoRA Adapters)
# We target all linear layers to capture your specific posting scent.
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # The Rank. 16 is standard fluff. 32 if you're complex. 64 is too heavy for this tiny frame.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Unsloth says NO shedding allowed (0 dropout)
    bias = "none",    # Unsloth prefers no bias in the treats
    use_gradient_checkpointing = "unsloth", # True or "unsloth" if the context gets too long to chew
    random_state = 3407,
    use_rslora = False,  # Rank Stabilization. Optional shiny collar.
    loftq_config = None, # LoftQ. Optional accessories.
)

# 🐾 Kibble Prep (Data Loading)
# Assuming data is in a JSONL file: {"content": "Bark bark bark..."}
# Make sure 'akkoma_posts.jsonl' is actually in the bowl (folder).
dataset = load_dataset("json", data_files="data.jsonl", split="train")

# We MUST add the EOS token. Unsloth handles EOS usually,
# but for raw text packing, we need to be explicit or the model won't know when to stop howling.
EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    posts = examples["content"] # Grab the meat from the bone
    # Just append EOS. No "User:" or "Assistant:" muzzles.
    texts = [post + EOS_TOKEN for post in posts]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched = True)

# 🐾 Agility Training (The Trainer)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = True, # THIS IS KEY. Packs posts together like puppies in a pile for warmth (and efficiency).
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 2, # max_steps = 60 for testing
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit", # 8-bit optimizer saves VRAM for more treats
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 67777,
        output_dir = "outputs",
    ),
)

# 🐾 ZOOMIES! (Train)
print("🦊 Ears up! Training started... hold onto your tail!")
trainer_stats = trainer.train()

# 🐾 Awoo Test (Inference)
# Does it sound like your friend, or just a confused fox?
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(["I was just born and I think"], return_tensors = "pt").to("cuda")

# Generate
print("🐾 Sniffing out a response...")
outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
print(tokenizer.batch_decode(outputs)[0])

# 🐾 Buried Treasure (Save)
model.save_pretrained("lora_model") # Stashing the goods locally
print("💾 Model stashed in 'lora_model'. Good boy/girl/entity! :3")
# model.push_to_hub("your_name/lora_model", token = "...") # Howling to the cloud
