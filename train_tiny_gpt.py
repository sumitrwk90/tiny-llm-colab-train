print("Hello world")

# ✅ tiny_llm_colab_train/train_tiny_gpt2.py

from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, PreTrainedTokenizerFast
from datasets import load_dataset
import torch

# Load Tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer/tokenizer.json")
tokenizer.pad_token = tokenizer.eos_token

# Load and Tokenize Dataset
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

dataset = load_dataset("text", data_files={"train": "data/data.txt"})
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Define Tiny GPT-2 Config
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=512,
    n_ctx=512,
    n_embd=256,
    n_layer=6,
    n_head=4,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

# Initialize Model
model = GPT2LMHeadModel(config)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    report_to="none"
)

# Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# Train
trainer.train()

# Save Final Model
model.save_pretrained("trained_model")
tokenizer.save_pretrained("trained_model")
