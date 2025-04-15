from tokenizers import ByteLevelBPETokenizer
import os

# Create tokenizer directory
os.makedirs("tokenizer", exist_ok=True)

# Train a tokenizer from scratch
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files="data/the-verdict.txt",
    vocab_size=50257,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)

# Save tokenizer
tokenizer.save_model("tokenizer")
print("Tokenizer saved to tokenizer/")