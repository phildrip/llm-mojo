import os
import sys
import numpy as np

try:
    from transformers import AutoTokenizer
    from datasets import load_dataset
except ImportError:
    print("Preparing data requires 'transformers' and 'datasets' libraries.")
    print("Please install them: pip install transformers datasets")
    sys.exit(1)

def main():
    os.makedirs("sandbox", exist_ok=True)
    
    print("Loading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # GPT-2 does not have a pad token by default, use eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Loading TinyStories dataset (validation split for a fast sandbox subset)...")
    # Using the validation split yields a beautiful, high-quality subset of ~20,000 stories (~20MB of text)
    # which is the perfect size to train our 10M parameter sandbox model to convergence in under 10 minutes!
    dataset = load_dataset("roneneldan/TinyStories", split="validation")
    
    print(f"Loaded {len(dataset)} stories.")
    
    token_ids = []
    print("Tokenizing stories...")
    for idx, example in enumerate(dataset):
        text = example["text"].strip()
        # Add <|endoftext|> to separate stories
        text = text + tokenizer.eos_token
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_ids.extend(ids)
        
        if (idx + 1) % 2000 == 0:
            print(f"Tokenized {idx + 1}/{len(dataset)} stories...")
            
    print(f"Total tokens generated: {len(token_ids)}")
    
    # Save as 16-bit unsigned integers (GPT-2 vocab is 50,257, which fits perfectly in uint16: 0 - 65,535)
    token_array = np.array(token_ids, dtype=np.uint16)
    
    output_path = "sandbox/tinystories_train.bin"
    print(f"Writing tokens to {output_path}...")
    token_array.tofile(output_path)
    print("Done! TinyStories data preparation completed successfully!")

if __name__ == "__main__":
    main()
