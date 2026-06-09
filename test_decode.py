import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained("google/gemma-4-12B-it")
print("Token 236751 is:", repr(tokenizer.decode([236751])))
print("Token 0 is:", repr(tokenizer.decode([0])))
print("Token 1 is:", repr(tokenizer.decode([1])))
print("Token 2 is:", repr(tokenizer.decode([2])))
print("Token 105 is:", repr(tokenizer.decode([105])))
print("Token 2364 is:", repr(tokenizer.decode([2364])))
print("Token 107 is:", repr(tokenizer.decode([107])))
