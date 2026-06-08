from dataloader import DataLoader
from mini_transformer import MiniTransformer, Activations
from std.memory import alloc
from std.python import Python, PythonObject
import std.time

def decode_tokens(py_tokenizer: PythonObject, token_ids: List[Int32]) raises -> String:
    var py_list = Python.list()
    for i in range(len(token_ids)):
        py_list.append(Int(token_ids[i]))
    var decoded = py_tokenizer.decode(py_list)
    return String(decoded)

def generate_sample[
    embed_dim: Int,
    intermediate_dim: Int,
    num_heads: Int,
    num_layers: Int,
    vocab_size: Int,
    max_seq_len: Int,
](
    model: MiniTransformer[embed_dim, intermediate_dim, num_heads, num_layers, vocab_size, max_seq_len],
    py_tokenizer: PythonObject,
    start_text: String,
    num_tokens: Int,
) raises:
    # 1. Encode prompt
    var encoded = py_tokenizer.encode(start_text)
    var prompt_len = len(encoded)
    
    var token_ids = List[Int32]()
    for i in range(prompt_len):
        token_ids.append(Int32(encoded[i]))
        
    print("Prompt:", start_text)
    
    # 2. Setup Activations for autoregressive generation
    var act = Activations(
        batch_size=1,
        seq_len=64, # max length
        embed_dim=embed_dim,
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        vocab_size=vocab_size,
    )
    
    # 3. Autoregressive loop
    for step in range(num_tokens):
        var current_len = len(token_ids)
        if current_len >= 64:
            break
            
        # Copy current sequence into activations
        for t in range(current_len):
            act.x_tokens[t] = token_ids[t]
            
        # Run forward pass for the active length
        # Note: self.forward expects seq_len elements, but we can temporarily mock S
        # For simplicity, we pad the unused tokens with 0, and run the model on S=64
        # and look at the logit at position current_len - 1
        for t in range(current_len, 64):
            act.x_tokens[t] = 0
            
        model.forward(act)
        
        # Greedy sampling of the next token
        var offset = (current_len - 1) * vocab_size
        var max_logit = act.logits[offset]
        var next_token = Int32(0)
        for k in range(1, vocab_size):
            if act.logits[offset + k] > max_logit:
                max_logit = act.logits[offset + k]
                next_token = Int32(k)
                
        token_ids.append(next_token)
        if next_token == 50256: # End of text token
            break
            
    # Decode and print
    var decoded = decode_tokens(py_tokenizer, token_ids)
    print("Generated:", decoded)
    print("-" * 50)

def main() raises:
    print("=" * 60)
    print("         MOJO TINYSTORIES SANDBOX TRAINING PIPELINE")
    print("=" * 60)
    
    # Initialize Python and Tokenizer
    print("Initializing GPT-2 Tokenizer via Python FFI...")
    var transformers = Python.import_module("transformers")
    var py_tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    
    # Setup model parameters
    comptime embed_dim = 256
    comptime intermediate_dim = 512
    comptime num_heads = 4
    comptime num_layers = 4
    comptime vocab_size = 50257
    comptime max_seq_len = 64
    
    # Initialize DataLoader
    print("Loading TinyStories validation dataset...")
    var loader = DataLoader("sandbox/tinystories_train.bin")
    
    # Initialize Model
    print("Instantiating 15M Parameter Sandbox Base Model...")
    var model = MiniTransformer[
        embed_dim=embed_dim,
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
    ]()
    
    # Pre-allocate activations for training batch
    comptime batch_size = 8
    comptime seq_len = 64
    print("Allocating zero-overhead activations buffer...")
    var act = Activations(
        batch_size=batch_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        vocab_size=vocab_size,
    )
    
    # Hyper-parameters
    var lr: Float32 = 5e-4
    var weight_decay: Float32 = 0.01
    var num_steps = 150
    
    print("\nStarting training loop for", num_steps, "optimization steps...\n")
    
    var start_time = std.time.perf_counter()
    for step in range(1, num_steps + 1):
        var t0 = std.time.perf_counter()
        
        # 1. Draw random training slice
        loader.get_batch(batch_size, seq_len, act.x_tokens, act.y_tokens)
        
        # 2. Forward pass
        model.forward(act)
        
        # 3. Backward pass
        var loss = model.backward(act)
        
        # 4. AdamW optimizer step
        model.adamw_step(t=step, lr=lr, weight_decay=weight_decay, beta1=0.9, beta2=0.95, eps=1e-8)
        
        var t1 = std.time.perf_counter()
        var elapsed_ms = (t1 - t0) * 1000.0
        var tokens_per_sec = Float64(batch_size * seq_len) / (t1 - t0)
        
        if step == 1 or step % 10 == 0:
            print("Step", step, "| Loss:", loss, "| Latency:", elapsed_ms, "ms | Performance:", tokens_per_sec, "tok/sec")
            
        if step % 50 == 0:
            print("\n" + "*" * 50)
            print("Sampling story snippet from model at Step", step, ":")
            print("*" * 50)
            generate_sample[
                embed_dim=embed_dim,
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                vocab_size=vocab_size,
                max_seq_len=max_seq_len,
            ](model, py_tokenizer, "Once upon a time, there was a little", 20)
            
    var end_time = std.time.perf_counter()
    var total_elapsed_sec = end_time - start_time
    print("\nTraining completed successfully in", total_elapsed_sec, "seconds!")
    print("=" * 60)
