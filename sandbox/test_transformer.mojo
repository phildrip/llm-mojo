from dataloader import DataLoader
from mini_transformer import MiniTransformer, Activations
from std.memory import alloc

def main() raises:
    print("Initializing DataLoader...")
    var loader = DataLoader("sandbox/tinystories_train.bin")
    
    comptime batch_size = 2
    comptime seq_len = 8
    
    print("Initializing MiniTransformer...")
    # Quick, lightweight 2-layer test transformer
    var model = MiniTransformer[
        embed_dim=128,
        intermediate_dim=256,
        num_heads=4,
        num_layers=2,
        vocab_size=50257,
        max_seq_len=64,
        reuse_attn=True,
    ]()
    
    print("Allocating Activations...")
    var act = Activations(
        batch_size=batch_size,
        seq_len=seq_len,
        embed_dim=128,
        intermediate_dim=256,
        num_heads=4,
        num_layers=2,
        vocab_size=50257,
    )
    
    print("Drawing random batch...")
    loader.get_batch(batch_size, seq_len, act.x_tokens, act.y_tokens)
    
    print("Running Forward Pass...")
    model.forward(act)
    
    print("Running Backward Pass...")
    var loss = model.backward(act)
    print("Initial Cross-Entropy Loss:", loss)
    
    print("Running AdamW Step...")
    model.adamw_step(t=1, lr=1e-4, weight_decay=0.01, beta1=0.9, beta2=0.99, eps=1e-8)
    
    print("Verification completed successfully!")
