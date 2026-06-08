from dataloader import DataLoader
from std.memory import alloc

def main() raises:
    print("Initializing DataLoader...")
    var loader = DataLoader("sandbox/tinystories_train.bin")
    
    comptime batch_size = 2
    comptime seq_len = 8
    
    var x = alloc[Int32](batch_size * seq_len)
    var y = alloc[Int32](batch_size * seq_len)
    
    print("Drawing random batch...")
    loader.get_batch(batch_size, seq_len, x, y)
    
    print("Batch X:")
    for b in range(batch_size):
        var s = String("Batch ") + String(b) + ": "
        for t in range(seq_len):
            s += String(x[b * seq_len + t]) + " "
        print(s)
        
    print("Batch Y (shifted targets):")
    for b in range(batch_size):
        var s = String("Batch ") + String(b) + ": "
        for t in range(seq_len):
            s += String(y[b * seq_len + t]) + " "
        print(s)
        
    x.free()
    y.free()
    print("Test passed successfully!")
