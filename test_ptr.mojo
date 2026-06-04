from std.memory import alloc

def main() raises:
    var test_file_path = "io_test.txt"
    var path_bytes = test_file_path.as_bytes()
    var path_ptr = alloc[UInt8](len(path_bytes) + 1)
    
    # Test __int__() method
    var addr_val = path_ptr.__int__()
    print("Pointer to Int conversion works! Address =", addr_val)

    path_ptr.free()
