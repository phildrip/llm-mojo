from std.sys.ffi import DLHandle

fn main():
    try:
        var handle = DLHandle("./liburing_pager.so")
        print("Success: Loaded liburing_pager.so via std.sys.ffi!")
    except e:
        print("Failed to load library:", e)
