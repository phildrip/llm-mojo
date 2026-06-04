from std.ffi import OwnedDLHandle

def main() raises:
    # Use libm.so.6 which is the actual compiled shared library on Ubuntu
    var lib = OwnedDLHandle("libm.so.6")
    var sqrt = lib.get_function[def(Float64) -> Float64]("sqrt")
    print("Square root of 4.0 is:", sqrt(4.0))
