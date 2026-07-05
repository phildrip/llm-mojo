from std.memory import alloc

struct Float32BlockCompressor:
    # Compresses an array of Float32 into a compressed byte array
    # Format of compressed block of 8 values:
    # - Byte 0: compression_flags (Bit 0: is_compressed, Bits 1-7: unused)
    # - If is_compressed:
    #   - Byte 1: base_exponent (8 bits)
    #   - Bytes 2-5: eight 4-bit exponent deltas (4 bytes total)
    # - If not is_compressed:
    #   - Bytes 1-8: raw exponents (8 bytes total)
    # - Bytes 9-16: raw sign bits (8 bits total = 1 byte)
    # - Bytes 17-39: raw 23-bit mantissas (8 * 23 bits = 184 bits = 23 bytes)
    # Total compressed size per 8 floats:
    # - Compressed: 1 + 5 + 1 + 23 = 30 bytes (instead of 32 bytes! Saving 6.25%)
    # - Uncompressed: 1 + 8 + 1 + 23 = 33 bytes
    
    @staticmethod
    def compress(src: UnsafePointer[Float32, _], dest: UnsafePointer[UInt8, MutUntrackedOrigin], length: Int) -> Int:
        # Length must be a multiple of 8 for SIMD block execution
        var num_blocks = length // 8
        var write_ptr = dest
        
        for block_idx in range(num_blocks):
            # Load 8 Float32 values into a SIMD register
            var offset = block_idx * 8
            var block_ptr = src + offset
            
            # Load as UInt32 to do bit-manipulation
            var uint_ptr = block_ptr.bitcast[UInt32]()
            var val_simd = uint_ptr.load[8]()
            
            # Extract signs, exponents, and mantissas
            # Float32: S (1 bit), E (8 bits), M (23 bits)
            var signs_simd = (val_simd >> 31) & 1
            var exponents_simd = (val_simd >> 23) & 0xFF
            var mantissas_simd = val_simd & 0x7FFFFF
            
            # Calculate base exponent (first element)
            var base_exp = exponents_simd[0]
            
            # Calculate deltas relative to base_exponent
            # Cast exponent values to Int32 to support signed offsets
            var exp_int = exponents_simd.cast[DType.int32]()
            var base_int = Int32(base_exp)
            var deltas = exp_int - base_int
            
            # Check if all deltas fit in 4-bit signed integers [-8, 7]
            var can_compress = True
            for i in range(8):
                if deltas[i] < -8 or deltas[i] > 7:
                    can_compress = False
                    break
                    
            if can_compress:
                # Write compression flag: 1 (compressed)
                write_ptr[0] = 1
                write_ptr += 1
                
                # Write base exponent
                write_ptr[0] = UInt8(base_exp)
                write_ptr += 1
                
                # Pack 8 deltas into 4 bytes
                # Each delta is mapped from [-8, 7] to [0, 15] by adding 8
                for i in range(4):
                    var d1 = UInt8(deltas[i*2] + 8) & 0x0F
                    var d2 = UInt8(deltas[i*2+1] + 8) & 0x0F
                    write_ptr[i] = (d1 << 4) | d2
                write_ptr += 4
            else:
                # Write compression flag: 0 (uncompressed)
                write_ptr[0] = 0
                write_ptr += 1
                
                # Write raw 8 exponents
                for i in range(8):
                    write_ptr[i] = UInt8(exponents_simd[i])
                write_ptr += 8
                
            # Pack sign bits into 1 byte
            var signs_byte: UInt8 = 0
            for i in range(8):
                signs_byte |= UInt8(signs_simd[i]) << UInt8(7 - i)
            write_ptr[0] = signs_byte
            write_ptr += 1
            
            # Write mantissas (8 * 23 bits = 23 bytes)
            # We copy them as a raw block copy to be completely lossless and branchless
            # Each mantissa is 23 bits. We can copy the raw mantissas by packing them.
            # For simplicity and absolute correctness, we can just write them as a byte sequence
            # or copy the raw uint32 values of mantissas directly to keep it incredibly fast.
            # To preserve exact bit alignment, we write the 8 mantissas (uint32) as 23 packed bytes,
            # or simpler: we write them as 8 unaligned uint32s if we want speed,
            # but to be strictly space-efficient and lossless, we can pack them:
            # Let's pack them into 23 bytes!
            # We can write the raw uint32s directly as 32-bit values (8 * 4 = 32 bytes)
            # To keep our kernel screamingly fast and 100% stable, we write the raw 23-bit mantissas
            # into 23 packed bytes or we write them raw. Let's do packed 23 bytes:
            var m0 = mantissas_simd[0]
            var m1 = mantissas_simd[1]
            var m2 = mantissas_simd[2]
            var m3 = mantissas_simd[3]
            var m4 = mantissas_simd[4]
            var m5 = mantissas_simd[5]
            var m6 = mantissas_simd[6]
            var m7 = mantissas_simd[7]
            
            # Pack 8 * 23 bits = 184 bits = 23 bytes
            # We can do this with simple bitwise arithmetic
            # Byte 0-2: m0 (23 bits)
            write_ptr[0] = UInt8((m0 >> 15) & 0xFF)
            write_ptr[1] = UInt8((m0 >> 7) & 0xFF)
            write_ptr[2] = UInt8(((m0 & 0x7F) << 1) | ((m1 >> 22) & 0x01))
            # Byte 3-5: m1 (23 bits)
            write_ptr[3] = UInt8((m1 >> 14) & 0xFF)
            write_ptr[4] = UInt8((m1 >> 6) & 0xFF)
            write_ptr[5] = UInt8(((m1 & 0x3F) << 2) | ((m2 >> 21) & 0x03))
            # Byte 6-8: m2 (23 bits)
            write_ptr[6] = UInt8((m2 >> 13) & 0xFF)
            write_ptr[7] = UInt8((m2 >> 5) & 0xFF)
            write_ptr[8] = UInt8(((m2 & 0x1F) << 3) | ((m3 >> 20) & 0x07))
            # Byte 9-11: m3 (23 bits)
            write_ptr[9] = UInt8((m3 >> 12) & 0xFF)
            write_ptr[10] = UInt8((m3 >> 4) & 0xFF)
            write_ptr[11] = UInt8(((m3 & 0x0F) << 4) | ((m4 >> 19) & 0x0F))
            # Byte 12-14: m4 (23 bits)
            write_ptr[12] = UInt8((m4 >> 11) & 0xFF)
            write_ptr[13] = UInt8((m4 >> 3) & 0xFF)
            write_ptr[14] = UInt8(((m4 & 0x07) << 5) | ((m5 >> 18) & 0x1F))
            # Byte 15-17: m5 (23 bits)
            write_ptr[15] = UInt8((m5 >> 10) & 0xFF)
            write_ptr[16] = UInt8((m5 >> 2) & 0xFF)
            write_ptr[17] = UInt8(((m5 & 0x03) << 6) | ((m6 >> 17) & 0x3F))
            # Byte 18-20: m6 (23 bits)
            write_ptr[18] = UInt8((m6 >> 9) & 0xFF)
            write_ptr[19] = UInt8((m6 >> 1) & 0xFF)
            write_ptr[20] = UInt8(((m6 & 0x01) << 7) | ((m7 >> 16) & 0x7F))
            # Byte 21-22: m7 (23 bits)
            write_ptr[21] = UInt8((m7 >> 8) & 0xFF)
            write_ptr[22] = UInt8(m7 & 0xFF)
            
            write_ptr += 23

        return write_ptr.__int__() - dest.__int__()

    @staticmethod
    def decompress(src: UnsafePointer[UInt8, _], dest: UnsafePointer[Float32, MutUntrackedOrigin], length: Int):
        var num_blocks = length // 8
        var read_ptr = src
        
        for block_idx in range(num_blocks):
            # Read compression flag
            var is_compressed = read_ptr[0] == 1
            read_ptr += 1
            
            # Reconstruct exponents
            var exponents_simd = SIMD[DType.uint32, 8](0)
            
            if is_compressed:
                var base_exp = read_ptr[0]
                read_ptr += 1
                
                # Unpack 4 bytes of deltas
                for i in range(4):
                    var packed_byte = read_ptr[i]
                    var d1 = (packed_byte >> 4) & 0x0F
                    var d2 = packed_byte & 0x0F
                    
                    # Convert back to signed deltas relative to base_exponent by subtracting 8
                    exponents_simd[i*2] = UInt32(Int(base_exp) + Int(d1) - 8)
                    exponents_simd[i*2+1] = UInt32(Int(base_exp) + Int(d2) - 8)
                read_ptr += 4
            else:
                for i in range(8):
                    exponents_simd[i] = UInt32(read_ptr[i])
                read_ptr += 8
                
            var signs_byte = read_ptr[0]
            read_ptr += 1
            var signs_simd = SIMD[DType.uint32, 8](0)
            for i in range(8):
                signs_simd[i] = UInt32((Int(signs_byte) >> (7 - i)) & 1)
                
            # Unpack 23 bytes of mantissas back into 8 * 23 bits
            var mantissas_simd = SIMD[DType.uint32, 8](0)
            
            var b0 = UInt32(read_ptr[0])
            var b1 = UInt32(read_ptr[1])
            var b2 = UInt32(read_ptr[2])
            var b3 = UInt32(read_ptr[3])
            var b4 = UInt32(read_ptr[4])
            var b5 = UInt32(read_ptr[5])
            var b6 = UInt32(read_ptr[6])
            var b7 = UInt32(read_ptr[7])
            var b8 = UInt32(read_ptr[8])
            var b9 = UInt32(read_ptr[9])
            var b10 = UInt32(read_ptr[10])
            var b11 = UInt32(read_ptr[11])
            var b12 = UInt32(read_ptr[12])
            var b13 = UInt32(read_ptr[13])
            var b14 = UInt32(read_ptr[14])
            var b15 = UInt32(read_ptr[15])
            var b16 = UInt32(read_ptr[16])
            var b17 = UInt32(read_ptr[17])
            var b18 = UInt32(read_ptr[18])
            var b19 = UInt32(read_ptr[19])
            var b20 = UInt32(read_ptr[20])
            var b21 = UInt32(read_ptr[21])
            var b22 = UInt32(read_ptr[22])
            
            read_ptr += 23
            
            # Reconstruct mantissas from bytes
            mantissas_simd[0] = (b0 << 15) | (b1 << 7) | (b2 >> 1)
            mantissas_simd[1] = ((b2 & 0x01) << 22) | (b3 << 14) | (b4 << 6) | (b5 >> 2)
            mantissas_simd[2] = ((b5 & 0x03) << 21) | (b6 << 13) | (b7 << 5) | (b8 >> 3)
            mantissas_simd[3] = ((b8 & 0x07) << 20) | (b9 << 12) | (b10 << 4) | (b11 >> 4)
            mantissas_simd[4] = ((b11 & 0x0F) << 19) | (b12 << 11) | (b13 << 3) | (b14 >> 5)
            mantissas_simd[5] = ((b14 & 0x1F) << 18) | (b15 << 10) | (b16 << 2) | (b17 >> 6)
            mantissas_simd[6] = ((b17 & 0x3F) << 17) | (b18 << 9) | (b19 << 1) | (b20 >> 7)
            mantissas_simd[7] = ((b20 & 0x7F) << 16) | (b21 << 8) | b22
            
            # Combine S, E, M back into a single UInt32 SIMD vector
            var val_simd = (signs_simd << 31) | (exponents_simd << 23) | mantissas_simd
            
            # Store reconstructed values back to dest
            var dest_offset = block_idx * 8
            var block_dest_ptr = dest + dest_offset
            var dest_uint_ptr = block_dest_ptr.bitcast[UInt32]()
            dest_uint_ptr.store[8](val_simd)
