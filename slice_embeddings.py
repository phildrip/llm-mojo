import os
import sys

def slice_embeddings():
    input_path = "diffusiongemma_embed_q4.bin"
    output_path = "diffusiongemma_embed_q4_32k.bin"
    
    V_src = 262144
    V_tgt = 32768
    D = 2816
    
    P_row = D // 2  # 1408 bytes
    S_row = (D // 32) * 4  # 352 bytes
    
    packed_src_size = V_src * P_row
    scales_src_size = V_src * S_row
    
    print(f"Reading from {input_path}...")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not exist!")
        sys.exit(1)
        
    with open(input_path, "rb") as f_in:
        # Read the first V_tgt rows of packed weights
        packed_tgt = f_in.read(V_tgt * P_row)
        
        # Seek to the beginning of scales in the source file
        f_in.seek(packed_src_size)
        # Read the first V_tgt rows of scales
        scales_tgt = f_in.read(V_tgt * S_row)
        
    print(f"Writing to {output_path}...")
    with open(output_path, "wb") as f_out:
        f_out.write(packed_tgt)
        f_out.write(scales_tgt)
        
    print(f"Successfully created sliced embedding file: {output_path}")
    print(f"Original size: {os.path.getsize(input_path)} bytes")
    print(f"Sliced size:   {os.path.getsize(output_path)} bytes")

if __name__ == "__main__":
    slice_embeddings()
