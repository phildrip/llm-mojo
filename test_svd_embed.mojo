from svd_embed import SvdEmbedding
from std.memory import alloc
import std.os

def main() raises:
    print("=========================================================")
    print("            TESTING MOJO SVD EMBEDDING READER")
    print("=========================================================")

    var filename = "gemma4_embed_svd_d32.bin"
    if not std.os.path.exists(filename):
        print("Error: ", filename, " does not exist.")
        return

    # 1. Load SVD embeddings
    var svd = SvdEmbedding(filename)

    # 2. Allocate a thread-local reconstruction buffer
    var recon_buf = alloc[Float32](svd.embed_dim)
    var recon_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=recon_buf.__int__())

    # 3. Reconstruct token 0
    print("\nReconstructing token 0...")
    svd.reconstruct(0, recon_ptr)

    # 4. Print some reconstructed dimensions to confirm it is populated
    print("Reconstructed embedding sample for token 0 (first 16 values):")
    for i in range(16):
        print("  dim", i, ": ", recon_ptr[i])

    # 5. Free temporary reconstruction buffer
    recon_buf.free()
    print("=========================================================")
    print("Test finished successfully!")
    print("=========================================================")
