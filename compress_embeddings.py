import os
import sys
import argparse
import struct
import numpy as np
from scipy.sparse.linalg import svds

def main():
    parser = argparse.ArgumentParser(description="Low-Rank SVD Embedding Factorization and Compression for Gemma-4.")
    parser.add_argument("--input", type=str, default="gemma4_embed.bin", help="Path to input raw Float32 embedding binary")
    parser.add_argument("--output", type=str, default="gemma4_embed_svd.bin", help="Output compressed factorized binary")
    parser.add_argument("--rank", type=int, default=32, help="The rank of the SVD approximation (default: 32)")
    parser.add_argument("--vocab_size", type=int, default=262144, help="Vocabulary size (default: 262144)")
    parser.add_argument("--embed_dim", type=int, default=3840, help="Embedding dimension (default: 3840)")
    args = parser.parse_args()

    print("=" * 60)
    print("      GEMMA-4 LOW-RANK SVD EMBEDDING COMPRESSION")
    print("=" * 60)

    # 1. Load the raw embedding matrix
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    print(f"Loading raw embeddings from {args.input}...")
    file_size = os.path.getsize(args.input)
    expected_size = args.vocab_size * args.embed_dim * 4
    if file_size != expected_size:
        print(f"Warning: File size ({file_size} bytes) does not match expected size ({expected_size} bytes).")
        print("Adjusting parameters based on actual file size...")
        # Auto-detect embed_dim if possible
        args.embed_dim = file_size // (args.vocab_size * 4)
        print(f"Auto-detected dimensions: Vocab={args.vocab_size}, Embed Dim={args.embed_dim}")

    # Load file as float32
    W = np.fromfile(args.input, dtype=np.float32).reshape(args.vocab_size, args.embed_dim)
    print(f"Loaded matrix of shape: {W.shape} (Total size: {file_size / 1024 / 1024:.2f} MB)")

    # 2. Perform SciPy Truncated SVD
    print(f"\nComputing truncated SVD for Rank d = {args.rank}...")
    import time
    t0 = time.time()
    
    # svds computes the k largest singular values/vectors
    u, s, vt = svds(W, k=args.rank)
    
    # Sort in descending order
    idx = np.argsort(s)[::-1]
    s = s[idx]
    u = u[:, idx]
    vt = vt[idx, :]
    
    t1 = time.time()
    print(f"SVD computation completed in {t1 - t0:.2f} seconds.")

    # 3. Factorize into symmetric A and B matrices
    # A = u * sqrt(s)
    # B = sqrt(s) * vt
    print("\nScaling factor matrices...")
    sqrt_s = np.sqrt(s)
    A = u * sqrt_s
    B = sqrt_s[:, np.newaxis] * vt

    print(f"  Factor A (Left Singular Matrix):  {A.shape} (floats: {A.size:,})")
    print(f"  Factor B (Right Singular Matrix): {B.shape} (floats: {B.size:,})")

    # 4. Evaluate Fidelity & Reconstruction Error
    print("\nEvaluating approximation fidelity...")
    # Reconstruct a sample to avoid memory pressure of full matrix multiplication if RAM is tight
    # We will reconstruct the first 10,000 rows to calculate statistics
    sample_rows = min(10000, args.vocab_size)
    W_recon_sample = np.dot(A[:sample_rows], B)
    W_orig_sample = W[:sample_rows]

    mse = np.mean((W_orig_sample - W_recon_sample) ** 2)
    rmse = np.sqrt(mse)
    max_err = np.max(np.abs(W_orig_sample - W_recon_sample))
    
    # Calculate cosine similarity of reconstructed vectors
    dot_prod = np.sum(W_orig_sample * W_recon_sample, axis=1)
    norm_orig = np.linalg.norm(W_orig_sample, axis=1)
    norm_recon = np.linalg.norm(W_recon_sample, axis=1)
    cosine_sim = np.mean(dot_prod / (norm_orig * norm_recon + 1e-9))

    # Calculate original variance to find R-squared (explained variance)
    total_variance = np.var(W_orig_sample)
    explained_variance = 1.0 - (mse / total_variance)

    print(f"  Reconstruction Statistics (over {sample_rows:,} sample rows):")
    print(f"    Mean Squared Error (MSE):     {mse:.6e}")
    print(f"    Root Mean Squared Error (RMSE): {rmse:.6e}")
    print(f"    Maximum Absolute Error:       {max_err:.6e}")
    print(f"    Mean Cosine Similarity:       {cosine_sim * 100:.4f}%")
    print(f"    Explained Variance (R²):       {explained_variance * 100:.4f}%")

    # 5. Save the factorized matrices
    # File structure:
    # Header (12 bytes): [Rank (4 bytes Int32)][Vocab (4 bytes Int32)][EmbedDim (4 bytes Int32)]
    # Followed by flat raw binary of Factor A (Float32) and Factor B (Float32)
    output_path = args.output
    if output_path == "gemma4_embed_svd.bin":
        output_path = f"gemma4_embed_svd_d{args.rank}.bin"

    print(f"\nWriting factorized matrices to {output_path}...")
    with open(output_path, "wb") as f:
        # Header
        f.write(struct.pack("<iii", args.rank, args.vocab_size, args.embed_dim))
        # Data
        f.write(A.astype(np.float32).tobytes())
        f.write(B.astype(np.float32).tobytes())

    compressed_size = os.path.getsize(output_path)
    compression_ratio = file_size / compressed_size
    print(f"Success! Output written successfully.")
    print(f"  Original Size:   {file_size / 1024 / 1024:.2f} MB")
    print(f"  Compressed Size: {compressed_size / 1024 / 1024:.2f} MB")
    print(f"  Compression Ratio: {compression_ratio:.2f}x")
    print("=" * 60)

if __name__ == "__main__":
    main()
