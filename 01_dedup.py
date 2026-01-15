import json
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import gc
from tqdm import tqdm

# --- Configuration ---
INPUT_FILE = "data/chess_annotation_filtered.jsonl"
OUTPUT_FILE = "data/step1_deduplicated.jsonl"
MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.90
# H100 80GB can handle large chunks.
# Matrix size: Chunk_Size * 400k * 4 bytes (float32).
# 5000 * 400,000 * 4 = ~8 GB VRAM per chunk. Safe.
SEARCH_BATCH_SIZE = 5000 

def dedup_pytorch():
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_json(INPUT_FILE, lines=True)
    texts = df["explanation"].tolist()
    
    # 1. Embed on GPU
    print(f"Embedding {len(df)} items on H100...")
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    
    # Encode directly to PyTorch tensor on GPU
    with torch.amp.autocast('cuda'):
        embeddings = model.encode(
            texts, 
            batch_size=4096, 
            show_progress_bar=True, 
            convert_to_tensor=True,
            normalize_embeddings=True
        )

    # 2. Brute Force Deduplication (Chunked)
    print("Running Similarity Search directly in PyTorch...")
    
    num_vectors = len(embeddings)
    keep_mask = torch.ones(num_vectors, dtype=torch.bool, device="cuda")
    duplicates_found = 0
    examples = []

    # Iterate through the dataset in chunks
    for start_idx in tqdm(range(0, num_vectors, SEARCH_BATCH_SIZE), desc="Deduplicating"):
        end_idx = min(start_idx + SEARCH_BATCH_SIZE, num_vectors)
        
        # Current batch of queries
        batch = embeddings[start_idx:end_idx] # Shape: [Batch, D]
        
        # Matrix Multiply: Batch x All_Vectors^T
        # Result Shape: [Batch, N]
        # This computes cosine similarity since vectors are normalized
        sim_matrix = torch.matmul(batch, embeddings.T)
        
        # Find pairs > threshold
        # We only care about columns (j) > rows (i) to avoid self-matches and double counting
        # Logic: global_row_idx < global_col_idx
        
        # Create a mask for valid upper-triangular comparisons
        # We want: (start_idx + r) < c
        rows, cols = torch.where(sim_matrix > SIMILARITY_THRESHOLD)
        
        # Convert local row indices to global indices
        global_rows = rows + start_idx
        
        # Filter: Only look at future duplicates (i < j)
        valid_pairs = global_rows < cols
        
        # Apply filter
        source_indices = global_rows[valid_pairs] # The "original"
        target_indices = cols[valid_pairs]        # The "duplicate"
        
        # Mark duplicates for removal
        if len(target_indices) > 0:
            # Check which sources are still valid (not already deleted)
            # We must be careful: if A is dupe of B, and B is dupe of C.
            # We keep A, delete B. When we process B, we shouldn't use it to delete C?
            # Actually for aggressive dedup, if A~B and B~C, removing B and C is fine.
            
            # We assume if i < j, i deletes j.
            # We only perform deletion if 'i' (source) hasn't been deleted yet.
            # However, in a vectorised way, simpler is: if ANYONE marks 'j' as a dupe, 'j' dies.
            keep_mask[target_indices] = False
            
            # Grab examples for logging (limit to first 5)
            if len(examples) < 5:
                # CPU copy for printing
                src_ex = source_indices[0].item()
                tgt_ex = target_indices[0].item()
                score = sim_matrix[rows[valid_pairs][0], tgt_ex].item()
                examples.append({
                    "score": score,
                    "original": texts[src_ex],
                    "duplicate": texts[tgt_ex]
                })

    # 3. Save
    keep_mask_cpu = keep_mask.cpu().numpy()
    duplicates_found = num_vectors - keep_mask_cpu.sum()
    
    print(f"Removed {duplicates_found} duplicates.")
    df_clean = df[keep_mask_cpu]
    df_clean.to_json(OUTPUT_FILE, orient="records", lines=True)

    print("\n--- DUPLICATE EXAMPLES ---")
    for ex in examples:
        print(f"[{ex['score']:.4f}]")
        print(f"   Keep: {ex['original'][:60]}...")
        print(f"   Drop: {ex['duplicate'][:60]}...")

if __name__ == "__main__":
    dedup_pytorch()