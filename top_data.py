import json
import re
import random
from tqdm import tqdm
from collections import Counter

# --- Configuration ---
INPUT_FILE = "chess_reasoning_tagged.jsonl"
OUTPUT_FILE = "chess_reasoning_platinum_10k.jsonl"
TARGET_SIZE = 10000

# High-value connector words that indicate reasoning
CAUSAL_KEYWORDS = {
    "because", "since", "so", "therefore", "thus", "order to", "aiming", 
    "prevent", "allows", "enables", "leads to", "consequently", "forcing",
    "threatens", "prepares", "defends", "exploits"
}

# Regex to find square mentions (e.g., "e4", "f7")
SQUARE_PATTERN = re.compile(r"\b[a-h][1-8]\b")

def calculate_score(entry):
    text = entry["explanation"].lower()
    tags = entry.get("tags", [])
    words = text.split()
    
    score = 0.0
    
    # 1. Causal Reasoning (+1.5 per causal word, max cap 3 points)
    causal_hits = sum(1 for w in CAUSAL_KEYWORDS if w in text)
    score += min(causal_hits * 1.5, 4.5)
    
    # 2. Concept Density (+2.0 per unique tag category found)
    # We prioritize entries that hit the keywords we defined earlier
    score += len(tags) * 2.0
    
    # 3. Grounding (+0.5 per square mention, max cap 2 points)
    square_hits = len(SQUARE_PATTERN.findall(text))
    score += min(square_hits * 0.5, 2.0)
    
    # 4. Length Penalty
    # We want concise explanations (10-40 words is the sweet spot)
    if len(words) < 10: 
        score -= 1.0 # Too short to be deep
    elif len(words) > 60:
        score -= 0.5 # Too verbose/story-telling
        
    return score

def select_top_data():
    data_pool = []
    
    print("Scoring data...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            entry = json.loads(line)
            score = calculate_score(entry)
            entry["score"] = score
            data_pool.append(entry)

    # Sort by score descending
    data_pool.sort(key=lambda x: x["score"], reverse=True)
    
    # --- STRATIFIED SAMPLING ---
    # We don't just want the top 10k (which might all be the same type).
    # We want a balance of Tactical and Positional.
    
    tactical_pool = [x for x in data_pool if "Tactical" in x["tags"]]
    positional_pool = [x for x in data_pool if "Positional" in x["tags"]]
    mixed_pool = [x for x in data_pool if not x["tags"]] # High score but no specific tag
    
    print(f"\nPool Sizes (Score > 0):")
    print(f"Tactical: {len(tactical_pool)}")
    print(f"Positional: {len(positional_pool)}")
    
    # Selection Target: 40% Tactic, 40% Strategy, 20% Best of the Rest
    final_selection = []
    final_selection.extend(tactical_pool[:4000])
    final_selection.extend(positional_pool[:4000])
    
    # Fill remaining 2000 from the highest scoring remaining items (any category)
    # Use a set to avoid duplicates
    selected_ids = {id(x) for x in final_selection}
    
    remaining_candidates = [x for x in data_pool if id(x) not in selected_ids]
    final_selection.extend(remaining_candidates[:2000])
    
    # Shuffle to mix them up
    #random.shuffle(final_selection)
    
    # Save
    print(f"\nSaving top {len(final_selection)} entries...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for entry in final_selection:
            f_out.write(json.dumps(entry) + "\n")

    # Show a few examples of "Platinum" data
    print("\n--- SAMPLE PLATINUM ENTRIES ---")
    for i in range(3):
        e = final_selection[i]
        print(f"Score: {e['score']:.1f} | Tags: {e['tags']}")
        print(f"Exp: {e['explanation']}")
        print("-" * 30)

if __name__ == "__main__":
    select_top_data()
