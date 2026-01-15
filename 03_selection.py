import pandas as pd

def select_best():
    df = pd.read_json("data/step2_tagged.jsonl", lines=True)
    
    # 1. Filter: Keep only high scores
    high_quality = df[df["slm_score"] >= 5]
    
    # 2. Stratify
    tactical = high_quality[high_quality["slm_tag"] == "Tactical"]
    positional = high_quality[high_quality["slm_tag"] == "Positional"]
    
    # 3. Sample (e.g. 5k each)
    # Using 'sample' randomly picks rows if we have more than needed
    best_tactical = tactical.sample(n=min(len(tactical), 500), random_state=42)
    best_positional = positional.sample(n=min(len(positional), 500), random_state=42)
    
    final_set = pd.concat([best_tactical, best_positional])
    
    final_set.to_json("data/platinum_1k.jsonl", orient="records", lines=True)
    print(f"Created dataset with {len(final_set)} records.")

if __name__ == "__main__":
    select_best()
