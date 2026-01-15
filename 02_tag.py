
import json
import re
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

INPUT_FILE = "data/step1_deduplicated.jsonl"
OUTPUT_FILE = "data/step2_qwen3_scored.jsonl"
MODEL_NAME = "Qwen/Qwen3-8B" 

SYSTEM_PROMPT = """You are a Grandmaster Chess Coach. 
Analyze the move explanation.
1. Identify and assign ALL relevant categories.
**Categories:**
- 'Tactical': Short-term calculation, threats, captures, forced sequences.
- 'Positional': Long-term strategy, structure, space, prophylaxis, piece improvement.
(An explanation can be BOTH, ONE, or NEITHER).
2. Provide a pedagogical score from 0 to 5 based on educational value and causal soundness.
**Grading (0-5):**
- 0: Garbage/Notation only.
- 1: Observation.
- 3: Basic Reasoning.
- 5: Deep Causal Insight.

**Output Format:** Return a valid JSON format
```
{
  "tags": ["Tag1", "Tag2"],
  "score": 4
}
```
"""
def format_prompt(tokenizer, explanation):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Explanation: {explanation}"}
    ]
    # Qwen3 specific: enable_thinking=True triggers the Chain-of-Thought
    return tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        enable_thinking=True  # <--- KEY UPGRADE
    )

def extract_and_parse_json(text):
    """
    Robustly extracts JSON from mixed text, handling:
    1. <think> blocks (removing them)
    2. Markdown code blocks (```json ... ```)
    3. Raw JSON strings embedded in text
    """
    try:
        # 1. Strip the <think> block if present (Chain-of-Thought)
        clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        
        # 2. Try to find JSON inside Markdown code blocks first
        # Matches ```json { ... } ``` or just ``` { ... } ```
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1)
        else:
            # 3. Fallback: Find the first '{' and the last '}'
            # This handles cases where the model just outputs "{...}" without markdown
            json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return [], 0 # No JSON found

        # 4. Parse
        data = json.loads(json_str)
        
        # Normalize keys (handling case sensitivity)
        tags = data.get("tags") or data.get("Tags") or []
        score = data.get("score") or data.get("Score") or 0
        
        # Sanitize tags
        clean_tags = []
        for t in tags:
            t_lower = str(t).lower()
            if "tact" in t_lower: clean_tags.append("Tactical")
            if "pos" in t_lower: clean_tags.append("Positional")
        
        return list(set(clean_tags)), int(score)
            
    except Exception as e:
        # print(f"Parsing Failed: {e}") # Uncomment to debug
        return [], 0
def process_qwen3():
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_json(INPUT_FILE, lines=True)
    
    print("Formatting prompts...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompts = [format_prompt(tokenizer, exp) for exp in df["explanation"]]

    print(f"Initializing {MODEL_NAME} on H100...")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        max_num_seqs=4096,
        max_model_len=8192, 
        trust_remote_code=True
    )
    
    # Allow enough tokens for reasoning + JSON
    params = SamplingParams(temperature=0.0, max_tokens=1024) 

    print(f"Inference on {len(df)} records...")
    outputs = llm.generate(prompts, params)

    results = []
    
    for i, o in enumerate(outputs):
        raw_output = o.outputs[0].text.strip()
        tags, score = extract_and_parse_json(raw_output)
        
        entry = df.iloc[i].to_dict()
        entry["slm_tags"] = tags
        entry["slm_score"] = score
        # Optional: Save the reasoning trace for auditing/debugging
        entry["judge_reasoning"] = re.findall(r"<think>(.*?)</think>", raw_output, re.DOTALL)
        
        results.append(entry)

    # Save
    result_df = pd.DataFrame(results)
    result_df.to_json(OUTPUT_FILE, orient="records", lines=True)
    print(f"Saved processed data to {OUTPUT_FILE}")

    # --- Verification ---
    print("\n>>> SAMPLE OUTPUT")
    sample = result_df[result_df['slm_score'] >= 4].head(1)
    if not sample.empty:
        print(f"Explanation: {sample.iloc[0]['explanation']}")
        print(f"Tags: {sample.iloc[0]['slm_tags']}")
        print(f"Score: {sample.iloc[0]['slm_score']}")
        if sample.iloc[0]['judge_reasoning']:
            print(f"Reasoning snippet: {sample.iloc[0]['judge_reasoning'][0][:100]}...")

if __name__ == "__main__":
    process_qwen3()