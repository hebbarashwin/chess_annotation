import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Configuration
model_id = "OutFlankShu/MATE"
checkpoint_path = "generate_strategy/checkpoint-19208"

print(f"Loading {model_id} (Checkpoint: {checkpoint_path})...")

# 2. Load Model & Tokenizer
# We specify the 'subfolder' to target the exact checkpoint you requested
tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder=checkpoint_path)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    subfolder=checkpoint_path,
    torch_dtype=torch.bfloat16, # GH200 supports bfloat16 natively
    device_map="auto"
)

# 3. Define Inference Function
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            repetition_penalty=1.2,
            do_sample=True, 
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def model_generate(
    tok, model, prompt
) -> str:
    messages = [{"role": "user", "content": prompt}]
    prompt = tok.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    gen_temp = 0.1
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=gen_temp,
            pad_token_id=tok.eos_token_id  
        )
    
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    response = tok.decode(new_tokens, skip_special_tokens=True)
    
    return response.strip()
# 4. Test Input (Chess Strategy)
# MATE is trained on chess positions (FEN strings). Here is a sample test case.
def build_prompt(fen: str, move: str) -> str:
       
    test_fen = f'''You are a chess expert. Given the chess position in FEN format and a candidate move, provide a comprehensive explanation. 
    Your analysis must be divided into two specific components:
    1. **Strategy**: Explain the long-term strategic reasoning (e.g., space, piece activity, pawn structure, king safety).
    2. **Tactic**: Explain the short-term tactical calculation (e.g., immediate threats, pins, forks, mate patterns, calculation depth of 1-6 moves).

    The FEN of the given chess board is \"{fen}\".
    Please generate explanation for the following move: {move}.
    '''.format(fen=fen, move=move)
    return test_fen

def build_prompt2(fen: str, move: str) -> str:
       
    test_fen = f'''
    The FEN of the given chess board is \"{fen}\".
    Please generate explanation for the following move: {move}.
    '''.format(fen=fen, move=move)
    return test_fen

print("\n--- Test Prompt 1 ---")
prompt = build_prompt('1r6/6pk/p2qp1rp/3p1p2/P1p5/4PN1P/2P2KP1/R2QR3 b - - 0 33', 'g6g2')
print(f"Input: {prompt}")
response = model_generate(tokenizer, model, prompt)
print(f"Output: {response}")


print("\n--- Test Prompt 2 ---")
prompt = build_prompt2('1r6/6pk/p2qp1rp/3p1p2/P1p5/4PN1P/2P2KP1/R2QR3 b - - 0 33', 'g6g2')
print(f"Input: {prompt}")
response = model_generate(tokenizer, model, prompt)
print(f"Output: {response}")
# Optional: Keep script running for interactive testing




while True:
    user_fen = input("\nEnter FEN:")
    user_move = input("Enter Move: ")
    print("\n--- Response 1 ---")
    print(model_generate(tokenizer, model, build_prompt(user_fen, user_move)))
    print("\n--- Response 2 ---")
    print(model_generate(tokenizer, model, build_prompt2(user_fen, user_move)))



#{"instruction": "You are an expert chess player. You are given a chess board with FEN format. Your goal is to choose a better move given two candidate moves with their strategy explanations and tactics.", "input": "The FEN of the given chess board is \"4r1k1/pp3ppp/3q1n2/8/2r4N/1Q1n4/PP2RPPP/1K2R3 w - - 0 25\". Which move is better? MoveA:h4g6, Relocate the piece to a dynamic square, it's more influential on the board. TacticA: h4g6 e8e2 e1d1 d6d4 g6e7 g8f8   MoveB:a2a3, When a white pawn advances, it widens the queenside, allowing white greater liberty in moving pieces. TacticB: a2a3 e8e2 e1e2 c4c1 b1a2 d3b4  ", "output": "MoveB:a2a3"}
# {"instruction": "You are an expert chess player. You are given a chess board with FEN format. Your goal is to choose a better move given two candidate moves with their strategy explanations and tactics.", "input": "The FEN of the given chess board is \"6k1/6p1/7p/5p2/4p3/1R1rP2P/5PP1/6K1 w - - 1 34\". The move is: b3b8, Marshall the piece to a better square, extending its control over the board. TacticA: b3b8 g8f7 g2g3 f7e6 g3g4 f5f4   MoveB:b3c3, Displace the piece to a more promising position, strengthening its board authority. TacticB: b3c3 d3c3 g2g4 g8f7 g1h2 f7f6  ", "output": "MoveA:b3b8"}

# {"instruction": "You are an expert chess player. You are given a chess board with FEN format. Your goal is to choose a better move given two candidate moves with their strategy explanations and tactics.", "input": "The FEN of the given chess board is \"1r6/6pk/p2qp1rp/3p1p2/P1p5/4PN1P/2P2KP1/R2QR3 b - - 0 33\". Which move is better? MoveA:d6g3, Transfer the piece to a lively area, boosting its impact on the board. TacticA: d6g3 f2e2 g3g2 Checkmate!  MoveB:g6g2, Surrender a piece to establish an open file or diagonal near the rival king TacticB: g6g2 f2g2 Trade the lower value piece for a higher value piece. ", "output": "MoveA:d6g3"}

#The FEN of the given chess board is \"1r6/6pk/p2qp1rp/3p1p2/P1p5/4PN1P/2P2KP1/R2QR3 b - - 0 33\". The move: g6g2.  