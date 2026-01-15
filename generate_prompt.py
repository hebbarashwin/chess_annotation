def build_prompt(fen: str, move: str) -> str:
       
    test_fen = f'''You are a chess expert. Given the chess position in FEN format and a candidate move in UCI format, provide a concise and pedagogical explanation. 
    The explanation can be from any of the following aspects:
    1. **Positional**: Explain the long-term strategic reasoning (e.g., space, piece activity, pawn structure, king safety).
    2. **Tactical**: Explain the short-term tactical calculation (e.g., immediate threats, pins, forks, mate patterns, calculation depth of 1-6 moves).

    The FEN of the given chess board is \"{fen}\".
    Please generate explanation for the following move: {move}.
    '''.format(fen=fen, move=move)
    return test_fen


while True:
    user_fen = input("\nEnter FEN:")
    user_move = input("Enter Move: ")
    print(build_prompt(user_fen, user_move))
    