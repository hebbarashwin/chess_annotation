import re
import chess.pgn

part = int(input("Enter part number (e.g., 1, 2): ").strip())
INPUT_TXT = f"data/on_my_great_predecessors_{part}.txt"    
INPUT_PGN = f"data/on_my_great_predecessors_{part}.pgn" 
OUTPUT_PGN = f"data/on_my_great_predecessors_{part}_annotated.pgn" 

if part == -1:
    INPUT_TXT = "data/logical_chess.txt"
    INPUT_PGN = "data/logical_chess_verified.pgn"
    OUTPUT_PGN = "data/logical_chess_annotated.pgn"

def split_text_into_games(text):
    pattern = re.compile(r'(^Game\s+(\d+).*?)(?=\nGame\s+\d+|\Z)', re.DOTALL | re.MULTILINE)
    games_text = {}
    for match in pattern.finditer(text):
        game_id = match.group(2) 
        full_text = match.group(1)
        games_text[game_id] = full_text
        
    print(f"Found {len(games_text)} text chunks in the OCR file.")
    return games_text

def is_move_sequences(check_str, print_result=False):
    check_str = re.sub(r'\(.*?\)', '', check_str)
    check_str = re.sub(r'\d+\.+', '', check_str)
    check_str = re.sub(r'[.KQRBNabcdefgh0-9O\-x\+\#\=\!\?]', '', check_str)
    check_str = re.sub(r'\s+', '', check_str)
    if print_result: print(check_str)
    return len(check_str) == 0
def align(game, game_text):
    raw_lines = game_text.split('\n')
    parsed_segments = [] 
    
    current_last_move = None
    current_comment_buffer = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        line_no_parens = re.sub(r'\(.*?\)', '', line)
        is_move_line = is_move_sequences(line_no_parens)
        #if is_move_line: print(line)
        if is_move_line:
            if current_last_move is not None or current_comment_buffer:
                full_comment = "\n".join(current_comment_buffer).strip()
                parsed_segments.append((current_last_move, full_comment))
                current_comment_buffer = []

            tokens = line_no_parens.split()
            
            found_move = None
            for token in reversed(tokens):
                if not re.match(r'^\d+\.+$', token):
                    found_move = token.rstrip('?!')
                    #print(found_move)
                    break
            
            current_last_move = found_move
        else:
            current_comment_buffer.append(line)
    if current_last_move or current_comment_buffer:
        parsed_segments.append((current_last_move, "\n".join(current_comment_buffer)))
    #print(parsed_segments)
    
    
    game_move_iter = iter(game.mainline())

    for target_san, comment_text in parsed_segments:
        
        if target_san is None:
            game.comment = comment_text
            continue
        try:
            for node in game_move_iter:
                board_before = node.parent.board()
                real_san = board_before.san(node.move)

                if real_san == target_san:
                    node.comment = comment_text
                    break
        except StopIteration:
            print("stop")
            pass


def main():
    with open(INPUT_TXT, "r", encoding="utf-8") as f:
        raw_text = f.read()
    game_chunks = split_text_into_games(raw_text)
    
    print(f"Processing {INPUT_PGN}...")
    output_f = open(OUTPUT_PGN, "w", encoding="utf-8")
    
    with open(INPUT_PGN, "r", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None: break
            
            book_id = game.headers.get("BookGame")
            if book_id and book_id in game_chunks:
                print(f"Aligning Game {book_id}...")
                align(game, game_chunks[book_id])
            else:
                print(f"Skipping Game {book_id} (No text or missing ID)")
                                
            print(game, file=output_f, end="\n\n")
            
    output_f.close()
    print(f"Done. Saved to {OUTPUT_PGN}")

if __name__ == "__main__":
    main()