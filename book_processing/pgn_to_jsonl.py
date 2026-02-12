import chess.pgn
import json

def pgn_to_jsonl(pgn_file_path, output_file_path, book_id_header="Part1"):
    
    with open(pgn_file_path, "r", encoding="utf-8") as pgn_in, \
         open(output_file_path, "a", encoding="utf-8") as json_out:

        while True:
            game = chess.pgn.read_game(pgn_in)
            if game is None:
                break  

            board = game.board()

            for node in game.mainline():
                current_fen = board.fen()
                
                move = node.move
                move_uci = move.uci()
                move_san = board.san(move) 
                if node.comment and node.comment.strip():
                    entry = {
                        "game_id": book_id_header + "Game" + game.headers.get("BookGame", "Unknown"),
                        "fen": current_fen,
                        "move_uci": move_uci,
                        "move_san": move_san,
                        "annotation": node.comment.strip(),
                        "metadata": dict(game.headers)
                    }
                    json_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

                board.push(move)

if __name__ == "__main__":
    for i in range(1, 6):
        pgn_to_jsonl(f"data/on_my_great_predecessors_{i}_annotated.pgn", "data/my_great_predecessors.jsonl", book_id_header=f"Part{i}_")