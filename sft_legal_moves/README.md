# SFT Legal Moves -- Next-Move Prediction Dataset

Extract chess positions from game databases for **next-move prediction** evaluation. Each record presents a position with a mix of legal and illegal candidate moves; the task is to identify which moves are legal.

## Data Source

Positions are extracted from the [Lichess Open Database](https://database.lichess.org/).

We use the January 2013 standard rated games:
1. Download: https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst
2. Decompress `.zst` → `.pgn` (e.g. `zstd -d lichess_db_standard_rated_2013-01.pgn.zst`)
3. Place the `.pgn` file in `data/`

## Output Format

Each record is a JSON line (`.jsonl`) with these fields:

| Field | Description |
|---|---|
| `fen` | Board position (FEN string) |
| `last_move_uci` | The move that led to this position |
| `game_move_uci` | The move actually played in the game |
| `next_move_candidates_uci` | All candidate moves (legal + illegal) |
| `correct_outputs_uci` | All legal moves (= correct answers) |
| `illegal_category_uci` | Category-specific illegal moves (see below) |
| `illegal_general_uci` | General distractor illegal moves (see below) |
| `tags` | Which categories this position belongs to |
| `phase` | Game phase: `opening`, `middlegame`, or `endgame` |
| `game_id` | Lichess game URL |
| `ply` | Half-move number in the game |

`next_move_candidates_uci` = `correct_outputs_uci` ∪ `illegal_category_uci` ∪ `illegal_general_uci`

## Position Categories

### 1. En Passant (`en_passant`)
Positions where en passant capture is legal. Tests whether the model knows this special pawn rule.

**Category illegals:** Pawn diagonal moves to empty squares that aren't the en passant square (looks like a capture but nothing is there).

### 2. Single Check Evasion (`check`)
In check by one piece, with at least 2 types of evasion available (king move, capture attacker, block). Tests understanding of check response options.

**Category illegals:** King moves to squares still attacked by opponent + castling while in check.

### 3. Double Check (`double_check`)
Two pieces give check simultaneously. Only king moves are legal -- you can't capture or block both checkers at once.

**Category illegals:** Non-king pseudo-legal moves that capture/block one checker (tempting but illegal since the other checker still gives check) + king moves to attacked squares + castling while in check.

### 4. Illegal King Moves (`illegal_king`)
Not in check, but the king has adjacent squares or castling paths controlled by the opponent.

**Category illegals:** King moves to attacked squares + castling through/onto attacked squares (path is clear and rights exist, but a transit or destination square is attacked).

### 5. Pin (`pin`)
The side to move has one of its own pieces pinned to its own king by an opponent sliding piece. Moving the pinned piece off the pin ray would expose the king.

**Category illegals:** Pseudo-legal moves of the pinned piece that leave the pin ray.

### 6. Promotion (`promotion`)
A pawn is on the 7th rank (or 2nd for black) and can promote.

**Category illegals:** Promotion push onto an occupied square (pawn blocked) + promotion capture to an empty square (nothing to capture diagonally).

### 7. Vanilla (`vanilla`)
Random positions with no special tag. Only general distractors, no category-specific illegals. Controlled by `NUM_VANILLA_POSITIONS` config knob.

## General Distractors

Added to every position (tagged and vanilla) to pad candidates with plausible-looking but fundamentally illegal moves. Controlled by `NUM_GENERAL_DISTRACTORS` config knob.

| Type | Description |
|---|---|
| `backward_pawn` | Pawn moves in the wrong direction |
| `friendly_fire` | Piece "captures" its own piece |
| `blocked_sliding` | Rook/bishop/queen moves through a blocking piece |
| `pawn_double_wrong_rank` | Pawn double-pushes from a non-starting rank |
| `pawn_push_onto_piece` | Pawn pushes forward into an occupied square |
| `wrong_geometry` | Piece moves in a way its type doesn't allow (knight diagonal, bishop straight, rook diagonal) |

Sampling is diversity-aware: one from each available type first, then fill randomly.

## Config Knobs

All at the top of the notebook:

| Variable | Default | Description |
|---|---|---|
| `PGN_PATH` | `data/lichess_db_standard_rated_2013-01.pgn` | Path to input PGN |
| `MAX_GAMES` | `50` | Number of games to scan |
| `NUM_GENERAL_DISTRACTORS` | `5` | General illegal moves per position |
| `NUM_VANILLA_POSITIONS` | `100` | Vanilla positions to sample |
| `SEED` | `50` | RNG seed for reproducibility |

## Usage

```bash
# activate environment with python-chess installed
conda activate dev  # or: pip install python-chess

# run the notebook
jupyter notebook extract_eval_positions.ipynb
```

Run all cells top-to-bottom. The notebook includes:
- Position extraction with per-category detection
- Summary statistics and tag co-occurrence
- SVG board visualization (yellow=last move, red=category illegals, orange=general illegals)
- Sanity checks (verifies legality/illegality of all moves)
- Interactive browser (`browse(rows)` or `browse(rows, tag_filter='pin')`)
- JSONL export to `data/eval_positions_preview.jsonl`

## Dependencies

- `python-chess` (tested with v1.9.4)
- `jupyter` / `ipython`

## File Structure

```
sft_legal_moves/
├── README.md
├── extract_eval_positions.ipynb   # main notebook
└── data/                          # put PGN here; JSONL output goes here
```
