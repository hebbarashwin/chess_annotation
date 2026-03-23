# Evaluation

Extract structured atoms from chess commentary and evaluate LLM-generated explanations.

## Pipeline

```
logical_chess.jsonl (1832 positions)
        │
        ▼
  extract_all.py prepare   ← Stockfish analysis
        │
        ▼
  extract_all.py submit    ← OpenAI Batch API
        │
        ▼
  extract_all.py collect   ← download results
        │
        ▼
  extract_all.py process   ← postprocess filter → included.jsonl / excluded.jsonl
        │
        ▼
  extract_all.py filter    ← LLM filter pass → included_filtered.jsonl / excluded_filtered.jsonl
        │
        ▼
  train.jsonl + test_unfiltered.jsonl  ← game-wise split (no leakage)
        │
        ▼
  review_gold.ipynb        ← human review → test_accepted / test_rejected / test_again
```

## Scripts

### `extract_all.py`

Batch extraction of commentary into structured atoms. No tools — single LLM call per position.

```bash
# 1. Run Stockfish, build batch request
python evaluation/extract_all.py prepare \
    --input evaluation/data/logical_chess.jsonl \
    --batch-file evaluation/data/logical_chess_atomize/batch_input.jsonl \
    --meta-file evaluation/data/logical_chess_atomize/batch_meta.jsonl

# 2. Submit to OpenAI Batch API (50% off)
python evaluation/extract_all.py submit \
    --batch-file evaluation/data/logical_chess_atomize/batch_input.jsonl

# 3. Poll + download results
python evaluation/extract_all.py collect \
    --batch-id <BATCH_ID> \
    --output evaluation/data/logical_chess_atomize/batch_output.jsonl --poll

# 4. Postprocess → included/excluded
python evaluation/extract_all.py process \
    --meta-file evaluation/data/logical_chess_atomize/batch_meta.jsonl \
    --batch-output evaluation/data/logical_chess_atomize/batch_output.jsonl \
    --out-included evaluation/data/logical_chess_atomize/included.jsonl \
    --out-excluded evaluation/data/logical_chess_atomize/excluded.jsonl
```

`sync` subcommand available for small runs without the batch API.

```bash
# 5. Filter atoms: contextualize, move to alternative, deduplicate
python evaluation/extract_all.py filter \
    --input evaluation/data/logical_chess_atomize/included.jsonl \
    --out-included evaluation/data/logical_chess_atomize/included_filtered.jsonl \
    --out-excluded evaluation/data/logical_chess_atomize/excluded_filtered.jsonl
```

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `extract_atoms.ipynb` | Interactive extraction pipeline with tool-augmented LLM. Contextual atom system prompt, hardcoded demos, generation loop. |
| `eval_tools.ipynb` | LLM-as-a-Judge: generate NL commentary, then evaluate via atom-level decomposition → verification → matching. |
| `compare_contextual.ipynb` | Comparison notebook — contextual atom style (atoms carry full move-sequence prefix). |
| `compare_flat.ipynb` | Comparison notebook — flat atom style (conclusions only, move sequences in `variation` field). |

## Data

### `data/logical_chess.jsonl`

Source dataset. 1832 annotated positions from *Logical Chess: Move by Move*.

```json
{"game_id": "...", "fen": "...", "move_uci": "e2e4", "move_san": "e4",
 "annotation": "This is an excellent opening move...",
 "metadata": {"White": "Scheve", "Black": "Teichmann", ...}}
```

### `data/logical_chess_atomize/`

Batch extraction outputs:

| File | Description |
|------|-------------|
| `batch_input.jsonl` | OpenAI Batch API requests |
| `batch_meta.jsonl` | Metadata (engine lines, wp_loss) keyed by `custom_id` |
| `batch_output.jsonl` | Raw LLM responses |
| `included.jsonl` | 1302 positions with extracted atoms (32 games) |
| `excluded.jsonl` | 530 positions excluded (too minimal, conflicts, etc.) |
| `included_filtered.jsonl` | Positions after filter pass (atoms cleaned up) |
| `excluded_filtered.jsonl` | Positions excluded by filter (0 reasoning atoms) |
| `filter_atoms.ipynb` | Interactive filter notebook (same logic as `extract_all.py filter`) |
| `train.jsonl` | 992 positions from 24 games (seed=99, game-wise split) |
| `test_unfiltered.jsonl` | 310 positions from 8 games (seed=99, game-wise split) |
| `review_gold.ipynb` | Human review UI for test set (randomized order, seed=42) |
| `test_accepted.jsonl` | Accepted after review |
| `test_rejected.jsonl` | Rejected after review |
| `test_again.jsonl` | Flagged for re-review |

### Extraction output schema

```json
{
  "position_number": 5,
  "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
  "move_uci": "f1c4", "move_san": "Bc4",
  "annotation": "The bishop seizes a valuable diagonal...",
  "wp_loss": 0.64, "quality": "good",
  "game": "Scheve – Teichmann",
  "engine_lines": [{"move_san": "Bb5", "eval": "+0.32", "cp": 32, ...}, ...],
  "extracted": {
    "include": true,
    "quality": "good",
    "reasoning": [
      "Bc4 develops White's king bishop and clears the way for early castling.",
      "Bc4 places the bishop on the a2-g8 diagonal through the center.",
      "Bc4 attacks the f7-pawn.",
      "The f7-pawn is defended only by the king, making it a vulnerable target."
    ],
    "book_commentary": "..."
  },
  "model": "gpt-5.4"
}
```

## Key concepts

- **Contextual atoms**: each reasoning atom is self-contained with full move-sequence prefix ("After Nxd7 Nh5, White can capture on g6") so it can be verified independently.
- **wp_loss**: win-percentage loss from Stockfish eval. `Win% = 50 + 50 * tanh(0.00368208 * cp / 2)`.
- **Quality**: good (≤10%), inaccuracy (>10%), mistake (>20%), blunder (>30%).
- **Postprocess filter**: catches quality/engine conflicts, missing reasoning, empty alternatives.
