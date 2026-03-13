#!/usr/bin/env python3
"""Extract a balanced binary eval dataset with a target count per subcategory.

Single pass through PGN(s).  Each subcategory has its own RNG (derived from a
master seed) and independently decides on-the-fly whether to keep a position.
No two-phase buffering — extraction and selection happen together.

Features:
  - Per-subcategory seed derived from master seed → different subcategories
    make independent random choices, avoiding correlated position overlap.
  - Per-game cap (default 3) per subcategory → diversity across games/phases.
  - One move per position per subcategory → maximises position diversity.
  - --subcategories flag → run a subset (for parallel jobs or rare types).

Usage:
    # All subcategories at once:
    python extract_balanced_eval.py \
        --pgn_paths data/lichess_2013-01.pgn data/lichess_2013-02.pgn \
        --out_path data/eval_binary_balanced.jsonl \
        --target 200 --max_games 5000 --seed 42

    # Just a few rare ones (can scan more games):
    python extract_balanced_eval.py \
        --pgn_paths data/big.pgn \
        --out_path data/eval_ep_wrong_pawn.jsonl \
        --subcategories ep_wrong_pawn non_king_double_check promo_push_blocked \
        --target 200 --max_games 100000 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

import chess
import chess.pgn

from legal_moves import get_phase, move_to_san
from legal_move_puzzles import (
    SUBCATEGORY_TO_CATEGORY,
    classify_legal_move,
    detect_en_passant, build_en_passant_illegals,
    analyze_check, build_check_candidates,
    detect_double_check, build_double_check_illegals,
    detect_illegal_king_moves, build_illegal_king_illegals,
    detect_pin, build_pin_illegals,
    detect_promotion, build_promotion_illegals,
    _gen_backward_pawn, _gen_friendly_fire, _gen_blocked_sliding,
    _gen_pawn_double_push_wrong_rank, _gen_pawn_double_push_blocked,
    _gen_pawn_push_onto_piece, _gen_pawn_diagonal_to_empty,
    _gen_pawn_capture_friendly, _gen_castling_path_occupied,
    _gen_wrong_geometry,
)

ALL_SUBCATEGORIES = sorted(SUBCATEGORY_TO_CATEGORY.keys())


# ── Position scanning ───────────────────────────────────────────────────────


def extract_position_subcategories(
    board: chess.Board,
    wanted: Set[str],
) -> Dict[str, List[str]]:
    """Return {subcategory: [uci, ...]} for wanted types present in this position.

    Only computes generators relevant to the wanted set for efficiency.
    """
    legal_ucis = set(m.uci() for m in board.legal_moves)
    result: Dict[str, List[str]] = defaultdict(list)
    seen_illegal: Set[str] = set()

    # Which families of generators do we need?
    need_ep = bool(wanted & {"ep_fake_diagonal", "ep_wrong_pawn"})
    need_check = bool(wanted & {"king_to_attacked", "castling_in_check", "non_evasion_in_check"})
    need_dc = bool(wanted & {"king_to_attacked", "non_king_double_check", "castling_in_check"})
    need_ik = bool(wanted & {"king_to_attacked", "castling_through_attacked"})
    need_pin = "pin_breaking" in wanted
    need_promo = bool(wanted & {"promo_push_blocked", "promo_capture_empty"})
    need_legal = bool(wanted & {
        "legal_move", "legal_capture", "legal_castling", "legal_en_passant",
        "legal_promotion", "legal_check", "legal_king_escape",
        "legal_capture_checker", "legal_block_check",
    })
    need_general = bool(wanted & {
        "backward_pawn", "friendly_fire", "blocked_sliding",
        "pawn_double_wrong_rank", "pawn_double_push_blocked",
        "pawn_push_onto_piece", "pawn_diagonal_to_empty",
        "pawn_capture_friendly", "wrong_ep",
        "castling_path_occupied",
        "wrong_geometry_knight", "wrong_geometry_bishop",
        "wrong_geometry_rook", "wrong_geometry_queen", "wrong_geometry_king",
    })

    def _add_illegal(uci: str, t: str):
        if t in wanted and uci not in legal_ucis and uci not in seen_illegal:
            seen_illegal.add(uci)
            result[t].append(uci)

    # ── Category-specific illegals ──
    if need_ep:
        ep = detect_en_passant(board)
        if ep:
            for uci, t in build_en_passant_illegals(board, ep):
                _add_illegal(uci, t)

    if need_dc or need_check:
        dc = detect_double_check(board)
        if dc and need_dc:
            for uci, t in build_double_check_illegals(board, dc):
                _add_illegal(uci, t)
        elif not dc and board.is_check() and need_check:
            ci = analyze_check(board)
            if ci and ci.evasion_types >= 2:
                for uci, t in build_check_candidates(board, ci):
                    _add_illegal(uci, t)

    if need_ik and not board.is_check():
        ik = detect_illegal_king_moves(board)
        if ik:
            for uci, t in build_illegal_king_illegals(board, ik):
                _add_illegal(uci, t)

    if need_pin:
        pin = detect_pin(board)
        if pin:
            for uci, t in build_pin_illegals(board, pin):
                _add_illegal(uci, t)

    if need_promo:
        promo = detect_promotion(board)
        if promo:
            for uci, t in build_promotion_illegals(board, promo):
                _add_illegal(uci, t)

    # ── General distractors ──
    if need_general:
        turn = board.turn
        all_gen = []
        all_gen += _gen_backward_pawn(board, turn)
        all_gen += _gen_friendly_fire(board, turn)
        all_gen += _gen_blocked_sliding(board, turn)
        all_gen += _gen_pawn_double_push_wrong_rank(board, turn)
        all_gen += _gen_pawn_double_push_blocked(board, turn)
        all_gen += _gen_pawn_push_onto_piece(board, turn)
        all_gen += _gen_pawn_diagonal_to_empty(board, turn)
        all_gen += _gen_pawn_capture_friendly(board, turn)
        all_gen += _gen_castling_path_occupied(board, turn)
        all_gen += _gen_wrong_geometry(board, turn)

        combined_seen = legal_ucis | seen_illegal
        for move, t in all_gen:
            uci = move.uci()
            if t in wanted and uci not in combined_seen:
                combined_seen.add(uci)
                result[t].append(uci)

    # ── Legal moves ──
    if need_legal:
        for m in board.legal_moves:
            subcat = classify_legal_move(board, m)
            if subcat in wanted:
                result[subcat].append(m.uci())

    return dict(result)


# ── Seed derivation ─────────────────────────────────────────────────────────


def derive_seed(master_seed: int, subcategory: str) -> int:
    """Deterministically derive a per-subcategory seed from the master seed."""
    h = hashlib.sha256(f"{master_seed}:{subcategory}".encode()).hexdigest()
    return int(h[:8], 16)


# ── Per-subcategory sampler state ───────────────────────────────────────────


class SubcategorySampler:
    """Independently samples positions for one subcategory on-the-fly."""

    def __init__(self, subcategory: str, master_seed: int,
                 target: int, per_game_cap: int):
        self.subcategory = subcategory
        self.target = target
        self.per_game_cap = per_game_cap
        self.rng = random.Random(derive_seed(master_seed, subcategory))
        self.label = "legal" if SUBCATEGORY_TO_CATEGORY[subcategory] == "legal" else "illegal"
        self.category = SUBCATEGORY_TO_CATEGORY[subcategory]
        self.rows: List[dict] = []
        self.game_counts: Dict[int, int] = Counter()

    @property
    def full(self) -> bool:
        return len(self.rows) >= self.target

    def offer(self, game_idx: int, fen: str, phase: str, ucis: List[str]) -> bool:
        """Offer a candidate position. Returns True if accepted."""
        if self.full:
            return False
        if self.game_counts[game_idx] >= self.per_game_cap:
            return False

        # Pick one random move from this position
        move_uci = self.rng.choice(ucis)
        board = chess.Board(fen)
        move_san = move_to_san(board, move_uci)

        self.rows.append({
            "fen": fen,
            "move_uci": move_uci,
            "move_san": move_san,
            "label": self.label,
            "category": self.category,
            "subcategory": self.subcategory,
            "phase": phase,
        })
        self.game_counts[game_idx] += 1
        return True


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Extract balanced binary eval, target N per subcategory."
    )
    parser.add_argument("--pgn_paths", type=str, nargs="+", required=True,
                        help="PGN file(s) to scan")
    parser.add_argument("--out_path", type=str, required=True,
                        help="Output JSONL path")
    parser.add_argument("--target", type=int, default=200,
                        help="Target (fen, move) pairs per subcategory (default: 200)")
    parser.add_argument("--max_games", type=int, default=50000,
                        help="Max total games to scan across all PGNs")
    parser.add_argument("--per_game_cap", type=int, default=3,
                        help="Max positions per (game, subcategory) pair (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--subcategories", type=str, nargs="*", default=None,
                        help="Subset of subcategories to extract (default: all)")
    args = parser.parse_args()

    # Validate subcategories
    if args.subcategories:
        unknown = set(args.subcategories) - set(ALL_SUBCATEGORIES)
        if unknown:
            parser.error(f"Unknown subcategories: {unknown}\nValid: {ALL_SUBCATEGORIES}")
        active_subs = sorted(set(args.subcategories))
    else:
        active_subs = ALL_SUBCATEGORIES

    print(f"Extracting {len(active_subs)} subcategories, target {args.target} each")

    # Create one sampler per subcategory
    samplers = {
        s: SubcategorySampler(s, args.seed, args.target, args.per_game_cap)
        for s in active_subs
    }
    wanted = set(active_subs)

    # ── Single PGN pass ──

    total_games = 0
    total_positions = 0

    for pgn_path in args.pgn_paths:
        if total_games >= args.max_games:
            break
        # Check if all samplers are full
        if all(samplers[s].full for s in active_subs):
            break
        print(f"Scanning {pgn_path} ...")

        with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
            while total_games < args.max_games:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                game_idx = total_games
                board = game.board()
                last_move = None

                # Figure out which subcategories still need data
                still_needed = {s for s in active_subs if not samplers[s].full}
                if not still_needed:
                    break

                for game_move in game.mainline_moves():
                    if last_move is not None:
                        total_positions += 1
                        subcats = extract_position_subcategories(board, still_needed)
                        if subcats:
                            fen = board.fen()
                            phase = get_phase(board)
                            for subcat, ucis in subcats.items():
                                samplers[subcat].offer(game_idx, fen, phase, ucis)

                    board.push(game_move)
                    last_move = game_move

                total_games += 1
                if total_games % 1000 == 0:
                    filled = sum(1 for s in active_subs if samplers[s].full)
                    still = len(active_subs) - filled
                    print(f"  {total_games} games, {total_positions} positions | "
                          f"{filled}/{len(active_subs)} full, {still} remaining")

                    # Recalculate what we still need (for efficiency)
                    still_needed = {s for s in active_subs if not samplers[s].full}

    # ── Write output ──

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for s in active_subs:
        all_rows.extend(samplers[s].rows)

    # Shuffle so subcategories are interleaved
    master_rng = random.Random(args.seed)
    master_rng.shuffle(all_rows)

    with out_path.open("w") as fout:
        for row in all_rows:
            fout.write(json.dumps(row) + "\n")

    # ── Report ──

    print(f"\nScanned {total_games} games, {total_positions} positions")
    print(f"\n{'Subcategory':<30} {'Selected':>8} / {'Target':>6}  {'Phase distribution'}")
    print("-" * 85)
    total_written = 0
    shortfall = []
    for subcat in active_subs:
        rows = samplers[subcat].rows
        n = len(rows)
        total_written += n
        phases = Counter(r["phase"] for r in rows)
        phase_str = "  ".join(f"{p}:{c}" for p, c in sorted(phases.items()))
        marker = "" if n >= args.target else f"  ** need {args.target - n} more"
        print(f"  {subcat:<30} {n:>6} / {args.target:>6}  {phase_str}{marker}")
        if n < args.target:
            shortfall.append((subcat, args.target - n))

    print(f"\nTotal rows: {total_written}")
    print(f"Output: {out_path}")

    if shortfall:
        print(f"\n{len(shortfall)} subcategories below target.")
        print("Increase --max_games or add more PGN files.")


if __name__ == "__main__":
    main()
