"""Position detectors, illegal-move builders, and extraction pipeline.

Each detect_* function identifies a position category (pin, check, etc.)
and each build_* function generates typed illegal move distractors as
(uci, type) pairs.

Depends on legal_moves.py for CASTLE_INFO, get_phase, iter_games.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import chess

from legal_moves import CASTLE_INFO, get_phase, iter_games

# ── Category 1: En Passant ───────────────────────────────────────────────────


def detect_en_passant(board: chess.Board) -> Optional[dict]:
    """If en passant is legal, return info about it."""
    if board.ep_square is None:
        return None
    ep_moves = [m for m in board.legal_moves if board.is_en_passant(m)]
    if not ep_moves:
        return None
    return {"ep_moves": ep_moves, "ep_square": board.ep_square}


def build_en_passant_illegals(board: chess.Board, info: dict) -> List[Tuple[str, str]]:
    """Category-specific illegal moves for en passant positions.

    Distractors: pawn diagonal moves to empty squares that aren't the ep square.
    Sub-typed as ep_wrong_pawn (adjacent enemy pawn that didn't just push) or
    ep_fake_diagonal (no adjacent enemy pawn at all).
    """
    legal_ucis = set(m.uci() for m in board.legal_moves)
    opp = not board.turn
    illegal = []
    for sq in board.pieces(chess.PAWN, board.turn):
        for target in board.attacks(sq):
            if board.piece_at(target) is None and target != board.ep_square:
                uci = chess.Move(sq, target).uci()
                if uci not in legal_ucis:
                    adjacent_sq = chess.square(chess.square_file(target), chess.square_rank(sq))
                    adj_piece = board.piece_at(adjacent_sq)
                    if adj_piece and adj_piece.piece_type == chess.PAWN and adj_piece.color == opp:
                        illegal.append((uci, "ep_wrong_pawn"))
                    else:
                        illegal.append((uci, "ep_fake_diagonal"))
    return illegal


# ── Category 2: Check Evasion (single check) ────────────────────────────────


@dataclass
class CheckInfo:
    king_moves: List[chess.Move]
    captures: List[chess.Move]
    blocks: List[chess.Move]
    illegal_king_moves: List[chess.Move]
    illegal_castling: List[chess.Move]
    num_checkers: int

    @property
    def evasion_types(self) -> int:
        return (
            int(len(self.king_moves) > 0)
            + int(len(self.captures) > 0)
            + int(len(self.blocks) > 0)
        )


def _find_castling_in_check(board: chess.Board) -> List[chess.Move]:
    """Castling moves illegal because the king is in check.

    Requires: castling rights exist, rook on square, path clear.
    """
    turn = board.turn
    illegal = []
    for info in CASTLE_INFO[turn]:
        if not getattr(board, info["rights_fn"])(turn):
            continue
        rook = board.piece_at(info["rook_sq"])
        if rook is None or rook.piece_type != chess.ROOK or rook.color != turn:
            continue
        if any(board.piece_at(sq) is not None for sq in info["clear_sqs"]):
            continue
        illegal.append(chess.Move(info["king_from"], info["king_to"]))
    return illegal


def analyze_check(board: chess.Board) -> Optional[CheckInfo]:
    """Detailed analysis of a single-check position."""
    if not board.is_check():
        return None

    king_sq = board.king(board.turn)
    checkers = list(board.checkers())

    block_squares: Set[int] = set()
    for att_sq in checkers:
        piece = board.piece_at(att_sq)
        if piece and piece.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            block_squares |= set(chess.SquareSet.between(king_sq, att_sq))

    king_moves, captures, blocks = [], [], []
    for move in board.legal_moves:
        mover = board.piece_at(move.from_square)
        if mover and mover.piece_type == chess.KING:
            king_moves.append(move)
        elif move.to_square in [s for s in checkers]:
            captures.append(move)
        elif move.to_square in block_squares and mover and mover.piece_type != chess.KING:
            blocks.append(move)

    illegal_king_moves = []
    for dest in chess.SquareSet(chess.BB_KING_ATTACKS[king_sq]):
        own_piece = board.piece_at(dest)
        if own_piece and own_piece.color == board.turn:
            continue
        m = chess.Move(king_sq, dest)
        if not board.is_pseudo_legal(m):
            continue
        if m not in board.legal_moves and board.is_attacked_by(not board.turn, dest):
            illegal_king_moves.append(m)

    illegal_castling = _find_castling_in_check(board)

    return CheckInfo(
        king_moves=king_moves,
        captures=captures,
        blocks=blocks,
        illegal_king_moves=illegal_king_moves,
        illegal_castling=illegal_castling,
        num_checkers=len(checkers),
    )


def build_check_candidates(board: chess.Board, info: CheckInfo) -> List[Tuple[str, str]]:
    """Category-specific illegal moves for single check positions."""
    return (
        [(m.uci(), "king_to_attacked") for m in info.illegal_king_moves]
        + [(m.uci(), "castling_in_check") for m in info.illegal_castling]
    )


# ── Category 3: Double Check ────────────────────────────────────────────────


@dataclass
class DoubleCheckInfo:
    checker_squares: List[int]
    legal_king_moves: List[chess.Move]
    illegal_king_moves: List[chess.Move]
    illegal_non_king: List[chess.Move]
    illegal_castling: List[chess.Move]


def detect_double_check(board: chess.Board) -> Optional[DoubleCheckInfo]:
    """Detect double check: 2+ pieces giving check simultaneously."""
    if not board.is_check():
        return None
    checkers = list(board.checkers())
    if len(checkers) < 2:
        return None

    king_sq = board.king(board.turn)
    legal_king_moves = list(board.legal_moves)

    illegal_king_moves = []
    for dest in chess.SquareSet(chess.BB_KING_ATTACKS[king_sq]):
        own_piece = board.piece_at(dest)
        if own_piece and own_piece.color == board.turn:
            continue
        m = chess.Move(king_sq, dest)
        if not board.is_pseudo_legal(m):
            continue
        if m not in board.legal_moves and board.is_attacked_by(not board.turn, dest):
            illegal_king_moves.append(m)

    block_or_capture_squares = set(checkers)
    for att_sq in checkers:
        piece = board.piece_at(att_sq)
        if piece and piece.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            block_or_capture_squares |= set(chess.SquareSet.between(king_sq, att_sq))

    illegal_non_king = []
    for m in board.pseudo_legal_moves:
        mover = board.piece_at(m.from_square)
        if mover and mover.piece_type == chess.KING:
            continue
        if m.to_square in block_or_capture_squares:
            if m not in board.legal_moves:
                illegal_non_king.append(m)

    illegal_castling = _find_castling_in_check(board)

    return DoubleCheckInfo(
        checker_squares=checkers,
        legal_king_moves=legal_king_moves,
        illegal_king_moves=illegal_king_moves,
        illegal_non_king=illegal_non_king,
        illegal_castling=illegal_castling,
    )


def build_double_check_illegals(board: chess.Board, info: DoubleCheckInfo) -> List[Tuple[str, str]]:
    """Category-specific illegal moves for double check positions."""
    return (
        [(m.uci(), "king_to_attacked") for m in info.illegal_king_moves]
        + [(m.uci(), "non_king_double_check") for m in info.illegal_non_king]
        + [(m.uci(), "castling_in_check") for m in info.illegal_castling]
    )


# ── Category 4: Illegal King Moves + Illegal Castling ───────────────────────


def _find_illegal_castling(board: chess.Board) -> List[chess.Move]:
    """Castling illegal because the king passes through/lands on attacked square."""
    turn = board.turn
    opp = not turn
    illegal = []
    for info in CASTLE_INFO[turn]:
        if not getattr(board, info["rights_fn"])(turn):
            continue
        rook = board.piece_at(info["rook_sq"])
        if rook is None or rook.piece_type != chess.ROOK or rook.color != turn:
            continue
        if any(board.piece_at(sq) is not None for sq in info["clear_sqs"]):
            continue
        castle_move = chess.Move(info["king_from"], info["king_to"])
        if castle_move in board.legal_moves:
            continue
        if any(board.is_attacked_by(opp, sq) for sq in info["safe_sqs"]):
            illegal.append(castle_move)
    return illegal


def detect_illegal_king_moves(board: chess.Board) -> Optional[dict]:
    """Find positions (not in check) with king pseudo-legal to attacked squares."""
    if board.is_check():
        return None
    king_sq = board.king(board.turn)
    if king_sq is None:
        return None

    legal_king, illegal_king = [], []
    for dest in chess.SquareSet(chess.BB_KING_ATTACKS[king_sq]):
        own_piece = board.piece_at(dest)
        if own_piece and own_piece.color == board.turn:
            continue
        m = chess.Move(king_sq, dest)
        if not board.is_pseudo_legal(m):
            continue
        if m in board.legal_moves:
            legal_king.append(m)
        elif board.is_attacked_by(not board.turn, dest):
            illegal_king.append(m)

    illegal_castling = _find_illegal_castling(board)

    if not (illegal_king or illegal_castling):
        return None
    has_legal_king = len(legal_king) > 0 or any(
        board.is_castling(m) for m in board.legal_moves
    )
    if not has_legal_king:
        return None

    return {
        "legal_king": legal_king,
        "illegal_king": illegal_king,
        "illegal_castling": illegal_castling,
    }


def build_illegal_king_illegals(board: chess.Board, info: dict) -> List[Tuple[str, str]]:
    """Category-specific illegal moves for illegal king positions."""
    return (
        [(m.uci(), "king_to_attacked") for m in info["illegal_king"]]
        + [(m.uci(), "castling_through_attacked") for m in info["illegal_castling"]]
    )


# ── Category 5: Pin Against King ────────────────────────────────────────────


def detect_pin(board: chess.Board) -> Optional[dict]:
    """Find positions where the side to move has a piece pinned to its own king."""
    pinned_pieces = []
    pinned_illegal_moves = []

    for sq, piece in board.piece_map().items():
        if piece.color != board.turn or piece.piece_type == chess.KING:
            continue
        if not board.is_pinned(board.turn, sq):
            continue
        pin_mask = board.pin(board.turn, sq)
        illegal_for_piece = []
        for m in board.pseudo_legal_moves:
            if m.from_square != sq:
                continue
            if m.to_square not in pin_mask and m not in board.legal_moves:
                illegal_for_piece.append(m)
        if illegal_for_piece:
            pinned_pieces.append({
                "square": sq, "piece": piece, "illegal_moves": illegal_for_piece,
            })
            pinned_illegal_moves.extend(illegal_for_piece)

    if not pinned_pieces:
        return None
    return {"pinned_pieces": pinned_pieces, "illegal_moves": pinned_illegal_moves}


def build_pin_illegals(board: chess.Board, info: dict) -> List[Tuple[str, str]]:
    """Category-specific illegal moves for pin positions."""
    return [(m.uci(), "pin_breaking") for m in info["illegal_moves"]]


# ── Category 6: Promotion ───────────────────────────────────────────────────

PROMO_PIECES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]


def detect_promotion(board: chess.Board) -> Optional[dict]:
    """Detect positions where a pawn can promote this move."""
    promo_moves = [m for m in board.legal_moves if m.promotion is not None]
    if not promo_moves:
        return None
    return {"promo_moves": promo_moves}


def build_promotion_illegals(board: chess.Board, info: dict) -> List[Tuple[str, str]]:
    """Category-specific illegal moves for promotion positions.

    Distractors: promotion push onto occupied square, promotion capture to empty.
    """
    legal_ucis = set(m.uci() for m in board.legal_moves)
    illegal = []
    turn = board.turn
    promo_rank = 6 if turn == chess.WHITE else 1

    for sq in board.pieces(chess.PAWN, turn):
        if chess.square_rank(sq) != promo_rank:
            continue
        file = chess.square_file(sq)
        dest_rank = 7 if turn == chess.WHITE else 0

        push_dest = chess.square(file, dest_rank)
        if board.piece_at(push_dest) is not None:
            for promo in PROMO_PIECES:
                uci = chess.Move(sq, push_dest, promotion=promo).uci()
                if uci not in legal_ucis:
                    illegal.append((uci, "promo_push_blocked"))

        for df in [-1, 1]:
            cap_file = file + df
            if not (0 <= cap_file <= 7):
                continue
            cap_dest = chess.square(cap_file, dest_rank)
            if board.piece_at(cap_dest) is None:
                for promo in PROMO_PIECES:
                    uci = chess.Move(sq, cap_dest, promotion=promo).uci()
                    if uci not in legal_ucis:
                        illegal.append((uci, "promo_capture_empty"))
    return illegal


# ── General Illegal Move Generator ───────────────────────────────────────────

PIECE_NAMES = {
    chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B",
    chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K",
}


def _on_board(sq: int) -> bool:
    return 0 <= sq <= 63


def _gen_backward_pawn(board: chess.Board, turn: chess.Color) -> List[Tuple[chess.Move, str]]:
    results = []
    direction = -8 if turn == chess.WHITE else 8
    for sq in board.pieces(chess.PAWN, turn):
        dest = sq + direction
        if _on_board(dest) and board.piece_at(dest) is None:
            results.append((chess.Move(sq, dest), "backward_pawn"))
    return results


def _gen_friendly_fire(board: chess.Board, turn: chess.Color) -> List[Tuple[chess.Move, str]]:
    results = []
    for sq, piece in board.piece_map().items():
        if piece.color != turn or piece.piece_type == chess.PAWN:
            continue
        for target in board.attacks(sq):
            target_piece = board.piece_at(target)
            if target_piece and target_piece.color == turn:
                results.append((chess.Move(sq, target), "friendly_fire"))
    return results


def _gen_blocked_sliding(board: chess.Board, turn: chess.Color) -> List[Tuple[chess.Move, str]]:
    results = []
    for sq, piece in board.piece_map().items():
        if piece.color != turn:
            continue
        if piece.piece_type not in (chess.ROOK, chess.BISHOP, chess.QUEEN):
            continue
        empty = chess.Board.empty()
        empty.set_piece_at(sq, piece)
        empty_attacks = set(empty.attacks(sq))
        real_attacks = set(board.attacks(sq))
        blocked = empty_attacks - real_attacks
        for dest in blocked:
            dest_piece = board.piece_at(dest)
            if dest_piece and dest_piece.color == turn:
                continue
            results.append((chess.Move(sq, dest), "blocked_sliding"))
    return results


def _gen_pawn_double_push_wrong_rank(board: chess.Board, turn: chess.Color) -> List[Tuple[chess.Move, str]]:
    results = []
    direction = 16 if turn == chess.WHITE else -16
    start_rank = 1 if turn == chess.WHITE else 6
    for sq in board.pieces(chess.PAWN, turn):
        if chess.square_rank(sq) == start_rank:
            continue
        dest = sq + direction
        if _on_board(dest) and board.piece_at(dest) is None:
            mid = sq + (8 if turn == chess.WHITE else -8)
            if _on_board(mid) and board.piece_at(mid) is None:
                results.append((chess.Move(sq, dest), "pawn_double_wrong_rank"))
    return results


def _gen_pawn_diagonal_to_empty(board: chess.Board, turn: chess.Color) -> List[Tuple[chess.Move, str]]:
    """Pawn moves diagonally to an empty square (no capture, not en passant)."""
    results = []
    direction = 1 if turn == chess.WHITE else -1
    promo_rank = 7 if turn == chess.WHITE else 0
    for sq in board.pieces(chess.PAWN, turn):
        rank, file = chess.square_rank(sq), chess.square_file(sq)
        dest_rank = rank + direction
        if not (0 <= dest_rank <= 7):
            continue
        # Skip promotion rank (handled by promo_capture_empty)
        if dest_rank == promo_rank:
            continue
        for df in [-1, 1]:
            dest_file = file + df
            if not (0 <= dest_file <= 7):
                continue
            dest = chess.square(dest_file, dest_rank)
            if board.piece_at(dest) is None and dest != board.ep_square:
                results.append((chess.Move(sq, dest), "pawn_diagonal_to_empty"))
    return results


def _gen_pawn_capture_friendly(board: chess.Board, turn: chess.Color) -> List[Tuple[chess.Move, str]]:
    """Pawn captures own piece diagonally."""
    results = []
    direction = 1 if turn == chess.WHITE else -1
    for sq in board.pieces(chess.PAWN, turn):
        rank, file = chess.square_rank(sq), chess.square_file(sq)
        dest_rank = rank + direction
        if not (0 <= dest_rank <= 7):
            continue
        for df in [-1, 1]:
            dest_file = file + df
            if not (0 <= dest_file <= 7):
                continue
            dest = chess.square(dest_file, dest_rank)
            target = board.piece_at(dest)
            if target and target.color == turn:
                results.append((chess.Move(sq, dest), "pawn_capture_friendly"))
    return results


def _gen_pawn_push_onto_piece(board: chess.Board, turn: chess.Color) -> List[Tuple[chess.Move, str]]:
    results = []
    direction = 8 if turn == chess.WHITE else -8
    for sq in board.pieces(chess.PAWN, turn):
        dest = sq + direction
        if _on_board(dest):
            dest_piece = board.piece_at(dest)
            if dest_piece and dest_piece.color != turn:
                results.append((chess.Move(sq, dest), "pawn_push_onto_piece"))
    return results


def _gen_wrong_geometry(board: chess.Board, turn: chess.Color) -> List[Tuple[chess.Move, str]]:
    results = []
    for sq, piece in board.piece_map().items():
        if piece.color != turn:
            continue
        rank, file = chess.square_rank(sq), chess.square_file(sq)

        if piece.piece_type == chess.KNIGHT:
            for dr, df in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nr, nf = rank + dr, file + df
                if 0 <= nr <= 7 and 0 <= nf <= 7:
                    dest = chess.square(nf, nr)
                    dp = board.piece_at(dest)
                    if dp is None or dp.color != turn:
                        results.append((chess.Move(sq, dest), "wrong_geometry_knight"))

        elif piece.piece_type == chess.BISHOP:
            for dr, df in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
                nr, nf = rank + dr, file + df
                if 0 <= nr <= 7 and 0 <= nf <= 7:
                    dest = chess.square(nf, nr)
                    dp = board.piece_at(dest)
                    if dp is None or dp.color != turn:
                        results.append((chess.Move(sq, dest), "wrong_geometry_bishop"))

        elif piece.piece_type == chess.ROOK:
            for dr, df in [(2, 2), (2, -2), (-2, 2), (-2, -2)]:
                nr, nf = rank + dr, file + df
                if 0 <= nr <= 7 and 0 <= nf <= 7:
                    dest = chess.square(nf, nr)
                    dp = board.piece_at(dest)
                    if dp is None or dp.color != turn:
                        results.append((chess.Move(sq, dest), "wrong_geometry_rook"))
    return results


def generate_general_distractors(
    board: chess.Board,
    legal_ucis: Set[str],
    rng: random.Random,
    num_target: int = 5,
) -> List[Tuple[str, str]]:
    """Generate diverse illegal distractors. Returns list of (uci, type) pairs."""
    turn = board.turn
    all_candidates: List[Tuple[chess.Move, str]] = []
    all_candidates += _gen_backward_pawn(board, turn)
    all_candidates += _gen_friendly_fire(board, turn)
    all_candidates += _gen_blocked_sliding(board, turn)
    all_candidates += _gen_pawn_double_push_wrong_rank(board, turn)
    all_candidates += _gen_pawn_push_onto_piece(board, turn)
    all_candidates += _gen_pawn_diagonal_to_empty(board, turn)
    all_candidates += _gen_pawn_capture_friendly(board, turn)
    all_candidates += _gen_wrong_geometry(board, turn)

    seen = set(legal_ucis)
    filtered = []
    for move, mtype in all_candidates:
        uci = move.uci()
        if uci not in seen:
            seen.add(uci)
            filtered.append((uci, mtype))

    by_type = defaultdict(list)
    for uci, mtype in filtered:
        by_type[mtype].append(uci)

    selected = []
    types_available = list(by_type.keys())
    rng.shuffle(types_available)

    for t in types_available:
        if len(selected) >= num_target:
            break
        choice = rng.choice(by_type[t])
        selected.append((choice, t))
        by_type[t].remove(choice)

    remaining = [(u, t) for t, ulist in by_type.items() for u in ulist]
    rng.shuffle(remaining)
    for u, t in remaining:
        if len(selected) >= num_target:
            break
        selected.append((u, t))

    return selected


# ── Row construction & extraction pipeline ───────────────────────────────────


def make_row(
    board: chess.Board,
    last_move: chess.Move,
    game_move: chess.Move,
    tags: List[str],
    legal_uci: List[str],
    cat_illegal: List[dict],
    gen_illegal: List[dict],
    game,
    ply: int,
    extra: Optional[dict] = None,
) -> dict:
    """Build one output record.

    cat_illegal / gen_illegal are lists of {"uci": str, "type": str}.
    """
    cat_ucis = [d["uci"] for d in cat_illegal]
    gen_ucis = [d["uci"] for d in gen_illegal]
    candidates = list(set(legal_uci + cat_ucis + gen_ucis))
    row = {
        "fen": board.fen(),
        "last_move_uci": last_move.uci(),
        "game_move_uci": game_move.uci(),
        "next_move_candidates_uci": candidates,
        "correct_outputs_uci": legal_uci,
        "illegal_category": cat_illegal,
        "illegal_general": gen_illegal,
        "tags": tags,
        "phase": get_phase(board),
        "game_id": game.headers.get("Site", ""),
        "ply": ply,
        "num_candidates": len(candidates),
        "num_correct": len(legal_uci),
        "num_illegal_category": len(cat_illegal),
        "num_illegal_general": len(gen_illegal),
    }
    if extra:
        row.update(extra)
    return row


def extract_all(
    pgn_path: str,
    max_games: int,
    num_general_distractors: int = 5,
    num_vanilla_positions: int = 100,
    rng: Optional[random.Random] = None,
) -> List[dict]:
    """Scan games and extract tagged + vanilla positions with distractors."""
    if rng is None:
        rng = random.Random(42)

    rows = []
    tag_counts = defaultdict(int)
    vanilla_candidates = []

    for gi, game in enumerate(iter_games(pgn_path, max_games)):
        board = game.board()
        last_move = None
        ply = 0

        for game_move in game.mainline_moves():
            if last_move is not None:
                tags = []
                cat_illegal_pairs: List[Tuple[str, str]] = []
                extra = {}

                # ── Cat 1: En passant ──
                ep_info = detect_en_passant(board)
                if ep_info:
                    tags.append("en_passant")
                    cat_illegal_pairs += build_en_passant_illegals(board, ep_info)
                    extra["ep_moves_uci"] = [m.uci() for m in ep_info["ep_moves"]]

                # ── Cat 3: Double check (before single check) ──
                dc_info = detect_double_check(board)
                if dc_info:
                    tags.append("double_check")
                    cat_illegal_pairs += build_double_check_illegals(board, dc_info)
                    extra["num_checkers"] = len(dc_info.checker_squares)
                    extra["checker_squares"] = [chess.square_name(s) for s in dc_info.checker_squares]
                    extra["checker_pieces"] = [board.piece_at(s).symbol() for s in dc_info.checker_squares]
                    extra["num_legal_king_moves"] = len(dc_info.legal_king_moves)

                # ── Cat 2: Single check (only if NOT double check) ──
                elif board.is_check():
                    check_info = analyze_check(board)
                    if check_info and check_info.evasion_types >= 2:
                        tags.append("check")
                        cat_illegal_pairs += build_check_candidates(board, check_info)
                        extra["check_king_moves"] = len(check_info.king_moves)
                        extra["check_captures"] = len(check_info.captures)
                        extra["check_blocks"] = len(check_info.blocks)
                        extra["check_illegal_king"] = len(check_info.illegal_king_moves)
                        extra["check_illegal_castling"] = len(check_info.illegal_castling)

                # ── Cat 4: Illegal king moves + castling (not in check) ──
                if not board.is_check():
                    ik_info = detect_illegal_king_moves(board)
                    if ik_info:
                        tags.append("illegal_king")
                        cat_illegal_pairs += build_illegal_king_illegals(board, ik_info)
                        extra["num_legal_king_moves"] = len(ik_info["legal_king"])
                        extra["num_illegal_king_moves"] = len(ik_info["illegal_king"])
                        extra["num_illegal_castling"] = len(ik_info["illegal_castling"])
                        if ik_info["illegal_castling"]:
                            extra["illegal_castling_uci"] = [
                                m.uci() for m in ik_info["illegal_castling"]
                            ]

                # ── Cat 5: Pin ──
                pin_info = detect_pin(board)
                if pin_info:
                    tags.append("pin")
                    cat_illegal_pairs += build_pin_illegals(board, pin_info)
                    extra["num_pinned_pieces"] = len(pin_info["pinned_pieces"])
                    extra["num_pin_illegal_moves"] = len(pin_info["illegal_moves"])
                    extra["pinned_details"] = [
                        {
                            "square": chess.square_name(p["square"]),
                            "piece": p["piece"].symbol(),
                            "num_illegal": len(p["illegal_moves"]),
                        }
                        for p in pin_info["pinned_pieces"]
                    ]

                # ── Cat 6: Promotion ──
                promo_info = detect_promotion(board)
                if promo_info:
                    tags.append("promotion")
                    cat_illegal_pairs += build_promotion_illegals(board, promo_info)
                    extra["promo_moves_uci"] = [m.uci() for m in promo_info["promo_moves"]]

                legal_uci = [m.uci() for m in board.legal_moves]

                if tags:
                    legal_set = set(legal_uci)
                    seen_ucis: Set[str] = set()
                    deduped_pairs = []
                    for uci, mtype in cat_illegal_pairs:
                        if uci not in legal_set and uci not in seen_ucis:
                            seen_ucis.add(uci)
                            deduped_pairs.append((uci, mtype))

                    cat_illegal_dicts = [{"uci": u, "type": t} for u, t in deduped_pairs]
                    cat_ucis = set(u for u, _ in deduped_pairs)

                    existing = legal_set | cat_ucis
                    gen_distractors = generate_general_distractors(
                        board, legal_ucis=existing, rng=rng,
                        num_target=num_general_distractors,
                    )
                    gen_illegal_dicts = [{"uci": u, "type": t} for u, t in gen_distractors]

                    row = make_row(
                        board, last_move, game_move, tags,
                        legal_uci, cat_illegal_dicts, gen_illegal_dicts,
                        game, ply, extra,
                    )
                    rows.append(row)
                    for t in tags:
                        tag_counts[t] += 1
                else:
                    vanilla_candidates.append(
                        (board.fen(), last_move.uci(), game_move.uci(),
                         game.headers.get("Site", ""), ply)
                    )

            board.push(game_move)
            last_move = game_move
            ply += 1

        if (gi + 1) % 10 == 0:
            print(f"  Processed {gi+1} games, {len(rows)} tagged positions so far...")

    # ── Vanilla positions ──
    rng.shuffle(vanilla_candidates)
    num_vanilla = min(num_vanilla_positions, len(vanilla_candidates))
    print(f"\nSampling {num_vanilla} vanilla positions from {len(vanilla_candidates)} candidates...")

    for fen, lm_uci, gm_uci, game_id, v_ply in vanilla_candidates[:num_vanilla]:
        vboard = chess.Board(fen)
        legal_uci = [m.uci() for m in vboard.legal_moves]
        gen_distractors = generate_general_distractors(
            vboard, legal_ucis=set(legal_uci), rng=rng,
            num_target=num_general_distractors,
        )
        gen_illegal_dicts = [{"uci": u, "type": t} for u, t in gen_distractors]
        gen_ucis = [d["uci"] for d in gen_illegal_dicts]

        candidates = list(set(legal_uci + gen_ucis))
        row = {
            "fen": fen,
            "last_move_uci": lm_uci,
            "game_move_uci": gm_uci,
            "next_move_candidates_uci": candidates,
            "correct_outputs_uci": legal_uci,
            "illegal_category": [],
            "illegal_general": gen_illegal_dicts,
            "tags": ["vanilla"],
            "phase": get_phase(chess.Board(fen)),
            "game_id": game_id,
            "ply": v_ply,
            "num_candidates": len(candidates),
            "num_correct": len(legal_uci),
            "num_illegal_category": 0,
            "num_illegal_general": len(gen_illegal_dicts),
        }
        rows.append(row)
        tag_counts["vanilla"] += 1

    print(f"\nDone: {gi+1} games, {len(rows)} total positions")
    print(f"Tag counts: {dict(tag_counts)}")
    return rows
