#!/usr/bin/env python3
"""
Commentary generation CLI for chess positions.

Generates natural language explanations for chess moves using LLMs with optional
engine analysis and tool augmentation.

Usage:
    # Generate with engine analysis
    python generate_commentary.py --input positions.jsonl --output commentary.jsonl --use-engine

    # Generate with tools only (no engine)
    python generate_commentary.py --input positions.jsonl --output commentary.jsonl

    # Use Qwen model hosted locally
    python generate_commentary.py --input positions.jsonl --provider qwen \
        --model Qwen/Qwen3-32B --base-url http://localhost:8000/v1

    # Use Claude
    python generate_commentary.py --input positions.jsonl --provider anthropic \
        --model claude-sonnet-4-5-20250929
"""

import argparse
import json
import sys
import chess

from generation import generate_commentary, generate_commentary_raw
from eval_utils import get_engine_analysis


def generate_for_positions(input_path, output_path, provider="openai", model="gpt-4o",
                          use_engine=False, use_tools=True, ascii=False,
                          base_url=None, api_key=None):
    """
    Generate commentary for positions in a JSONL file.

    Args:
        input_path: Input JSONL file with positions
        output_path: Output JSONL file for results
        provider: LLM provider ('openai', 'anthropic', 'qwen')
        model: Model name
        use_engine: Use engine analysis in prompts
        use_tools: Enable tool calling
        ascii: Include ASCII board in prompts
        base_url: Base URL for OpenAI-compatible APIs (for Qwen)
        api_key: API key (optional)
    """
    with open(input_path) as f:
        positions = [json.loads(line) for line in f]

    print(f"Loaded {len(positions)} positions from {input_path}")
    print(f"Provider: {provider}, Model: {model}")
    print(f"Engine: {use_engine}, Tools: {use_tools}")

    with open(output_path, 'w') as out_f:
        for i, pos in enumerate(positions):
            print(f"\n[{i+1}/{len(positions)}] Processing position...")

            # Build entry
            board = chess.Board(pos['fen'])
            move_san = pos.get('move_san', board.san(chess.Move.from_uci(pos['move_uci'])))
            entry = {
                'fen': pos['fen'],
                'move_uci': pos['move_uci'],
                'move_san': move_san,
                'wp_loss': pos.get('wp_loss', 0),
                'annotation': pos.get('annotation', ''),
            }

            # Generate
            if use_engine:
                # Get engine lines if not already present
                engine_lines = pos.get('engine_lines')
                if engine_lines is None:
                    print(f"  Getting engine analysis...")
                    engine_lines = get_engine_analysis(pos['fen'], pos['move_uci'])

                text, tool_log = generate_commentary(
                    entry, engine_lines, provider=provider, model=model,
                    ascii=ascii, use_tools=use_tools, base_url=base_url, api_key=api_key)
            else:
                text, tool_log = generate_commentary_raw(
                    entry, provider=provider, model=model,
                    ascii=ascii, use_tools=use_tools, base_url=base_url, api_key=api_key)

            print(f"  Generated ({len(tool_log)} tool calls): {text[:80]}...")

            # Write result
            result = {
                **pos,
                'generated_commentary': text,
                'n_tool_calls': len(tool_log),
                'tool_log': tool_log,
                'gen_model': model,
                'gen_provider': provider,
                'used_engine': use_engine,
            }
            out_f.write(json.dumps(result) + '\n')
            out_f.flush()

    print(f"\n✓ Wrote {len(positions)} results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate chess move commentary')
    parser.add_argument('--input', required=True, help='Input JSONL file with positions')
    parser.add_argument('--output', required=True, help='Output JSONL file for results')
    parser.add_argument('--provider', choices=['openai', 'anthropic', 'qwen'], default='openai',
                       help='LLM provider')
    parser.add_argument('--model', default='gpt-4o', help='Model name')
    parser.add_argument('--use-engine', action='store_true',
                       help='Use engine analysis in prompts')
    parser.add_argument('--no-tools', action='store_true',
                       help='Disable tool calling')
    parser.add_argument('--ascii', action='store_true',
                       help='Include ASCII board in prompts')
    parser.add_argument('--base-url', help='Base URL for OpenAI-compatible API (for Qwen)')
    parser.add_argument('--api-key', help='API key (optional)')

    args = parser.parse_args()

    generate_for_positions(
        args.input,
        args.output,
        provider=args.provider,
        model=args.model,
        use_engine=args.use_engine,
        use_tools=not args.no_tools,
        ascii=args.ascii,
        base_url=args.base_url,
        api_key=args.api_key,
    )


if __name__ == '__main__':
    main()
