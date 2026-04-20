"""Train a BPE tokenizer and save the learned merges to a JSON file.

Usage:
    python scripts/train_bpe.py \
        --input data/TinyStoriesV2-GPT4-train.txt \
        --output data/bpe_ts_10k.json \
        --num-merges 9743 \
        --special-tokens "<|endoftext|>"
"""

import argparse
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm_e2e.tokenizer import BPEOptimTokenizer

GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def parse_args():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    parser.add_argument("--input", required=True, help="Path to training text file.")
    parser.add_argument("--output", required=True, help="Path to save tokenizer JSON.")
    parser.add_argument("--num-merges", type=int, default=9743, help="Number of BPE merges.")
    parser.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"], help="Special tokens.")
    parser.add_argument("--pattern", default=GPT2_PATTERN, help="Regex pre-tokenization pattern.")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Input  : {args.input}")
    print(f"Output : {args.output}")
    print(f"Merges : {args.num_merges}")
    print(f"Special tokens: {args.special_tokens}")
    print()

    t0 = time.time()
    tokenizer = BPEOptimTokenizer.from_file(
        filepath=args.input,
        pattern=args.pattern,
        special_tokens=args.special_tokens,
        num_merges=args.num_merges,
    )
    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s | merges={len(tokenizer.bpe_merges)} | vocab_size={tokenizer.vocab_size}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    tokenizer.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
