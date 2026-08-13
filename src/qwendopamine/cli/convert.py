from __future__ import annotations

import argparse


def main() -> None:
    r"""Convert or inspect Hugging Face checkpoints into the research format."""
    parser = argparse.ArgumentParser(description="Convert or inspect HF checkpoints into the research format.")
    parser.add_argument("source", help="HF model name or path")
    parser.add_argument("target", help="Target directory for converted artifacts")
    args = parser.parse_args()
    print(f"Converting {args.source} -> {args.target}")


if __name__ == "__main__":
    main()
