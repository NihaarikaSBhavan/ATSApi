from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ATSConfig
from .engine import ATSEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ATSv4 resume scorer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a resume against a job description")
    evaluate_parser.add_argument("--resume", required=True, help="Path to resume file")
    evaluate_parser.add_argument("--job", required=True, help="Path to job description file")
    evaluate_parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    evaluate_parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable LLM job structuring and candidate evaluation",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "evaluate":
        config = ATSConfig(embedding_model_name=args.embedding_model)
        engine = ATSEngine(config)
        result = engine.evaluate(
            resume_path=Path(args.resume),
            job_path=Path(args.job),
            use_llm=args.use_llm,
        )
        print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
