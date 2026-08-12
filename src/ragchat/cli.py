"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence

from .chunkers import available_strategies
from .config import Config
from .indexer import ingest
from .models import Answer, SearchHit
from .pipeline import RAGPipeline
from .retriever import Retriever
from .store import VectorStore, namespace_for


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragchat", description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="path to config.yaml")
    parser.add_argument("--index-dir", type=Path, default=None, help="override the index directory")
    parser.add_argument(
        "--strategy",
        default=None,
        help=f"chunking strategy ({', '.join(available_strategies())})",
    )
    parser.add_argument("--chunk-size", type=int, default=None, help="override chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="override chunk overlap in characters")

    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="chunk, embed and append documents to the index")
    p_ingest.add_argument("paths", nargs="*", type=Path, help="files or directories (default: the new articles dir)")
    p_ingest.add_argument("--label", default="ingest", help="label recorded in the index history")

    p_search = sub.add_parser("search", help="search only: no generation")
    p_search.add_argument("question")
    p_search.add_argument("--top-k", type=int, default=None)
    p_search.add_argument("--json", action="store_true", help="emit JSON instead of text")

    p_ask = sub.add_parser("ask", help="retrieve, then answer with citations or refuse")
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k", type=int, default=None)
    p_ask.add_argument("--json", action="store_true")

    sub.add_parser("stats", help="show what is in the index")

    return parser


def load_config(args: argparse.Namespace) -> Config:
    config = Config.load(args.config)
    if args.chunk_size is not None or args.chunk_overlap is not None:
        config = config.with_chunking(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    return config


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args)
    strategy = args.strategy or config.chunking.default_strategy

    if args.command == "ingest":
        paths: List[Path] = list(args.paths) or [config.paths.articles_dir]
        report = ingest(config, paths, strategy=strategy, label=args.label, index_dir=args.index_dir)
        print(json.dumps(report.to_dict(), indent=2))
        return 0

    if args.command == "stats":
        namespace = namespace_for(strategy, config.chunking.chunk_size, config.chunking.chunk_overlap)
        store = VectorStore.load(args.index_dir or config.paths.index_dir, namespace)
        print(json.dumps(store.stats(), indent=2))
        return 0

    if args.command == "search":
        retriever = Retriever.open(config, strategy=strategy, index_dir=args.index_dir)
        hits = retriever.search(args.question, top_k=args.top_k or config.retrieval.top_k)
        if args.json:
            print(json.dumps([hit.to_dict() for hit in hits], indent=2))
        else:
            print(format_hits(args.question, retriever.namespace, hits))
        return 0

    if args.command == "ask":
        pipeline = RAGPipeline.open(config, strategy=strategy, index_dir=args.index_dir)
        answer = pipeline.ask(args.question, top_k=args.top_k)
        if args.json:
            print(json.dumps(answer.to_dict(), indent=2))
        else:
            print(format_answer(answer))
        return 0

    parser.error(f"unknown command {args.command}")
    return 2


def format_hits(question: str, namespace: str, hits: Sequence[SearchHit]) -> str:
    lines = [f"query: {question}", f"index: {namespace}", ""]
    if not hits:
        lines.append("(no matches)")
        return "\n".join(lines)
    for hit in hits:
        chunk = hit.chunk
        preview = " ".join(chunk.text.split())
        lines.append(
            f"#{hit.rank}  score={hit.score:.4f}  {chunk.chunk_id}\n"
            f"    article={chunk.article_id}  file={chunk.source_file}\n"
            f"    section={chunk.section or '(none)'}\n"
            f"    {preview[:240]}{'…' if len(preview) > 240 else ''}"
        )
    return "\n".join(lines)


def format_answer(answer: Answer) -> str:
    lines = [f"Q: {answer.question}", ""]
    if answer.refused:
        lines += [f"REFUSED: {answer.text}", f"reason: {answer.refusal_reason}"]
    else:
        lines += [answer.text, "", "Sources:"]
        for citation in answer.citations:
            lines.append(
                f"  [{citation.chunk_id}] {citation.article_id} · {citation.source_file} · {citation.section}"
            )
    lines += ["", f"backend: {answer.backend}"]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
