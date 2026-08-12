"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence

from .chunkers import available_strategies
from .config import Config, ConfigError
from .indexer import ingest
from .loader import IngestError
from .models import Answer, SearchHit
from .pipeline import RAGPipeline
from .retriever import Retriever
from .store import StoreError, VectorStore, namespace_for


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
    _add_filter_args(p_search)

    p_ask = sub.add_parser("ask", help="retrieve, then answer with citations or refuse")
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k", type=int, default=None)
    p_ask.add_argument("--json", action="store_true")
    _add_filter_args(p_ask)

    sub.add_parser("stats", help="show what is in the index")

    p_eval = sub.add_parser("eval", help="run the full evaluation and write results")
    p_eval.add_argument("--questions", type=Path, default=None, help="path to questions.yaml")
    p_eval.add_argument(
        "--strategies",
        nargs="*",
        default=None,
        help="strategies to compare (default: baseline structure-aware)",
    )
    p_eval.add_argument("--out", type=Path, default=None, help="directory for the generated report")
    p_eval.add_argument("--skip-sweep", action="store_true", help="skip the chunk-size sweep")

    p_sweep = sub.add_parser("sweep", help="chunk-size sweep only")
    p_sweep.add_argument("--sizes", nargs="+", type=int, default=[400, 800, 1200, 1600])
    p_sweep.add_argument("--questions", type=Path, default=None)
    p_sweep.add_argument("--strategies", nargs="*", default=None)

    p_serve = sub.add_parser("serve", help="run the web UI")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=5000)
    p_serve.add_argument("--debug", action="store_true")

    return parser


def _add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--product-area", default=None, help="restrict retrieval to a product_area")
    parser.add_argument("--article-id", default=None, help="restrict retrieval to an article_id")


def filters_from_args(args: argparse.Namespace) -> dict:
    filters = {
        "product_area": getattr(args, "product_area", None),
        "article_id": getattr(args, "article_id", None),
    }
    return {key: value for key, value in filters.items() if value}


def load_config(args: argparse.Namespace) -> Config:
    config = Config.load(args.config)
    if args.chunk_size is not None or args.chunk_overlap is not None:
        config = config.with_chunking(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    return config


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Known failures become a one-line message, not a traceback."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _dispatch(parser, args)
    except (ConfigError, IngestError, StoreError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _dispatch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    config = load_config(args)
    strategy = args.strategy or config.chunking.default_strategy
    if strategy not in available_strategies():
        parser.error(f"unknown strategy {strategy!r}; available: {', '.join(available_strategies())}")

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
        filters = filters_from_args(args)
        hits = retriever.search(args.question, top_k=args.top_k or config.retrieval.top_k, filters=filters)
        if args.json:
            print(json.dumps([hit.to_dict() for hit in hits], indent=2))
        else:
            print(format_hits(args.question, retriever.namespace, hits, filters))
        return 0

    if args.command == "ask":
        pipeline = RAGPipeline.open(config, strategy=strategy, index_dir=args.index_dir)
        answer = pipeline.ask(args.question, top_k=args.top_k, filters=filters_from_args(args))
        if args.json:
            print(json.dumps(answer.to_dict(), indent=2))
        else:
            print(format_answer(answer))
        return 0

    if args.command == "eval":
        from .evaluation import run_evaluation

        report = run_evaluation(
            config,
            questions_path=args.questions,
            strategies=args.strategies,
            out_dir=args.out,
            index_dir=args.index_dir,
            with_sweep=not args.skip_sweep,
        )
        print(report.summary_text())
        return 0

    if args.command == "sweep":
        from .evaluation import load_questions, run_sweep

        questions = load_questions(args.questions or (config.repo_root / "eval" / "questions.yaml"))
        rows = run_sweep(
            config,
            questions,
            sizes=args.sizes,
            strategies=args.strategies or ["baseline", "structure-aware"],
            index_dir=args.index_dir,
        )
        print(json.dumps(rows, indent=2))
        return 0

    if args.command == "serve":
        from .webapp import create_app

        app = create_app(config, index_dir=args.index_dir)
        app.run(host=args.host, port=args.port, debug=args.debug)
        return 0

    parser.error(f"unknown command {args.command}")
    return 2


def format_hits(
    question: str,
    namespace: str,
    hits: Sequence[SearchHit],
    filters: dict | None = None,
) -> str:
    lines = [f"query: {question}", f"index: {namespace}", f"filters: {filters or '(none)'}", ""]
    if not hits:
        lines.append("(no matches)")
        return "\n".join(lines)
    for hit in hits:
        chunk = hit.chunk
        preview = " ".join(chunk.text.split())
        lines.append(
            f"#{hit.rank}  score={hit.score:.4f}  {chunk.chunk_id}\n"
            f"    article={chunk.article_id}  file={chunk.source_file}\n"
            f"    product_area={chunk.product_area}  last_updated={chunk.last_updated}\n"
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
