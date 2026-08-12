"""Minimal Flask UI: ask a question, see the answer, its sources, or the refusal."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template_string, request

from .chunkers import available_strategies
from .config import Config
from .pipeline import RAGPipeline

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Help-centre assistant</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; max-width: 60rem; margin: 2rem auto; padding: 0 1rem;
         line-height: 1.5; }
  form { display: grid; gap: .75rem; margin-bottom: 2rem; }
  .row { display: flex; gap: .75rem; flex-wrap: wrap; }
  input[type=text] { flex: 1 1 24rem; padding: .6rem; font-size: 1rem; }
  select, button { padding: .6rem; font-size: 1rem; }
  .answer { border-left: 4px solid #3a7; padding: .5rem 1rem; background: rgba(51,170,119,.08); }
  .refused { border-left: 4px solid #c53; padding: .5rem 1rem; background: rgba(204,85,51,.08); }
  .meta { font-size: .85rem; opacity: .8; }
  table { border-collapse: collapse; width: 100%; margin-top: 1rem; font-size: .85rem; }
  th, td { border: 1px solid rgba(128,128,128,.4); padding: .35rem .5rem; text-align: left;
           vertical-align: top; }
  code { font-size: .85em; }
</style>
</head>
<body>
<h1>Help-centre assistant</h1>
<p class="meta">Answers come only from the indexed articles. Anything the corpus does not cover is
refused rather than guessed.</p>

<form method="get">
  <input type="text" name="q" value="{{ question|e }}" placeholder="e.g. What does ERR-4032 mean?" autofocus>
  <div class="row">
    <select name="strategy">
      {% for s in strategies %}<option value="{{ s }}" {% if s == strategy %}selected{% endif %}>{{ s }}</option>{% endfor %}
    </select>
    <select name="product_area">
      <option value="">all product areas</option>
      {% for a in product_areas %}<option value="{{ a }}" {% if a == product_area %}selected{% endif %}>{{ a }}</option>{% endfor %}
    </select>
    <button type="submit">Ask</button>
  </div>
</form>

{% if answer %}
  {% if answer.refused %}
    <div class="refused">
      <strong>No answer given.</strong>
      <p>{{ answer.text }}</p>
      <p class="meta">Reason: {{ answer.refusal_reason }}</p>
    </div>
  {% else %}
    <div class="answer">
      {% for line in answer.text.split('\n') %}<p>{{ line }}</p>{% endfor %}
    </div>
    <h2>Sources</h2>
    <table>
      <tr><th>chunk_id</th><th>Article</th><th>Source file</th><th>Section</th><th>Product area</th></tr>
      {% for c in answer.citations %}
      <tr><td><code>{{ c.chunk_id }}</code></td><td>{{ c.article_id }}</td><td>{{ c.source_file }}</td>
          <td>{{ c.section }}</td><td>{{ areas.get(c.chunk_id, '') }}</td></tr>
      {% endfor %}
    </table>
  {% endif %}

  <h2>Retrieved chunks</h2>
  <table>
    <tr><th>#</th><th>Score</th><th>chunk_id</th><th>Product area</th><th>Section</th><th>Preview</th></tr>
    {% for h in answer.hits %}
    <tr><td>{{ h.rank }}</td><td>{{ '%.4f'|format(h.score) }}</td>
        <td><code>{{ h.chunk.chunk_id }}</code></td>
        <td>{{ h.chunk.metadata.get('product_area','') }}</td>
        <td>{{ h.chunk.section }}</td>
        <td>{{ h.chunk.text[:180] }}…</td></tr>
    {% endfor %}
  </table>
  <p class="meta">index: <code>{{ namespace }}</code> · backend: <code>{{ answer.backend }}</code></p>
{% endif %}
</body>
</html>
"""


def create_app(config: Config, index_dir: Path | None = None) -> Flask:
    app = Flask(__name__)
    pipelines: Dict[str, RAGPipeline] = {}

    def get_pipeline(strategy: str) -> RAGPipeline:
        if strategy not in pipelines:
            pipelines[strategy] = RAGPipeline.open(config, strategy=strategy, index_dir=index_dir)
        return pipelines[strategy]

    def answer_for(question: str, strategy: str, product_area: str):
        pipeline = get_pipeline(strategy)
        filters = {"product_area": product_area} if product_area else None
        return pipeline, pipeline.ask(question, filters=filters)

    @app.get("/")
    def home() -> str:
        question = (request.args.get("q") or "").strip()
        strategy = request.args.get("strategy") or config.chunking.default_strategy
        product_area = (request.args.get("product_area") or "").strip()

        answer = None
        namespace = ""
        areas: Dict[str, str] = {}
        pipeline = get_pipeline(strategy)
        if question:
            pipeline, answer = answer_for(question, strategy, product_area)
            namespace = pipeline.namespace
            areas = {h.chunk.chunk_id: h.chunk.product_area for h in answer.hits}

        return render_template_string(
            PAGE,
            question=question,
            strategy=strategy,
            product_area=product_area,
            strategies=available_strategies(),
            product_areas=pipeline.retriever.distinct_values("product_area"),
            answer=answer,
            namespace=namespace,
            areas=areas,
        )

    @app.get("/api/ask")
    def api_ask():
        question = (request.args.get("q") or "").strip()
        if not question:
            return jsonify({"error": "missing q parameter"}), 400
        strategy = request.args.get("strategy") or config.chunking.default_strategy
        product_area = (request.args.get("product_area") or "").strip()
        _pipeline, answer = answer_for(question, strategy, product_area)
        return jsonify(answer.to_dict())

    @app.get("/api/search")
    def api_search():
        question = (request.args.get("q") or "").strip()
        if not question:
            return jsonify({"error": "missing q parameter"}), 400
        strategy = request.args.get("strategy") or config.chunking.default_strategy
        product_area = (request.args.get("product_area") or "").strip()
        pipeline = get_pipeline(strategy)
        filters = {"product_area": product_area} if product_area else None
        hits = pipeline.search(question, filters=filters)
        return jsonify([hit.to_dict() for hit in hits])

    @app.get("/healthz")
    def healthz():
        pipeline = get_pipeline(config.chunking.default_strategy)
        return jsonify({"status": "ok", "namespace": pipeline.namespace,
                        "chunks": len(pipeline.retriever.store.chunks)})

    return app
