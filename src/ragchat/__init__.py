"""A retrieval-augmented help-centre assistant with forced refusal and citations."""

from .config import Config
from .indexer import ingest
from .models import Answer, Chunk, Citation, Document, SearchHit
from .pipeline import RAGPipeline
from .retriever import Retriever

__version__ = "1.0.0"

__all__ = [
    "Answer",
    "Chunk",
    "Citation",
    "Config",
    "Document",
    "RAGPipeline",
    "Retriever",
    "SearchHit",
    "ingest",
]
