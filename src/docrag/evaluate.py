"""Retrieval evaluation on TQuAD: does a chunk holding the gold answer come back in the top k?

    python -m docrag.evaluate                     # bm25, nomic-embed-text, bge-m3, hybrid × 500/1000/1500
    python -m docrag.evaluate --retrievers bge-m3 --chunk-sizes 1000

Corpus: every dev article plus every train article that shares no title and no paragraph with
dev (the 72 dev articles alone would make retrieval too easy). Queries: the dev questions.

A chunk is relevant only if it comes from the question's own article and contains the whole
gold answer at its annotated position, so a chunk that cuts the answer in half is a miss.
Search is exact cosine over all chunks (no HNSW), so the numbers measure chunking and
embeddings, not index approximation. Embeddings are cached in data/cache/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from .chunking import split_text

TQUAD = Path("data/raw/tquad")
CACHE = Path("data/cache")
OUT = Path("outputs/eval")
KS = (1, 3, 5, 10)


@dataclass
class Article:
    title: str
    text: str  # paragraphs with collapsed whitespace, joined by "\n"


@dataclass
class Question:
    qid: str
    question: str
    article: int
    start: int  # answer span in Article.text
    end: int


@dataclass
class Chunk:
    article: int
    start: int
    end: int
    text: str


def flatten(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def load_tquad(root: Path = TQUAD) -> tuple[list[Article], list[Question], dict]:
    dev = json.loads((root / "dev-v0.1.json").read_text(encoding="utf-8"))["data"]
    train = json.loads((root / "train-v0.1.json").read_text(encoding="utf-8"))["data"]
    dev_titles = {a["title"].strip().casefold() for a in dev}
    dev_paragraphs = {p["context"] for a in dev for p in a["paragraphs"]}

    articles: list[Article] = []
    questions: list[Question] = []
    excluded = unmatched = 0
    for split, source in (("dev", dev), ("train", train)):
        for art in source:
            if split == "train" and (
                art["title"].strip().casefold() in dev_titles
                or any(p["context"] in dev_paragraphs for p in art["paragraphs"])
            ):
                excluded += 1
                continue
            parts: list[str] = []
            offset = 0
            for p in art["paragraphs"]:
                context = p["context"]
                flat = flatten(context)
                lead = len(flat) - len(flat.lstrip())
                flat = flat.strip()
                if not flat:
                    continue
                for qa in p["qas"] if split == "dev" else []:
                    answer = qa["answers"][0]
                    raw = answer["text"]
                    s = int(answer["answer_start"]) + len(raw) - len(raw.lstrip())
                    text = flatten(raw).strip()
                    start = len(flatten(context[:s])) - lead
                    if not text or flat[start : start + len(text)] != text:
                        unmatched += 1
                        continue
                    questions.append(
                        Question(str(qa["id"]), qa["question"].strip(), len(articles),
                                 offset + start, offset + start + len(text))
                    )
                parts.append(flat)
                offset += len(flat) + 1
            articles.append(Article(art["title"].strip(), "\n".join(parts)))

    corpus = {
        "dev_articles": len(dev),
        "train_articles_kept": len(articles) - len(dev),
        "train_articles_excluded": excluded,
        "characters": sum(len(a.text) for a in articles),
        "questions": len(questions),
        "questions_unmatched": unmatched,
    }
    return articles, questions, corpus


def chunk_articles(articles: list[Article], size: int, overlap: int) -> tuple[list[Chunk], int]:
    """Chunk every article with the production chunker and locate each chunk in its article.

    split_text joins units with single spaces, so on whitespace-collapsed text a chunk is an
    exact substring once "\\n" is read as a space. The exception is a unit longer than `size`,
    which is cut hard and re-joined with a space; such chunks are anchored on their first words.
    """
    chunks: list[Chunk] = []
    approximate = 0
    for i, article in enumerate(articles):
        flat = article.text.replace("\n", " ")
        cursor = 0
        for piece in split_text(article.text, size, overlap):
            pos = flat.find(piece, cursor)
            if pos < 0:
                approximate += 1
                pos = flat.find(piece[:40], cursor)
                pos = cursor if pos < 0 else pos
            chunks.append(Chunk(i, pos, pos + len(piece), piece))
            cursor = pos + 1
    return chunks, approximate


def relevant_sets(chunks: list[Chunk], questions: list[Question]) -> list[set[int]]:
    by_article: dict[int, list[int]] = defaultdict(list)
    for idx, chunk in enumerate(chunks):
        by_article[chunk.article].append(idx)
    return [
        {i for i in by_article[q.article] if chunks[i].start <= q.start and q.end <= chunks[i].end}
        for q in questions
    ]


def tokenize(text: str) -> list[str]:
    # Turkish casing: İ → i and I → ı before lowercasing
    return re.findall(r"\w+", text.replace("İ", "i").replace("I", "ı").lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = np.zeros(len(docs))
        for i, doc in enumerate(docs):
            counts = Counter(tokenize(doc))
            self.lengths[i] = sum(counts.values())
            for term, tf in counts.items():
                self.postings[term].append((i, tf))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}
        self.norm = k1 * (1 - b + b * self.lengths / self.lengths.mean())

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.lengths))
        for term in set(tokenize(query)):
            for i, tf in self.postings.get(term, []):
                out[i] += self.idf[term] * tf * (self.k1 + 1) / (tf + self.norm[i])
        return out


def ollama_embed(model: str, texts: list[str], base_url: str, batch_size: int = 32) -> np.ndarray:
    key = hashlib.sha1("\x00".join([model, *texts]).encode()).hexdigest()[:16]
    path = CACHE / f"{model.replace(':', '_')}-{key}.npy"
    if path.exists():
        return np.load(path)
    vectors = []
    with httpx.Client(base_url=base_url, timeout=600) as client:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = client.post("/api/embed", json={"model": model, "input": batch})
            response.raise_for_status()
            vectors.extend(response.json()["embeddings"])
    matrix = np.asarray(vectors, dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)
    return matrix


def rrf(*score_matrices: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion: sum of 1 / (k + rank) over retrievers (Cormack et al., 2009)."""
    fused = np.zeros_like(score_matrices[0], dtype=float)
    for scores in score_matrices:
        ranks = np.empty_like(scores, dtype=float)
        order = np.argsort(-scores, axis=1)
        np.put_along_axis(ranks, order, np.arange(1, scores.shape[1] + 1, dtype=float)[None, :], axis=1)
        fused += 1.0 / (k + ranks)
    return fused


def score_chunks(retriever: str, chunks: list[Chunk], questions: list[Question], base_url: str) -> np.ndarray:
    """(questions × chunks) relevance scores. "hybrid" fuses BM25 and bge-m3 with RRF."""
    if retriever == "bm25":
        index = BM25([c.text for c in chunks])
        return np.stack([index.scores(q.question) for q in questions])
    if retriever == "hybrid":
        return rrf(score_chunks("bm25", chunks, questions, base_url),
                   score_chunks("bge-m3", chunks, questions, base_url))
    doc_vectors = ollama_embed(retriever, [c.text for c in chunks], base_url)
    query_vectors = ollama_embed(retriever, [q.question for q in questions], base_url)
    return query_vectors @ doc_vectors.T


def rank_chunks(retriever: str, chunks: list[Chunk], questions: list[Question], base_url: str,
                depth: int = max(KS)) -> np.ndarray:
    """Return a (questions × depth) array of chunk indices, best first."""
    scores = score_chunks(retriever, chunks, questions, base_url)
    top = np.argpartition(-scores, depth, axis=1)[:, :depth]
    order = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(top, order, axis=1)


def first_relevant_ranks(ranking: np.ndarray, relevant: list[set[int]]) -> np.ndarray:
    """1-based rank of the first relevant chunk in each row, 0 if none in the ranking."""
    ranks = np.zeros(len(relevant), dtype=int)
    for row, (indices, gold) in enumerate(zip(ranking, relevant, strict=True)):
        for position, idx in enumerate(indices, start=1):
            if idx in gold:
                ranks[row] = position
                break
    return ranks


def bootstrap_ci(values: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_difference(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> list[float]:
    """Mean of a − b over the same questions with a paired bootstrap 95 % CI: [diff, lo, hi]."""
    diff = a.astype(float) - b.astype(float)
    return [float(diff.mean()), *bootstrap_ci(diff, n, seed)]


def retrieval_metrics(ranks: np.ndarray, ks: tuple[int, ...] = KS) -> dict[str, float | list[float]]:
    out: dict[str, float | list[float]] = {}
    for k in ks:
        hits = ((ranks > 0) & (ranks <= k)).astype(float)
        out[f"recall@{k}"] = float(hits.mean())
        out[f"recall@{k}_ci"] = list(bootstrap_ci(hits))
    rr = np.where((ranks > 0) & (ranks <= max(ks)), 1.0 / np.maximum(ranks, 1), 0.0)
    out[f"mrr@{max(ks)}"] = float(rr.mean())
    out[f"mrr@{max(ks)}_ci"] = list(bootstrap_ci(rr))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retrievers", nargs="+", default=["bm25", "nomic-embed-text", "bge-m3", "hybrid"])
    parser.add_argument("--baseline", default="bm25", help="retriever for paired comparisons")
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[500, 1000, 1500])
    parser.add_argument("--overlap-frac", type=float, default=0.15)
    parser.add_argument("--ollama", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    articles, questions, corpus = load_tquad()
    print(f"corpus  {corpus}")

    runs, per_question = [], {"qid": [q.qid for q in questions]}
    for size in args.chunk_sizes:
        overlap = round(size * args.overlap_frac)
        chunks, approximate = chunk_articles(articles, size, overlap)
        relevant = relevant_sets(chunks, questions)
        answerable = float(np.mean([bool(r) for r in relevant]))
        baseline_ranks = None
        for retriever in args.retrievers:
            t0 = time.time()
            ranks = first_relevant_ranks(rank_chunks(retriever, chunks, questions, args.ollama), relevant)
            run = {
                "retriever": retriever, "chunk_size": size, "overlap": overlap, "chunks": len(chunks),
                "chunks_located_approximately": approximate, "answerable": answerable,
                **retrieval_metrics(ranks), "seconds": round(time.time() - t0, 1),
            }
            if retriever == args.baseline:
                baseline_ranks = ranks
            elif baseline_ranks is not None:
                for k in (1, 5):
                    run[f"recall@{k}_vs_{args.baseline}"] = paired_difference(
                        (ranks > 0) & (ranks <= k), (baseline_ranks > 0) & (baseline_ranks <= k))
            runs.append(run)
            per_question[f"{retriever}@{size}"] = ranks.tolist()
            print(f"{retriever:>17} size {size:>4}  chunks {len(chunks):>5}  answerable {answerable:.3f}  "
                  f"R@1 {run['recall@1']:.3f}  R@5 {run['recall@5']:.3f}  R@10 {run['recall@10']:.3f}  "
                  f"MRR@10 {run['mrr@10']:.3f}  ({run['seconds']} s)")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps({"corpus": corpus, "runs": runs}, indent=2))
    with (args.out / "per_question.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(per_question.keys())
        writer.writerows(zip(*per_question.values(), strict=True))
    print(f"wrote   {args.out / 'summary.json'}, {args.out / 'per_question.csv'}")


if __name__ == "__main__":
    main()
