"""Answer-quality evaluation on TQuAD: does the LLM answer correctly from the retrieved chunks,
and does it fall back instead of inventing an answer when the chunks do not contain one?

    python -m docrag.evaluate_answers                                  # 300 + 150 questions, ~75 min
    python -m docrag.evaluate_answers --answerable 50 --unanswerable 25

Two conditions on disjoint random samples of dev questions (seed 0):

* answerable: the production setup. bge-m3 retrieves the top 5 chunks (1000 characters, 150
  overlap) from the whole corpus and the LLM answers with the API's system prompt.
* unanswerable: the same, but every chunk of the question's own article is removed first, so the
  context should not hold the answer and the expected reply is the fallback sentence.

Scoring is deterministic (no LLM judge):

* exact: the normalized gold answer appears as a phrase in the generated answer (a Turkish
  suffix may follow its last word)
* token recall: share of gold answer tokens found in the answer; tokens match on their first five
  characters, a rough way to accept Turkish suffixes ("Musul" / "Musul'da" / "Musulda")
* correct: exact, or token recall of at least 0.5
* refused: the answer gives the fallback ("bulunmuyor", "bulunmamaktadır", ...)
* cited / cited_gold: the answer has an [n] marker / a cited chunk holds the gold answer span

Answers that are neither correct nor refused are split by whether the gold answer text appears
anywhere in the context: "grounded_other" (taken from another chunk that happens to hold it) does
not count as a hallucination, "ungrounded_correct" means the model answered from its own memory.
Generations are cached in data/cache/, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from .evaluate import CACHE, OUT, bootstrap_ci, chunk_articles, load_tquad, ollama_embed, relevant_sets
from .prompt import NO_ANSWER, build_messages

LLM = "qwen2.5:7b-instruct"
EMBED = "bge-m3"
CHUNK_SIZE, CHUNK_OVERLAP, TOP_K = 1000, 150, 5
OPTIONS = {"temperature": 0.1, "seed": 0}  # API temperature; the seed makes reruns reproducible

REFUSAL = re.compile(r"\b(bulunmuyor|bulunmamaktadır|bulunmamakta|yer almıyor|yer almamaktadır|"
                     r"bilgi verilmemiş|belirtilmemiş)\b")


@dataclass
class Passage:
    document_title: str
    page: int | None
    content: str


def normalize(text: str) -> str:
    text = text.replace("İ", "i").replace("I", "ı").lower()
    return " ".join(re.findall(r"\w+", text))


def exact_match(gold: str, answer: str) -> bool:
    """Gold phrase in the text, starting on a word boundary.

    TQuAD spans often stop inside a word ("Buzgan kasabası" for "Buzgan kasabasında"), so a Turkish
    suffix may follow the last word; after a final digit nothing may follow ("12" is not in "1200").
    """
    g = normalize(gold)
    if not g:
        return False
    pattern = r"(?<!\w)" + re.escape(g) + (r"(?!\d)" if g[-1].isdigit() else "")
    return re.search(pattern, normalize(answer)) is not None


def token_recall(gold: str, answer: str, prefix: int = 5) -> float:
    gold_tokens = normalize(gold).split()
    if not gold_tokens:
        return 0.0
    stems = {t[:prefix] for t in normalize(answer).split()}
    return sum(t[:prefix] in stems for t in gold_tokens) / len(gold_tokens)


def is_refusal(answer: str) -> bool:
    return normalize(NO_ANSWER) in normalize(answer) or bool(REFUSAL.search(answer.lower()))


def citations(answer: str) -> list[int]:
    return [int(n) for n in re.findall(r"\[(\d+)\]", answer)]


def classify(correct: bool, refused: bool, gold_text_in_context: bool) -> str:
    if refused and not correct:
        return "refused"
    if correct:
        return "correct" if gold_text_in_context else "ungrounded_correct"
    return "hallucinated"


class AnswerCache:
    def __init__(self, path: Path):
        self.path = path
        self.items: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                self.items[item["key"]] = item

    @staticmethod
    def key(messages: list[dict[str, str]]) -> str:
        return hashlib.sha1(json.dumps([LLM, OPTIONS, messages], ensure_ascii=False).encode()).hexdigest()

    def get_or_generate(self, client: httpx.Client, messages: list[dict[str, str]]) -> dict:
        key = self.key(messages)
        if key not in self.items:
            t0 = time.time()
            response = client.post("/api/chat", json={"model": LLM, "messages": messages, "stream": False,
                                                      "options": OPTIONS})
            response.raise_for_status()
            item = {"key": key, "answer": response.json()["message"]["content"].strip(),
                    "seconds": round(time.time() - t0, 2)}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            self.items[key] = item
        return self.items[key]


def rate(values: list[bool]) -> dict[str, float | int | list[float]]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"n": 0}
    return {"n": len(arr), "rate": float(arr.mean()), "ci": list(bootstrap_ci(arr))}


def summarize(records: list[dict]) -> dict:
    out: dict = {}
    for condition in ("answerable", "unanswerable"):
        rows = [r for r in records if r["condition"] == condition]
        part: dict = {"n": len(rows), "end_to_end_correct": rate([r["correct"] for r in rows])}
        groups = {"gold_in_context": [r for r in rows if r["gold_in_context"]],
                  "gold_not_in_context": [r for r in rows if not r["gold_in_context"]]}
        for name, group in groups.items():
            part[name] = {
                "n": len(group),
                **{label: rate([r["outcome"] == label for r in group])
                   for label in ("correct", "refused", "ungrounded_correct", "hallucinated")},
                "exact": rate([r["exact"] for r in group]),
                "mean_token_recall": float(np.mean([r["token_recall"] for r in group])) if group else None,
                "cited": rate([bool(r["cited"]) for r in group]),
                "cited_gold_when_correct": rate([r["cited_gold"] for r in group
                                                 if r["outcome"] == "correct"]),
            }
        out[condition] = part
    out["seconds_per_answer_median"] = float(np.median([r["seconds"] for r in records]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answerable", type=int, default=300)
    parser.add_argument("--unanswerable", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ollama", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    articles, questions, _ = load_tquad()
    chunks, _ = chunk_articles(articles, CHUNK_SIZE, CHUNK_OVERLAP)
    relevant = relevant_sets(chunks, questions)
    chunk_article = np.array([c.article for c in chunks])
    doc_vectors = ollama_embed(EMBED, [c.text for c in chunks], args.ollama)
    query_vectors = ollama_embed(EMBED, [q.question for q in questions], args.ollama)

    order = np.random.default_rng(args.seed).permutation(len(questions))
    plan = [("answerable", int(i)) for i in order[: args.answerable]]
    plan += [("unanswerable", int(i)) for i in order[args.answerable : args.answerable + args.unanswerable]]

    cache = AnswerCache(CACHE / f"answers-{LLM.replace(':', '_')}.jsonl")
    records: list[dict] = []
    started = time.time()
    with httpx.Client(base_url=args.ollama, timeout=600) as client:
        for n, (condition, qi) in enumerate(plan, start=1):
            q = questions[qi]
            scores = query_vectors[qi] @ doc_vectors.T
            if condition == "unanswerable":
                scores = np.where(chunk_article == q.article, -np.inf, scores)
            top = np.argsort(-scores)[:TOP_K]
            passages = [Passage(articles[chunks[j].article].title, None, chunks[j].text) for j in top]
            item = cache.get_or_generate(client, build_messages(passages, q.question))

            answer = item["answer"]
            gold = articles[q.article].text[q.start : q.end]
            exact, recall = exact_match(gold, answer), token_recall(gold, answer)
            correct = exact or recall >= 0.5
            refused = is_refusal(answer)
            cited = [c for c in citations(answer) if 1 <= c <= len(top)]
            gold_text_in_context = any(exact_match(gold, p.content) for p in passages)
            records.append({
                "condition": condition, "qid": q.qid, "question": q.question, "gold": gold, "answer": answer,
                "gold_in_context": any(int(j) in relevant[qi] for j in top),
                "gold_text_in_context": gold_text_in_context,
                "exact": exact, "token_recall": round(recall, 3), "correct": correct, "refused": refused,
                "outcome": classify(correct, refused, gold_text_in_context),
                "cited": " ".join(map(str, cited)),
                "cited_gold": any(int(top[c - 1]) in relevant[qi] for c in cited),
                "seconds": item["seconds"],
            })
            if n % 25 == 0 or n == len(plan):
                eta = (time.time() - started) / n * (len(plan) - n) / 60
                done = [r for r in records if r["condition"] == "answerable"]
                acc = np.mean([r["correct"] for r in done]) if done else float("nan")
                print(f"{n:>4}/{len(plan)}  answerable correct so far {acc:.3f}  eta {eta:.0f} min",
                      flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = summarize(records)
    summary["setup"] = {"llm": LLM, "embedding": EMBED, "chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP,
                        "top_k": TOP_K, "options": OPTIONS, "seed": args.seed}
    (args.out / "answers_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with (args.out / "answers.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({k: v for k, v in summary.items() if k != "setup"}, indent=1, ensure_ascii=False)[:3000])
    print(f"wrote   {args.out / 'answers_summary.json'}, {args.out / 'answers.csv'}")


if __name__ == "__main__":
    main()
