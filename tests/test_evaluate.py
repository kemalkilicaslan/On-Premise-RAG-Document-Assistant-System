import json

import numpy as np
import pytest

from docrag.evaluate import (
    BM25,
    Article,
    Question,
    chunk_articles,
    first_relevant_ranks,
    load_tquad,
    paired_difference,
    relevant_sets,
    retrieval_metrics,
    rrf,
    tokenize,
)


def write_tquad(tmp_path, dev, train):
    for name, data in (("dev-v0.1.json", dev), ("train-v0.1.json", train)):
        (tmp_path / name).write_text(json.dumps({"data": data}), encoding="utf-8")


def article(title, *paragraphs):
    return {"title": title, "paragraphs": [{"context": c, "qas": qas} for c, qas in paragraphs]}


def qa(qid, question, text, start):
    return {"id": qid, "question": question, "answers": [{"text": text, "answer_start": str(start)}]}


def test_load_maps_answers_through_whitespace_collapse_and_excludes_overlap(tmp_path):
    context = "Uluğ  Bey\n bir astronomdur. Semerkant'ta  rasathane kurdu."
    start = context.index("rasathane")
    dev = [article("Uluğ Bey", (context, [qa(1, "Ne kurdu?", "rasathane", start)]))]
    train = [
        article("uluğ bey", ("Aynı başlık, elenmeli.", [])),
        article("Başka", (context, [])),  # same paragraph as dev, must be excluded
        article("Ali Kuşçu", ("Ali Kuşçu bir matematikçidir.", [])),
    ]
    write_tquad(tmp_path, dev, train)
    articles, questions, corpus = load_tquad(tmp_path)
    assert [a.title for a in articles] == ["Uluğ Bey", "Ali Kuşçu"]
    assert corpus["train_articles_excluded"] == 2
    q = questions[0]
    assert articles[0].text[q.start : q.end] == "rasathane"


def test_chunks_are_located_and_relevance_needs_the_whole_answer():
    text = "Birinci cümle burada. İkinci cümle cevabı içerir: rasathane.\nÜçüncü paragraf başka bir konu."
    articles = [Article("A", text)]
    chunks, approximate = chunk_articles(articles, size=40, overlap=0)
    flat = text.replace("\n", " ")
    assert approximate == 0
    assert all(flat[c.start : c.end] == c.text for c in chunks)

    start = text.index("rasathane")
    relevant = relevant_sets(chunks, [Question("1", "q", 0, start, start + len("rasathane"))])[0]
    assert relevant and all("rasathane" in chunks[i].text for i in relevant)

    # an answer spanning two chunks is in no single chunk
    cut = text.index("İkinci")
    assert relevant_sets(chunks, [Question("2", "q", 0, cut - 10, cut + 10)])[0] == set()


def test_first_relevant_rank_and_metrics():
    ranking = np.array([[5, 2, 9], [1, 3, 4], [7, 8, 6]])
    ranks = first_relevant_ranks(ranking, [{2}, {1}, {0}])
    assert ranks.tolist() == [2, 1, 0]
    m = retrieval_metrics(ranks, ks=(1, 3))
    assert m["recall@1"] == pytest.approx(1 / 3)
    assert m["recall@3"] == pytest.approx(2 / 3)
    assert m["mrr@3"] == pytest.approx((1 / 2 + 1) / 3)
    lo, hi = m["recall@3_ci"]
    assert 0 <= lo <= m["recall@3"] <= hi <= 1


def test_rrf_rewards_agreement_between_retrievers():
    lexical = np.array([[3.0, 2.0, 1.0]])   # ranks: doc0, doc1, doc2
    dense = np.array([[0.1, 0.9, 0.5]])     # ranks: doc1, doc2, doc0
    fused = rrf(lexical, dense, k=60)[0]
    assert fused.argmax() == 1               # second for lexical, first for dense
    assert fused[1] == pytest.approx(1 / 62 + 1 / 61)


def test_paired_difference_ci_brackets_the_mean():
    a = np.array([1, 1, 1, 0, 1, 1, 0, 1])
    b = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    diff, lo, hi = paired_difference(a, b)
    assert diff == pytest.approx(0.25)
    assert lo <= diff <= hi


def test_bm25_prefers_matching_document_and_handles_turkish_casing():
    assert tokenize("İSTANBUL ve IŞIK") == ["istanbul", "ve", "ışık"]
    index = BM25(["Uluğ Bey Semerkant'ta rasathane kurdu.", "Ali Kuşçu matematikçidir."])
    scores = index.scores("Semerkant rasathanesini kim kurdu")
    assert scores[0] > scores[1]
