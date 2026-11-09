import pytest

from docrag.evaluate_answers import (
    citations,
    classify,
    exact_match,
    is_refusal,
    normalize,
    summarize,
    token_recall,
)
from docrag.prompt import NO_ANSWER, build_context


def test_normalize_handles_turkish_casing_and_punctuation():
    assert normalize("İBN-İ Yunus, (1156) Musul'da!") == "ibn i yunus 1156 musul da"


def test_exact_match_needs_a_whole_phrase():
    assert exact_match("1156", "1156 yılında Musul'da doğdu. [1]")
    assert exact_match("Musul", "Musul'da doğdu.")
    assert exact_match("Buzgan kasabası", "İran'da bulunan Buzgan kasabasında doğmuştur.")
    assert not exact_match("asaba", "Buzgan kasabası")
    assert not exact_match("12", "1200 yılında")
    assert not exact_match("", "herhangi bir cevap")


def test_token_recall_accepts_turkish_suffixes():
    assert token_recall("Musul", "Musulda doğdu") == 1.0
    assert token_recall("astronom ve matematikçi", "Astronomdu.") == pytest.approx(1 / 3)
    assert token_recall("", "x") == 0.0


def test_refusal_detection():
    assert is_refusal(NO_ANSWER)
    assert is_refusal("Bu bilgi bağlamda bulunmamaktadır.")
    assert not is_refusal("Uluğ Bey Semerkant'ta rasathane kurdu. [1]")


def test_citations_are_parsed():
    assert citations("Evet. [1][3] Ayrıca [2].") == [1, 3, 2]


def test_classify():
    assert classify(correct=True, refused=False, gold_text_in_context=True) == "correct"
    assert classify(correct=True, refused=False, gold_text_in_context=False) == "ungrounded_correct"
    assert classify(correct=False, refused=True, gold_text_in_context=True) == "refused"
    assert classify(correct=False, refused=False, gold_text_in_context=False) == "hallucinated"
    # an answer that states the gold answer and then hedges still counts as correct
    assert classify(correct=True, refused=True, gold_text_in_context=True) == "correct"


def test_context_omits_page_when_unknown():
    class P:
        def __init__(self, title, page, content):
            self.document_title, self.page, self.content = title, page, content

    assert build_context([P("Uluğ Bey", None, "metin"), P("Kanun", 7, "madde")]) == (
        "[1] Uluğ Bey\nmetin\n\n[2] Kanun, sayfa 7\nmadde"
    )


def test_summarize_splits_by_condition_and_context():
    base = {"exact": True, "token_recall": 1.0, "cited": "1", "cited_gold": True, "seconds": 5.0}
    records = [
        {**base, "condition": "answerable", "gold_in_context": True, "correct": True, "outcome": "correct"},
        {**base, "condition": "answerable", "gold_in_context": False, "correct": False, "outcome": "refused"},
        {**base, "condition": "unanswerable", "gold_in_context": False, "correct": False,
         "outcome": "hallucinated"},
    ]
    s = summarize(records)
    assert s["answerable"]["end_to_end_correct"]["rate"] == 0.5
    assert s["answerable"]["gold_in_context"]["correct"]["rate"] == 1.0
    assert s["unanswerable"]["gold_not_in_context"]["hallucinated"]["rate"] == 1.0
