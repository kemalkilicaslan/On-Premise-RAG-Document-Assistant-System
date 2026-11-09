from docrag.chunking import chunk_pages, normalize, split_text


def test_normalize_joins_hyphenated_words_and_collapses_spaces():
    assert normalize("güven-\nliği   ve\t\tsağlık\n\n\n\nMADDE 1") == "güvenliği ve sağlık\n\nMADDE 1"


def test_normalize_drops_space_before_soft_g():
    assert normalize("risk de ğerlendirmesi bu şekilde") == "risk değerlendirmesi bu şekilde"


def test_short_text_is_single_chunk():
    assert split_text("Kısa bir metin.", size=100, overlap=10) == ["Kısa bir metin."]


def test_empty_text_has_no_chunks():
    assert split_text("", size=100, overlap=10) == []


def test_chunks_respect_size_plus_overlap():
    text = " ".join(f"Bu {i}. cümledir." for i in range(200))
    chunks = split_text(text, size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(chunk) <= 240 for chunk in chunks)


def test_consecutive_chunks_overlap():
    text = " ".join(f"Cümle numarası {i}." for i in range(100))
    chunks = split_text(text, size=150, overlap=50)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current.split(" ", 1)[0] in previous


def test_long_unit_is_cut_hard():
    chunks = split_text("a" * 250, size=100, overlap=0)
    assert [len(c) for c in chunks] == [100, 100, 50]


def test_chunk_pages_keeps_page_numbers_and_global_index():
    chunks = chunk_pages(["Birinci sayfa.", "", "Üçüncü sayfa."], size=100, overlap=10)
    assert [(c.page, c.index) for c in chunks] == [(1, 0), (3, 1)]
