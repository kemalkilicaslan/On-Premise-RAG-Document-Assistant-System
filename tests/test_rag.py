from docrag.rag import RetrievedChunk, build_context, version_at_least


def test_version_at_least_handles_pgvector_versions():
    assert version_at_least("0.8.0", (0, 8))
    assert version_at_least("0.10.1", (0, 8))
    assert not version_at_least("0.6.2", (0, 8))
    assert not version_at_least("0", (0, 8))


def test_context_numbers_chunks_with_title_and_page():
    chunks = [
        RetrievedChunk(1, 1, "6331 İSG Kanunu", 7, "MADDE 10 – İşveren...", 0.9),
        RetrievedChunk(2, 1, "6331 İSG Kanunu", 2, "Risk değerlendirmesi: ...", 0.8),
    ]
    assert build_context(chunks) == (
        "[1] 6331 İSG Kanunu, sayfa 7\nMADDE 10 – İşveren...\n\n"
        "[2] 6331 İSG Kanunu, sayfa 2\nRisk değerlendirmesi: ..."
    )
