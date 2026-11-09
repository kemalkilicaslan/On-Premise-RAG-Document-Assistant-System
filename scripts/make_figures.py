"""Draw the README figures into the repository root.

    python scripts/make_figures.py

The architecture and role-based access figures are diagrams of what the code does. The chunk
figure is computed from data/raw/6331-is-sagligi-ve-guvenligi-kanunu.pdf with the default
CHUNK_SIZE / CHUNK_OVERLAP, so run scripts/download_data.py first. The retrieval figure reads
outputs/eval/summary.json, written by `python -m docrag.evaluate`.
"""

from __future__ import annotations

import json
import os
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# chunking/ingest import the settings module, which requires secrets the figures do not use
os.environ.setdefault("JWT_SECRET", "figures-only-" + "x" * 32)
os.environ.setdefault("ADMIN_PASSWORD", "figures-only")
from docrag.chunking import chunk_pages  # noqa: E402
from docrag.ingest import extract_pages  # noqa: E402

PREFIX = "On-Premise-RAG-Document-Assistant"
LAW_PDF = Path("data/raw/6331-is-sagligi-ve-guvenligi-kanunu.pdf")
EVAL_SUMMARY = Path("outputs/eval/summary.json")
ANSWERS_SUMMARY = Path("outputs/eval/answers_summary.json")

# Same palette as the other projects' figures
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
GRAY = "#8a8985"
GRAY_LIGHT = "#c3c2b7"
BLUE = "#2a78d6"
GREEN = "#1baf7a"
ORANGE = "#eb6834"
PURPLE = "#4a3aa7"

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": TEXT,
    "axes.labelcolor": TEXT_2,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
})


def titled(fig, title: str, subtitle: str) -> None:
    fig.text(0.012, 0.965, title, fontsize=15, fontweight="bold", va="top")
    fig.text(0.012, 0.905, subtitle, fontsize=11.5, color=TEXT_2, va="top", linespacing=1.4)


def box(ax, x, y, w, h, text, color=GRAY, fill="white", text_color=TEXT, size=9.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                linewidth=1.3, edgecolor=color, facecolor=fill))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=text_color,
            fontweight=weight, linespacing=1.35)


def arrow(ax, start, end, color=GRAY, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=13, linewidth=1.4,
                                 color=color, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def diagram_axes(fig):
    ax = fig.add_axes([0.012, 0.02, 0.976, 0.80])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")
    return ax


def architecture() -> None:
    fig = plt.figure(figsize=(10, 5.4), dpi=150)
    titled(fig, "Documents, vectors and models stay on the company server",
           "Upload once: PDF → pages → chunks → embeddings.\n"
           "Every question: role check → vector search filtered by role → grounded answer "
           "with page citations.")
    ax = diagram_axes(fig)

    ax.add_patch(FancyBboxPatch((0.5, 0.5), 99, 48.5, boxstyle="round,pad=0,rounding_size=1.5",
                                linewidth=1.1, edgecolor=GRAY_LIGHT, facecolor="none", linestyle=(0, (4, 3))))
    ax.text(98.5, 1.6, "on-premise · Docker Compose", ha="right", fontsize=8.5, color=GRAY)

    w, h = 15.5, 8
    xs = [3, 21.5, 40, 58.5]
    model_fill = "#edeaf7"

    # ingestion row
    ax.text(3, 46, "INGESTION  ·  once per document", fontsize=8.5, color=TEXT_2, fontweight="bold")
    y1 = 35.5
    box(ax, xs[0], y1, w, h, "PDF upload\nallowed_roles\nset by admin", size=9)
    box(ax, xs[1], y1, w, h, "pypdf\ntext per page")
    box(ax, xs[2], y1, w, h, "Chunking\n1000 chars\n150 overlap", size=9)
    box(ax, xs[3], y1, w, h, "bge-m3\nembeddings (1024-d)", color=PURPLE, fill=model_fill)
    for a, b in zip(xs, xs[1:], strict=False):
        arrow(ax, (a + w, y1 + h / 2), (b, y1 + h / 2))

    # question row
    ax.text(3, 29, "QUESTION  ·  every request", fontsize=8.5, color=TEXT_2, fontweight="bold")
    y2 = 18.5
    box(ax, xs[0], y2, w, h, "Question\n+ JWT", color=BLUE)
    box(ax, xs[1], y2, w, h, "Role check\nrole read from DB", color=BLUE)
    box(ax, xs[3], y2, w, h, "bge-m3\nquery vector", color=PURPLE, fill=model_fill)
    arrow(ax, (xs[0] + w, y2 + h / 2), (xs[1], y2 + h / 2), color=BLUE)
    arrow(ax, (xs[1] + w, y2 + h / 2), (xs[3], y2 + h / 2), color=BLUE)

    # database
    db_x, db_w = 79.5, 17.5
    box(ax, db_x, 16, db_w, 30,
        "PostgreSQL 17\n+ pgvector\n\nusers · documents\nchunks (vector)\n\n"
        "HNSW cosine top-k\nWHERE role =\nANY(allowed_roles)",
        color=BLUE, fill="#eaf1fb", weight="normal")
    arrow(ax, (xs[3] + w, y1 + h / 2), (db_x, y1 + h / 2))
    arrow(ax, (xs[3] + w, y2 + h / 2), (db_x, y2 + h / 2), color=BLUE)

    # answer row
    ax.text(3, 13, "ANSWER", fontsize=8.5, color=TEXT_2, fontweight="bold")
    y3 = 3
    box(ax, xs[3], y3, w, h, "Top-k chunks\nnumbered [1] … [k]", color=GREEN)
    box(ax, xs[2], y3, w, h, "qwen2.5:7b-instruct\ngrounded TR prompt", color=PURPLE, fill=model_fill)
    box(ax, xs[1], y3, w, h, "Answer\n+ page citations", color=GREEN, fill="#e8f7f1", weight="bold")
    arrow(ax, (db_x + db_w / 2, 16), (xs[3] + w, y3 + h / 2), color=GREEN, rad=-0.35)
    arrow(ax, (xs[3], y3 + h / 2), (xs[2] + w, y3 + h / 2), color=GREEN)
    arrow(ax, (xs[2], y3 + h / 2), (xs[1] + w, y3 + h / 2), color=GREEN)

    ax.text(xs[0] + w / 2, y3 + h / 2, "Ollama serves\nthe purple boxes", ha="center", va="center",
            fontsize=8.5, color=PURPLE, linespacing=1.3)

    fig.savefig(f"{PREFIX}-Architecture.png")
    plt.close(fig)


def role_based_access() -> None:
    fig = plt.figure(figsize=(10, 5), dpi=150)
    titled(fig, "Same question, different roles: restricted chunks never reach the LLM",
           "Law No. 6331 uploaded with allowed_roles = {isg, personel}. The role filter is part of the "
           "vector query,\nso an ik user gets the fallback answer and the LLM is not called.")
    ax = diagram_axes(fig)

    box(ax, 1, 19, 19, 12, "“İşveren risk\ndeğerlendirmesi yapmakla\nyükümlü müdür?”", color=GRAY, size=9.5)

    lanes = [
        (33, GREEN, "#e8f7f1", "mehmet  ·  role isg", "isg ∈ {isg, personel}  ✓",
         "5 chunks retrieved\ntop: MADDE 10, p. 7 (0.78)",
         "“Evet, işveren risk\ndeğerlendirmesi yapmakla\nyükümlüdür. [1][2]”"),
        (5, ORANGE, "#fdeee7", "ayse  ·  role ik", "ik ∉ {isg, personel}  ✗",
         "0 chunks retrieved\nLLM not called",
         "“Bu sorunun cevabı\nerişebildiğiniz dokümanlarda\nbulunmuyor.”"),
    ]
    for y, color, fill, user, check, retrieved, answer in lanes:
        box(ax, 23, y, 13, 10, user, color=color, weight="bold", size=8.8)
        box(ax, 39.5, y, 16, 10, f"pgvector search\n{check}", color=color, size=9)
        box(ax, 59, y, 16.5, 10, retrieved, color=color, size=8.6)
        box(ax, 79, y, 20.5, 10, answer, color=color, fill=fill, size=8.6)
        arrow(ax, (20, 25), (23, y + 5), color=color)
        arrow(ax, (36, y + 5), (39.5, y + 5), color=color)
        arrow(ax, (55.5, y + 5), (59, y + 5), color=color)
        arrow(ax, (75.5, y + 5), (79, y + 5), color=color)

    ax.text(89.25, 45.5, "ANSWER", ha="center", fontsize=8.5, color=TEXT_2, fontweight="bold")
    ax.text(47.5, 45.5, "ROLE FILTER (SQL)", ha="center", fontsize=8.5, color=TEXT_2, fontweight="bold")
    ax.text(1, 1, "Score and answer from the live demo (bge-m3, qwen2.5:7b-instruct, PostgreSQL + pgvector). "
            "Usernames as in the demo.", fontsize=8, color=GRAY)

    fig.savefig(f"{PREFIX}-Role-Based-Access.png")
    plt.close(fig)


def chunk_length_distribution(size: int = 1000, overlap: int = 150) -> None:
    pages = extract_pages(LAW_PDF.read_bytes())
    chunks = chunk_pages(pages, size, overlap)
    lengths = [len(c.content) for c in chunks]

    # first chunk in which each "MADDE n" heading appears
    first_chunk: dict[int, int] = {}
    for c in chunks:
        for m in re.finditer(r"MADDE\s+(\d+)", c.content):
            first_chunk.setdefault(int(m.group(1)), c.index)
    starts = sorted(first_chunk.values())
    ends = starts[1:] + [len(chunks)]
    articles = len(starts)
    # an article continues past its first chunk when the next article does not start in that same chunk
    split = sum(end > start for start, end in zip(starts, ends, strict=True))
    mixed = sum(len(re.findall(r"MADDE\s+\d+", c.content)) >= 2 for c in chunks)

    fig = plt.figure(figsize=(10, 5), dpi=150)
    titled(fig, f"Fixed-size chunks cut {split} of {articles} articles of Law No. 6331 apart",
           f"{len(chunks)} chunks from {len(pages)} pages ({size} characters, {overlap} overlap, "
           f"never crossing a page), median {statistics.median(lengths):.0f} characters.\n"
           "Splitting at article boundaries (MADDE n) is the next step.")

    ax1 = fig.add_axes([0.07, 0.12, 0.50, 0.62])
    # bins (950, 1000] etc., so chunks of exactly CHUNK_SIZE land left of the limit line
    edges = [size - 50 * k for k in range(size // 50 - 4, -1, -1)]
    ax1.hist(lengths, bins=[e + 0.5 for e in edges], color=BLUE, edgecolor=SURFACE, linewidth=1)
    ax1.axvline(size + 0.5, color=ORANGE, linewidth=1.4, linestyle=(0, (4, 3)))
    ax1.set_xlim(edges[0], size + 130)
    ax1.text(size + 12, ax1.get_ylim()[1] * 0.96, f"CHUNK_SIZE\n= {size}", ha="left", va="top",
             fontsize=9, color=ORANGE, linespacing=1.3)
    ax1.set_xlabel("Chunk length (characters)")
    ax1.set_ylabel("Chunks")
    ax1.set_title("Chunk length", loc="left", fontsize=11, color=TEXT, pad=8)

    ax2 = fig.add_axes([0.66, 0.12, 0.31, 0.62])
    labels = ["Article continues\ninto another chunk", "Article fits in\nits first chunk",
              "Chunks holding\n2+ article starts"]
    values = [split, articles - split, mixed]
    colors = [ORANGE, GREEN, GRAY]
    bars = ax2.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.6)
    for bar, value in zip(bars, values[::-1], strict=True):
        ax2.text(bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2, str(value), va="center",
                 fontsize=10, color=TEXT)
    ax2.set_xlim(0, max(values) * 1.18)
    ax2.set_title("Articles vs chunks", loc="left", fontsize=11, color=TEXT, pad=8)
    ax2.tick_params(axis="y", length=0, labelsize=9)

    for ax in (ax1, ax2):
        ax.grid(axis="x" if ax is ax2 else "y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.savefig(f"{PREFIX}-Chunk-Length-Distribution.png")
    plt.close(fig)


def retrieval_evaluation(summary_path: Path = EVAL_SUMMARY, chunk_size: int = 1000) -> None:
    summary = json.loads(summary_path.read_text())
    runs, corpus = summary["runs"], summary["corpus"]
    styles = {  # retriever -> (label, color)
        "hybrid": ("Hybrid BM25 + bge-m3 (RRF)", PURPLE),
        "bge-m3": ("bge-m3", BLUE),
        "bm25": ("BM25", GRAY),
        "nomic-embed-text": ("nomic-embed-text", ORANGE),
    }
    at_size = {r["retriever"]: r for r in runs if r["chunk_size"] == chunk_size}
    best = at_size["hybrid"]
    gain = best["recall@5_vs_bm25"]
    nomic_gap = at_size["nomic-embed-text"]["recall@5_vs_bm25"][0]

    fig = plt.figure(figsize=(10, 5.2), dpi=150)
    titled(fig, f"Hybrid retrieval puts the answer chunk in the top 5 for {best['recall@5']:.0%} "
                "of questions",
           f"TQuAD dev: {corpus['questions']} questions over "
           f"{corpus['dev_articles'] + corpus['train_articles_kept']} articles "
           f"({corpus['characters'] / 1e6:.1f} M characters). Hybrid vs BM25 at {chunk_size} characters: "
           f"{gain[0] * 100:+.1f} points [{gain[1] * 100:+.1f}, {gain[2] * 100:+.1f}].\n"
           f"nomic-embed-text finds {-nomic_gap * 100:.0f} points fewer than BM25; bge-m3 alone is "
           "not significantly better than BM25 on this lexical-overlap-heavy set.")

    ks = [1, 3, 5, 10]
    ax1 = fig.add_axes([0.07, 0.12, 0.42, 0.58])
    for retriever, (label, color) in styles.items():
        run = at_size[retriever]
        values = [run[f"recall@{k}"] for k in ks]
        lo = [run[f"recall@{k}_ci"][0] for k in ks]
        hi = [run[f"recall@{k}_ci"][1] for k in ks]
        ax1.fill_between(ks, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax1.plot(ks, values, marker="o", markersize=4.5, color=color, linewidth=2, label=label)
    ax1.set_xticks(ks)
    ax1.set_xlabel("k (chunks put into the prompt)")
    ax1.set_ylabel("Recall@k")
    ax1.set_ylim(0.2, 1.0)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax1.set_title(f"Recall@k at chunk size {chunk_size} (95 % CI)", loc="left", fontsize=11, color=TEXT,
                  pad=8)
    ax1.legend(frameon=False, fontsize=8.5, loc="lower right")

    sizes = sorted({r["chunk_size"] for r in runs})
    ax2 = fig.add_axes([0.58, 0.12, 0.39, 0.58])
    end_labels = []
    for retriever, (label, color) in styles.items():
        by_size = {r["chunk_size"]: r for r in runs if r["retriever"] == retriever}
        values = [by_size[s]["mrr@10"] for s in sizes]
        ax2.plot(sizes, values, marker="o", markersize=4.5, color=color, linewidth=2, label=label)
        end_labels.append((values[-1], color))
    # spread end labels that would overlap, top to bottom
    previous = None
    for value, color in sorted(end_labels, reverse=True):
        y = value if previous is None else min(value, previous - 0.03)
        ax2.text(sizes[-1] + 40, y, f"{value:.3f}", va="center", fontsize=8.5, color=color)
        previous = y
    ax2.set_xticks(sizes)
    ax2.set_xlim(sizes[0] - 100, sizes[-1] + 200)
    ax2.set_ylim(0.2, 1.0)
    ax2.set_xlabel("Chunk size (characters, 15 % overlap)")
    ax2.set_ylabel("MRR@10")
    ax2.set_title("MRR@10 by chunk size", loc="left", fontsize=11, color=TEXT, pad=8)

    for ax in (ax1, ax2):
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.savefig(f"{PREFIX}-Retrieval-Evaluation.png")
    plt.close(fig)


def answer_quality(summary_path: Path = ANSWERS_SUMMARY) -> None:
    summary = json.loads(summary_path.read_text())
    groups = [  # (label, summary part)
        ("Answer chunk retrieved", summary["answerable"]["gold_in_context"]),
        ("Answer chunk not retrieved", summary["answerable"]["gold_not_in_context"]),
        ("Own article removed\n(unanswerable)", summary["unanswerable"]["gold_not_in_context"]),
    ]
    outcomes = [  # (key, label, color)
        ("correct", "Correct, answer in context", GREEN),
        ("refused", "Fallback (\"not in the documents\")", GRAY_LIGHT),
        ("ungrounded_correct", "Correct, but not in context", "#eda100"),
        ("hallucinated", "Wrong answer", ORANGE),
    ]
    retrieved = groups[0][1]
    unanswerable = groups[2][1]

    fig = plt.figure(figsize=(10, 4.8), dpi=150)
    titled(fig, f"With the answer in context, the LLM answers correctly {retrieved['correct']['rate']:.0%} "
                "of the time",
           f"qwen2.5:7b-instruct with the API prompt, top 5 bge-m3 chunks, TQuAD dev. "
           f"{summary['answerable']['n']} answerable and {summary['unanswerable']['n']} unanswerable "
           f"questions.\nWithout the answer in context it gives the fallback "
           f"{unanswerable['refused']['rate']:.0%} of the time and a wrong answer "
           f"{unanswerable['hallucinated']['rate']:.0%} of the time.")

    ax = fig.add_axes([0.2, 0.2, 0.77, 0.52])
    labels = [f"{label}\nn = {part['n']}" for label, part in groups]
    left = np.zeros(len(groups))
    for key, label, color in outcomes:
        values = np.array([part[key]["rate"] if part["n"] else 0.0 for _, part in groups])
        ax.barh(labels, values, left=left, color=color, height=0.6, label=label, edgecolor=SURFACE)
        for y, (x0, v) in enumerate(zip(left, values, strict=True)):
            if v >= 0.06:
                ax.text(x0 + v / 2, y, f"{v:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if color in (GREEN, ORANGE) else TEXT)
        left += values
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.tick_params(axis="y", length=0, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(ncol=4, frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(-0.02, -0.12))

    fig.savefig(f"{PREFIX}-Answer-Quality.png")
    plt.close(fig)


def main() -> None:
    architecture()
    role_based_access()
    chunk_length_distribution()
    if EVAL_SUMMARY.exists():
        retrieval_evaluation()
    else:
        print(f"skipped Retrieval-Evaluation: run `python -m docrag.evaluate` to create {EVAL_SUMMARY}")
    if ANSWERS_SUMMARY.exists():
        answer_quality()
    else:
        print(f"skipped Answer-Quality: run `python -m docrag.evaluate_answers` to create {ANSWERS_SUMMARY}")
    print("\n".join(sorted(str(p) for p in Path(".").glob(f"{PREFIX}-*.png"))))


if __name__ == "__main__":
    main()
