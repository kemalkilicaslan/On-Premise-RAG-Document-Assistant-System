"""Download the sample document and the TQuAD evaluation set into data/raw.

    python scripts/download_data.py

Law No. 6331 is published by the Turkish Ministry of Labour and Social Security.
TQuAD is distributed by its authors on GitHub; read its terms before use.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

RAW = Path("data/raw")
LAW_URL = "https://csgb.gov.tr/Media/fldk2co0/6331_isgkanunu_tr.pdf"
TQUAD_URL = "https://raw.githubusercontent.com/TQuad/turkish-nlp-qa-dataset/master"
TQUAD_FILES = ["train-v0.1.json", "dev-v0.1.json"]


def fetch(url: str, dst: Path) -> Path:
    if dst.exists():
        print(f"exists  {dst}")
        return dst
    print(f"get     {url}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dst)
    return dst


def main() -> None:
    fetch(LAW_URL, RAW / "6331-is-sagligi-ve-guvenligi-kanunu.pdf")
    for name in TQUAD_FILES:
        fetch(f"{TQUAD_URL}/{name}", RAW / "tquad" / name)


if __name__ == "__main__":
    main()
