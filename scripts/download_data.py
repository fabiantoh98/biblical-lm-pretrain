"""Download all corpus datasets from public domain sources.

Datasets:
  ASV Bible         — GitHub (openbibleinfo/American-Standard-Version-Bible)
  Matthew Henry     — CCEL (ccel.org/ccel/henry), 6 volumes
  Calvin            — CCEL (ccel.org/ccel/calvin), Institutes + Commentaries
  Spurgeon          — CCEL (ccel.org/ccel/spurgeon), sermons and devotionals
  Augustine         — CCEL (ccel.org/ccel/augustine), Confessions, City of God, etc.

All sources are public domain. Running this script downloads everything to
data/raw/. Use DataConfig in config.py to control which datasets are included
when running scripts/prepare_data.py.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

RAW_DIR = Path("data/raw")
MH_DIR = RAW_DIR / "matthew_henry"

ASV_BASE_URL = (
    "https://raw.githubusercontent.com/openbibleinfo/American-Standard-Version-Bible"
    "/master/usx/"
)

# All 66 USX book files in canonical order.
ASV_USX_FILES = [
    "01-GEN.usx", "02-EXO.usx", "03-LEV.usx", "04-NUM.usx", "05-DEU.usx",
    "06-JOS.usx", "07-JDG.usx", "08-RUT.usx", "09-1SA.usx", "10-2SA.usx",
    "11-1KI.usx", "12-2KI.usx", "13-1CH.usx", "14-2CH.usx", "15-EZR.usx",
    "16-NEH.usx", "17-EST.usx", "18-JOB.usx", "19-PSA.usx", "20-PRO.usx",
    "21-ECC.usx", "22-SNG.usx", "23-ISA.usx", "24-JER.usx", "25-LAM.usx",
    "26-EZK.usx", "27-DAN.usx", "28-HOS.usx", "29-JOL.usx", "30-AMO.usx",
    "31-OBA.usx", "32-JON.usx", "33-MIC.usx", "34-NAM.usx", "35-HAB.usx",
    "36-ZEP.usx", "37-HAG.usx", "38-ZEC.usx", "39-MAL.usx",
    "40-MAT.usx", "41-MRK.usx", "42-LUK.usx", "43-JHN.usx", "44-ACT.usx",
    "45-ROM.usx", "46-1CO.usx", "47-2CO.usx", "48-GAL.usx", "49-EPH.usx",
    "50-PHP.usx", "51-COL.usx", "52-1TH.usx", "53-2TH.usx", "54-1TI.usx",
    "55-2TI.usx", "56-TIT.usx", "57-PHM.usx", "58-HEB.usx", "59-JAS.usx",
    "60-1PE.usx", "61-2PE.usx", "62-1JN.usx", "63-2JN.usx", "64-3JN.usx",
    "65-JUD.usx", "66-REV.usx",
]

# CCEL plain text URLs for Matthew Henry's Complete Commentary on the Whole Bible.
# Each entry is (volume_number, ccel_slug, description).
# Source: https://ccel.org/ccel/henry — public domain, freely distributable.
MATTHEW_HENRY_VOLUMES: list[tuple[int, str, str]] = [
    (1, "mhc1", "Genesis to Deuteronomy"),
    (2, "mhc2", "Joshua to Esther"),
    (3, "mhc3", "Job to Song of Solomon"),
    (4, "mhc4", "Isaiah to Malachi"),
    (5, "mhc5", "Matthew to John"),
    (6, "mhc6", "Acts to Revelation"),
]

CCEL_MH_URL = "https://ccel.org/ccel/h/henry/{slug}/cache/{slug}.txt"

# Extra datasets from CCEL. Each entry is (filename_stem, description, url).
# Files are saved as data/raw/{dataset_name}/{filename_stem}.txt
CALVIN_WORKS: list[tuple[str, str, str]] = [
    (
        "institutes",
        "Institutes of the Christian Religion",
        "https://ccel.org/ccel/c/calvin/institutes/cache/institutes.txt",
    ),
]

# Calvin's Commentaries are split into 46 individual volumes on CCEL.
# The "commentaries" meta-file is just a table of contents — each volume
# must be downloaded separately as calcom01.txt through calcom46.txt.
# Volumes are saved as data/raw/calvin/calcom_NN.txt.
CALVIN_COMMENTARY_COUNT = 46

SPURGEON_WORKS: list[tuple[str, str, str]] = [
    (
        "morning_evening",
        "Morning and Evening",
        "https://ccel.org/ccel/s/spurgeon/morneve/cache/morneve.txt",
    ),
    (
        "treasury_1",
        "Treasury of David Vol 1",
        "https://ccel.org/ccel/s/spurgeon/treasury1/cache/treasury1.txt",
    ),
    (
        "treasury_2",
        "Treasury of David Vol 2",
        "https://ccel.org/ccel/s/spurgeon/treasury2/cache/treasury2.txt",
    ),
    (
        "treasury_3",
        "Treasury of David Vol 3",
        "https://ccel.org/ccel/s/spurgeon/treasury3/cache/treasury3.txt",
    ),
    (
        "all_of_grace",
        "All of Grace",
        "https://ccel.org/ccel/s/spurgeon/grace/cache/grace.txt",
    ),
    (
        "faith_checkbook",
        "Faith's Checkbook",
        "https://ccel.org/ccel/s/spurgeon/checkbook/cache/checkbook.txt",
    ),
]

AUGUSTINE_WORKS: list[tuple[str, str, str]] = [
    (
        "confessions",
        "Confessions",
        "https://ccel.org/ccel/a/augustine/confess/cache/confess.txt",
    ),
    (
        "city_of_god",
        "City of God",
        "https://ccel.org/s/schaff/npnf102/cache/npnf102.txt",
    ),
    (
        "christian_doctrine",
        "On Christian Doctrine",
        "https://ccel.org/ccel/a/augustine/doctrine/cache/doctrine.txt",
    ),
]

HEADERS = {"User-Agent": "biblical-lm-pretrain/1.0 (academic research)"}
REQUEST_DELAY_SECONDS = 1


def _extract_para_text(para: ET.Element) -> list[str]:
    """Extract plain text words from a USX <para> element, stripping footnotes.

    In USX, verse text is stored as the tail of inline <verse> and <note>
    elements. Footnote content is inside <note> children and must be excluded,
    but text that follows the closing </note> tag (the note's tail) is valid
    verse text and must be kept.

    Args:
        para: A <para> XML element from a USX file.

    Returns:
        List of whitespace-normalised words from the paragraph.
    """
    parts: list[str] = []

    if para.text:
        parts.append(para.text)

    for child in para:
        if child.tag == "note":
            # skip footnote body; tail is text after </note> — keep it
            if child.tail:
                parts.append(child.tail)
        elif child.tag == "char":
            if child.text:
                parts.append(child.text)
            if child.tail:
                parts.append(child.tail)
        else:
            # verse markers and other inline elements: only the tail matters
            if child.tail:
                parts.append(child.tail)

    return " ".join(parts).split()


def _parse_usx_book(xml_text: str) -> str:
    """Parse a USX XML file and return clean plain text for one Bible book.

    Extracts text from all <para> elements, skipping structural elements
    (chapter headers, section titles) that carry style codes like 'ms', 's',
    's1', 'r'. Verse text paragraphs use styles 'p', 'q', 'q1', 'q2', 'b',
    'pi', 'li', etc.

    Args:
        xml_text: Raw USX XML string.

    Returns:
        Plain text of the book with paragraphs separated by newlines.
    """
    root = ET.fromstring(xml_text)

    # styles that carry running verse text (not headings/titles)
    verse_para_styles = {
        "p", "m", "pi", "pi1", "pi2",
        "q", "q1", "q2", "q3", "qc", "qr",
        "b", "li", "li1", "li2",
        "pc", "pr", "cls",
    }

    paragraphs: list[str] = []
    for para in root.iter("para"):
        style = para.get("style", "")
        if style not in verse_para_styles:
            continue
        words = _extract_para_text(para)
        if words:
            paragraphs.append(" ".join(words))

    return "\n".join(paragraphs)


def download_asv(output_path: Path) -> None:
    """Download and clean the ASV Bible from the openbibleinfo GitHub repository.

    Downloads all 66 USX XML files, parses out plain verse text (footnotes
    stripped), and writes one book per block separated by a blank line.

    Args:
        output_path: Destination path for asv.txt.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    book_texts: list[str] = []
    failed: list[str] = []

    asv_cache = output_path.parent / ".asv_cache"
    if output_path.exists() and asv_cache.exists():
        print(f"  Skipping ASV — already downloaded at {output_path}")
        return

    for filename in ASV_USX_FILES:
        url = ASV_BASE_URL + filename
        print(f"  Downloading {filename} ...", end=" ", flush=True)
        try:
            response = requests.get(url, headers=HEADERS, timeout=60)
            response.raise_for_status()
            book_text = _parse_usx_book(response.text)
            book_texts.append(book_text)
            print(f"{len(book_text):,} chars")
        except Exception as exc:
            print(f"FAILED ({exc})")
            failed.append(filename)

    with output_path.open("w", encoding="utf-8") as f:
        for text in book_texts:
            f.write(text)
            f.write("\n\n")

    asv_cache.touch()  # sentinel file so subsequent runs skip the download
    print(f"\nASV: {len(book_texts)} books saved to {output_path}")
    if failed:
        print(f"  Failed books: {failed}")


def download_matthew_henry(output_dir: Path) -> None:
    """Download Matthew Henry Complete Commentary volumes from CCEL.

    Each volume is fetched as a plain text file from ccel.org and saved
    as vol_N.txt. CCEL hosts the authoritative public domain text.

    Args:
        output_dir: Directory to save the volume text files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    successes: list[tuple[int, str]] = []
    failures: list[tuple[int, str]] = []

    for vol_num, slug, description in MATTHEW_HENRY_VOLUMES:
        output_path = output_dir / f"vol_{vol_num}.txt"
        if output_path.exists():
            print(f"  Skipping Vol {vol_num} ({description}) — already downloaded")
            successes.append((vol_num, description))
            continue
        url = CCEL_MH_URL.format(slug=slug)
        print(f"Downloading Matthew Henry Vol {vol_num} ({description}) ...")
        print(f"  {url}")
        try:
            response = requests.get(url, headers=HEADERS, timeout=120)
            response.raise_for_status()
            text = response.text.strip()
            output_path.write_text(text, encoding="utf-8")
            print(f"  Saved {len(text):,} chars ({output_path.stat().st_size // 1024}KB)")
            successes.append((vol_num, description))
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failures.append((vol_num, description))
        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nSummary: {len(successes)}/{len(MATTHEW_HENRY_VOLUMES)} volumes downloaded.")

    if failures:
        print("\nFailed volumes — download manually from https://ccel.org/ccel/henry :")
        for vol_num, description in failures:
            print(f"  Vol {vol_num}: {description}")
        print("  Save each as data/raw/matthew_henry/vol_N.txt")


def download_ccel_dataset(
    works: list[tuple[str, str, str]],
    output_dir: Path,
    dataset_name: str,
) -> None:
    """Download a set of CCEL plain text works into a named subdirectory.

    Args:
        works: List of (filename_stem, description, url) tuples.
        output_dir: Directory to save the downloaded files.
        dataset_name: Human-readable name for progress output.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    successes: list[str] = []
    failures: list[str] = []

    for filename_stem, description, url in works:
        output_path = output_dir / f"{filename_stem}.txt"
        if output_path.exists():
            print(f"  Skipping {description} — already downloaded")
            successes.append(description)
            continue
        print(f"Downloading {dataset_name} — {description} ...")
        print(f"  {url}")
        try:
            response = requests.get(url, headers=HEADERS, timeout=120)
            response.raise_for_status()
            text = response.text.strip()
            output_path.write_text(text, encoding="utf-8")
            size_kb = output_path.stat().st_size // 1024
            print(f"  Saved {len(text):,} chars ({size_kb}KB) to {output_path}")
            successes.append(description)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failures.append(description)
        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\n{dataset_name}: {len(successes)}/{len(works)} works downloaded.")
    if failures:
        print(f"  Failed: {failures}")


def download_calvin_commentaries(output_dir: Path, n_volumes: int = CALVIN_COMMENTARY_COUNT) -> None:
    """Download Calvin's Commentary volumes from CCEL.

    The CCEL "commentaries" meta-file is just a table of contents linking to
    46 individual volumes (calcom01–calcom46). This function fetches each
    volume separately and saves it as calcom_NN.txt. Volumes that return a
    404 or suspiciously small response (<10 KB) are skipped with a warning.

    Args:
        output_dir: Directory to save the volume text files.
        n_volumes: Number of volumes to attempt (default 46).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Remove the stale TOC-only file if present
    stale = output_dir / "commentaries.txt"
    if stale.exists() and stale.stat().st_size < 10_000:
        stale.unlink()
        print("  Removed stale commentaries.txt (was a table of contents, not text)")

    successes: list[int] = []
    failures: list[int] = []

    for n in range(1, n_volumes + 1):
        slug = f"calcom{n:02d}"
        output_path = output_dir / f"calcom_{n:02d}.txt"
        if output_path.exists():
            print(f"  Skipping Calvin Commentary vol {n:02d} — already downloaded")
            successes.append(n)
            continue
        url = f"https://ccel.org/ccel/c/calvin/{slug}/cache/{slug}.txt"
        print(f"Downloading Calvin Commentary vol {n:02d} ({slug}) ...")
        try:
            response = requests.get(url, headers=HEADERS, timeout=120)
            response.raise_for_status()
            text = response.text.strip()
            if len(text) < 10_000:
                print(f"  SKIPPED vol {n:02d} — response too small ({len(text)} chars), likely a 404 page")
                failures.append(n)
                time.sleep(REQUEST_DELAY_SECONDS)
                continue
            output_path.write_text(text, encoding="utf-8")
            size_kb = output_path.stat().st_size // 1024
            print(f"  Saved {len(text):,} chars ({size_kb}KB) to {output_path}")
            successes.append(n)
        except Exception as exc:
            print(f"  FAILED vol {n:02d}: {exc}")
            failures.append(n)
        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nCalvin Commentaries: {len(successes)}/{n_volumes} volumes downloaded.")
    if failures:
        print(f"  Skipped/failed volumes: {failures}")


if __name__ == "__main__":
    download_asv(RAW_DIR / "asv.txt")
    download_matthew_henry(MH_DIR)
    download_ccel_dataset(CALVIN_WORKS, RAW_DIR / "calvin", "Calvin")
    download_calvin_commentaries(RAW_DIR / "calvin")
    download_ccel_dataset(SPURGEON_WORKS, RAW_DIR / "spurgeon", "Spurgeon")
    download_ccel_dataset(AUGUSTINE_WORKS, RAW_DIR / "augustine", "Augustine")
    print("\nAll downloads complete.")
