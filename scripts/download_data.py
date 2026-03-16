"""Download ASV Bible and Matthew Henry commentary from public domain sources."""

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

CCEL_URL_PATTERN = "https://ccel.org/ccel/h/henry/{slug}/cache/{slug}.txt"

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
        url = CCEL_URL_PATTERN.format(slug=slug)
        print(f"Downloading Matthew Henry Vol {vol_num} ({description}) ...")
        print(f"  {url}")
        try:
            response = requests.get(url, headers=HEADERS, timeout=120)
            response.raise_for_status()
            text = response.text.strip()
            output_path = output_dir / f"vol_{vol_num}.txt"
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


if __name__ == "__main__":
    download_asv(RAW_DIR / "asv.txt")
    download_matthew_henry(MH_DIR)
    print("\nData download complete.")
