from __future__ import annotations

import re
import subprocess
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile

PDF_DIR = Path(__file__).resolve().parents[1] / "docs"
OUT_DIR = Path(__file__).resolve().parents[1] / "agents" / "marketlogic" / "docs" / "scripts"

SCENE_RE = re.compile(r"^(INT\.|EXT\.|INT/EXT\.|EST\.)", re.IGNORECASE)


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "script"


def cleaned_title(filename: str) -> str:
    base = Path(filename).stem
    base = re.sub(r"[\-_]+", " ", base).strip()
    tokens = [t for t in base.split() if t.lower() not in {"script", "pdf"}]
    if tokens and tokens[-1].lower() == "the":
        tokens = tokens[:-1]
    title = " ".join(tokens).strip()
    return title.title() if title else "Untitled"


def extract_text(pdf_path: Path) -> str:
    with NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), str(tmp_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return tmp_path.read_text(encoding="utf-8", errors="ignore")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def build_scene_index(pages: list[str]) -> list[tuple[int, str]]:
    index: list[tuple[int, str]] = []
    for page_num, page in enumerate(pages, start=1):
        for line in page.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if SCENE_RE.match(stripped):
                index.append((page_num, stripped))
    return index


def write_markdown(title: str, slug: str, pages: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{slug}.md"
    scene_index = build_scene_index(pages)

    lines: list[str] = [
        "---",
        f"title: \"{title} - Script\"",
        f"film: \"{title}\"",
        f"date: \"{date.today().isoformat()}\"",
        "tags: [\"script\", \"pdf\"]",
        "---",
        "",
        "## Scene Index",
    ]

    if scene_index:
        for page_num, heading in scene_index:
            lines.append(f"- Page {page_num}: {heading}")
    else:
        lines.append("- No scene headings detected.")

    lines.extend(["", "## Script", ""])

    for page_num, page in enumerate(pages, start=1):
        lines.append(f"<!-- Page {page_num} -->")
        lines.append("")
        lines.append(page.rstrip())
        lines.append("")

    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for md in OUT_DIR.glob("*.md"):
        md.unlink()

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {PDF_DIR}")

    for pdf in pdfs:
        title = cleaned_title(pdf.name)
        slug = slugify(title)
        text = extract_text(pdf)
        pages = text.split("\f") if text else [""]
        write_markdown(title, slug, pages)
        print(f"Converted {pdf.name} -> {slug}.md")


if __name__ == "__main__":
    main()
