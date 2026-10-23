"""Prepare the MkDocs source tree without moving any repository files.

Copies every Markdown file (and the images they reference) into ``docs_src/``,
renames ``README.md``/``README.id.md`` to ``index.md``/``index.id.md`` so that
mkdocs-static-i18n serves them as section pages, rewrites links to non-doc
files (``.py``, ``.ipynb``, ``.csv`` ...) into GitHub links, and adds an
"Open in Colab" box to module pages that ship an ``exercise.ipynb``.

Usage::

    python scripts/build_docs.py && mkdocs build --strict
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs_src"
REPO = "imam-db/ml-algo"
BLOB = f"https://github.com/{REPO}/blob/main/"
TREE = f"https://github.com/{REPO}/tree/main/"
SITE = "https://imam-db.github.io/ml-algo/"
COLAB = f"https://colab.research.google.com/github/{REPO}/blob/main/"

SKIP_DIRS = {
    ".git",
    ".github",
    ".venv",
    "docs_src",
    "site",
    "datasets",
    "tests",
    "node_modules",
    "scripts",
}
ASSETS = {".png", ".jpg", ".jpeg", ".gif", ".svg"}
LINK = re.compile(r"(!?\[[^\]]*\]\()([^)\s]+)(\s+\"[^\"]*\")?\)")

COLAB_TEXT = {
    "en": '!!! tip "Hands-on notebook"\n    Run the exercise directly in your browser: '
    "[Open in Colab]({url}) · [View on GitHub]({gh})\n\n",
    "id": '!!! tip "Notebook latihan"\n    Jalankan latihan langsung di browser: '
    "[Buka di Colab]({url}) · [Lihat di GitHub]({gh})\n\n",
}


def skipped(path: Path) -> bool:
    parts = path.relative_to(ROOT).parts[:-1]
    return any(part in SKIP_DIRS or part.startswith(".") for part in parts)


def target_name(src: Path) -> str:
    if src.name == "README.md":
        return "index.md"
    if src.name == "README.id.md":
        return "index.id.md"
    return src.name


def counterpart_url(src: Path, target: Path) -> str | None:
    """Links between an English page and its Indonesian twin become site URLs."""
    if target.parent != src.parent or not target.name.endswith(".md"):
        return None
    base = lambda p: p.name.replace(".id.md", ".md")
    if base(src) != base(target) or src.name == target.name:
        return None
    lang = "id/" if target.name.endswith(".id.md") else ""
    rel_dir = target.parent.relative_to(ROOT).as_posix()
    rel_dir = "" if rel_dir == "." else rel_dir + "/"
    page = "" if base(target) == "README.md" else base(target)[:-3] + "/"
    return SITE + lang + rel_dir + page


def rewrite_link(link: str, src: Path) -> str:
    if re.match(r"^[a-z]+:|^#|^/", link):
        return link
    path, _, anchor = link.partition("#")
    anchor = f"#{anchor}" if anchor else ""
    if not path:
        return link
    resolved = (src.parent / path).resolve()
    try:
        rel = resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return link
    counterpart = counterpart_url(src, resolved)
    if counterpart:
        return counterpart + anchor
    if resolved.is_dir():
        if (resolved / "README.md").exists() and not skipped(resolved / "README.md"):
            prefix = path if path.endswith("/") else path + "/"
            return f"{prefix}index.md{anchor}"
        return TREE + rel
    if resolved.suffix == ".md" and resolved.exists() and not skipped(resolved):
        if resolved.name in ("README.md", "README.id.md"):
            new = path[: -len(resolved.name)] + target_name(resolved)
            return new + anchor
        return link
    if resolved.suffix.lower() in ASSETS and resolved.exists():
        return link
    return BLOB + rel + anchor


def colab_box(src: Path, lang: str) -> str:
    nb = src.parent / "exercise.ipynb"
    if src.name not in ("README.md", "README.id.md") or not nb.exists():
        return ""
    rel = nb.relative_to(ROOT).as_posix()
    return COLAB_TEXT[lang].format(url=COLAB + rel, gh=BLOB + rel)


SWITCHER = re.compile(r"(English|Bahasa Indonesia|Inggris)")
TRANSLATION_LINK = re.compile(
    r"\]\(\.?/?[\w.]*?(README|CODE_OF_CONDUCT|CONTRIBUTING)?(\.id)?\.md\)"
)


def drop_language_switcher(text: str) -> str:
    """Remove the in-file EN/ID switcher line; the site has its own language toggle."""
    kept = []
    for line in text.split("\n")[:12]:
        if SWITCHER.search(line) and TRANSLATION_LINK.search(line) and len(line) < 120:
            continue
        kept.append(line)
    return "\n".join(kept + text.split("\n")[12:])


def convert(src: Path) -> str:
    text = drop_language_switcher(src.read_text(encoding="utf-8"))
    text = LINK.sub(
        lambda m: f"{m.group(1)}{rewrite_link(m.group(2), src)}{m.group(3) or ''})", text
    )
    box = colab_box(src, "id" if src.name.endswith(".id.md") else "en")
    if box:
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("# "):
                lines.insert(i + 1, "\n" + box.rstrip("\n"))
                break
        text = "\n".join(lines)
    return text


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    count = 0
    for src in sorted(ROOT.rglob("*")):
        if not src.is_file() or skipped(src):
            continue
        rel_dir = src.parent.relative_to(ROOT)
        if src.suffix == ".md":
            dest = OUT / rel_dir / target_name(src)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(convert(src), encoding="utf-8")
            count += 1
        elif src.suffix.lower() in ASSETS:
            dest = OUT / rel_dir / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    for extra in (ROOT / "docs_overrides").glob("**/*"):
        if extra.is_file():
            dest = OUT / extra.relative_to(ROOT / "docs_overrides")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(extra, dest)
    print(f"Prepared {count} Markdown pages in {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
