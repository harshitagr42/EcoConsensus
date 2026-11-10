"""Deterministic report writing (plain Python, no AI). Nothing is ever overwritten.

- Timestamped reports go through `safe_write` (create-only, `_2`, `_3` on a clash).
- "Latest" files such as test/result.txt are first moved to an archive folder under their own timestamp,
  then the new file is written.
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from src.utils import safe_write, unique_path

SECTION_ORDER = [
    "Header", "Data summary", "System comparison", "Per-class results", "Confusion matrices",
    "Results by script group", "LLM usage", "Fallback usage", "Error analysis", "Explainability",
    "Environment", "Configuration", "Warnings", "References",
]


def md_table(rows: list[dict], columns: list[str] | None = None, floatfmt: str = ".3f") -> str:
    if not rows:
        return "_(no rows)_"
    columns = columns or list(rows[0])

    def fmt(v):
        if isinstance(v, float):
            return format(v, floatfmt)
        return "" if v is None else str(v).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in columns) + " |" for r in rows]
    return "\n".join(lines)


class Report:
    """Collect sections, then render them in SECTION_ORDER (unknown sections go before Warnings)."""

    def __init__(self, title: str):
        self.title = title
        self.created = datetime.now()
        self.sections: dict[str, list[str]] = {}
        self.warnings: list[str] = []

    def add(self, section: str, text: str) -> "Report":
        self.sections.setdefault(section, []).append(text.rstrip())
        return self

    def warn(self, msg: str) -> "Report":
        self.warnings.append(msg)
        return self

    def render(self) -> str:
        out = [f"# {self.title}", "", f"Generated: {self.created:%Y-%m-%d %H:%M:%S}", ""]
        tail = ("Warnings", "References")
        known = [s for s in SECTION_ORDER if s in self.sections and s not in tail]
        extra = [s for s in self.sections if s not in SECTION_ORDER]
        for s in known + extra:
            out += [f"## {s}", "", *[t + "\n" for t in self.sections[s]]]
        if self.warnings or "Warnings" in self.sections:
            out += ["## Warnings", ""] + [f"- {w}" for w in self.warnings] + self.sections.get("Warnings", []) + [""]
        if "References" in self.sections:
            out += ["## References", "", *self.sections["References"], ""]
        return "\n".join(out)

    def save(self, path: Path) -> Path:
        """Write to a NEW timestamped file next to `path` (e.g. outputs/reports/meta_report.md)."""
        return safe_write(Path(path), self.render())


def archive_latest(latest: Path, archive_dir: Path) -> Path | None:
    """Move an existing 'latest' file into archive_dir as <stem>_<its mtime>.<ext>. Returns the archived path."""
    latest = Path(latest)
    if not latest.exists():
        return None
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.fromtimestamp(latest.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
    target = unique_path(archive_dir / f"{latest.stem}_{ts}{latest.suffix}", add_timestamp=False)
    shutil.move(str(latest), str(target))
    return target


def write_latest(latest: Path, content: str, archive_dir: Path) -> tuple[Path, Path | None]:
    """Archive the previous `latest`, then create the new one (create-only)."""
    archived = archive_latest(latest, archive_dir)
    Path(latest).parent.mkdir(parents=True, exist_ok=True)
    with open(latest, "x", encoding="utf-8") as f:
        f.write(content)
    return Path(latest), archived
