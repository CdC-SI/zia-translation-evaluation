#!/usr/bin/env python3
"""

The script accepts MQM evaluation data (JSON files such as
``lettre_manu_cyr_sr_it_dual.json``) in one
of these shapes:

1. A folder path (``str`` / ``Path``) — every ``*.json`` file directly inside
   it is loaded and treated as one report.
2. A single JSON file path (``str`` / ``Path``).
3. A list/tuple of JSON file paths — each path is treated as ONE independent
   report, i.e. one row of the global summary table.

Output
------
A single Markdown string containing:

- **Section 1 — Global Summary**: one row per report, with a bold
  ``TOTAL / AVERAGE`` footer row (mean MQM score, summed error counts).
- **Section 2 — Detailed Per-Report Breakdown**: for every report, the MQM
  error-category legend followed by three tables (Minor / Major / Critical)
  listing every individual error found in that report.

Usage
-----
Scan a specific folder and write the result to a chosen directory::

    python src/pipeline/report_md.py data/evaluation_results --output-dir data/reports/md

Aggregate a handful of specific files::

    python src/pipeline/report_md.py file1.json file2.json --output-dir data/reports/md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import markdown as md
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# All input/output locations are supplied explicitly via CLI arguments (see
# `parse_args`) — no environment-variable-based defaults are used here.

# MQM severity → penalty weight (WMT / GEMBA-MQM standard weights).
SEVERITY_WEIGHTS: dict[str, int] = {"MINOR": 1, "MAJOR": 5, "CRITICAL": 25}

# A document passes the quality gate when its normalized score meets this bar.
QUALITY_GATE_THRESHOLD = 75.0

logger = logging.getLogger("zia-report-md")

# ─────────────────────────────────────────────────────────────────────────────
# STATIC CONTENT — MQM ERROR CATEGORY REFERENCE
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_REFERENCE_MD = """\
- **ACC — Accuracy**
  - `MISTRANSLATION`: Translated text conveys a different meaning.
  - `OMISSION`: Source content is missing in the translation.
  - `ADDITION`: Content is added that has no basis in the source.
  - `UNTRANSLATED`: Source text left in the original language.
- **FLU — Fluency & Style**
  - `GRAMMAR`: Unnatural phrasing, syntactic errors, or grammatical mistakes.
  - `TERMINOLOGY`: Incorrect domain vocabulary or failure to maintain tone.
- **FOR — Formatting & Structure (Markdown)**
  - `STRUCTURE_LOSS`: Headings, lists, tables, or emphasis are not preserved.
  - `SEGMENTATION`: Paragraphs or sections are wrongly merged, split, or reordered.
- **COM — Completeness**
  - `HALLUCINATION`: Content in the translation that does not exist in the source.

| Severity | Weight | Meaning                                                                    |
|----------|--------|----------------------------------------------------------------------------|
| MINOR    | 1      | Blemish; does not affect comprehension.                                    |
| MAJOR    | 5      | Misrepresents source meaning or disrupts reading flow.                     |
| CRITICAL | 25     | Dangerous misinformation, complete meaning reversal, or unreadable output. |

**Sample Reliability** — how much a document's word count can be trusted to
produce a stable, representative MQM score:

| Reliability | Total Words     | Meaning                                                        |
|-------------|------------------|---------------------------------------------------------------|
| VERY_LOW    | < 200            | Sample too small; a single error can swing the score sharply. |
| LOW         | 200 – 999        | Small sample; score should be interpreted with caution.       |
| MEDIUM      | 1,000 – 4,999    | Reasonable sample size for a fairly stable score.             |
| HIGH        | ≥ 5,000          | Large sample; score is statistically robust.                  |

"""

GLOBAL_SUMMARY_HEADERS: list[str] = [
    "File Name",
    "Strategy",
    "Pages",
    "Doc MQM Score",
    "Sample Reliability",
    "QA",
    "Total Penalty",
    "Total Words",
    "Total Min. Err.",
    "Total Maj. Err.",
    "Total Crit. Err.",
]

ERROR_TABLE_HEADERS: list[str] = [
    "Page",
    "Segment",
    "Dimension",
    "Type",
    "Source Excerpt",
    "Target Excerpt",
    "Justification",
]


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ErrorRecord:
    """One individual MQM error, flattened out of a page's segments."""
    page: Optional[int]
    segment_id: str
    error_id: str
    dimension: str
    error_type: str
    severity: str
    source_excerpt: str
    target_excerpt: str
    justification: str


@dataclass
class ReportEntry:
    """One aggregated, document-level report (one row of the global table)."""
    name: str
    strategy: str
    pages_evaluated: int
    total_words: int
    total_penalty: int
    minor_count: int
    major_count: int
    critical_count: int
    final_score: float
    quality_gate: str
    normalized_penalty: float = 0.0
    errors: "list[ErrorRecord]" = field(default_factory=list)
    judge_model: str = "unknown"
    prompt_hash: str = "unknown"
    temperature: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MQM SCORE RECALCULATION  (mirrors report.py's arithmetic philosophy)
# ─────────────────────────────────────────────────────────────────────────────

def compute_document_mqm_score(total_penalty: int, total_words: int) -> dict[str, Any]:
    """Aggregate a document's total penalty/words into a normalized MQM score.

    Implements the WMT standard formula (identical to ``report.py``):

        Final MQM Score = max(0, 100 − (ΣPenalty / ΣWords × 1000))

    Falls back to the raw penalty when no word count is available so a
    missing ``translation_word_count`` never raises a ``ZeroDivisionError``.
    """
    if total_words > 0:
        normalized_penalty = (total_penalty / total_words) * 1000
    else:
        normalized_penalty = float(total_penalty)

    final_score = max(0.0, 100.0 - normalized_penalty)
    quality_gate = "PASS" if final_score >= QUALITY_GATE_THRESHOLD else "FAIL"

    return {
        "final_score": round(final_score, 2),
        "quality_gate": quality_gate,
        "normalized_penalty": round(normalized_penalty, 2),
    }


def compute_sample_reliability(total_words: int) -> str:
    """Classify how reliable the MQM score is based on the sample size.

    A small word count means a handful of errors can swing the normalized
    score wildly, so the confidence in the score should be flagged.
    """
    if total_words < 200:
        return "VERY_LOW"
    if total_words < 1000:
        return "LOW"
    if total_words < 5000:
        return "MEDIUM"
    return "HIGH"


def detect_strategy(name: str) -> str:
    """Parse the pipeline strategy tag ("Dual" / "Single") from a filename."""
    stem = name.lower()
    if "_dual_" in stem:
        return "Dual"
    if "_single_" in stem:
        return "Single"
    return "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# ROBUST FILE / STRING LOADING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _read_json_file(path: Path) -> Optional[Any]:
    """Read and parse a JSON file, logging a warning instead of raising."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except OSError as exc:
        logger.warning("Could not read file '%s': %s", path, exc)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON in file '%s': %s", path, exc)
    return None


def load_evaluations(
    source: Union[str, Path, Sequence[Union[str, Path]]],
) -> "list[tuple[str, Any]]":
    """Load evaluation JSON files into ``(file_name, json_data)`` pairs.

    Accepted shapes for *source*:

    - A directory path  → every ``*.json`` file directly inside it (sorted
      by name), e.g. ``lettre_manu_cyr_sr_it_dual.json``.
    - A single file path → that one file.
    - A list/tuple of file paths → each path is loaded individually.

    Unreadable or invalid JSON files are skipped with a warning rather than
    raising, so one bad file never aborts the whole run.
    """
    loaded: "list[tuple[str, Any]]" = []

    def _load_one(path: Path) -> None:
        data = _read_json_file(path)
        if data is not None:
            loaded.append((path.name, data))

    if isinstance(source, (str, Path)):
        path = Path(source)

        if path.is_dir():
            json_files = sorted(path.glob("*.json"))
            if not json_files:
                logger.warning("No '.json' files found in directory '%s'.", path)
            for file_path in json_files:
                _load_one(file_path)
            return loaded

        if path.is_file():
            _load_one(path)
        else:
            logger.warning("Path does not exist: '%s'.", path)
        return loaded

    if isinstance(source, (list, tuple)):
        for item in source:
            path = Path(item)
            if path.is_file():
                _load_one(path)
            else:
                logger.warning("Skipping non-existent file: %s", item)
        return loaded

    logger.warning("Unsupported type for `source`: %s", type(source).__name__)
    return loaded


# ─────────────────────────────────────────────────────────────────────────────
# PAGE / REPORT VALIDATION & EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _process_page(page: Any, source_name: str) -> Optional[dict[str, Any]]:
    """Validate and normalize one page-level MQM record.

    Recomputes ``minor_count`` / ``major_count`` / ``critical_count`` and
    ``total_penalty`` directly from ``segments[*].errors[*].severity``
    (mirroring ``report.py``'s recalculation philosophy) so that any
    arithmetic drift introduced upstream by an LLM is corrected here.

    Returns ``None`` (after logging a warning) when the page lacks the
    minimum structural fields required to compute an MQM score, so the
    caller can skip it gracefully instead of crashing.
    """
    if not isinstance(page, dict):
        logger.warning("Skipping a non-object page entry found in %r.", source_name)
        return None

    segments = page.get("segments")
    mqm_score = page.get("mqm_score")

    if not isinstance(segments, list) and not isinstance(mqm_score, dict):
        logger.warning(
            "Skipping page %r in %r: missing required MQM fields "
            "('segments' and 'mqm_score' are both absent).",
            page.get("document_id", "unknown"), source_name,
        )
        return None

    minor = major = critical = 0
    errors: "list[dict[str, Any]]" = []

    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            segment_id = segment.get("segment_id", "unknown")

            for err in segment.get("errors") or []:
                if not isinstance(err, dict):
                    continue

                severity = str(err.get("severity") or "").upper()
                if severity == "MINOR":
                    minor += 1
                elif severity == "MAJOR":
                    major += 1
                elif severity == "CRITICAL":
                    critical += 1
                else:
                    logger.warning(
                        "Unrecognized severity %r for error %s in %r.",
                        severity, err.get("error_id", "?"), source_name,
                    )

                location = err.get("location")
                if isinstance(location, dict):
                    source_excerpt = location.get("source_substring", "")
                    target_excerpt = location.get("target_substring", "")
                else:
                    source_excerpt = str(location) if location else ""
                    target_excerpt = ""

                errors.append({
                    "page":            page.get("page"),
                    "segment_id":      segment_id,
                    "error_id":        err.get("error_id", "?"),
                    "dimension":       err.get("dimension", ""),
                    "error_type":      err.get("type", ""),
                    "severity":        severity,
                    "source_excerpt":  source_excerpt,
                    "target_excerpt":  target_excerpt,
                    "justification":   err.get("justification", ""),
                })
    elif isinstance(mqm_score, dict):
        # Degenerate fallback: no segments, but pre-computed counts exist.
        try:
            minor = int(mqm_score.get("minor_count") or 0)
            major = int(mqm_score.get("major_count") or 0)
            critical = int(mqm_score.get("critical_count") or 0)
        except (TypeError, ValueError):
            minor = major = critical = 0

    total_penalty = (
        minor * SEVERITY_WEIGHTS["MINOR"]
        + major * SEVERITY_WEIGHTS["MAJOR"]
        + critical * SEVERITY_WEIGHTS["CRITICAL"]
    )

    metadata = page.get("metadata") or {}
    try:
        word_count = int(metadata.get("translation_word_count") or 0)
    except (TypeError, ValueError):
        word_count = 0

    return {
        "word_count":      word_count,
        "minor_count":      minor,
        "major_count":      major,
        "critical_count":   critical,
        "total_penalty":    total_penalty,
        "errors":           errors,
        "judge_model":      metadata.get("judge_model", "unknown"),
        "prompt_hash":      metadata.get("prompt_hash", "unknown"),
        "temperature":      metadata.get("temperature", 0.0),
    }


def build_report_entries(loaded: "list[tuple[str, Any]]") -> "list[ReportEntry]":
    """Convert raw ``(name, json_data)`` pairs into validated ``ReportEntry`` objects.

    A report that yields no valid page data at all (e.g. every page is
    malformed) is logged and skipped entirely, so one bad file can never
    abort the whole run.
    """
    entries: "list[ReportEntry]" = []

    for name, raw_data in loaded:
        if isinstance(raw_data, dict):
            pages: "list[Any]" = [raw_data]
        elif isinstance(raw_data, list):
            pages = raw_data
        else:
            logger.warning(
                "Skipping report %r: unexpected JSON root type '%s' (expected "
                "object or array).", name, type(raw_data).__name__,
            )
            continue

        processed_pages: "list[dict[str, Any]]" = []
        for page in pages:
            try:
                processed = _process_page(page, name)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "Unexpected error while processing a page in %r: %s", name, exc
                )
                continue
            if processed is not None:
                processed_pages.append(processed)

        if not processed_pages:
            logger.warning(
                "Report %r contains no valid MQM page data; skipping.", name
            )
            continue

        total_words = sum(p["word_count"] for p in processed_pages)
        total_penalty = sum(p["total_penalty"] for p in processed_pages)
        minor = sum(p["minor_count"] for p in processed_pages)
        major = sum(p["major_count"] for p in processed_pages)
        critical = sum(p["critical_count"] for p in processed_pages)

        doc_score = compute_document_mqm_score(total_penalty, total_words)

        errors = [
            ErrorRecord(**err) for p in processed_pages for err in p["errors"]
        ]

        # Take the first non-null value found across pages (should be
        # consistent for a single run, but a page may lack this metadata).
        judge_model = next(
            (p["judge_model"] for p in processed_pages if p.get("judge_model")), "unknown"
        )
        prompt_hash = next(
            (p["prompt_hash"] for p in processed_pages if p.get("prompt_hash")), "unknown"
        )
        temperature = next(
            (p["temperature"] for p in processed_pages if p.get("temperature") is not None),
            0.0,
        )

        entries.append(ReportEntry(
            name=name,
            strategy=detect_strategy(name),
            pages_evaluated=len(processed_pages),
            total_words=total_words,
            total_penalty=total_penalty,
            minor_count=minor,
            major_count=major,
            critical_count=critical,
            final_score=doc_score["final_score"],
            quality_gate=doc_score["quality_gate"],
            normalized_penalty=doc_score["normalized_penalty"],
            errors=errors,
            judge_model=judge_model,
            prompt_hash=prompt_hash,
            temperature=temperature,
        ))

    return entries


# ─────────────────────────────────────────────────────────────────────────────
# MARKDOWN RENDERING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _bold(value: Any) -> str:
    """Wrap *value* in Markdown bold markers."""
    return f"**{value}**"


def _escape_cell(value: Any) -> str:
    """Sanitize a value for safe embedding inside a Markdown table cell."""
    text = "" if value is None else str(value)
    text = text.replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    return text.strip()


def _make_md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a GitHub-flavored Markdown table from headers and row values."""
    if not rows:
        return "*No data available.*\n"

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = [
        "| " + " | ".join(_escape_cell(v) for v in row) + " |" for row in rows
    ]
    return "\n".join([header_line, separator_line, *body_lines]) + "\n"


def _render_run_metadata_line(entries: "list[ReportEntry]") -> str:
    """Build the ``Judge model · Prompt hash · Temperature`` metadata line.

    Values are collected from every entry's metadata; when multiple distinct
    values are found across reports (e.g. a mixed run), they are joined with
    ``/`` so drift between runs is visible instead of silently hidden.
    """
    def _unique_values(values: "list[Optional[Any]]") -> str:
        seen = [str(v) for v in values if v is not None and str(v) != ""]
        unique = sorted(set(seen))
        return " / ".join(unique) if unique else "—"

    judge_model = _unique_values([e.judge_model for e in entries])
    prompt_hash = _unique_values([e.prompt_hash for e in entries])
    temperature = _unique_values([e.temperature for e in entries])

    return (
        f"**Judge model:** {judge_model}  ·  "
        f"**Prompt hash:** `{prompt_hash}`  ·  "
        f"**Temperature:** {temperature}"
    )


def render_global_summary_table(entries: "list[ReportEntry]") -> str:
    """Build Section 1's global summary table, including the bold footer row."""
    if not entries:
        return "*No reports were successfully processed.*\n"

    rows: "list[list[Any]]" = []
    for entry in entries:
        gate_display = "✅ PASS" if entry.quality_gate == "PASS" else "❌ FAIL"
        reliability = compute_sample_reliability(entry.total_words)
        rows.append([
            entry.name[:30],  # Truncated to keep the summary table readable.
            entry.strategy,
            entry.pages_evaluated,
            f"{entry.final_score:.2f}",
            reliability,
            gate_display,
            entry.total_penalty,
            entry.total_words,
            entry.minor_count,
            entry.major_count,
            entry.critical_count,
        ])

    # ── Footer: Document MQM Score → mean; error counts → sum. ─────────────
    # Pages / Total Penalty / Total Words are also summed since the row is
    # explicitly a "TOTAL" row for every count-like column.
    mean_score = statistics.fmean(entry.final_score for entry in entries)
    total_words_all = sum(entry.total_words for entry in entries)
    rows.append([
        _bold("TOTAL / AVERAGE"),
        _bold("—"),
        _bold(sum(entry.pages_evaluated for entry in entries)),
        _bold(f"{mean_score:.2f}"),
        _bold(compute_sample_reliability(total_words_all)),
        _bold("—"),
        _bold(sum(entry.total_penalty for entry in entries)),
        _bold(total_words_all),
        _bold(sum(entry.minor_count for entry in entries)),
        _bold(sum(entry.major_count for entry in entries)),
        _bold(sum(entry.critical_count for entry in entries)),
    ])

    return _make_md_table(GLOBAL_SUMMARY_HEADERS, rows)


def _errors_table_for_severity(errors: "list[ErrorRecord]", severity: str) -> str:
    """Render the error table for a single severity level."""
    filtered = [e for e in errors if e.severity == severity]
    if not filtered:
        return f"*No {severity.lower()} errors detected.*\n"

    rows = [
        [
            e.page, e.segment_id, e.dimension, e.error_type,
            e.source_excerpt, e.target_excerpt, e.justification,
        ]
        for e in filtered
    ]
    return _make_md_table(ERROR_TABLE_HEADERS, rows)


def render_report_section(entry: ReportEntry) -> str:
    """Build the full Section-2 breakdown for a single report."""
    gate_display = "✅ PASS" if entry.quality_gate == "PASS" else "❌ FAIL"

    lines = [
        f"## 📄 {entry.name}",
        "",
        f"**Strategy:** {entry.strategy}  ",
        f"**Document MQM Score:** {entry.final_score:.2f}  |  "
        f"**Quality Gate:** {gate_display}  ",
        f"**Pages Evaluated:** {entry.pages_evaluated}  |  "
        f"**Total Penalty:** {entry.total_penalty}  |  "
        f"**Total Words:** {entry.total_words}",
        "",
        "<details>",
        "<summary>MQM Score Calculation</summary>",
        "",
        "```",
        f"normalized_penalty = (total_penalty / total_words) * 1000",
        f"                   = ({entry.total_penalty} / {entry.total_words}) * 1000",
        f"                   = {entry.normalized_penalty}",
        "",
        f"final_score = max(0.0, 100.0 - normalized_penalty)",
        f"            = max(0.0, 100.0 - {entry.normalized_penalty})",
        f"            = {entry.final_score:.2f}",
        "```",
        "",
        "</details>",
        "",
        "",
        f"#### 🟡 Minor Errors ({entry.minor_count})",
        "",
        _errors_table_for_severity(entry.errors, "MINOR"),
        f"#### 🟠 Major Errors ({entry.major_count})",
        "",
        _errors_table_for_severity(entry.errors, "MAJOR"),
        f"#### 🔴 Critical Errors ({entry.critical_count})",
        "",
        _errors_table_for_severity(entry.errors, "CRITICAL"),
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def write_markdown_file(markdown: str, output_path: Path) -> Path:
    """Write *markdown* to *output_path*, creating parent directories."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    logger.info("Markdown report written -> %s", output_path)
    return output_path


# HTML page chrome wrapped around the converted Markdown body so the
# resulting file is a fully self-contained, nicely styled document.
_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    max-width: 1200px;
    margin: 2rem auto;
    padding: 0 1.5rem;
    line-height: 1.6;
    color: #1f2328;
  }}
  h1, h2, h3, h4 {{ border-bottom: 1px solid #d0d7de; padding-bottom: .3em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; display: block; overflow-x: auto; }}
  th, td {{ border: 1px solid #d0d7de; padding: 6px 10px; text-align: left; }}
  th {{ background-color: #f6f8fa; }}
  tr:nth-child(even) {{ background-color: #f6f8fa; }}
  code {{ background-color: #f6f8fa; padding: .2em .4em; border-radius: 6px; }}
  pre {{ background-color: #f6f8fa; padding: 1em; border-radius: 6px; overflow-x: auto; }}
  details {{ margin: .5em 0; }}
  hr {{ border: none; border-top: 1px solid #d0d7de; margin: 2em 0; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def markdown_to_html(markdown_text: str, title: str = "Report") -> str:
    """Convert a Markdown string into a fully self-contained HTML document."""
    body = md.markdown(
        markdown_text,
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    return _HTML_TEMPLATE.format(title=title, body=body)


def write_html_file(html: str, output_path: Path) -> Path:
    """Write *html* to *output_path*, creating parent directories."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("HTML report written -> %s", output_path)
    return output_path


def generate_markdown_report(
    reports: Union[str, Path, Sequence[Union[str, Path]]],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Translation Evaluation — Markdown Summary Report",
    html_output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Build the complete Markdown report string from *reports*.

    Parameters
    ----------
    reports:
        A directory path (scanned for ``*.json`` files), a single JSON file
        path, or a list/tuple of JSON file paths.
    output_path:
        If given, the resulting Markdown is also written to this file path
        (parent directories are created automatically).
    title:
        Top-level H1 heading for the generated document.
    html_output_path:
        If given, the Markdown report is additionally converted to a
        self-contained HTML document and written to this file path.

    Returns
    -------
    The complete Markdown report as a single string.
    """
    evaluations = load_evaluations(reports)
    entries = build_report_entries(evaluations)

    total_pages = sum(entry.pages_evaluated for entry in entries)
    total_errors = sum(len(entry.errors) for entry in entries)
    generated_at = datetime.now().strftime("%B %d, %Y at %H:%M")

    run_meta_line = _render_run_metadata_line(entries)

    parts: "list[str]" = [
        f"# {title}",
        "",
        f"*Generated on {generated_at}*  ·  *{len(entries)} report(s) processed*  ·  "
        f"*{total_pages} page(s) evaluated*  ·  *{total_errors} error(s) recorded*",
        "",
        run_meta_line,
        "",
        "---",
        "",
        "## 1. Global Summary",
        "",
        render_global_summary_table(entries),
        "---",
        "<br>",
        "",
        "## 2. Detailed Per-Report Breakdown",
        "",
        "### MQM Error Category Reference",
        "",
        CATEGORY_REFERENCE_MD,
        "",
        "<br>",
        "",
    ]

    if not entries:
        parts.append("*No valid reports to display.*")
    else:
        section_blocks = [render_report_section(entry) for entry in entries]
        parts.append("\n\n---\n\n".join(section_blocks))

    markdown = "\n".join(parts).rstrip() + "\n"

    if output_path is not None:
        if not entries:
            logger.error(
                "No valid reports were loaded from the given input(s); "
                "skipping write of '%s'.", output_path,
            )
        else:
            write_markdown_file(markdown, Path(output_path))

    if html_output_path is not None:
        if not entries:
            logger.error(
                "No valid reports were loaded from the given input(s); "
                "skipping write of '%s'.", html_output_path,
            )
        else:
            html = markdown_to_html(markdown, title=title)
            write_html_file(html, Path(html_output_path))

    return markdown


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Markdown summary report from one or more MQM "
            "evaluation JSON files (or a directory containing them)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/pipeline/report.py data/evaluation_results --output-dir data/reports/md\n"
            "  python src/pipeline/report.py file1.json file2.json --output-dir data/reports/md\n"
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help=(
            "One or more JSON file paths, or a single directory containing "
            "'.json' evaluation files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory to write auto-named 'report_YYYYMMDD_HHMMSS.md' and "
            "'report_YYYYMMDD_HHMMSS.html' files into."
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    # A single positional argument is passed through as-is (it may be a
    # directory, in which case `load_evaluations` performs its own folder scan).
    # Multiple positional arguments are forwarded as a list of file paths.
    reports_input: Union[str, "list[str]"]
    if len(args.paths) == 1:
        reports_input = args.paths[0]
    else:
        reports_input = list(args.paths)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output_dir / f"report_{timestamp}.md"
    html_output_path = args.output_dir / f"report_{timestamp}.html"

    evaluations = load_evaluations(reports_input)
    if not evaluations:
        logger.error(
            "No valid reports were loaded from the given input(s): %s. "
            "Check that the path(s) exist and contain valid JSON.",
            reports_input,
        )
        return 1

    generate_markdown_report(
        reports_input,
        output_path=output_path,
        html_output_path=html_output_path,
    )

    print(
        f"\n{'='*60}\n"
        f"  ✅  Markdown report ready: {output_path}\n"
        f"  ✅  HTML report ready: {html_output_path}\n"
        f"{'='*60}\n"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
