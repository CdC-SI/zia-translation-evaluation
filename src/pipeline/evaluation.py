#!/usr/bin/env python3
"""LLM-as-a-judge evaluation pipeline for the OCR + translation process.

Given an *original* document (PDF or PNG) and the *translated Markdown*
produced by the OCR + translation pipeline, this script:

1. Extracts text from the original page by page (PyMuPDF; a PNG is treated
   as a single-page document).
2. Splits the translated Markdown into pages using the page-break divider
   ``<div style="page-break-after: always;"></div>``.
   Single-page Markdowns (no divider) are treated as one page.
3. Verifies that both documents have the same number of pages.
4. Sends each original page paired with its corresponding Markdown page
   to a vision/language model (OpenAI-compatible vLLM endpoint).
5. Receives a structured JSON report per page, then aggregates all page
   reports into a single JSON list written to ``data/evaluation_results``.

Two modes are selected automatically based on ``--original``:

* **Single-file mode** — ``--original`` is a single ``.pdf``/``.png`` file.
  ``--translated`` may be a single ``.md`` file, or a directory (the most
  recently modified ``.md`` file inside it is then used). This is the
  original 1-to-1 behaviour.

* **Batch (1-to-many) mode** — ``--original`` is a *directory*. Every
  ``.pdf``/``.png`` file directly inside it is evaluated. ``--translated``
  must then also be a directory: for each original file, every Markdown
  file whose name *starts with* that original file's stem (its filename
  without extension) is treated as a match and evaluated as its own
  independent run (one output report per match). Original files with no
  matching translation are logged as a warning and skipped.

Examples
--------
Single-file mode — evaluate one translated Markdown against one original::

    python evaluation.py \\
        --original data/pdfs/weisungen_1.pdf \\
        --translated data/translations/weisungen_1_fr_dual_20260619_110746.md

If ``--translated`` (or ``--original``) is a directory in single-file mode,
the most recently modified file of the expected type inside it is used::

    python evaluation.py -O data/pdfs/weisungen_1.pdf -T data/translations

Batch mode — evaluate every original in a folder against every translation
that matches it in another folder (e.g. ``certificat_travail_01_fr.png``
matches both ``certificat_travail_01_fr_it_dual_...md`` and
``certificat_travail_01_fr_de_dual_...md``)::

    python evaluation.py -O data/validation_set -T data/translations/validation_set
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Union

# Ensure the project root is on sys.path so both
#   python src/pipeline/evaluation.py   and
#   python -m src.pipeline.evaluation
# resolve imports correctly.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import fitz  # PyMuPDF
import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import ValidationError

from src.pipeline.schemas import LLMPageReport, ParsedPageReport
from src.pipeline.word_count import count_translation_words

# Load environment variables from a `.env` file (if present) before defaults
# are computed below.
load_dotenv(_PROJECT_ROOT / ".env")


# --- Defaults ---------------------------------------------------------------
# Every DEFAULT_* value can be overridden via an environment variable of the
# same name (e.g. in a `.env` file at the project root). See `.env.example`.

LLM_JUDGE_URL = os.getenv("LLM_JUDGE_URL")
LLM_JUDGE_MODEL_NAME = os.getenv("LLM_JUDGE_MODEL_NAME")
CA_BUNDLE = os.getenv("CA_BUNDLE")
LLM_JUDGE_API_KEY = os.getenv("LLM_JUDGE_API_KEY")

LLM_JUDGE_PROMPT = Path(os.getenv("LLM_JUDGE_PROMPT", "prompts/evaluation_prompt.md"))
LLM_JUDGE_TEMPERATURE = float(os.getenv("LLM_JUDGE_TEMPERATURE", "0.0"))          # deterministic evaluation
LLM_JUDGE_MAX_TOKEN = int(os.getenv("LLM_JUDGE_MAX_TOKEN", "8192"))          # budget for dense pages (18+ segments, 13+ errors)
LLM_JUDGE_MAX_CHAR = int(os.getenv("LLM_JUDGE_MAX_CHAR", "24000"))          # per markdown page, to stay within context
LLM_JUDGE_RENDER_DPI = int(os.getenv("LLM_JUDGE_RENDER_DPI", "200"))           # 200 DPI is the sweet spot for VLM OCR quality
LLM_JUDGE_REQUEST_TIMEOUT = float(os.getenv("LLM_JUDGE_REQUEST_TIMEOUT", "300.0"))    # seconds to wait for the model response
LLM_JUDGE_FREQUENCY_PENALTY = float(os.getenv("LLM_JUDGE_FREQUENCY_PENALTY", "0"))    # set to 0 so far but can be tuned if needed
LLM_JUDGE_PRESENCE_PENALTY = float(os.getenv("LLM_JUDGE_PRESENCE_PENALTY", "0"))    # set to 0 so far but can be tuned if needed

DATE_FORMAT = "%Y%m%d_%H%M%S"
PDF_SUFFIX = ".pdf"
PNG_SUFFIX = ".png"
# Original documents may be a native PDF or a scanned/rasterised PNG.
ORIGINAL_SUFFIXES = (PDF_SUFFIX, PNG_SUFFIX)
MD_SUFFIX = ".md"
# Divider the translation pipeline inserts between pages in Markdown output.
MD_PAGE_BREAK = '<div style="page-break-after: always;"></div>'

# Translated filenames are built as
# "{stem}_target-{language}_{strategy}_{date}{suffix}" — {language} may be
# an ISO 639-1 code (e.g. "fr") or a full language name (e.g. "french").
TARGET_LANGUAGE_PATTERN = re.compile(r"_target-([^_]+)_")

logger = logging.getLogger("zia-evaluation")


def parse_target_language_from_filename(filename: str) -> str:
    """Extract the ``{language}`` token from a translated filename built as
    ``{stem}_target-{language}_{strategy}_{date}{suffix}``.

    ``{language}`` may be an ISO 639-1 code (e.g. ``fr``) or a full language
    name (e.g. ``french``).

    Raises:
        ValueError: if the filename does not contain a ``_target-<lang>_``
            segment.
    """
    match = TARGET_LANGUAGE_PATTERN.search(filename)
    if not match:
        raise ValueError(
            f"Could not parse target language from filename: {filename!r} "
            "(expected pattern '..._target-<language>_<strategy>_<date>...')"
        )
    return match.group(1)


# --- Prompt loader ----------------------------------------------------------

_MD_FENCE = re.compile(r"^```[a-zA-Z]*\n?(.*?)\n?```\s*$", re.DOTALL)


def load_prompt(path: Path) -> str:
    """Load a system prompt from a Markdown file.

    Strips any surrounding markdown code fence (```markdown ... ```) that
    some prompt files use as a wrapper.
    """
    raw = path.read_text(encoding="utf-8").strip()
    m = _MD_FENCE.match(raw)
    if m:
        raw = m.group(1).strip()
    return raw


def compute_prompt_hash(prompt_text: str) -> str:
    """Return a short, stable hash of a prompt's content (drift detection)."""
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:12]



# --- PDF/PNG helpers ---------------------------------------------------------

def resolve_original_file(path: Path, *, label: str) -> Optional[Path]:
    """Resolve ``path`` to a single concrete original PDF/PNG file.

    ``path`` may be a ``.pdf``/``.png`` file or a directory; for a directory
    the most recently modified PDF/PNG is chosen. Returns ``None`` (and logs
    an error) if no suitable file is found.

    Used for single-file mode only; batch (directory) mode uses
    ``find_original_files`` instead, which returns *every* match.
    """
    if path.is_dir():
        candidates = sorted(
            (p for p in path.glob("*") if p.is_file() and p.suffix.lower() in ORIGINAL_SUFFIXES),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            logger.error("No PDF/PNG found in %s directory: %s", label, path)
            return None
        chosen = candidates[-1]
        logger.info("Using newest %s file: %s", label, chosen)
        return chosen
    if path.is_file() and path.suffix.lower() in ORIGINAL_SUFFIXES:
        return path
    logger.error("%s path is not a PDF/PNG file or directory: %s", label, path)
    return None


def find_original_files(directory: Path) -> list[Path]:
    """Return every PDF/PNG file directly inside ``directory``, sorted by name.

    Used by batch (1-to-many) mode, where every original document in the
    folder is evaluated against its matching translation(s).
    """
    return sorted(
        p for p in directory.glob("*")
        if p.is_file() and p.suffix.lower() in ORIGINAL_SUFFIXES
    )


def find_matching_translations(original_stem: str, translated_dir: Path) -> list[Path]:
    """Return every Markdown file in ``translated_dir`` matching ``original_stem``.

    A translation file is considered a match if its filename *starts with*
    the exact ``original_stem`` — the original file's name with its
    extension stripped. This lets a single original document match several
    translations, e.g. one per target language and/or strategy::

        original_stem = "certificat_travail_01_fr"
        matches:
            certificat_travail_01_fr_it_dual_20260708_113921.md
            certificat_travail_01_fr_de_dual_20260708_113921.md

    Results are sorted by name for deterministic run order.
    """
    matches = [
        p for p in translated_dir.glob(f"*{MD_SUFFIX}")
        if p.is_file() and p.name.startswith(original_stem)
    ]
    return sorted(matches)


def resolve_translated_md(path: Path) -> Optional[Path]:
    """Resolve ``path`` to a concrete Markdown file.

    ``path`` may be a ``.md`` file or a directory; for a directory the most
    recently modified ``.md`` file is chosen. Returns ``None`` (and logs an
    error) if no suitable file is found.
    """
    if path.is_dir():
        candidates = sorted(
            (p for p in path.glob("*.md") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            logger.error("No Markdown file found in translated directory: %s", path)
            return None
        chosen = candidates[-1]
        logger.info("Using newest translated Markdown: %s", chosen)
        return chosen
    if path.is_file() and path.suffix.lower() == MD_SUFFIX:
        return path
    logger.error("Translated path is not a .md file or directory: %s", path)
    return None


def split_markdown_pages(
    path: Path, *, max_chars: int
) -> list[tuple[int, str, int]]:
    """Split a translated Markdown file into per-page chunks.

    Pages are delimited by ``<div style="page-break-after: always;"></div>``.
    If the file contains no such divider (single-page document), the entire
    content is returned as page 1. Each chunk is individually truncated to
    ``max_chars`` (0 disables truncation).

    The word count is computed with ``count_translation_words`` on the full,
    untruncated page text *before* the ``max_chars`` truncation is applied,
    so the reported ``translation_word_count`` always reflects the complete
    translated page, not the (possibly truncated) text sent to the judge.

    Returns a list of ``(page_num, text, translation_word_count)`` tuples
    (1-based).
    """
    raw = path.read_text(encoding="utf-8")
    chunks = raw.split(MD_PAGE_BREAK)
    pages: list[tuple[int, str, int]] = []
    for index, chunk in enumerate(chunks, start=1):
        text = chunk.strip()
        word_count = count_translation_words(text)
        if max_chars > 0 and len(text) > max_chars:
            logger.warning(
                "Truncating markdown page %d of %s from %d to %d characters",
                index, path.name, len(text), max_chars,
            )
            text = text[:max_chars]
        pages.append((index, text, word_count))
    return pages


def render_original_pages(path: Path, *, dpi: int) -> list[tuple[int, str]]:
    """Render every page of ``path`` to a PNG and return base64-encoded strings.

    Accepts either a PDF (rendered page by page) or a PNG (opened by PyMuPDF
    as a single-page pseudo-document, so no special-casing is needed here).
    Each page is rasterised in-memory at ``dpi`` resolution using PyMuPDF —
    no temporary files are written. Returns a list of ``(page_num, b64_png)``
    tuples (1-based). 200 DPI is the recommended sweet spot for VLM OCR quality.
    """
    pages: list[tuple[int, str]] = []
    with fitz.open(path) as doc:
        for index, page in enumerate(doc, start=1):
            matrix = fitz.Matrix(dpi / 72, dpi / 72)
            pixmap = page.get_pixmap(matrix=matrix)
            png_bytes = pixmap.tobytes("png")
            b64 = base64.b64encode(png_bytes).decode("ascii")
            logger.debug(
                "Rendered page %d of %s at %d DPI (%d bytes PNG)",
                index, path.name, dpi, len(png_bytes),
            )
            pages.append((index, b64))
    return pages


def build_user_message(
    page_image_b64: str, translated_text: str, expected_target_language: str
) -> list[dict]:
    """Build a multimodal user message for the VLM.

    The original document is sent as a base64-encoded PNG image so the model
    can read it visually (handles scanned / image-based PDFs). The translated
    Markdown is appended as plain text, preceded by the target language the
    pipeline was configured to produce (parsed from the translated file's
    name), so the model can verify the translation is actually in that
    language.
    """
    return [
        {"type": "text", "text": "ORIGINAL_DOCUMENT (image of PDF page):"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{page_image_b64}"},
        },
        {
            "type": "text",
            "text": (
                f"EXPECTED_TARGET_LANGUAGE: {expected_target_language}\n\n"
                f"TRANSLATED_DOCUMENT:\n{translated_text}"
            ),
        },
    ]


# --- JSON parsing & validation ----------------------------------------------

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def parse_and_validate_report(
    raw: str,
    *,
    page_num: int,
    doc_stem: str,
    expected_target_language: str,
    translation_word_count: int,
    judge_model: str,
    prompt_hash: str,
    temperature: float,
) -> Optional[ParsedPageReport]:
    r"""Parse raw model output and validate it against the ParsedPageReport schema.

    Applies a two-tier recovery strategy:

    1. **Strip noise** — remove Qwen3 ``<think>...</think>`` blocks and any
       surrounding ``\`\`\`json`` code fences.
    2. **Standard parse** — ``json.loads``.  Succeeds for well-formed output.
    3. **Brace-slicing** — last resort; extracts the substring between the
       outermost ``{`` and ``}`` in case prose surrounds the JSON object.

    Truncation recovery (e.g. via ``json_repair``) is intentionally absent.
    A truncated response means the model's ``summary`` block is incomplete or
    garbage; silently patching it would hide the failure and produce wrong data
    in the report.  Instead, ``max_tokens`` is set high enough (8 192) that
    truncation should not occur, and any remaining failures are saved as raw
    files for manual inspection.

    On success the dict is stamped with ``document_id`` and ``page`` (the model
    is not expected to know the file stem) and validated with Pydantic.
    """
    if not raw:
        return None
    text = raw.strip()

    # 1. Strip Qwen3-style <think>...</think> reasoning block
    text = _THINK_BLOCK.sub("", text).strip()
    if not text:
        logger.debug("Model output contained only a thinking block and no JSON.")
        return None

    # Strip optional ```json ... ``` fences
    fenced = _JSON_FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    data: Optional[object] = None

    # 2. Standard parse
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3. Brace-slicing — last resort for prose-wrapped JSON
    if not isinstance(data, dict):
        starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
        end = max(text.rfind("}"), text.rfind("]"))
        if starts and end != -1 and end > min(starts):
            try:
                data = json.loads(text[min(starts): end + 1])
            except json.JSONDecodeError:
                pass

    if not isinstance(data, dict):
        logger.warning(
            "Page %d: could not extract a JSON object from model output "
            "(output length: %d chars). Raw file will be saved.",
            page_num, len(raw),
        )
        return None

    # Stamp page identity — the model is not expected to know the file stem.
    data["document_id"] = f"{doc_stem}_p{page_num}"
    data["page"] = page_num

    # The target language is authoritative from the filename, not the
    # model's guess — enforce it regardless of what the model returned.
    # The word count is computed deterministically by the pipeline, not by
    # the model — enforce it here regardless of what (if anything) the
    # model returned.
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        metadata["target_language"] = expected_target_language
        metadata["translation_word_count"] = translation_word_count
        metadata["judge_model"] = judge_model
        metadata["prompt_hash"] = prompt_hash
        metadata["temperature"] = temperature
    else:
        data["metadata"] = {
            "target_language": expected_target_language,
            "translation_word_count": translation_word_count,
            "judge_model": judge_model,
            "prompt_hash": prompt_hash,
            "temperature": temperature,
        }

    try:
        return ParsedPageReport.model_validate(data)
    except ValidationError as exc:
        logger.warning(
            "Page %d: JSON parsed but failed schema validation: %s",
            page_num,
            exc.errors()[0],
        )
        return None


# --- Model call -------------------------------------------------------------

async def request_evaluation(
    *,
    page_image_b64: str,
    translated_text: str,
    expected_target_language: str,
    system_prompt: str,
    base_url: str,
    model_name: str,
    api_key: str,
    ca_bundle: Optional[str],
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    enable_thinking: Optional[bool],
    frequency_penalty: float = LLM_JUDGE_FREQUENCY_PENALTY,
    presence_penalty: float = LLM_JUDGE_PRESENCE_PENALTY,
) -> str:
    """Call the vLLM chat-completions endpoint and return the raw text content."""
    verify: Union[str, bool] = ca_bundle if ca_bundle else True

    http_client = httpx.AsyncClient(verify=verify, timeout=request_timeout)
    client = AsyncOpenAI(
        base_url=f"{base_url.rstrip('/')}/v1",
        api_key=api_key,
        http_client=http_client,
    )

    kwargs: dict = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": build_user_message(
                    page_image_b64, translated_text, expected_target_language
                ),
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        # Discourage degenerate repetition loops (observed to occasionally
        # exhaust the entire max_tokens budget on a single repeated
        # justification sentence, truncating the JSON before it can close).
        # Configurable via LLM_JUDGE_FREQUENCY_PENALTY / LLM_JUDGE_PRESENCE_PENALTY
        # (default 0, but can be tuned if needed).
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }
    # Constrain output to the exact PageReport JSON schema.
    kwargs["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": "LLMPageReport",
            "schema": LLMPageReport.model_json_schema(),
            "strict": True,
        },
    }
    # Optionally control Qwen3-style thinking mode via chat_template_kwargs.
    if enable_thinking is not None:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}

    try:
        completion = await client.chat.completions.create(**kwargs)
    finally:
        await http_client.aclose()

    return completion.choices[0].message.content or ""


# --- Report output ----------------------------------------------------------

def write_raw_failure(
    raw: str,
    result_dir: Path,
    translated: Path,
    run_date: str,
    page_num: int,
) -> Path:
    """Persist the raw model output for a page that failed JSON parsing."""
    result_dir.mkdir(parents=True, exist_ok=True)
    out = result_dir / f"{translated.stem}_p{page_num}_{run_date}.raw.txt"
    out.write_text(raw, encoding="utf-8")
    logger.error("Model output was not valid JSON; wrote raw output -> %s", out)
    return out


# --- Core pipeline ----------------------------------------------------------

async def evaluate_pair(
    *,
    original: Path,
    original_pages: list[tuple[int, str]],
    translated_md: Path,
    system_prompt: str,
    run_date: str,
    args: argparse.Namespace,
) -> int:
    """Evaluate a single (original, translated) pair and write its report.

    ``original_pages`` is rendered by the caller (rather than re-rendered
    here) so batch mode can render each original once and reuse it across
    every translation that matches it.

    Returns ``0`` on full success, ``1`` if a report was written but some
    (or all) pages failed to parse, or ``2`` on a fatal error that prevented
    any report from being produced (e.g. a page-count mismatch).
    """
    try:
        translated_pages = split_markdown_pages(translated_md, max_chars=args.max_chars)
    except Exception as exc:  # pragma: no cover - depends on file contents
        logger.error("Failed to read translated markdown %s: %s", translated_md, exc)
        return 2

    try:
        expected_target_language = parse_target_language_from_filename(translated_md.name)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    # --- Page-count parity check -------------------------------------------
    if len(original_pages) != len(translated_pages):
        logger.error(
            "Page count mismatch for %s vs %s: original has %d page(s), "
            "translated has %d page(s). Both must have the same number of pages.",
            original.name, translated_md.name,
            len(original_pages), len(translated_pages),
        )
        return 2

    logger.info(
        "Original=%s (%d page(s)), Translated=%s (%d page(s))",
        original.name, len(original_pages),
        translated_md.name, len(translated_pages),
    )
    logger.info("Requesting evaluation from %s (model=%s)", args.base_url, args.model)

    doc_stem = original.stem          # e.g. "weisungen_1"
    page_reports: list[ParsedPageReport] = []
    failed_pages: list[int] = []
    prompt_hash = compute_prompt_hash(system_prompt)

    for (page_num, orig_b64), (_, trans_text, word_count) in zip(
        original_pages, translated_pages
    ):
        logger.info("Evaluating page %d / %d …", page_num, len(original_pages))

        if not trans_text:
            logger.warning("Page %d: no text in translated Markdown, skipping.", page_num)
            continue

        report = None
        raw = ""
        try:
            raw = await request_evaluation(
                page_image_b64=orig_b64,
                translated_text=trans_text,
                expected_target_language=expected_target_language,
                system_prompt=system_prompt,
                base_url=args.base_url,
                model_name=args.model,
                api_key=args.api_key,
                ca_bundle=args.ca_bundle,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                request_timeout=args.request_timeout,
                enable_thinking=args.enable_thinking,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
            )
        except Exception as exc:  # pragma: no cover - network/endpoint dependent
            logger.error("Evaluation request failed for page %d: %s", page_num, exc)

        if raw:
            report = parse_and_validate_report(
                raw,
                page_num=page_num,
                doc_stem=doc_stem,
                expected_target_language=expected_target_language,
                translation_word_count=word_count,
                judge_model=args.model,
                prompt_hash=prompt_hash,
                temperature=args.temperature,
            )

        if report is None:
            write_raw_failure(raw, args.result_dir, translated_md, run_date, page_num)
            failed_pages.append(page_num)
            continue

        page_reports.append(report)

    if not page_reports:
        logger.error("No pages produced a valid JSON report for %s.", translated_md.name)
        return 1

    # --- Write the final aggregated report (flat list, one entry per page) ---
    out = args.result_dir / f"{translated_md.stem}_{run_date}.json"
    args.result_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            [r.model_dump(mode="json") for r in page_reports],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "Wrote aggregated report (%d page(s)) -> %s", len(page_reports), out
    )

    if failed_pages:
        logger.warning(
            "Pages with parse failures (raw files saved): %s",
            ", ".join(str(p) for p in failed_pages),
        )
        return 1

    return 0


async def run_single(args: argparse.Namespace) -> int:
    """Single-file mode: evaluate exactly one (original, translated) pair.

    ``args.original`` is a single ``.pdf``/``.png`` file. ``args.translated``
    may be a single ``.md`` file, or a directory whose newest ``.md`` file is
    used. This preserves the original 1-to-1 CLI behaviour.
    """
    original = resolve_original_file(args.original, label="original")
    translated_md = resolve_translated_md(args.translated)
    if original is None or translated_md is None:
        return 2

    try:
        system_prompt = load_prompt(args.prompt)
    except OSError as exc:
        logger.error("Failed to load prompt file %s: %s", args.prompt, exc)
        return 2

    logger.info("Using prompt: %s", args.prompt)

    try:
        original_pages = render_original_pages(original, dpi=args.render_dpi)
    except Exception as exc:  # pragma: no cover - depends on file contents
        logger.error("Failed to render %s: %s", original.name, exc)
        return 2

    run_date = datetime.now().strftime(DATE_FORMAT)
    return await evaluate_pair(
        original=original,
        original_pages=original_pages,
        translated_md=translated_md,
        system_prompt=system_prompt,
        run_date=run_date,
        args=args,
    )


async def run_batch(args: argparse.Namespace) -> int:
    """Batch (1-to-many) mode: evaluate every original against every match.

    ``args.original`` is a directory: every ``.pdf``/``.png`` file directly
    inside it is evaluated. ``args.translated`` must also be a directory —
    for each original file, every Markdown file whose name starts with the
    original's stem (filename without extension) is evaluated as its own
    independent (original, translated) pair. Original files with no
    matching translation are logged as a warning and skipped.
    """
    if not args.translated.is_dir():
        logger.error(
            "When --original is a directory, --translated must also be a "
            "directory (got: %s)",
            args.translated,
        )
        return 2

    originals = find_original_files(args.original)
    if not originals:
        logger.error("No PDF/PNG files found in original directory: %s", args.original)
        return 2

    try:
        system_prompt = load_prompt(args.prompt)
    except OSError as exc:
        logger.error("Failed to load prompt file %s: %s", args.prompt, exc)
        return 2

    logger.info("Using prompt: %s", args.prompt)
    logger.info("Found %d original file(s) in %s", len(originals), args.original)

    run_date = datetime.now().strftime(DATE_FORMAT)
    worst_rc = 0
    evaluated_pairs = 0

    for original in originals:
        matches = find_matching_translations(original.stem, args.translated)
        if not matches:
            logger.warning(
                "No matching translation found for %s (expected a file in %s "
                "starting with %r); skipping.",
                original.name, args.translated, original.stem,
            )
            worst_rc = max(worst_rc, 1)
            continue

        try:
            original_pages = render_original_pages(original, dpi=args.render_dpi)
        except Exception as exc:  # pragma: no cover - depends on file contents
            logger.error("Failed to render %s: %s", original.name, exc)
            worst_rc = 2
            continue

        logger.info(
            "%s: found %d matching translation(s): %s",
            original.name, len(matches), ", ".join(m.name for m in matches),
        )

        for translated_md in matches:
            evaluated_pairs += 1
            logger.info("=== Evaluating %s + %s ===", original.name, translated_md.name)
            rc = await evaluate_pair(
                original=original,
                original_pages=original_pages,
                translated_md=translated_md,
                system_prompt=system_prompt,
                run_date=run_date,
                args=args,
            )
            worst_rc = max(worst_rc, rc)

    if evaluated_pairs == 0:
        logger.error("No (original, translation) pairs were evaluated.")
        return 2

    logger.info(
        "Batch evaluation complete: %d pair(s) evaluated across %d original file(s).",
        evaluated_pairs, len(originals),
    )
    return worst_rc


async def run(args: argparse.Namespace) -> int:
    """Dispatch to batch mode when ``--original`` is a directory, else single-file mode."""
    if args.original.is_dir():
        return await run_batch(args)
    return await run_single(args)


# --- CLI --------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an OCR + translation result by comparing the "
        "original (PDF/PNG) and translated Markdown with an LLM judge, writing "
        "strict-JSON report(s) into data/evaluation_results. --original may be a "
        "single file (single-file mode) or a directory (batch mode: every "
        "original is matched against every translation whose name starts with "
        "its stem).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-O",
        "--original",
        required=True,
        type=Path,
        help="Original source PDF/PNG file, or a directory of them. A file "
        "triggers single-file mode; a directory triggers batch mode, "
        "evaluating every PDF/PNG inside against its matching translation(s).",
    )
    parser.add_argument(
        "-T",
        "--translated",
        required=True,
        type=Path,
        help="Translated Markdown file (.md), or a directory. In single-file "
        "mode a directory falls back to its newest .md file; in batch mode "
        "(--original is a directory) it must be a directory, and every .md "
        "file whose name starts with an original file's stem is evaluated "
        "as a separate match.",
    )
    parser.add_argument(
        "-r",
        "--result-dir",
        dest="result_dir",
        required=True,
        type=Path,
        help="Folder where the JSON evaluation report is written.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=Path,
        default=LLM_JUDGE_PROMPT,
        help="Path to the system prompt Markdown file (prompts/ folder).",
    )
    parser.add_argument(
        "-u",
        "--base-url",
        default=LLM_JUDGE_URL,
        help="Base URL of the OpenAI-compatible (vLLM) endpoint.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=LLM_JUDGE_MODEL_NAME,
        help="Model name served by the endpoint.",
    )
    parser.add_argument(
        "--api-key",
        default=LLM_JUDGE_API_KEY,
        help="API key (vLLM ignores the value but requires a non-empty string).",
    )
    parser.add_argument(
        "--ca-bundle",
        default=CA_BUNDLE,
        help="Path to the CA bundle used to verify the endpoint's TLS certificate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=LLM_JUDGE_TEMPERATURE,
        help="Sampling temperature (0 = deterministic).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=LLM_JUDGE_MAX_TOKEN,
        help="Maximum number of tokens in the model's response.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=LLM_JUDGE_MAX_CHAR,
        help="Truncate each markdown page to this many characters (0 disables).",
    )
    parser.add_argument(
        "--render-dpi",
        type=int,
        default=LLM_JUDGE_RENDER_DPI,
        help="DPI used when rendering PDF pages to PNG images for the VLM (200 recommended).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=LLM_JUDGE_REQUEST_TIMEOUT,
        help="Seconds to wait for the model response.",
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=LLM_JUDGE_FREQUENCY_PENALTY,
        help="Frequency penalty sent to the model (discourages repetition loops).",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=LLM_JUDGE_PRESENCE_PENALTY,
        help="Presence penalty sent to the model (discourages repetition loops).",
    )
    parser.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        type=lambda v: {"true": True, "false": False}[v.lower()],
        default=False,
        metavar="{true,false}",
        help="Set chat_template_kwargs enable_thinking (true/false). Omit to leave unset.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
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
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
