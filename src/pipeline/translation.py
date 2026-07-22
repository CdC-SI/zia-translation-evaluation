#!/usr/bin/env python3
"""Translation evaluation pipeline.

This script sends PDF documents to the Zia translation endpoint and collects the
translated PDFs into a results folder, giving each output a normalized file name
that encodes the target language, the translation *strategy* and the run date.

How it works
------------
The endpoint translates an uploaded PDF and writes the result to a folder that is
configured *on the server side* (it is not part of the request). Because of that,
the script:

1. POSTs each input PDF together with the ``targetLanguage`` and ``strategy``
   fields.
2. Detects the translated PDF the server just wrote to its output folder
   (``--server-output-dir``). As a convenience it also handles servers that
   return the PDF directly in the HTTP response body.
3. Copies that translated PDF into the results folder (default ``data/translations``)
   using the name ``{original_stem}_{language}_{strategy}_{date}.pdf``.

The *strategy* (``single`` or ``dual``) is sent to the endpoint as the
``strategy`` field, and is also used to tag the produced file names so
different runs can be compared.

Examples
--------
Translate every PDF in ``data/pdf`` to French, tagging the run as ``baseline`` and
picking up the files the server wrote to ``/tmp/translations``::

    python translate.py \
        --input data/pdf \
        --target-language fr \
        --strategy baseline \
        --server-output-dir /tmp/translations

Translate a single file::

    python translate.py -i data/pdf/pdf_01.pdf -l fr -s baseline -o /tmp/translations
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import requests
from dotenv import load_dotenv

# Ensure the project root is on sys.path and load environment variables from
# a `.env` file (if present) before defaults are computed below.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
load_dotenv(_PROJECT_ROOT / ".env")

# --- Defaults ---------------------------------------------------------------
# Every DEFAULT_* value can be overridden via an environment variable of the
# same name (e.g. in a `.env` file at the project root). See `.env.example`.

ZIA_TRANSLATION_URL = os.getenv("ZIA_TRANSLATION_URL", "http://localhost:8080/zia-trad/api/translation")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "300.0"))  # seconds to wait for each HTTP response
POLL_TIMEOUT = float(os.getenv("POLL_TIMEOUT", "600.0"))     # overall seconds to wait for a job to finish
POLL_INITIAL_INTERVAL = float(os.getenv("POLL_INITIAL_INTERVAL", "1.0"))   # first delay between status polls
POLL_MAX_INTERVAL = float(os.getenv("POLL_MAX_INTERVAL", "30.0"))      # cap for the exponentially growing delay
POLL_BACKOFF_FACTOR = float(os.getenv("POLL_BACKOFF_FACTOR", "2.0"))     # multiply the delay by this after each poll
DATE_FORMAT = "%Y%m%d_%H%M%S"
PDF_SUFFIX = ".pdf"
PDF_MAGIC = b"%PDF"

# Job status values reported by GET .../{jobId}/status.
# Documented contract: status is one of PENDING, PROCESSING, COMPLETED, FAILED.
STATUS_PENDING = "PENDING"
STATUS_PROCESSING = "PROCESSING"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"

# If the result of the translation should be in markdown or PDF format.
class OutputFormat(StrEnum):
    MD = "md"
    PDF = "pdf"

class Strategy(StrEnum):
    SINGLE = "single"
    DUAL = "dual"

# PENDING/PROCESSING are non-terminal (keep polling); the others are terminal.
PENDING_STATUSES = frozenset({STATUS_PENDING, STATUS_PROCESSING})
SUCCESS_STATUSES = frozenset({STATUS_COMPLETED})
FAILURE_STATUSES = frozenset({STATUS_FAILED})

# A 404 means the job id is unknown. It can briefly occur right after the job is
# accepted, so tolerate this many consecutive 404s before failing fast.
NOT_FOUND_GRACE = 3

logger = logging.getLogger("zia-eval")


@dataclass
class TranslationResult:
    """Outcome of translating a single source document."""

    source: Path
    output: Optional[Path]
    ok: bool
    detail: str = ""


# --- File helpers -----------------------------------------------------------

def build_translation_url(base_url: str, output_format: OutputFormat) -> str:
    """
    Build the translation URL by appending the output format to the base URL.
    """
    return f"{base_url}/{output_format.lower()}"

def build_download_url(base_url: str, job_id: str) -> str:
    """Build the url top download the translation job."""
    return f"{base_url}/jobs/{job_id}"

VALID_INPUT_SUFFIXES = (PDF_SUFFIX, ".png", ".jpg", ".jpeg")


def discover_inputs(input_path: Path) -> List[Path]:
    """Return the list of PDF, PNG or JPEG files to translate.

    ``input_path`` may be a single file or a directory containing such files.
    When given a directory, every file with a ``.pdf``, ``.png``, ``.jpg`` or
    ``.jpeg`` suffix (case-insensitive) is returned in sorted order.
    """
    if input_path.is_dir():
        candidates = [
            p for p in input_path.glob("*")
            if p.is_file() and p.suffix.lower() in VALID_INPUT_SUFFIXES
        ]
        return sorted(candidates)
    if input_path.is_file() and input_path.suffix.lower() in VALID_INPUT_SUFFIXES:
        return [input_path]
    return []


def sanitize(value: str) -> str:
    """Make ``value`` safe to embed in a file name."""
    cleaned = "".join(
        char if (char.isalnum() or char in "-._") else "-"
        for char in value.strip()
    )
    # Collapse repeated separators and trim leading/trailing ones.
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-_") or "x"


def build_output_name(
    source: Path,
    target_language: str,
    strategy: str,
    run_date: str,
    output_suffix: str,
) -> str:
    """Build ``{stem}_target-{language}_{strategy}_{date}{suffix}`` for ``source``."""
    parts = [
        source.stem,
        f"target-{sanitize(target_language)}",
        sanitize(strategy),
        sanitize(run_date),
    ]
    return "_".join(parts) + output_suffix


def response_is_pdf(response: requests.Response) -> bool:
    """Heuristically decide whether ``response`` carries a PDF body."""
    content_type = response.headers.get("Content-Type", "").lower()
    if "application/pdf" in content_type:
        return True
    # Some servers reply with octet-stream; fall back to the PDF magic header.
    return response.content[:4] == PDF_MAGIC


# --- Asynchronous job polling ----------------------------------------------

def build_status_url(base_url: str, job_id: str) -> str:
    """Build the status URL ``{base}/{jobId}/status`` for a translation job."""
    return f"{base_url}/jobs/{job_id}/status"


def extract_job_id(payload: object) -> Optional[str]:
    """Pull a job id out of a JSON response payload, tolerating key variants."""
    if not isinstance(payload, dict):
        return None
    for key in ("jobId", "job_id", "id", "uuid", "taskId", "task_id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def poll_job_status(
    session: requests.Session,
    status_url: str,
    *,
    timeout: float,
    initial_interval: float,
    max_interval: float,
    backoff_factor: float,
    request_timeout: float,
) -> tuple[bool, str, Dict]:
    """Poll ``status_url`` until the job reaches a terminal state.

    Uses *exponential backoff*: the delay starts at ``initial_interval`` and is
    multiplied by ``backoff_factor`` after every poll, capped at ``max_interval``.

    Status semantics follow the endpoint contract:

    * ``200 OK`` with ``status`` ``PENDING`` or ``PROCESSING`` -> keep polling.
    * ``200 OK`` with ``status`` ``COMPLETED`` -> success (terminal).
    * ``200 OK`` with ``status`` ``FAILED`` -> failure (terminal).
    * ``404 Not Found`` -> the job id is unknown. Tolerated for a few consecutive
      polls (it can race with job creation), then treated as a failure.

    Transient connection errors and other ``>= 400`` responses are retried until
    ``timeout`` seconds elapse. Returns ``(ok, status, payload)`` where ``ok`` is
    ``True`` only for a terminal *success* status.
    """
    deadline = time.monotonic() + timeout
    interval = max(initial_interval, 0.0)
    last_status = "UNKNOWN"
    last_payload: Dict = {}
    consecutive_not_found = 0

    while True:
        try:
            response = session.get(status_url, timeout=request_timeout)
        except requests.RequestException as exc:
            last_status = f"request error: {exc}"
        else:
            if response.status_code == 404:
                # The job id is unknown; tolerate a brief window after creation.
                consecutive_not_found += 1
                last_status = "HTTP 404 (job not found)"
                if consecutive_not_found >= NOT_FOUND_GRACE:
                    return False, "job not found (HTTP 404)", last_payload
            elif response.status_code >= 400:
                # Other server-side errors (e.g. 5xx) may be transient; retry.
                consecutive_not_found = 0
                last_status = f"HTTP {response.status_code}"
            else:
                consecutive_not_found = 0
                try:
                    payload = response.json()
                except ValueError:
                    payload = {}
                if isinstance(payload, dict):
                    last_payload = payload
                    status = str(payload.get("status", "")).strip().upper()
                    if status:
                        last_status = status
                    if status in SUCCESS_STATUSES:
                        return True, status, payload
                    if status in FAILURE_STATUSES:
                        return False, status, payload
                    if status and status not in PENDING_STATUSES:
                        # Unknown status: treat as non-terminal but flag it.
                        logger.warning("Unexpected job status %r; still polling", status)

        now = time.monotonic()
        if now >= deadline:
            return False, f"timeout (last status: {last_status})", last_payload

        sleep_for = min(interval, deadline - now)
        logger.info(
            "Job not finished (status=%s); next poll in %.1fs", last_status, sleep_for
        )
        time.sleep(sleep_for)
        interval = min(interval * backoff_factor, max_interval) if interval > 0 else max_interval


# --- Core pipeline ----------------------------------------------------------

def download_translated_file(
    job_id: str, 
    dest_file_path: str, 
    base_url: str) -> None:
    """
    Downloads a translated file from the translation API and saves it locally.
    
    Returns:
        Path: The local path to the saved file if successful.
        None: If the download failed due to an API error.
    """
    download_url = build_download_url(base_url, job_id)
    headers = {}  # Add {"Authorization": "Bearer <token>"} here once auth is required.
    
    try:
        # stream=True is ideal for files to avoid loading the whole thing into memory
        response = requests.get(download_url, headers=headers, stream=True)
        
        # Handle specific API states based on status codes
        if response.status_code == 404:
            logger.error(f"Download failed: Unknown job ID '{job_id}'.")
        elif response.status_code == 409:
            logger.warning(f"Download pending: Job '{job_id}' is still PENDING or PROCESSING.")
        elif response.status_code == 422:
            logger.error(f"Download failed: The translation job '{job_id}' FAILED during processing.")
        elif response.status_code == 410:
            logger.error(f"Download failed: Job record exists, but the file is GONE (expired).")
            
        # Raise an exception for any other 4xx/5xx errors we didn't explicitly catch
        response.raise_for_status()
        
        # --- 200 OK: Process the file download ---
        # Stream chunks to disk
        with open(dest_file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    file.write(chunk)
                    
        logger.info(f"Successfully downloaded: {job_id} to {dest_file_path}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Network or connection error occurred while fetching job {job_id}: {e}")


def translate_one(
    session: requests.Session,
    source: Path,
    *,
    url: str,
    target_language: str,
    output_format: OutputFormat,
    strategy: str,
    run_date: str,
    output: Path,
    request_timeout: float,
    poll_timeout: float,
    poll_initial_interval: float,
    poll_max_interval: float,
    poll_backoff_factor: float,
) -> TranslationResult:
    """Translate a single PDF and copy the result into ``output``.

    The endpoint is asynchronous: it answers ``202 Accepted`` with a ``jobId``.
    We then poll ``GET {url}/{jobId}/status`` with exponential backoff until the
    job reaches a terminal state, and finally collect the translated PDF.
    """
    logger.info("Translating %s -> %s [format=%s]", source.name, target_language, output_format)

    # Derive the output file suffix from the requested format.
    output_suffix = "." + output_format.lower()

    # Pick the correct MIME type based on the file extension.
    mime_types = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    content_type = mime_types.get(source.suffix.lower(), "application/octet-stream")

    translation_url = build_translation_url(url, output_format)

    try:
        with source.open("rb") as handle:
            response = session.post(
                translation_url,
                files={"file": (source.name, handle, content_type)},
                data={"targetLanguage": target_language, "strategy": strategy},
                timeout=request_timeout,
            )
    except requests.RequestException as exc:
        return TranslationResult(source, None, False, f"request failed: {exc}")

    if response.status_code >= 400:
        snippet = response.text[:200].replace("\n", " ")
        return TranslationResult(
            source,
            None,
            False,
            f"server returned HTTP {response.status_code}: {snippet}",
        )

    # Case 1: the endpoint returned the translated PDF directly in the response.

    try:
        accept_payload = response.json()
    except ValueError:
        accept_payload = None

    job_id = extract_job_id(accept_payload)

    if job_id is None:
        logger.error(
            "Aborting file download: Response did not contain a jobId. Payload=%s", accept_payload
        )
        return TranslationResult(source, None, False, "response did not contain a jobId")

    initial_status = str((accept_payload or {}).get("status", "?"))
    status_url = build_status_url(url, job_id)
    logger.info(
        "Job %s accepted (status=%s); polling %s", job_id, initial_status, status_url
    )

    ok, status, status_payload = poll_job_status(
        session,
        status_url,
        timeout=poll_timeout,
        initial_interval=poll_initial_interval,
        max_interval=poll_max_interval,
        backoff_factor=poll_backoff_factor,
        request_timeout=request_timeout,
    )
    if not ok:
        message = ""
        if isinstance(status_payload, dict):
            message = str(
                status_payload.get("message") or status_payload.get("error") or ""
            )
        detail = f"job {job_id} did not succeed: {status}"
        if message:
            detail += f" ({message})"
        return TranslationResult(source, None, False, detail)

    logger.info("Job %s finished with status=%s", job_id, status)
    
    dest_file_path = output / build_output_name(source, target_language, strategy, run_date, output_suffix)

    try:
        download_translated_file(job_id=job_id, dest_file_path=dest_file_path, base_url=url)
    except Exception as e:
        return TranslationResult(source, None, False, f"could not downlopad translation jon {job_id}: {e}")
    return TranslationResult(source, dest_file_path, True, f"Successfully translated {source.name}")


def run(args: argparse.Namespace) -> int:
    """Execute the pipeline for every discovered file x language x strategy combo."""
    input_path: Path = args.input
    output: Path = args.output

    inputs = discover_inputs(input_path)
    if not inputs:
        logger.error("No input files found at %s", input_path)
        return 2

    output.mkdir(parents=True, exist_ok=True)
    run_date = datetime.now().strftime(DATE_FORMAT)

    combos = list(itertools.product(args.target_language, args.strategy))
    logger.info(
        "Found %d input file(s); %d language/strategy combo(s); format=%s endpoint=%s",
        len(inputs),
        len(combos),
        args.output_format,
        ZIA_TRANSLATION_URL,
    )

    results: List[TranslationResult] = []
    with requests.Session() as session:
        for target_language, strategy in combos:
            logger.info("=== Combo: language=%s strategy=%s ===", target_language, strategy)
            for source in inputs:
                result = translate_one(
                    session,
                    source,
                    url=ZIA_TRANSLATION_URL,
                    target_language=target_language,
                    output_format=args.output_format,
                    strategy=strategy,
                    run_date=run_date,
                    output=output,
                    request_timeout=REQUEST_TIMEOUT,
                    poll_timeout=POLL_TIMEOUT,
                    poll_initial_interval=POLL_INITIAL_INTERVAL,
                    poll_max_interval=POLL_MAX_INTERVAL,
                    poll_backoff_factor=POLL_BACKOFF_FACTOR,
                )
                results.append(result)
                if not result.ok:
                    logger.error(
                        "FAILED %s [%s/%s]: %s", source.name, target_language, strategy, result.detail
                    )

    succeeded = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    logger.info("Done: %d succeeded, %d failed (out of %d total runs)", len(succeeded), len(failed), len(results))
    for result in succeeded:
        logger.info("  OK   %s -> %s", result.source.name, result.output)
    for result in failed:
        logger.info("  FAIL %s (%s)", result.source.name, result.detail)

    return 0 if not failed else 1


# --- CLI --------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send PDFs to the Zia translation endpoint and collect the "
        "translated files into the results folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="PDF file or directory of PDFs to translate.",
    )
    parser.add_argument(
        "-l",
        "--target-language",
        required=True,
        nargs="+",
        help="One or more target language codes passed to the endpoint (e.g. 'fr it de').",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        required=True,
        type=Strategy,
        choices=list(Strategy),
        nargs="+",
        help="One or more translation strategies ('single'/'dual'). Sent to the "
        "endpoint as the 'strategy' field and also used to tag output file names.",
    )
    parser.add_argument(
        "-f",
        "--output-format",
        dest="output_format",
        type=OutputFormat,
        choices=list(OutputFormat),
        default=OutputFormat.MD,
        help="Output format sent to the endpoint as 'outputFormat' (MD or PDF).",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=Path,
        help="Folder where renamed translated files are collected.",
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
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
