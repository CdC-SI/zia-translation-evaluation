#!/usr/bin/env python3
"""End-to-end orchestrator for the Zia translation evaluation pipeline.

Chains the three independent scripts in this package into a single command:

1. :mod:`translation` — translates every input PDF/PNG into one or more
   target languages, using one or more strategies.
2. :mod:`evaluation` — runs the LLM-as-a-judge evaluation of every produced
   translation against its original document (batch mode).
3. :mod:`report` — aggregates every evaluation JSON report into a single
   Markdown summary report.

The output directory structure is::

    <output>/
        translations/   <- produced by translation.py (Markdown by default)
        evaluations/     <- produced by evaluation.py (one JSON per translation)
        reports/         <- produced by report.py (one aggregated Markdown file)

``--output`` must already exist (an error is raised otherwise); the three
sub-folders above are created automatically if missing.

Example
-------
    python src/pipeline/run_pipeline.py \\
        -i data/pdf \\
        -l fr it de \\
        -s single dual \\
        -o data/pipeline_run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

# Ensure the project root is on sys.path so both
#   python src/pipeline/run_pipeline.py   and
#   python -m src.pipeline.run_pipeline
# resolve imports correctly.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline import evaluation, report, translation

DATE_FORMAT = "%Y%m%d_%H%M%S"

TRANSLATIONS_SUBDIR = "translations"
EVALUATIONS_SUBDIR = "evaluations"
REPORTS_SUBDIR = "reports"

logger = logging.getLogger("zia-pipeline")


# --- Setup -------------------------------------------------------------------

def ensure_output_dir(output: Path) -> None:
    """Ensure ``output`` already exists; raise otherwise (per pipeline contract)."""
    if not output.exists():
        raise FileNotFoundError(
            f"Output directory does not exist: {output}. "
            "Create it first (this script will not create the top-level output folder)."
        )
    if not output.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output}")


def ensure_subdirs(output: Path) -> tuple[Path, Path, Path]:
    """Create (if missing) and return the translations/evaluations/reports sub-folders."""
    translations_dir = output / TRANSLATIONS_SUBDIR
    evaluations_dir = output / EVALUATIONS_SUBDIR
    reports_dir = output / REPORTS_SUBDIR
    for directory in (translations_dir, evaluations_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return translations_dir, evaluations_dir, reports_dir


# --- Pipeline steps ------------------------------------------------------------

def run_translation_step(
    *,
    input_path: Path,
    languages: Sequence[str],
    strategies: Sequence[str],
    output_format: str,
    translations_dir: Path,
    extra_argv: Sequence[str],
) -> int:
    """Run the translation step for every input x language x strategy combo."""
    logger.info(
        "running translation on %s [languages=%s strategies=%s format=%s] -> %s",
        input_path, list(languages), list(strategies), output_format, translations_dir,
    )
    argv = [
        "-i", str(input_path),
        "-l", *[str(l) for l in languages],
        "-s", *[str(s) for s in strategies],
        "-f", str(output_format),
        "-o", str(translations_dir),
        *extra_argv,
    ]
    translation_args = translation.parse_args(argv)
    rc = translation.run(translation_args)
    if rc != 0:
        logger.error("Translation step finished with errors (exit code %d)", rc)
    else:
        logger.info("Translation step completed successfully")
    return rc


async def run_evaluation_step(
    *,
    input_path: Path,
    translations_dir: Path,
    evaluations_dir: Path,
    extra_argv: Sequence[str],
) -> int:
    """Run the LLM-as-a-judge evaluation of every produced translation."""
    logger.info(
        "running evaluation on %s (translations from %s) -> %s",
        input_path, translations_dir, evaluations_dir,
    )
    argv = [
        "-O", str(input_path),
        "-T", str(translations_dir),
        "-r", str(evaluations_dir),
        *extra_argv,
    ]
    evaluation_args = evaluation.parse_args(argv)
    rc = await evaluation.run(evaluation_args)
    if rc != 0:
        logger.error("Evaluation step finished with errors (exit code %d)", rc)
    else:
        logger.info("Evaluation step completed successfully")
    return rc


def run_report_step(
    *,
    evaluations_dir: Path,
    reports_dir: Path,
    run_date: str,
) -> Optional[Path]:
    """Generate the final aggregated Markdown report from every evaluation JSON."""
    logger.info("generating the final report from %s -> %s", evaluations_dir, reports_dir)
    md_output_path = reports_dir / f"report_{run_date}.md"
    html_output_path = md_output_path.with_suffix(".html")

    evaluations = report.load_evaluations(evaluations_dir)
    if not evaluations:
        logger.error(
            "No evaluation reports found in %s; skipping report generation.",
            evaluations_dir,
        )
        return None

    report.generate_markdown_report(
        evaluations_dir,
        output_path=md_output_path,
        html_output_path=html_output_path,
    )
    logger.info(
        "Report generation completed successfully -> %s and %s",
        md_output_path, html_output_path,
    )
    return md_output_path


# --- Orchestration -------------------------------------------------------------

async def run(args: argparse.Namespace) -> int:
    ensure_output_dir(args.output)
    translations_dir, evaluations_dir, reports_dir = ensure_subdirs(args.output)

    run_date = datetime.now().strftime(DATE_FORMAT)

    translation_rc = run_translation_step(
        input_path=args.input,
        languages=args.target_language,
        strategies=args.strategy,
        output_format=args.output_format,
        translations_dir=translations_dir,
        extra_argv=args.translation_arg,
    )

    evaluation_rc = await run_evaluation_step(
        input_path=args.input,
        translations_dir=translations_dir,
        evaluations_dir=evaluations_dir,
        extra_argv=args.evaluation_arg,
    )

    report_path = run_report_step(
        evaluations_dir=evaluations_dir,
        reports_dir=reports_dir,
        run_date=run_date,
    )

    if translation_rc != 0 or evaluation_rc != 0 or report_path is None:
        logger.error("Pipeline finished with errors.")
        return 1

    logger.info("Pipeline finished successfully. Final report: %s", report_path)
    return 0


# --- CLI ------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full translation -> evaluation -> report pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="PDF/PNG file or directory of PDFs/PNGs to translate and evaluate.",
    )
    parser.add_argument(
        "-l",
        "--target-language",
        required=True,
        nargs="+",
        help="One or more target language codes (e.g. 'fr it de').",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        required=True,
        nargs="+",
        choices=["single", "dual"],
        help="One or more translation strategies ('single'/'dual').",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Existing output directory. 'translations', 'evaluations' and "
        "'reports' sub-folders are created inside it if missing.",
    )
    parser.add_argument(
        "-f",
        "--output-format",
        dest="output_format",
        default="md",
        choices=["md", "pdf"],
        help="Output format requested from the translation endpoint.",
    )
    parser.add_argument(
        "--translation-arg",
        dest="translation_arg",
        action="append",
        default=[],
        help="Extra raw argument forwarded to translation.py's CLI (repeatable). "
        "Each occurrence supplies exactly one token, so a flag and its value "
        "must be passed as two separate --translation-arg entries, e.g. "
        "--translation-arg --output-format --translation-arg pdf.",
    )
    parser.add_argument(
        "--evaluation-arg",
        dest="evaluation_arg",
        action="append",
        default=[],
        help="Extra raw argument forwarded to evaluation.py's CLI (repeatable).",
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
    try:
        return asyncio.run(run(args))
    except (FileNotFoundError, NotADirectoryError) as exc:
        logger.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
