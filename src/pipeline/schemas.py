"""Pydantic models for the LLM-as-a-judge evaluation report.

Three separate schemas enforce strict separation of concerns:

1. **LLMPageReport** — the schema passed to vLLM as ``response_format``.
   Contains only what the model is expected to fill in.  All arithmetic is
   intentionally absent so the model is never asked to do math.

2. **FinalPageReport** — one record per PDF page, written to disk.
   Carries raw error counts and the page-level total penalty only.
   No per-page ``final_score`` or ``quality_gate`` — those are degenerate on
   short pages and are computed at document level instead.

3. **DocumentMQMScore** — the single aggregated score for an entire document
   (or evaluation run).  Computed in ``report.py`` by summing penalties and
   words across *all* pages before applying the WMT normalization formula:

       Final MQM Score = max(0, 100 − (ΣPenalty / ΣWords × 1000))

   This follows the standard GEMBA-MQM / Rubric-MQM approach used in the
   WMT shared tasks, where a long clean page naturally offsets the penalty
   spike of a short noisy page.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Dimension(str, Enum):
    ACC = "ACC"   # Accuracy
    FLU = "FLU"   # Fluency & Style
    FOR = "FOR"   # Formatting & Structure
    COM = "COM"   # Completeness


class ErrorType(str, Enum):
    MISTRANSLATION = "MISTRANSLATION"
    OMISSION = "OMISSION"
    ADDITION = "ADDITION"
    UNTRANSLATED = "UNTRANSLATED"
    GRAMMAR = "GRAMMAR"
    TERMINOLOGY = "TERMINOLOGY"
    STRUCTURE_LOSS = "STRUCTURE_LOSS"
    SEGMENTATION = "SEGMENTATION"
    HALLUCINATION = "HALLUCINATION"


class Severity(str, Enum):
    MINOR = "MINOR"      # weight 1
    MAJOR = "MAJOR"      # weight 5
    CRITICAL = "CRITICAL"  # weight 25


class RootCause(str, Enum):
    TRANSLATION_MODEL = "TRANSLATION_MODEL"
    FORMATTING = "FORMATTING"
    NONE = "NONE"


class QualityGate(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class TranslationQuality(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class FormattingQuality(str, Enum):
    HIGH = "HIGH"
    DEGRADED = "DEGRADED"
    POOR = "POOR"


class OverallRisk(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class ErrorLocation(BaseModel):
    """Pinpoints the error using exact verbatim spans — no fuzzy descriptions.

    Empty strings are intentionally still allowed here: an OMISSION error has
    no target_substring (nothing was translated), and an ADDITION error has
    no source_substring (nothing to point to in the source). Every other
    error type must populate both.
    """
    source_substring: str = Field(
        description="Verbatim span from the SOURCE text that contains the error. "
        "Empty only for ADDITION errors."
    )
    target_substring: str = Field(
        description="Verbatim span from the TRANSLATED text that contains the error. "
        "Empty only for OMISSION errors."
    )


class TranslationError(BaseModel):
    """
    Field order is intentional: the model writes `evidence` first so it anchors
    its own reasoning before committing to a dimension / severity classification.
    """
    error_id: str = Field(min_length=1, description="Unique identifier within the segment, e.g. E01.")
    evidence: str = Field(
        min_length=1,
        description=(
            "Write this FIRST. Quote the relevant spans from both original and "
            "translation that reveal the problem, then use that evidence to drive "
            "the classification below. Must never be empty."
        )
    )
    justification: str = Field(
        min_length=1,
        description=(
            "A short, single-sentence explanation of what went wrong and why it "
            "maps to this dimension and severity. Written after evidence, before "
            "the classification labels. Must never be empty."
        )
    )
    dimension: Dimension
    type: ErrorType
    severity: Severity
    location: ErrorLocation
    root_cause: RootCause


class Comparison(BaseModel):
    original: str = Field(min_length=1, description="Source text for this segment. Must never be empty.")
    final_translation: str = Field(
        min_length=1, description="Translated text for this segment. Must never be empty."
    )


class Segment(BaseModel):
    segment_id: str = Field(description="e.g. paragraph_1 or section_2.3")
    comparison: Comparison
    errors: List[TranslationError] = Field(
        default_factory=list,
        description="Empty list if the segment has no errors.",
    )


class LLMMetadata(BaseModel):
    """Metadata fields the LLM judge is actually asked to produce.

    Pipeline-only fields (``translation_word_count``, ``judge_model``,
    ``prompt_hash``, ``temperature``) are intentionally excluded — they are
    stamped in afterward by ``parse_and_validate_report()`` in evaluation.py,
    never generated by the model.
    """
    source_language: str = Field(description="Detected ISO 639-1 code, e.g. sr.")
    target_language: str = Field(
        description=(
            "Copy EXPECTED_TARGET_LANGUAGE exactly as provided — do not detect "
            "or substitute your own language guess. This value is overwritten "
            "by the pipeline regardless, but must still be populated."
        )
    )
    pipeline: str = Field(default="OCR + Translation")
    evaluator: str = Field(default="LLM-as-a-Judge (MQM)")


class Metadata(LLMMetadata):
    """Full metadata for the final stored report, including pipeline-stamped
    fields that are never generated by the LLM judge.
    """
    translation_word_count: int = Field(
        description=(
            "Word count of the translated Markdown page text, computed "
            "deterministically by the pipeline (not by the LLM judge)."
        )
    )
    judge_model: str = Field(description="Name of the LLM judge model used for this evaluation.")
    prompt_hash: str = Field(
        description="Short hash of the system prompt content, to detect prompt drift across runs."
    )
    temperature: float = Field(description="Sampling temperature used for this evaluation request.")


class FinalMQMScore(BaseModel):
    """Page-level math block: raw error counts + total penalty only.

    ``final_score`` and ``quality_gate`` are intentionally absent — they are
    computed at document level by ``compute_document_mqm_score()`` in report.py
    once all pages are aggregated, following the WMT normalization approach.
    """
    minor_count: int
    major_count: int
    critical_count: int
    total_penalty: int


class DocumentMQMScore(BaseModel):
    """Document-level aggregated MQM score (WMT standard formula).

    Computed exclusively by ``compute_document_mqm_score()`` in report.py
    after all page penalties and word counts are summed together.

        Final MQM Score = max(0, 100 − (ΣPenalty / ΣWords × 1000))
    """
    total_penalty: int = Field(description="Sum of all page penalties.")
    total_words: int = Field(description="Sum of all page word counts.")
    final_score: float = Field(ge=0, le=100, description="Normalized 0–100 score.")
    quality_gate: QualityGate = Field(description="PASS if final_score >= 75.")


class Summary(BaseModel):
    translation_quality: TranslationQuality
    formatting_quality: FormattingQuality
    pipeline_assessment: str = Field(
        description="One-sentence overall assessment for this page."
    )
    overall_risk: OverallRisk


# ---------------------------------------------------------------------------
# Top-level reports
# ---------------------------------------------------------------------------

class BasePageReport(BaseModel):
    """Fields shared by both the LLM output and the final stored report."""
    document_id: str = Field(
        description="Page reference, e.g. weisungen_1_p1. Stamped by the pipeline."
    )
    page: int = Field(description="1-based page number.")
    segments: List[Segment] = Field(
        description="One object per logical paragraph or section of the page."
    )
    summary: Summary


class LLMPageReport(BasePageReport):
    """Schema used for constrained generation and initial validation.

    Passed as ``response_format`` to the vLLM endpoint.  Contains exactly
    what the model is expected to produce — no arithmetic fields.

    ``metadata`` uses ``LLMMetadata`` (a subset of the full ``Metadata``) so
    the model is never asked to invent pipeline-only values such as
    ``translation_word_count``, ``judge_model``, ``prompt_hash``, or
    ``temperature`` — those are stamped in afterward by
    ``parse_and_validate_report()`` in evaluation.py.

    ``mqm_score`` is intentionally excluded — severity counts and penalties
    are computed deterministically by ``report.py`` from the errors listed
    in ``segments``, never by the LLM itself.

    ``summary`` is optional so that a page truncated mid-JSON before the
    summary block is written still validates successfully.  The segments and
    error data — which drive all scoring — are preserved.
    """
    metadata: LLMMetadata
    summary: Optional[Summary] = None


class FinalPageReport(BasePageReport):
    """Schema written to disk / database.

    Structurally parallel to ``LLMPageReport`` but carries the full
    ``Metadata`` (with pipeline-stamped fields) and ``FinalMQMScore``
    (pipeline-computed error counts + total penalty).  Populated exclusively
    by ``process_and_recalculate_report()`` in ``report.py``.

    ``summary`` inherits the Optional from ``LLMPageReport`` — pages that were
    truncated before the summary block will have ``summary=None``.
    """
    metadata: Metadata
    mqm_score: FinalMQMScore
    summary: Optional[Summary] = None


class ParsedPageReport(BasePageReport):
    """Schema used by ``evaluation.py`` to validate + serialize model output
    to disk, *after* pipeline metadata has been stamped in.

    Unlike ``LLMPageReport`` (used only for the vLLM ``response_format``,
    where the model must not see pipeline-only metadata fields), this model
    uses the full ``Metadata`` so that ``translation_word_count``,
    ``judge_model``, ``prompt_hash``, and ``temperature`` — stamped in by
    ``parse_and_validate_report()`` — survive ``model_dump()`` into the
    aggregated JSON report on disk.

    ``mqm_score`` is omitted: severity counts/penalties are computed later,
    at report-generation time, directly from ``segments`` — never stored
    redundantly in the per-page evaluation JSON.
    """
    metadata: Metadata
    summary: Optional[Summary] = None
