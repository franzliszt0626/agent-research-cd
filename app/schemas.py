"""Validated contracts shared by the six agents and the final JSON report."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class LiteratureMetadata(Contract):
    found: bool = False
    status: Literal["open_source", "close_source", "metadata_only", "not_found", "mcp_error", "network_timeout", "network_error", "invalid_response"] = "not_found"
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str = ""
    arxiv_id: str = ""
    venue: str = ""
    abstract: str = ""
    access_status: str = "access-undetermined"
    is_open_access: bool | None = None
    pdf_url: str = ""
    landing_page_url: str = ""
    source: str = ""
    error: str = ""
    provider_errors: list[str] = Field(default_factory=list)
    full_text_available: bool = False
    full_text_status: str = "not_requested"
    full_text_error: str = ""
    full_text_truncated: bool = False
    word_count: int = 0


class ReferenceItem(Contract):
    reference_id: str = Field(pattern=r"^REF\d+$")
    original_reference: str = Field(min_length=1)
    bib_key: str = ""
    label: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = Field(default=None, ge=1000, le=2999)
    title: str = ""
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""
    access_date: str = ""
    normalized_citation: str
    citation_number: int = Field(gt=0)
    literature_metadata: LiteratureMetadata | None = None


class StandardizedResponse(Contract):
    judgmental_statement: str
    supporting_paper: str
    mapping_basis: str
    verification_scope: str
    doi: str = ""
    abstract: str = ""
    access_status: str = "access-undetermined"
    pdf_url: str = ""
    access_note: str = ""


class ClaimPair(Contract):
    pair_id: str = Field(pattern=r"^PAIR\d+$")
    statement_original: str = Field(min_length=1)
    statement_summary: str
    claim_type: str
    reference_id: str
    mapping_method: str
    standardized_response: StandardizedResponse


class ReviewPayload(Contract):
    confidence: float = Field(ge=0, le=1)
    verdict: Literal["supports", "partially_supports", "does_not_support", "uncertain"]
    rationale: str = Field(min_length=1)
    limitations: str = ""
    evidence_from_abstract: str = ""
    supporting_passage: str = ""


class Review(Contract):
    status: Literal["completed", "not_available", "error"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    verdict: Literal["supports", "partially_supports", "does_not_support", "uncertain"] = "uncertain"
    rationale: str
    limitations: str = ""
    evidence_from_abstract: str = ""
    supporting_passage: str = ""
    basis: str
    failure_reason: str = ""

    @model_validator(mode="after")
    def check_evaluability(self):
        if self.status == "completed" and self.confidence is None:
            raise ValueError("Completed review requires confidence")
        if self.status != "completed" and (self.confidence is not None or self.verdict != "uncertain"):
            raise ValueError("Unavailable review must have null confidence and uncertain verdict")
        return self


class FinalDecision(Contract):
    passed: bool
    passed_by: Literal["initial_review", "deep_review", "none"]
    threshold: float = Field(ge=0, le=1)
    effective_confidence: float | None = Field(default=None, ge=0, le=1)
    explanation: str
    review_stage: Literal["preliminary", "deep", "manual_required"]
    reason_code: Literal["supported", "below_threshold", "literature_unavailable", "abstract_unavailable", "full_text_unavailable", "model_error", "invalid_response"]
    rule: str


class PairResult(ClaimPair):
    initial_review: Review
    deep_review: Review | None = None
    final_decision: FinalDecision


class Statistics(Contract):
    references_total: int
    pairs_total: int
    passed: int
    failed: int
    manual_required: int
    rejected: int
    failure_reasons: dict[str, int] = Field(default_factory=dict)
    deep_reviewed: int
    unmatched_references: int


class ReportSchema(Contract):
    schema_version: str = "1.2.0"
    task_id: str
    source_file: str
    status: Literal["completed", "completed_with_warnings", "failed"]
    threshold: float = Field(ge=0, le=1)
    model: str
    started_at: str
    completed_at: str
    statistics: Statistics
    references: list[ReferenceItem]
    pairs: list[PairResult]
    unmatched_references: list[ReferenceItem]
    warnings: list[str]
    errors: list[str]

    @model_validator(mode="after")
    def check_links(self):
        ids = [ref.reference_id for ref in self.references]
        pair_ids = [pair.pair_id for pair in self.pairs]
        if len(ids) != len(set(ids)) or len(pair_ids) != len(set(pair_ids)):
            raise ValueError("Duplicate reference/pair IDs")
        if any(p.reference_id not in ids for p in self.pairs):
            raise ValueError("Pair refers to missing reference")
        if any(r.reference_id not in ids for r in self.unmatched_references):
            raise ValueError("Unmatched reference does not exist")
        s = self.statistics
        if s.pairs_total != len(self.pairs) or s.references_total != len(ids):
            raise ValueError("Report statistics do not match records")
        if s.passed + s.failed != s.pairs_total or s.manual_required + s.rejected != s.failed:
            raise ValueError("Inconsistent decision statistics")
        if s.failure_reasons and sum(s.failure_reasons.values()) != s.failed:
            raise ValueError("Failure reasons do not match failed count")
        return self
