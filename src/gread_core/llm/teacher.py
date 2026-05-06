"""Offline LLM teacher: generate ERRs, verify, cache, and return accepted."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from gread_core.llm.cache import PromptCache
from gread_core.llm.clients import LLMClient
from gread_core.llm.prompt_builder import PromptBuilder
from gread_core.schemas.err import EvidenceRationaleRecord
from gread_core.schemas.evidence import MinimalEvidencePackage
from gread_core.verification.verifier import EvidenceContractVerifier, VerificationResult

logger = logging.getLogger(__name__)

_MAX_PARSE_RETRIES = 2


@dataclass(frozen=True)
class ErrGenerationResult:
    """Holds a verified ERR and the associated MEP."""

    node_id: str
    err: EvidenceRationaleRecord
    verification: VerificationResult


@dataclass(frozen=True)
class ErrAttemptResult:
    """Holds one raw LLM attempt and its deterministic verification outcome."""

    node_id: str
    raw_response: str
    err: EvidenceRationaleRecord | None
    verification: VerificationResult
    parse_error: str | None = None


class LLMTeacher:
    """Generate and verify ERRs offline via an LLM client with caching."""

    def __init__(
        self,
        client: LLMClient,
        verifier: EvidenceContractVerifier,
        cache_dir: str,
        score_blind: bool = True,
        model_name: str | None = None,
        batch_size: int = 1,
        show_progress: bool = True,
    ) -> None:
        self._client = client
        self._verifier = verifier
        self._cache = PromptCache(cache_dir)
        self._builder = PromptBuilder(score_blind=score_blind)
        self._model_name = model_name
        self._batch_size = max(1, batch_size)
        self._show_progress = show_progress
        self._tqdm = None
        if show_progress:
            try:
                from tqdm.auto import tqdm
            except ImportError:
                self._tqdm = None
            else:
                self._tqdm = tqdm
        if not score_blind:
            logger.warning(
                "ABLATION: score_blind=False — prediction_score will leak into prompts"
            )

    def generate_err(
        self,
        meps: list[MinimalEvidencePackage],
        labels: list[int] | None = None,
        *,
        contract_version: str | None = None,
    ) -> list[ErrGenerationResult]:
        """Generate accepted ERRs for *meps*.

        Rejected ERRs are logged but not returned as training targets.
        """
        attempts = self.generate_err_attempts(
            meps,
            labels,
            contract_version=contract_version,
        )
        return [
            ErrGenerationResult(
                node_id=attempt.node_id,
                err=attempt.err,
                verification=attempt.verification,
            )
            for attempt in attempts
            if attempt.err is not None and attempt.verification.accepted
        ]

    def generate_err_attempts(
        self,
        meps: list[MinimalEvidencePackage],
        labels: list[int] | None = None,
        *,
        contract_version: str | None = None,
    ) -> list[ErrAttemptResult]:
        """Generate ERR attempts for *meps*, preserving accepted and rejected output."""
        attempts: list[ErrAttemptResult] = []
        prompts = [self._builder.build(mep.to_teacher_payload()) for mep in meps]
        raw_responses: list[str | None] = [None] * len(prompts)
        missing_indices: list[int] = []

        for idx, prompt in enumerate(prompts):
            raw = self._cache.get(prompt)
            if raw is None:
                missing_indices.append(idx)
            else:
                raw_responses[idx] = raw

        logger.info(
            "LLM prompt cache summary: total=%d hits=%d misses=%d",
            len(prompts),
            len(prompts) - len(missing_indices),
            len(missing_indices),
        )

        batch_ranges = range(0, len(missing_indices), self._batch_size)
        batch_total = (len(missing_indices) + self._batch_size - 1) // self._batch_size
        if self._tqdm is not None and len(missing_indices) > 0:
            batch_ranges = self._tqdm(
                batch_ranges,
                total=batch_total,
                desc="LLM generate",
                unit="batch",
            )
        for batch_num, start in enumerate(batch_ranges, start=1):
            batch_indices = missing_indices[start : start + self._batch_size]
            if not batch_indices:
                continue
            batch_prompts = [prompts[idx] for idx in batch_indices]
            batch_raw = self._client.complete_batch(batch_prompts)
            if len(batch_raw) != len(batch_indices):
                raise RuntimeError(
                    "LLM batch response count does not match prompt count"
                )
            for idx, raw in zip(batch_indices, batch_raw, strict=True):
                raw_responses[idx] = raw
            logger.info(
                "LLM batch %d/%d complete | size=%d",
                batch_num,
                batch_total,
                len(batch_indices),
            )

        verify_indices = range(len(meps))
        if self._tqdm is not None and len(meps) > 0:
            verify_indices = self._tqdm(
                verify_indices,
                total=len(meps),
                desc="ERR verify",
                unit="mep",
            )
        for idx in verify_indices:
            mep = meps[idx]
            label = labels[idx] if labels is not None else None
            prompt = prompts[idx]
            raw = raw_responses[idx]
            if raw is None:
                raise RuntimeError("Missing LLM response for prompt")
            err, parse_error = self._parse_with_retry_details(prompt, raw)
            if err is None:
                verification = VerificationResult(
                    accepted=False,
                    reasons=[f"Parse failed: {parse_error or 'invalid ERR JSON'}"],
                )
                self._cache.put(
                    prompt,
                    raw,
                    verification_result="rejected",
                    contract_version=contract_version,
                    model=self._model_name,
                )
                logger.warning(
                    "ERR rejected for node=%s reasons=%s",
                    mep.node_id,
                    verification.reasons,
                )
                attempts.append(
                    ErrAttemptResult(
                        node_id=mep.node_id,
                        raw_response=raw,
                        err=None,
                        verification=verification,
                        parse_error=parse_error,
                    )
                )
                continue

            verification = self._verifier.verify(err, mep, label)

            # Cache with verification metadata (may overwrite bare response).
            ver_str = "accepted" if verification.accepted else "rejected"
            self._cache.put(
                prompt,
                raw,
                verification_result=ver_str,
                contract_version=contract_version,
                model=self._model_name,
            )

            if verification.accepted:
                logger.info(
                    "ERR accepted for node=%s reasons=%s",
                    mep.node_id,
                    verification.reasons,
                )
            else:
                logger.warning(
                    "ERR rejected for node=%s reasons=%s",
                    mep.node_id,
                    verification.reasons,
                )
            attempts.append(
                ErrAttemptResult(
                    node_id=mep.node_id,
                    raw_response=raw,
                    err=err,
                    verification=verification,
                    parse_error=None,
                )
            )
        return attempts

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _parse_with_retry(
        self, prompt: str, raw: str
    ) -> EvidenceRationaleRecord | None:
        err, _ = self._parse_with_retry_details(prompt, raw)
        return err

    def _parse_with_retry_details(
        self, prompt: str, raw: str
    ) -> tuple[EvidenceRationaleRecord | None, str | None]:
        text = raw.strip()
        last_error: str | None = None
        for attempt in range(_MAX_PARSE_RETRIES + 1):
            try:
                data = json.loads(text)
                return EvidenceRationaleRecord(**data), None
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                last_error = str(exc)
                logger.warning(
                    "JSON parse attempt %d failed: %s", attempt + 1, exc
                )
                if attempt < _MAX_PARSE_RETRIES:
                    text = self._extract_json_candidate(
                        self._strip_markdown_fences(text)
                    )
                    continue
        logger.error("Failed to parse LLM output after retries")
        return None, last_error

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove leading/trailing markdown code fences if present."""
        stripped = text.strip()
        if stripped.startswith("```"):
            # remove opening fence line
            first_newline = stripped.find("\n")
            if first_newline != -1:
                stripped = stripped[first_newline + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[: -len("```")]
        return stripped.strip()

    @staticmethod
    def _extract_json_candidate(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end >= start:
            return text[start : end + 1]
        return text
