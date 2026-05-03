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


class LLMTeacher:
    """Generate and verify ERRs offline via an LLM client with caching."""

    def __init__(
        self,
        client: LLMClient,
        verifier: EvidenceContractVerifier,
        cache_dir: str,
        score_blind: bool = True,
    ) -> None:
        self._client = client
        self._verifier = verifier
        self._cache = PromptCache(cache_dir)
        self._builder = PromptBuilder(score_blind=score_blind)
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
        results: list[ErrGenerationResult] = []
        for idx, mep in enumerate(meps):
            label = labels[idx] if labels is not None else None
            payload = mep.to_teacher_payload()
            prompt = self._builder.build(payload)

            raw = self._cache.get(prompt)
            if raw is None:
                raw = self._client.complete(prompt)
            err = self._parse_with_retry(prompt, raw)
            if err is None:
                continue

            verification = self._verifier.verify(err, mep, label)

            # Cache with verification metadata (may overwrite bare response).
            ver_str = "accepted" if verification.accepted else "rejected"
            self._cache.put(
                prompt,
                raw,
                verification_result=ver_str,
                contract_version=contract_version,
            )

            if verification.accepted:
                logger.info(
                    "ERR accepted for node=%s reasons=%s",
                    mep.node_id,
                    verification.reasons,
                )
                results.append(
                    ErrGenerationResult(
                        node_id=mep.node_id,
                        err=err,
                        verification=verification,
                    )
                )
            else:
                logger.warning(
                    "ERR rejected for node=%s reasons=%s",
                    mep.node_id,
                    verification.reasons,
                )
        return results

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _parse_with_retry(
        self, prompt: str, raw: str
    ) -> EvidenceRationaleRecord | None:
        text = raw.strip()
        for attempt in range(_MAX_PARSE_RETRIES + 1):
            try:
                data = json.loads(text)
                return EvidenceRationaleRecord(**data)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning(
                    "JSON parse attempt %d failed: %s", attempt + 1, exc
                )
                if attempt < _MAX_PARSE_RETRIES:
                    text = self._strip_markdown_fences(text)
                    continue
        logger.error("Failed to parse LLM output after retries")
        return None

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
