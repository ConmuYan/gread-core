"""Stage 2: Offline ERR generation + verification.

CRITICAL CONSTRAINT: Stage 2 is the ONLY stage that calls LLM.
This stage:
1. Runs base detector to get embeddings and scores.
2. Selects trace nodes using TraceSelector.
3. Builds MEPs using detector adapter.
4. Generates ERRs via LLMTeacher.
5. Verifies ERRs via EvidenceContractVerifier.
6. Saves accepted (and rejected) ERRs for Stage 3.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import torch

try:
    from torch_geometric.data import Data
except ImportError:
    Data = Any

from gread_core.tracing.selector import TraceSelector

logger = logging.getLogger(__name__)


def _strip_masks(data: Data) -> Data:
    """Remove all masks so forward_with_embedding returns logits for all nodes."""
    try:
        import copy
        d = copy.copy(data)
    except Exception:
        d = data
    for attr in ("train_mask", "val_mask", "test_mask"):
        if hasattr(d, attr):
            setattr(d, attr, None)
    return d


def _resolve_trace_split_mask(data: Data, split: str | None) -> torch.Tensor | None:
    """Return a boolean mask restricting Stage 2 trace nodes.

    ``prediction_score`` may be used internally for bucket assignment, but ERR
    generation should usually be restricted to training nodes so label
    compatibility checks cannot consume validation/test labels.
    """
    normalized = (split or "all").lower()
    if normalized in {"all", "none"}:
        return None
    attr = f"{normalized}_mask"
    if normalized not in {"train", "val", "test"} or not hasattr(data, attr):
        raise ValueError(
            "stage2.trace_split must be one of: train, val, test, all"
        )
    mask = getattr(data, attr)
    if mask is None:
        raise ValueError(f"Requested stage2.trace_split={normalized!r} but {attr} is None")
    return cast(torch.Tensor, mask.cpu().bool())


class Stage2Result:
    """Holds all ERR results from Stage 2 generation."""

    def __init__(
        self,
        accepted_errs: list[dict[str, Any]],
        rejected_errs: list[dict[str, Any]],
        rejection_report: list[dict[str, Any]] | None = None,
        raw_err_audit: list[dict[str, Any]] | None = None,
    ) -> None:
        self.accepted_errs = accepted_errs
        self.rejected_errs = rejected_errs
        self.rejection_report = rejection_report or rejected_errs
        self.raw_err_audit = raw_err_audit or []

    @property
    def num_accepted(self) -> int:
        return len(self.accepted_errs)

    @property
    def num_rejected(self) -> int:
        return len(self.rejected_errs)

    def save(self, output_dir: str | Path) -> Path:
        """Save ERR results to disk."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "accepted_errs.json", "w") as f:
            json.dump(self.accepted_errs, f, indent=2)

        with open(out / "rejected_errs.json", "w") as f:
            json.dump(self.rejected_errs, f, indent=2)

        _write_jsonl(out / "rejection_report.jsonl", self.rejection_report)
        _write_jsonl(out / "raw_err_audit.jsonl", self.raw_err_audit)
        with open(out / "rejection_summary.json", "w") as f:
            json.dump(
                _build_rejection_summary(
                    self.accepted_errs,
                    self.rejection_report,
                    self.raw_err_audit,
                ),
                f,
                indent=2,
                sort_keys=True,
            )

        logger.info(
            "Saved ERRs: %d accepted, %d rejected to %s",
            self.num_accepted, self.num_rejected, out,
        )
        return out

    @classmethod
    def load(cls, output_dir: str | Path) -> Stage2Result:
        """Load ERR results from disk."""
        out = Path(output_dir)
        with open(out / "accepted_errs.json") as f:
            accepted = json.load(f)
        with open(out / "rejected_errs.json") as f:
            rejected = json.load(f)
        rejection_report = _read_jsonl_if_exists(out / "rejection_report.jsonl")
        raw_err_audit = _read_jsonl_if_exists(out / "raw_err_audit.jsonl")
        return cls(
            accepted_errs=accepted,
            rejected_errs=rejected,
            rejection_report=rejection_report,
            raw_err_audit=raw_err_audit,
        )


def generate_errs(
    detector: Any,
    data: Data,
    adapter: Any,
    teacher: Any,
    verifier: Any,
    config: dict[str, Any],
    seed: int = 1,
) -> Stage2Result:
    """Stage 2: Generate and verify ERRs via LLM.

    Args:
        detector: Trained base detector (Stage 1 output).
        data: PyG Data object.
        adapter: Evidence adapter for building MEPs.
        teacher: LLMTeacher instance for generating ERRs.
        verifier: EvidenceContractVerifier for checking ERRs.
        config: Configuration dict.
        seed: Random seed for trace selection.

    Returns:
        Stage2Result with accepted and rejected ERRs.
    """
    detector.eval()

    # Get detector outputs for ALL nodes (strip masks so no filtering)
    no_mask_data = _strip_masks(data)
    with torch.no_grad():
        logits, embeddings = detector.forward_with_embedding(no_mask_data)
        scores = torch.sigmoid(logits).cpu()
        uncertainties = _compute_uncertainties(scores, embeddings).cpu()

    labels = data.y.cpu() if hasattr(data, "y") else None
    trace_split = config.get("stage2", {}).get("trace_split", "all")
    trace_mask = _resolve_trace_split_mask(data, trace_split)

    # Select trace nodes
    # Performance fix: extract MEPs only for bucket candidates, not all nodes.
    # Bucket assignment is cheap (scores/uncertainties only), then we extract
    # MEPs only for the much smaller candidate set.
    selector = TraceSelector(config, seed)
    from gread_core.tracing.buckets import assign_buckets

    bucket_assignments = assign_buckets(scores, uncertainties, labels)
    if trace_mask is not None:
        bucket_assignments = [
            assignment if bool(trace_mask[i]) else None
            for i, assignment in enumerate(bucket_assignments)
        ]
    bucket_nodes: dict[str, list[int]] = {
        "uncertain": [],
        "high_conf_fraud": [],
        "high_conf_benign": [],
    }
    for i, bl in enumerate(bucket_assignments):
        if bl is not None:
            bucket_nodes[bl].append(i)

    all_candidates = []
    for v in bucket_nodes.values():
        all_candidates.extend(v)

    # Extract MEPs only for bucket candidates (not all N nodes)
    candidate_meps_list = adapter.extract(all_candidates)
    candidate_meps = dict(zip(all_candidates, candidate_meps_list, strict=True))

    # Build full meps list indexed by node id for selector
    # (selector expects meps[node_id] to work)
    class _MEPProxy(Sequence[Any]):
        """Lazy MEP lookup that only has entries for candidates."""
        def __init__(self, mep_map: dict[int, Any]) -> None:
            self._map = mep_map
        def __getitem__(self, idx: int | slice) -> Any:
            if isinstance(idx, slice):
                indices = sorted(self._map.keys())[idx]
                return [self._map[i] for i in indices]
            return self._map[idx]
        def __len__(self) -> int:
            return len(self._map)
        def __iter__(self) -> Iterator[Any]:
            for idx in sorted(self._map.keys()):
                yield self._map[idx]

    meps = _MEPProxy(candidate_meps)
    selection = selector.select(
        scores,
        uncertainties,
        labels,
        meps,
        bucket_assignments=bucket_assignments,
    )  # type: ignore[arg-type]

    logger.info("Selected %d trace nodes for ERR generation", len(selection.node_ids))

    # Generate ERRs via LLM
    selected_meps = [candidate_meps[i] for i in selection.node_ids]
    selected_labels = (
        [int(labels[i]) for i in selection.node_ids] if labels is not None else None
    )

    if config.get("stage2", {}).get("release_detector_before_llm", False):
        with suppress(Exception):
            detector = detector.cpu()
        with suppress(Exception):
            data = data.cpu()
        del adapter
        del logits
        del embeddings
        del scores
        del uncertainties
        del bucket_assignments
        del bucket_nodes
        del all_candidates
        del candidate_meps_list
        del candidate_meps
        del meps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Released detector/data tensors before LLM generation")

    contract_version = config.get("verifier", {}).get("contract_version")
    if hasattr(teacher, "generate_err_attempts"):
        err_attempts = teacher.generate_err_attempts(
            selected_meps,
            selected_labels,
            contract_version=contract_version,
        )
        err_results = [
            attempt
            for attempt in err_attempts
            if attempt.err is not None and attempt.verification.accepted
        ]
    else:
        err_results = teacher.generate_err(
            selected_meps, selected_labels, contract_version=contract_version,
        )
        err_attempts = [
            _AcceptedOnlyAttempt(
                node_id=result.node_id,
                raw_response=None,
                err=result.err,
                verification=result.verification,
                parse_error=None,
            )
            for result in err_results
        ]

    # Separate accepted vs rejected
    # teacher.generate_err already returns only accepted, but we also need
    # to track rejected ones for the loss masking contract
    accepted_errs: list[dict[str, Any]] = []
    rejected_errs: list[dict[str, Any]] = []
    rejection_report: list[dict[str, Any]] = []
    raw_err_audit: list[dict[str, Any]] = []
    attempts_by_node = {attempt.node_id: attempt for attempt in err_attempts}

    for i, mep in enumerate(selected_meps):
        # Try to find matching accepted result
        accepted_match = None
        for result in err_results:
            if result.node_id == mep.node_id:
                accepted_match = result
                break

        attempt = attempts_by_node.get(mep.node_id)
        raw_response = getattr(attempt, "raw_response", None)
        parsed_err = (
            attempt.err.model_dump()
            if attempt is not None and attempt.err is not None
            else None
        )
        verification = getattr(attempt, "verification", None)
        verifier_reasons = list(getattr(verification, "reasons", []))
        failed_checks = _failed_checks_from_reasons(verifier_reasons)
        parse_error = getattr(attempt, "parse_error", None)
        audit_entry = {
            "node_id": mep.node_id,
            "node_idx": selection.node_ids[i],
            "bucket": selection.bucket_labels[i],
            "accepted": bool(verification.accepted) if verification is not None else False,
            "raw_response": raw_response,
            "parsed_err": parsed_err,
            "verifier_reasons": verifier_reasons,
            "failed_checks": failed_checks,
            "parse_error": parse_error,
        }
        raw_err_audit.append(audit_entry)

        if accepted_match is not None:
            accepted_errs.append({
                "node_id": mep.node_id,
                "node_idx": selection.node_ids[i],
                "bucket": selection.bucket_labels[i],
                "err": accepted_match.err.model_dump(),
                "accepted": True,
            })
        else:
            # This MEP was rejected or failed parsing
            rejected_errs.append({
                "node_id": mep.node_id,
                "node_idx": selection.node_ids[i],
                "bucket": selection.bucket_labels[i],
                "err": parsed_err,
                "raw_response": raw_response,
                "verifier_reasons": verifier_reasons,
                "failed_checks": failed_checks,
                "parse_error": parse_error,
                "accepted": False,
            })
            rejection_report.append(audit_entry)

    logger.info(
        "ERR generation complete: %d accepted, %d rejected",
        len(accepted_errs), len(rejected_errs),
    )

    return Stage2Result(
        accepted_errs=accepted_errs,
        rejected_errs=rejected_errs,
        rejection_report=rejection_report,
        raw_err_audit=raw_err_audit,
    )


def _compute_uncertainties(scores: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
    """Compute per-node uncertainty from scores and embeddings.

    Uses entropy-like measure: uncertainty = 1 - |2*score - 1|.
    High uncertainty when score is near 0.5.
    """
    return 1.0 - torch.abs(2.0 * scores - 1.0)  # type: ignore[no-any-return]


class _AcceptedOnlyAttempt:
    def __init__(
        self,
        node_id: str,
        raw_response: str | None,
        err: Any,
        verification: Any,
        parse_error: str | None,
    ) -> None:
        self.node_id = node_id
        self.raw_response = raw_response
        self.err = err
        self.verification = verification
        self.parse_error = parse_error


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _build_rejection_summary(
    accepted_errs: list[dict[str, Any]],
    rejection_report: list[dict[str, Any]],
    raw_err_audit: list[dict[str, Any]],
) -> dict[str, Any]:
    by_failed_check: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    parse_failed = 0
    for entry in rejection_report:
        failed_checks = entry.get("failed_checks", {})
        for check_name in failed_checks:
            by_failed_check[check_name] += 1
        for reason in entry.get("verifier_reasons", []):
            by_reason[reason] += 1
        if entry.get("parse_error") is not None or "parse" in failed_checks:
            parse_failed += 1

    total_attempts = (
        len(raw_err_audit)
        if raw_err_audit
        else len(accepted_errs) + len(rejection_report)
    )
    return {
        "total_attempts": total_attempts,
        "accepted": len(accepted_errs),
        "rejected": len(rejection_report),
        "parse_failed": parse_failed,
        "verifier_rejected": len(rejection_report) - parse_failed,
        "by_failed_check": dict(sorted(by_failed_check.items())),
        "by_reason": dict(sorted(by_reason.items())),
    }


def _failed_checks_from_reasons(reasons: list[str]) -> dict[str, list[str]]:
    failed: dict[str, list[str]] = {}
    for reason in reasons:
        check_name = _failed_check_from_reason(reason)
        failed.setdefault(check_name, []).append(reason)
    return failed


def _failed_check_from_reason(reason: str) -> str:
    if reason.startswith("Parse failed:"):
        return "parse"
    if reason.startswith("Unknown risk_type:") or reason.endswith("must be a list"):
        return "schema"
    if (
        reason.startswith("Evidence ID not available:")
        or reason.startswith("Supporting evidence has unavailable value:")
    ):
        return "availability"
    if (
        "not in allowed_support_ids" in reason
        or "not in allowed_counter_ids" in reason
        or reason.startswith("Forbidden as supporting_evidence:")
        or reason.startswith("Forbidden as counter_evidence:")
        or reason.startswith("Evidence in both support and counter:")
    ):
        return "role_consistency"
    if (
        reason.startswith("No required condition satisfied for")
        or reason.startswith("Forbidden condition met for")
    ):
        return "contract"
    if reason.startswith("Score-related ID in evidence:"):
        return "score_blindness"
    if "incompatible with fraud label" in reason or "incompatible with benign label" in reason:
        return "label_compatibility"
    return "unknown"
