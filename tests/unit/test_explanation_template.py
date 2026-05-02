from gread_core.inference.explanation_template import generate_explanation


def test_explanation_contains_risk_type() -> None:
    result = generate_explanation(
        risk_type="spectral_anomaly",
        supporting_evidence=["detector_signal", "neighbor_consistency"],
        counter_evidence=["counter_signal"],
    )
    assert "spectral_anomaly" in result


def test_explanation_contains_evidence_names() -> None:
    result = generate_explanation(
        risk_type="structural_discrepancy",
        supporting_evidence=["degree_level"],
        counter_evidence=[],
    )
    assert "degree_level" in result


def test_explanation_is_deterministic() -> None:
    args = dict(
        risk_type="spectral_anomaly",
        supporting_evidence=["detector_signal"],
        counter_evidence=["counter_signal"],
    )
    assert generate_explanation(**args) == generate_explanation(**args)
