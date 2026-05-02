def generate_explanation(
    risk_type: str,
    supporting_evidence: list[str],
    counter_evidence: list[str],
) -> str:
    support = ", ".join(supporting_evidence) if supporting_evidence else "none"
    counter = ", ".join(counter_evidence) if counter_evidence else "none"
    return (
        f"The node is classified as {risk_type} based on supporting evidence "
        f"[{support}] and counter evidence [{counter}]."
    )


def format_evidence_list(evidence: list[str], prefix: str = "- ") -> str:
    """Format evidence list as bullet points."""
    if not evidence:
        return f"{prefix}none"
    return "\n".join(f"{prefix}{e}" for e in evidence)


def generate_structured_explanation(
    risk_type: str,
    supporting_evidence: list[str],
    counter_evidence: list[str],
) -> dict[str, object]:
    """Return structured explanation dict for programmatic consumption."""
    return {
        "explanation": generate_explanation(risk_type, supporting_evidence, counter_evidence),
        "risk_type": risk_type,
        "supporting_evidence": supporting_evidence,
        "counter_evidence": counter_evidence,
        "num_supporting": len(supporting_evidence),
        "num_counter": len(counter_evidence),
    }
