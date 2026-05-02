# Evidence Contract Verifier

## Purpose

The Evidence Contract Verifier deterministically decides whether an LLM-generated ERR can be used for reasoning distillation.

The verifier is the main trust boundary.

## Verifier Output

```python
VerificationResult(
    accepted: bool,
    reasons: list[str]
)
```

## Acceptance Function

An ERR is accepted only if all checks pass:

```text
a_v =
  V_schema
  ∧ V_availability
  ∧ V_role
  ∧ V_contract
  ∧ V_score_blind
  ∧ V_label
```

## Check 1: Schema Validity

Requires:

* valid JSON parse;
* known risk type;
* `supporting_evidence` is list;
* `counter_evidence` is list;
* `summary` is string.

## Check 2: Evidence Availability

Every cited evidence ID must appear in the MEP reasoning channel.

Reject if:

```text
cited evidence is absent
cited evidence is unavailable but used as support
unknown evidence ID appears
```

## Check 3: Role Consistency

Rules:

```text
supporting_evidence ⊆ allowed_support_ids
counter_evidence ⊆ allowed_counter_ids
supporting_evidence ∩ counter_evidence = ∅
counter_signal cannot be supporting evidence
prediction_score cannot appear anywhere
uncertainty_level cannot be sole support for strong fraud type
```

## Check 4: Risk-Evidence Contract

Each risk type has a contract:

```text
C_t = (required_any, optional, forbidden)
```

### spectral_anomaly

Required any:

```text
detector_signal in {
  high_frequency_response_high,
  spectral_energy_shift_high,
  bandpass_response_high
}
```

Optional:

```text
neighbor_consistency in {low, medium}
detector_signal_strength in {moderate, strong}
```

Forbidden:

```text
detector_signal = unavailable
detector_signal_strength = weak
supporting_evidence only contains uncertainty_level
```

### camouflage_neighbor

Required any:

```text
neighbor_consistency = low
detector_signal in {
  camouflage_neighbor_filter_high,
  neighbor_selection_disagreement_high,
  relation_aware_camouflage_signal
}
```

Optional:

```text
degree_level in {high, burst}
feature_neighbor_discrepancy = high
```

Forbidden:

```text
neighbor_consistency = high
counter_signal = benign_neighbor_signal_high with no positive detector-native support
```

### feature_structure_conflict

Required any:

```text
feature_neighbor_discrepancy = high
```

Optional:

```text
neighbor_consistency in {low, medium}
detector_signal_strength in {moderate, strong}
```

Forbidden:

```text
feature_neighbor_discrepancy in {low, unavailable}
```

### structural_discrepancy

Required any:

```text
degree_level in {very_low, high, burst}
neighbor_consistency = low
```

Optional:

```text
feature_neighbor_discrepancy in {medium, high}
```

Forbidden:

```text
degree_level = normal
neighbor_consistency = high
no detector-native support
```

### relation_or_burst_anomaly

Required any:

```text
degree_level = burst
detector_signal indicates relation anomaly
detector_signal indicates burst anomaly
```

Optional:

```text
neighbor_consistency = low
```

Forbidden:

```text
all structural evidence normal
```

### weak_or_uncertain_evidence

Required any:

```text
uncertainty_level = high
detector_signal = unavailable
no positive risk contract is satisfied
```

Optional:

```text
counter_signal in {
  benign_neighbor_signal_high,
  benign_neighbor_signal_medium
}
```

Forbidden:

```text
strong positive detector_signal with no counter evidence
```

## Check 5: Score-Blindness

Reject if ERR includes:

```text
prediction_score
fraud_score
probability_score
base_score
model_score
```

in any evidence field.

## Check 6: Label Compatibility

When ground-truth label is available:

For fraud label:

```text
y = 1
risk_type cannot be weak_or_uncertain_evidence unless uncertainty_level = high and detector_signal is unavailable
```

For benign label:

```text
y = 0
risk_type cannot be strong malicious type unless contract support is strong and label-noise mode is enabled
```

Default label-noise mode:

```yaml
label_noise_mode: false
```

## Forbidden Main-Method Behavior

Do not implement in main verifier:

* LLM-as-judge
* learned verifier
* semantic embedding similarity verifier
* soft acceptance score
* dynamic self-consistency voting
* DHEF

These may be experimental only.

## Implementation Files

Expected files:

```text
src/gread_core/verification/schema.py
src/gread_core/verification/availability.py
src/gread_core/verification/role_consistency.py
src/gread_core/verification/contract.py
src/gread_core/verification/score_blindness.py
src/gread_core/verification/label_compatibility.py
src/gread_core/verification/verifier.py
configs/contracts/gread_v1.yaml
tests/unit/test_contract_verifier.py
tests/paper_alignment/test_accepted_err_only.py
```

## Required Tests

1. valid spectral anomaly ERR is accepted.
2. counter_signal as supporting evidence is rejected.
3. prediction_score in evidence is rejected.
4. unavailable detector_signal supporting spectral anomaly is rejected.
5. unknown evidence ID is rejected.
6. benign label with strong malicious risk is rejected.
7. fraud label with weak_or_uncertain risk is rejected unless allowed.
8. summary changes do not affect verification.
9. verifier is deterministic.

## Acceptance Criteria

This module is complete when:

* all six checks are implemented;
* rejection reasons are explicit;
* contract config is YAML-driven;
* no LLM code is imported;
* positive and negative fixtures exist;
* all verifier and paper-alignment tests pass.
