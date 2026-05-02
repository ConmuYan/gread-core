# Detector-Evidence Adapter Protocol

## Purpose

The adapter protocol converts base detector signals into score-blind Minimal Evidence Packages.

## Claim Boundary

GReaD-Core is detector-adaptable only when the base detector exposes at least one computable detector-native evidence signal.

Do not claim universal any-detector support.

## Adapter Interface

Every adapter must implement:

```python
class EvidenceAdapter(ABC):
    detector_name: str

    def extract(self, node_ids: list[int]) -> list[MinimalEvidencePackage]:
        ...

    def supports_detector_signal(self) -> bool:
        ...
```

## Evidence Categories

Every adapter output must include:

```text
E_v = E_generic(v) ∪ E_detector(v) ∪ E_counter(v)
```

### Generic Evidence

Required when computable:

```text
degree_level
neighbor_consistency
feature_neighbor_discrepancy
uncertainty_level
```

### Detector-Native Evidence

Detector-specific signal.

Examples:

#### BWGNN

```text
detector_signal = high_frequency_response_high
detector_signal = bandpass_response_high
detector_signal = spectral_energy_shift_high
```

#### CARE-GNN

```text
detector_signal = camouflage_neighbor_filter_high
detector_signal = neighbor_selection_disagreement_high
detector_signal = relation_aware_camouflage_signal
```

#### GAT

```text
detector_signal = attention_concentration_high
```

#### GCN / GraphSAGE

```text
detector_signal = embedding_neighbor_discrepancy_high
detector_signal = message_disagreement_high
```

#### Tree + Neighborhood Aggregation

```text
detector_signal = feature_importance_risk_high
detector_signal = neighborhood_aggregation_discrepancy_high
```

### Counter Evidence

Examples:

```text
benign_neighbor_signal_low
benign_neighbor_signal_medium
benign_neighbor_signal_high
high_uncertainty_signal
evidence_unavailable_signal
```

## Adapter Output Requirements

For every node, adapter must output:

```text
MinimalEvidencePackage(
  calibration=CalibrationChannel(...),
  reasoning=ReasoningChannel(...)
)
```

The `calibration` channel may include `prediction_score`.

The `reasoning` channel must not include `prediction_score`.

## Quantization

Continuous detector signals must be quantized into:

```text
weak
moderate
strong
unavailable
```

Quantization thresholds must be config-driven.

Hard-coded thresholds in adapter implementation are forbidden.

## Implementation Files

Expected files:

```text
src/gread_core/adapters/base.py
src/gread_core/adapters/bwgnn_adapter.py
src/gread_core/adapters/caregnn_adapter.py
src/gread_core/adapters/pyg_gnn_adapter.py
src/gread_core/adapters/tree_adapter.py
src/gread_core/evidence/generic_signals.py
src/gread_core/evidence/quantization.py
configs/detectors/*.yaml
tests/integration/test_adapter_protocol.py
```

## Required Tests

1. Every adapter returns valid MEPs.
2. Every adapter output includes generic, detector-native, and counter evidence fields.
3. Adapter outputs do not leak `prediction_score` to reasoning channel.
4. Quantization is deterministic.
5. Missing detector-native evidence produces `detector_signal=unavailable`.
6. Unsupported detector does not silently produce fake evidence.

## Acceptance Criteria

This module is complete when:

* mock adapter passes protocol tests;
* at least one real adapter or stub adapter is implemented;
* adapter outputs validate against MEP schema;
* score leakage tests pass;
* detector-specific thresholds are config-driven.
