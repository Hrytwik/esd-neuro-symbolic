"""
SymbolicFeatureAdapter — symbolic reasoning execution and signal extraction.

This module is the core bridge between the dataset and the symbolic diagnostic
reasoning infrastructure. For each patient record, it executes the full
reasoning pipeline on the 12 clinical features and extracts a rich set of
reasoning-aware signals that become augmentation inputs for Model C.

This is NOT simple feature engineering. The adapter runs the complete
symbolic reasoning stack — clinical grading, evidence activation, contradiction
analysis, certainty propagation, differential competition, evidence sufficiency,
instability monitoring, FSM transition, safety gate, and escalation — and then
reads the structured outputs of that multi-stage inference process.

The resulting SymbolicFeatureVector captures:
  · Terminal certainty metrics (from CertaintyPropagator)
  · Contradiction load and bilateral conflict signal (from ConflictAnalyzer)
  · FSM state and escalation decision (from StateTracker + EscalationEngine)
  · Trajectory dynamics (from TrajectoryGraph / CertaintyGraph)
  · Safety gate activation status (from SafetyGate)

These signals represent genuine symbolic inference over the clinical data,
not hand-crafted numerical transformations of the raw ordinal features.

Usage
-----
  from src.dataset_integration.symbolic_feature_adapter import SymbolicFeatureAdapter

  adapter = SymbolicFeatureAdapter()
  vectors = adapter.adapt_batch(dataset.records)
  feature_dict = adapter.to_feature_dict(vectors[0])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.dataset_integration.dataset_loader import DermatologyRecord
from src.dataset_integration.clinical_feature_mapper import ClinicalFeatureMapper


# ── FSM state ordinal encoding ────────────────────────────────────────────────

_FSM_STATE_ORDER: dict[str, int] = {
    "INITIAL_EVIDENCE":        0,
    "PARTIAL_ALIGNMENT":       1,
    "REINFORCING_ALIGNMENT":   2,
    "CONTRADICTION_DETECTED":  3,
    "AMBIGUITY_ESCALATION":    4,
    "CERTAINTY_STABILIZATION": 5,
    "SAFE_TRIAGE":             6,
    "BIOPSY_ESCALATION":       7,
    "UNSTABLE_REASONING":      8,
}

_RECOMMENDATION_ORDER: dict[str, int] = {
    "SAFE_NON_INVASIVE_TRIAGE": 0,
    "MODERATE_CERTAINTY":       1,
    "AMBIGUOUS_PRESENTATION":   2,
    "BIOPSY_RECOMMENDED":       3,
    "HIGH_RISK_CONTRADICTION":  4,
}

_DISEASE_ORDER: dict[str, int] = {
    "psoriasis":                 0,
    "seborrheic_dermatitis":     1,
    "lichen_planus":             2,
    "pityriasis_rosea":          3,
    "chronic_dermatitis":        4,
    "pityriasis_rubra_pilaris":  5,
    "unknown":                   -1,
}

_MAX_ENTROPY_6CLASS: float = math.log2(6)  # ≈ 2.585 bits


# ── Symbolic feature vector ───────────────────────────────────────────────────

@dataclass(frozen=True)
class SymbolicFeatureVector:
    """
    Reasoning-aware signals extracted from one full pipeline execution.

    This is the primary output of SymbolicFeatureAdapter.adapt().
    All fields are derived from genuine symbolic inference over the
    patient's 12 clinical features.

    Attributes (terminal reasoning outputs)
    ----------------------------------------
    certainty:
        Leading hypothesis certainty at terminal stage [0, 1].
    certainty_gap:
        Gap between leading and second hypothesis [0, 1].
    contradiction_load:
        Bilateral contradiction load at terminal stage [0, ∞).
    ambiguity_index:
        Shannon entropy of certainty distribution (bits).
    requires_biopsy:
        True if the terminal recommendation mandates biopsy.
    is_safe_triage:
        True if the terminal recommendation is SAFE_NON_INVASIVE_TRIAGE.
    leading_disease:
        Canonical name of the leading hypothesis at terminal stage.
    recommendation:
        Terminal triage recommendation string.
    final_state:
        Terminal FSM state string.

    Attributes (trajectory dynamics)
    ----------------------------------
    convergence_index:
        Ratio of final certainty to peak certainty [0, 1].
        1.0 = perfect convergence; < 0.70 = significant certainty decay.
    oscillation_count:
        Number of direction reversals in the certainty trajectory.
    trajectory_length:
        Number of pipeline stages executed (snapshot count).
    peak_certainty:
        Maximum certainty observed across all trajectory stages.
    certainty_delta_total:
        Certainty change from first to final stage.
    contradiction_emerged:
        True if contradiction load was non-zero at any stage.
    leadership_changed:
        True if the leading disease changed across trajectory stages.
    leadership_changes_count:
        Number of leadership transitions observed.
    entropy_reduction:
        Total entropy drop from peak to final stage (bits).
    stabilisation_stage:
        Stage index where certainty first stabilised, or -1 if never.
    was_dampened:
        True if certainty was suppressed by contradiction dampening.

    Attributes (encoded for ML)
    ----------------------------
    fsm_state_encoded:
        Ordinal integer encoding of the terminal FSM state [0, 8].
    recommendation_encoded:
        Ordinal integer encoding of the terminal recommendation [0, 4].
    leading_disease_encoded:
        Integer encoding of the leading disease [0, 5]; -1 if unknown.
    normalised_entropy:
        ambiguity_index / log2(6) — normalised to [0, 1].
    certainty_sufficiency:
        1.0 if certainty ≥ 0.55 and gap ≥ 0.20, else 0.0.

    Attributes (provenance)
    ------------------------
    patient_id:
        Source patient identifier.
    disease_label:
        Ground-truth disease label (canonical form).
    pipeline_success:
        True if the reasoning pipeline completed without error.
    error_message:
        Non-empty string if pipeline_success is False.
    """

    # Terminal outputs
    certainty:                float
    certainty_gap:            float
    contradiction_load:       float
    ambiguity_index:          float
    requires_biopsy:          bool
    is_safe_triage:           bool
    leading_disease:          str
    recommendation:           str
    final_state:              str

    # Trajectory dynamics
    convergence_index:        float
    oscillation_count:        int
    trajectory_length:        int
    peak_certainty:           float
    certainty_delta_total:    float
    contradiction_emerged:    bool
    leadership_changed:       bool
    leadership_changes_count: int
    entropy_reduction:        float
    stabilisation_stage:      int       # -1 if never stabilised
    was_dampened:             bool

    # ML-encoded signals
    fsm_state_encoded:        int
    recommendation_encoded:   int
    leading_disease_encoded:  int
    normalised_entropy:       float
    certainty_sufficiency:    float

    # Provenance
    patient_id:               str
    disease_label:            str
    pipeline_success:         bool
    error_message:            str       = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict of all signal fields (excluding provenance)."""
        return {
            "certainty":                self.certainty,
            "certainty_gap":            self.certainty_gap,
            "contradiction_load":       self.contradiction_load,
            "ambiguity_index":          self.ambiguity_index,
            "requires_biopsy":          int(self.requires_biopsy),
            "is_safe_triage":           int(self.is_safe_triage),
            "convergence_index":        self.convergence_index,
            "oscillation_count":        self.oscillation_count,
            "trajectory_length":        self.trajectory_length,
            "peak_certainty":           self.peak_certainty,
            "certainty_delta_total":    self.certainty_delta_total,
            "contradiction_emerged":    int(self.contradiction_emerged),
            "leadership_changed":       int(self.leadership_changed),
            "leadership_changes_count": self.leadership_changes_count,
            "entropy_reduction":        self.entropy_reduction,
            "stabilisation_stage":      self.stabilisation_stage,
            "was_dampened":             int(self.was_dampened),
            "fsm_state_encoded":        self.fsm_state_encoded,
            "recommendation_encoded":   self.recommendation_encoded,
            "leading_disease_encoded":  self.leading_disease_encoded,
            "normalised_entropy":       self.normalised_entropy,
            "certainty_sufficiency":    self.certainty_sufficiency,
        }

    @property
    def signal_names(self) -> tuple[str, ...]:
        """Ordered tuple of reasoning signal names."""
        return tuple(self.to_dict().keys())


# ── Symbolic feature adapter ──────────────────────────────────────────────────

class SymbolicFeatureAdapter:
    """
    Executes the symbolic reasoning pipeline on dataset records and
    extracts structured reasoning signals for Model C augmentation.

    The adapter wraps a single PipelineRunner instance and reuses it
    across all records. The pipeline is reset between records so that
    per-run state (FSM cursor, instability window) does not carry over.

    Parameters
    ----------
    pipeline_runner:
        Pre-constructed PipelineRunner. If None, a default runner with
        standard PipelineConfig is built at first use.
    age_imputation_value:
        Age imputation value passed to the clinical feature mapper.
        Should match DatasetSummary.age_median.
    suppress_errors:
        If True, pipeline exceptions produce a fallback zero-signal vector
        rather than propagating. Recommended for batch processing.
    """

    def __init__(
        self,
        pipeline_runner: Any | None = None,
        age_imputation_value: float = 33.0,
        suppress_errors: bool = True,
    ) -> None:
        self._runner         = pipeline_runner
        self._mapper         = ClinicalFeatureMapper(age_imputation_value)
        self._suppress       = suppress_errors
        self._runner_ready   = pipeline_runner is not None

    # ── Public API ────────────────────────────────────────────────────────────

    def adapt(
        self,
        record: DermatologyRecord,
        run_id: str | None = None,
    ) -> SymbolicFeatureVector:
        """
        Run the symbolic reasoning pipeline on one patient record and
        return a SymbolicFeatureVector of reasoning signals.

        Parameters
        ----------
        record:
            A DermatologyRecord from DermatologyDatasetLoader.
        run_id:
            Optional run ID passed to PipelineRunner. Auto-generated if None.
        """
        self._ensure_runner()
        clinical_input = self._mapper.map_clinical(record)

        try:
            result = self._runner.run(
                case_id=record.patient_id,
                feature_values=clinical_input.feature_values,
                run_id=run_id,
            )
            return self._extract_signals(result, record)
        except Exception as exc:  # noqa: BLE001
            if self._suppress:
                return self._fallback_vector(record, str(exc))
            raise

    def adapt_batch(
        self,
        records: list[DermatologyRecord],
    ) -> list[SymbolicFeatureVector]:
        """
        Run the symbolic reasoning pipeline on every record in a list.

        Returns vectors in the same order as the input records.
        Failed records produce fallback zero-signal vectors.
        """
        return [self.adapt(r) for r in records]

    def to_feature_dict(self, vector: SymbolicFeatureVector) -> dict[str, float]:
        """
        Return the reasoning signals as a flat float dict.

        Suitable for concatenation with clinical feature arrays in Model C.
        All boolean signals are encoded as 0.0/1.0.
        """
        return {k: float(v) for k, v in vector.to_dict().items()}

    def to_feature_matrix(
        self,
        vectors: list[SymbolicFeatureVector],
    ) -> list[list[float]]:
        """
        Return all vectors as a patient × signal float matrix.

        Row i corresponds to vectors[i]. Signal ordering is consistent
        with SymbolicFeatureVector.signal_names.
        """
        if not vectors:
            return []
        keys = list(vectors[0].to_dict().keys())
        return [[float(v.to_dict()[k]) for k in keys] for v in vectors]

    @staticmethod
    def signal_names() -> tuple[str, ...]:
        """Return the ordered tuple of symbolic signal names."""
        # Build from a representative instance
        dummy = _fallback_zero_vector("_", "_")
        return tuple(dummy.to_dict().keys())

    # ── Pipeline initialisation ───────────────────────────────────────────────

    def _ensure_runner(self) -> None:
        """Lazily initialise the pipeline runner on first use."""
        if self._runner_ready:
            return
        from src.pipeline.pipeline_runner import PipelineRunner
        from src.pipeline.pipeline_config import PipelineConfig
        self._runner       = PipelineRunner(config=PipelineConfig())
        self._runner_ready = True

    # ── Signal extraction ─────────────────────────────────────────────────────

    def _extract_signals(
        self,
        result: Any,   # PipelineResult
        record: DermatologyRecord,
    ) -> SymbolicFeatureVector:
        """
        Extract SymbolicFeatureVector from a completed PipelineResult.

        Uses the TrajectoryGraph and CertaintyGraph infrastructure to
        compute trajectory-level dynamics beyond the terminal snapshot.
        """
        # Terminal scalars — always present on a successful result
        certainty         = float(result.max_certainty)
        gap               = float(result.certainty_gap)
        contra_load       = float(result.contradiction_load)
        ambiguity         = float(result.ambiguity_index)
        requires_biopsy   = bool(result.requires_biopsy)
        is_safe           = bool(result.is_safe_triage)
        leading           = result.leading_disease or "unknown"
        recommendation    = result.recommendation or "AMBIGUOUS_PRESENTATION"
        final_state       = result.final_state or "INITIAL_EVIDENCE"

        # Trajectory-level dynamics via CertaintyGraph
        conv_idx    = 0.0
        osc_count   = 0
        traj_len    = 1
        peak_cert   = certainty
        cert_delta  = 0.0
        contra_emrg = contra_load > 0.0
        lead_chg    = False
        lead_chg_n  = 0
        ent_red     = 0.0
        stab_stage  = -1
        was_dampend = False

        traj = getattr(result, "trajectory", None)
        if traj is not None and traj.snapshots:
            snaps = traj.snapshots
            traj_len = len(snaps)

            certainties = [s.max_certainty for s in snaps]
            peak_cert   = max(certainties)
            final_cert  = certainties[-1]
            cert_delta  = final_cert - certainties[0]
            conv_idx    = final_cert / peak_cert if peak_cert > 0 else 0.0

            # Oscillation: direction reversals
            if len(certainties) >= 3:
                dirs = [
                    1 if certainties[i] > certainties[i - 1]
                    else (-1 if certainties[i] < certainties[i - 1] else 0)
                    for i in range(1, len(certainties))
                ]
                osc_count = sum(
                    1 for i in range(1, len(dirs))
                    if dirs[i] != 0 and dirs[i - 1] != 0
                    and dirs[i] != dirs[i - 1]
                )

            # Contradiction emergence
            contra_emrg = any(s.contradiction_load > 0.0 for s in snaps)

            # Leadership changes
            diseases = [s.leading_disease for s in snaps]
            lead_chg_n = sum(
                1 for i in range(1, len(diseases)) if diseases[i] != diseases[i - 1]
            )
            lead_chg = lead_chg_n > 0

            # Entropy reduction
            entropies = [s.ambiguity_index for s in snaps]
            ent_red = max(entropies) - entropies[-1] if entropies else 0.0

            # Stabilisation stage: first stage where cert≥0.55 and gap≥0.20
            for s in snaps:
                if s.max_certainty >= 0.55 and s.certainty_gap >= 0.20:
                    stab_stage = s.stage
                    break

            # Dampening detection (load > 0.20 at any stage)
            was_dampend = any(s.contradiction_load > 0.20 for s in snaps)

        # ML-encoded ordinals
        fsm_enc  = _FSM_STATE_ORDER.get(final_state, 0)
        rec_enc  = _RECOMMENDATION_ORDER.get(recommendation, 2)
        dis_enc  = _DISEASE_ORDER.get(leading, -1)
        norm_ent = ambiguity / _MAX_ENTROPY_6CLASS
        cert_suf = 1.0 if (certainty >= 0.55 and gap >= 0.20) else 0.0

        return SymbolicFeatureVector(
            certainty=certainty,
            certainty_gap=gap,
            contradiction_load=contra_load,
            ambiguity_index=ambiguity,
            requires_biopsy=requires_biopsy,
            is_safe_triage=is_safe,
            leading_disease=leading,
            recommendation=recommendation,
            final_state=final_state,
            convergence_index=conv_idx,
            oscillation_count=osc_count,
            trajectory_length=traj_len,
            peak_certainty=peak_cert,
            certainty_delta_total=cert_delta,
            contradiction_emerged=contra_emrg,
            leadership_changed=lead_chg,
            leadership_changes_count=lead_chg_n,
            entropy_reduction=ent_red,
            stabilisation_stage=stab_stage,
            was_dampened=was_dampend,
            fsm_state_encoded=fsm_enc,
            recommendation_encoded=rec_enc,
            leading_disease_encoded=dis_enc,
            normalised_entropy=norm_ent,
            certainty_sufficiency=cert_suf,
            patient_id=record.patient_id,
            disease_label=record.disease_label,
            pipeline_success=result.success,
            error_message="",
        )

    def _fallback_vector(
        self,
        record: DermatologyRecord,
        error_msg: str,
    ) -> SymbolicFeatureVector:
        """Return an all-zero signal vector when pipeline execution fails."""
        v = _fallback_zero_vector(record.patient_id, record.disease_label)
        # Can't mutate frozen dataclass — rebuild with error message
        return SymbolicFeatureVector(
            **{k: getattr(v, k) for k in v.__dataclass_fields__ if k not in ("error_message", "pipeline_success")},
            pipeline_success=False,
            error_message=error_msg[:200],
        )


# ── Module-level helper ───────────────────────────────────────────────────────

def _fallback_zero_vector(patient_id: str, disease_label: str) -> SymbolicFeatureVector:
    """Create an all-zero SymbolicFeatureVector (used for failed runs)."""
    return SymbolicFeatureVector(
        certainty=0.0,
        certainty_gap=0.0,
        contradiction_load=0.0,
        ambiguity_index=_MAX_ENTROPY_6CLASS,
        requires_biopsy=False,
        is_safe_triage=False,
        leading_disease="unknown",
        recommendation="AMBIGUOUS_PRESENTATION",
        final_state="INITIAL_EVIDENCE",
        convergence_index=0.0,
        oscillation_count=0,
        trajectory_length=0,
        peak_certainty=0.0,
        certainty_delta_total=0.0,
        contradiction_emerged=False,
        leadership_changed=False,
        leadership_changes_count=0,
        entropy_reduction=0.0,
        stabilisation_stage=-1,
        was_dampened=False,
        fsm_state_encoded=0,
        recommendation_encoded=2,
        leading_disease_encoded=-1,
        normalised_entropy=1.0,
        certainty_sufficiency=0.0,
        patient_id=patient_id,
        disease_label=disease_label,
        pipeline_success=False,
        error_message="",
    )
