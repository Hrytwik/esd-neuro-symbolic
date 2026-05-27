"""
SymbolicSignalEnrichmentV2 — expanded reasoning trajectory signals for Model C.

Extends the original 22 symbolic signals with 18 additional derived signals
that capture higher-order reasoning trajectory properties. All new signals
are derived exclusively from the existing pipeline outputs — no raw clinical
features are re-engineered.

Original 22 signals (from SymbolicFeatureAdapter):
  certainty, certainty_gap, contradiction_load, ambiguity_index,
  requires_biopsy, is_safe_triage, convergence_index, oscillation_count,
  trajectory_length, peak_certainty, certainty_delta_total,
  contradiction_emerged, leadership_changed, leadership_changes_count,
  entropy_reduction, stabilisation_stage, was_dampened,
  fsm_state_encoded, recommendation_encoded, leading_disease_encoded,
  normalised_entropy, certainty_sufficiency

New 18 signals (this module):

  Trajectory dynamics:
    trajectory_quality_index    — composite trajectory quality score
    certainty_velocity          — certainty_delta_total / trajectory_length
    certainty_peak_ratio        — certainty / peak_certainty
    entropy_trajectory_gradient — entropy_reduction / trajectory_length
    convergence_stability       — convergence_index * (1 - oscillation_rate)

  Competition topology:
    gap_entropy_product         — certainty_gap * (1 - normalised_entropy)
    resolution_confidence       — gap * certainty * convergence_index
    leadership_stability_index  — 1 / (1 + leadership_changes_count)
    competition_entropy_ratio   — certainty_gap / max(normalised_entropy, 0.01)

  Contradiction topology:
    contradiction_certainty_product  — contradiction_load * certainty
    dampening_impact                 — was_dampened * (peak_certainty - certainty)
    contradiction_trajectory_index   — contradiction_emerged * oscillation_count

  Clinical context:
    safe_trajectory_score       — (not requires_biopsy) * convergence_index
    escalation_confidence       — requires_biopsy * contradiction_load
    clinical_evidence_density   — certainty * trajectory_length / 8.0
    ambiguity_gap_ratio         — ambiguity_index / max(certainty_gap, 0.01)
    certainty_stability_product — certainty * (1 - oscillation_count * 0.1)
    reasoning_depth_index       — trajectory_length * convergence_index / 8.0

Total expanded feature set: 22 + 18 = 40 symbolic reasoning signals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.dataset_integration.symbolic_feature_adapter import SymbolicFeatureVector


# ── New signal definitions ────────────────────────────────────────────────────

_NEW_SIGNAL_NAMES: list[str] = [
    # Trajectory dynamics
    "trajectory_quality_index",
    "certainty_velocity",
    "certainty_peak_ratio",
    "entropy_trajectory_gradient",
    "convergence_stability",
    # Competition topology
    "gap_entropy_product",
    "resolution_confidence",
    "leadership_stability_index",
    "competition_entropy_ratio",
    # Contradiction topology
    "contradiction_certainty_product",
    "dampening_impact",
    "contradiction_trajectory_index",
    # Clinical context
    "safe_trajectory_score",
    "escalation_confidence",
    "clinical_evidence_density",
    "ambiguity_gap_ratio",
    "certainty_stability_product",
    "reasoning_depth_index",
]

_MAX_TRAJECTORY_LENGTH: float = 8.0


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class EnrichedSignalSet:
    """
    Complete enriched signal set: original 22 + new 18 signals.

    Provides the combined to_dict() for Model C matrix construction.
    """

    # All original signals
    original: dict[str, float]
    # New enrichment signals
    new_signals: dict[str, float]

    def to_combined_dict(self) -> dict[str, float]:
        """Return all 40 signals in a flat dict."""
        return {**self.original, **self.new_signals}

    @staticmethod
    def combined_signal_names() -> list[str]:
        """Ordered list of all 40 signal names."""
        # Original 22 names from SymbolicFeatureVector.to_dict()
        from src.dataset_integration.symbolic_feature_adapter import SymbolicFeatureVector
        dummy = SymbolicFeatureVector(
            certainty=0.0, certainty_gap=0.0, contradiction_load=0.0,
            ambiguity_index=0.0, requires_biopsy=False, is_safe_triage=False,
            leading_disease="", recommendation="", final_state="",
            convergence_index=0.0, oscillation_count=0, trajectory_length=1,
            peak_certainty=0.0, certainty_delta_total=0.0,
            contradiction_emerged=False, leadership_changed=False,
            leadership_changes_count=0, entropy_reduction=0.0,
            stabilisation_stage=-1, was_dampened=False,
            fsm_state_encoded=0, recommendation_encoded=0,
            leading_disease_encoded=-1, normalised_entropy=0.0,
            certainty_sufficiency=0.0, patient_id="", disease_label="",
            pipeline_success=True,
        )
        original_names = list(dummy.to_dict().keys())
        return original_names + _NEW_SIGNAL_NAMES


@dataclass
class EnrichmentReport:
    """
    Analysis report for the v2 signal enrichment.

    Attributes
    ----------
    n_signals_original:
        Number of original symbolic signals.
    n_signals_new:
        Number of new signals added.
    n_signals_total:
        Total combined signals.
    signal_variance:
        Per-signal variance (higher = more informative).
    top_variance_signals:
        New signals with highest variance (most informative).
    low_variance_signals:
        New signals with near-zero variance (potentially redundant).
    redundancy_pairs:
        New signals with |correlation| >= 0.90 against originals.
    per_disease_mean_signals:
        Mean of each new signal per disease (for discrimination analysis).
    """

    n_signals_original:    int = 22
    n_signals_new:         int = 0
    n_signals_total:       int = 0
    signal_variance:       dict[str, float] = field(default_factory=dict)
    top_variance_signals:  list[str]        = field(default_factory=list)
    low_variance_signals:  list[str]        = field(default_factory=list)
    redundancy_pairs:      list[tuple[str, str, float]] = field(default_factory=list)
    per_disease_mean_signals: dict[str, dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 72,
            "SYMBOLIC SIGNAL ENRICHMENT V2 REPORT",
            "=" * 72,
            f"  Original signals : {self.n_signals_original}",
            f"  New signals      : {self.n_signals_new}",
            f"  Total signals    : {self.n_signals_total}",
            "-" * 72,
            "  TOP VARIANCE NEW SIGNALS (most informative):",
        ]
        for sig in self.top_variance_signals[:8]:
            lines.append(
                f"    {sig:40s} var={self.signal_variance.get(sig, 0.0):.5f}"
            )
        lines += [
            "-" * 72,
            "  LOW VARIANCE NEW SIGNALS (potentially redundant):",
        ]
        for sig in self.low_variance_signals[:5]:
            lines.append(f"    {sig}")
        if self.redundancy_pairs:
            lines += [
                "-" * 72,
                "  HIGH-CORRELATION PAIRS (|r|>=0.90):",
            ]
            for s1, s2, r in self.redundancy_pairs[:5]:
                lines.append(f"    {s1[:20]:20s} vs {s2[:20]:20s} r={r:.3f}")
        lines.append("=" * 72)
        return "\n".join(lines)


# ── Enricher ──────────────────────────────────────────────────────────────────

class SymbolicSignalEnricherV2:
    """
    Derives 18 additional symbolic reasoning signals from SymbolicFeatureVectors.

    The enriched signal set (40 total) provides richer trajectory and
    competition information for Model C classification.
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def enrich(
        self,
        vectors: list[SymbolicFeatureVector],
    ) -> list[EnrichedSignalSet]:
        """
        Derive enriched signal sets for all vectors.

        Parameters
        ----------
        vectors:
            Symbolic feature vectors from the pipeline.

        Returns
        -------
        List of EnrichedSignalSet (one per vector).
        """
        return [self._enrich_one(v) for v in vectors]

    def build_feature_matrix(
        self,
        X_clinical: np.ndarray,
        enriched:   list[EnrichedSignalSet],
    ) -> tuple[np.ndarray, list[str]]:
        """
        Build the expanded feature matrix: clinical + 40 symbolic signals.

        Parameters
        ----------
        X_clinical:
            Clinical feature matrix (n × 12).
        enriched:
            Enriched signal sets.

        Returns
        -------
        (matrix, signal_names) where matrix is n × (12 + 40).
        """
        combined_names = EnrichedSignalSet.combined_signal_names()
        X_sym = np.array([
            [float(e.to_combined_dict().get(n, 0.0)) for n in combined_names]
            for e in enriched
        ])
        X_combined = np.hstack([X_clinical, X_sym])
        all_names  = [f"clin_{i}" for i in range(X_clinical.shape[1])] + combined_names
        return X_combined, all_names

    def build_report(
        self,
        vectors:  list[SymbolicFeatureVector],
        enriched: list[EnrichedSignalSet],
    ) -> EnrichmentReport:
        """Generate analysis report for the enriched signal set."""
        if not enriched:
            return EnrichmentReport()

        new_sigs = _NEW_SIGNAL_NAMES
        X_new    = np.array([
            [float(e.new_signals.get(n, 0.0)) for n in new_sigs]
            for e in enriched
        ])
        sig_var  = {n: float(np.var(X_new[:, i])) for i, n in enumerate(new_sigs)}

        # Original signals for redundancy check
        orig_names = list(vectors[0].to_dict().keys())
        X_orig     = np.array([
            [float(v.to_dict()[n]) for n in orig_names]
            for v in vectors
        ])

        # Redundancy pairs
        redundant: list[tuple[str, str, float]] = []
        for ni, n_new in enumerate(new_sigs):
            col_new = X_new[:, ni]
            if np.std(col_new) < 1e-9:
                continue
            for oi, n_orig in enumerate(orig_names):
                col_orig = X_orig[:, oi]
                if np.std(col_orig) < 1e-9:
                    continue
                r = float(np.corrcoef(col_new, col_orig)[0, 1])
                if abs(r) >= 0.90:
                    redundant.append((n_new, n_orig, r))
        redundant.sort(key=lambda x: abs(x[2]), reverse=True)

        # Per-disease mean new signals
        diseases = list({v.disease_label for v in vectors})
        per_dis: dict[str, dict[str, float]] = {}
        for dis in diseases:
            idxs = [i for i, v in enumerate(vectors) if v.disease_label == dis]
            if not idxs:
                continue
            per_dis[dis] = {
                n: float(np.mean(X_new[idxs, ni]))
                for ni, n in enumerate(new_sigs)
            }

        top_var = sorted(new_sigs, key=lambda n: -sig_var.get(n, 0))[:8]
        low_var = [n for n in new_sigs if sig_var.get(n, 0) < 1e-5]

        return EnrichmentReport(
            n_signals_original=len(orig_names),
            n_signals_new=len(new_sigs),
            n_signals_total=len(orig_names) + len(new_sigs),
            signal_variance=sig_var,
            top_variance_signals=top_var,
            low_variance_signals=low_var,
            redundancy_pairs=redundant[:10],
            per_disease_mean_signals=per_dis,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _enrich_one(self, v: SymbolicFeatureVector) -> EnrichedSignalSet:
        """Derive all 18 new signals for a single vector."""
        cert     = float(v.certainty)
        gap      = float(v.certainty_gap)
        load     = float(v.contradiction_load)
        amb      = float(v.ambiguity_index)
        n_ent    = float(v.normalised_entropy)
        conv     = float(v.convergence_index)
        peak     = float(v.peak_certainty)
        delta    = float(v.certainty_delta_total)
        traj     = max(int(v.trajectory_length), 1)
        n_osc    = int(v.oscillation_count)
        lc_cnt   = int(v.leadership_changes_count)
        ent_red  = float(v.entropy_reduction)
        damped   = float(v.was_dampened)
        req_bio  = float(v.requires_biopsy)
        cont_em  = float(v.contradiction_emerged)
        stab_stg = int(v.stabilisation_stage)

        osc_rate = n_osc / max(traj, 1)

        # Trajectory dynamics
        traj_qual   = conv * (1 - osc_rate) * (cert / max(peak, 0.01))
        cert_vel    = delta / traj
        cert_pr     = cert / max(peak, 0.01)
        ent_grad    = ent_red / traj
        conv_stab   = conv * (1.0 - min(osc_rate, 1.0))

        # Competition topology
        gap_ent     = gap * (1.0 - n_ent)
        res_conf    = gap * cert * conv
        lead_stab   = 1.0 / (1.0 + lc_cnt)
        comp_ent    = gap / max(n_ent, 0.01)

        # Contradiction topology
        contr_cert  = load * cert
        damp_imp    = damped * (peak - cert)
        contr_traj  = cont_em * n_osc

        # Clinical context
        safe_traj   = (1.0 - req_bio) * conv
        esc_conf    = req_bio * load
        clin_dense  = cert * traj / _MAX_TRAJECTORY_LENGTH
        amb_gap     = amb / max(gap, 0.01)
        cert_stab   = cert * max(0.0, 1.0 - n_osc * 0.1)
        reason_dep  = traj * conv / _MAX_TRAJECTORY_LENGTH

        new_signals = {
            "trajectory_quality_index":     float(np.clip(traj_qual, 0.0, 1.0)),
            "certainty_velocity":           float(np.clip(cert_vel, -1.0, 1.0)),
            "certainty_peak_ratio":         float(np.clip(cert_pr, 0.0, 1.0)),
            "entropy_trajectory_gradient":  float(np.clip(ent_grad, -1.0, 1.0)),
            "convergence_stability":        float(np.clip(conv_stab, 0.0, 1.0)),
            "gap_entropy_product":          float(np.clip(gap_ent, 0.0, 1.0)),
            "resolution_confidence":        float(np.clip(res_conf, 0.0, 1.0)),
            "leadership_stability_index":   float(np.clip(lead_stab, 0.0, 1.0)),
            "competition_entropy_ratio":    float(np.clip(comp_ent, 0.0, 10.0)),
            "contradiction_certainty_product": float(np.clip(contr_cert, 0.0, 2.0)),
            "dampening_impact":             float(np.clip(damp_imp, 0.0, 1.0)),
            "contradiction_trajectory_index": float(np.clip(contr_traj, 0.0, 5.0)),
            "safe_trajectory_score":        float(np.clip(safe_traj, 0.0, 1.0)),
            "escalation_confidence":        float(np.clip(esc_conf, 0.0, 2.0)),
            "clinical_evidence_density":    float(np.clip(clin_dense, 0.0, 1.0)),
            "ambiguity_gap_ratio":          float(np.clip(amb_gap, 0.0, 20.0)),
            "certainty_stability_product":  float(np.clip(cert_stab, 0.0, 1.0)),
            "reasoning_depth_index":        float(np.clip(reason_dep, 0.0, 1.0)),
        }

        return EnrichedSignalSet(
            original=v.to_dict(),
            new_signals=new_signals,
        )
