"""
CompetitionSharpener — inter-hypothesis competition signal enrichment.

Post-processes symbolic feature vectors to amplify differential competition
signals for Model C classification. Does NOT modify the symbolic pipeline
internals — instead, derives enhanced competition signals from the existing
certainty, gap, and entropy outputs.

Clinical reasoning context
--------------------------
The differential competition stage of the reasoning pipeline assigns
certainty scores to all competing hypotheses and applies suppression
to non-leading candidates. When clinical features overlap between
diseases (e.g., psoriasis vs. seborrheic_dermatitis both activate
scaling + erythema rules), the suppression is insufficient and
hypotheses coexist at similar certainty levels.

This module:
  1. Amplifies the certainty_gap signal using a non-linear transform
     that widens the gap for already-separated cases and narrows it
     for tightly-competing cases.
  2. Creates gap × certainty interaction features that capture the
     quality of competition outcome.
  3. Derives leader_confidence: a composite signal combining gap,
     convergence, and leadership stability.
  4. Produces an enriched feature matrix with additional competition
     signals for Model C.

All signals remain genuine reasoning trajectory outputs — they are
derived exclusively from the pipeline's certainty, gap, entropy, and
leadership outputs without engineering raw clinical features directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.dataset_integration.symbolic_feature_adapter import SymbolicFeatureVector


# ── Competition sharpening parameters ────────────────────────────────────────

_GAP_AMPLIFICATION_GAMMA:  float = 2.0   # Power transform for gap amplification
_ENTROPY_SHARPENING_BETA:  float = 1.5   # Entropy contrast sharpening
_LEADER_WEIGHT_CERT:       float = 0.40  # Weight of certainty in leader_confidence
_LEADER_WEIGHT_GAP:        float = 0.35  # Weight of gap
_LEADER_WEIGHT_CONV:       float = 0.25  # Weight of convergence_index
_MAX_ENTROPY_6CLASS:       float = math.log2(6)  # ≈ 2.585 bits


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class CompetitionEnrichedSignals:
    """
    Additional competition signals derived from a SymbolicFeatureVector.

    These are DERIVED from genuine pipeline outputs (certainty, gap,
    entropy, convergence, leadership). None are engineered from raw
    clinical features.

    Attributes
    ----------
    gap_amplified:
        Non-linearly amplified certainty gap: gap ** gamma.
        Widens separation between high-gap and low-gap cases.
    gap_certainty_product:
        certainty * certainty_gap — high only when both are high.
    entropy_contrast:
        1 - (normalised_entropy ** beta) — sharper entropy contrast.
    leader_confidence:
        Composite: cert * gap_weight + gap * gap_weight + conv * conv_weight.
    competition_resolution:
        Binary: 1 if gap >= 0.20 and certainty >= 0.40, else 0.
        Indicates a resolved (non-tied) competition outcome.
    certainty_gap_ratio:
        certainty_gap / max(certainty, 1e-6) — relative gap magnitude.
    convergence_quality:
        convergence_index * leader_confidence — combined stability signal.
    entropy_certainty_divergence:
        |normalised_entropy - (1 - certainty)| — measures alignment
        between entropy and certainty (high divergence = signal inconsistency).
    leadership_persistence:
        1.0 if leadership_changes_count == 0, decaying for each change.
    dampening_resilience:
        1 - int(was_dampened) * 0.5 * (1 - convergence_index).
        Captures how well the leading hypothesis resisted dampening.
    """

    gap_amplified:               float
    gap_certainty_product:       float
    entropy_contrast:            float
    leader_confidence:           float
    competition_resolution:      float
    certainty_gap_ratio:         float
    convergence_quality:         float
    entropy_certainty_divergence: float
    leadership_persistence:      float
    dampening_resilience:        float

    @staticmethod
    def signal_names() -> list[str]:
        return [
            "gap_amplified",
            "gap_certainty_product",
            "entropy_contrast",
            "leader_confidence",
            "competition_resolution",
            "certainty_gap_ratio",
            "convergence_quality",
            "entropy_certainty_divergence",
            "leadership_persistence",
            "dampening_resilience",
        ]

    def to_dict(self) -> dict[str, float]:
        return {n: getattr(self, n) for n in self.signal_names()}


@dataclass
class CompetitionSharpeningReport:
    """
    Analysis report for competition signal enrichment.

    Attributes
    ----------
    n_resolved_cases:
        Cases where competition_resolution == 1.0.
    n_tied_cases:
        Cases where certainty_gap < 0.10 (near-tie).
    mean_gap_amplified:
        Mean amplified gap across all cases.
    mean_leader_confidence:
        Mean leader_confidence score.
    per_disease_resolution_rate:
        Per-disease fraction of resolved competition outcomes.
    signal_variance:
        Variance of each new signal (higher = more informative).
    """

    n_resolved_cases:          int   = 0
    n_tied_cases:              int   = 0
    mean_gap_amplified:        float = 0.0
    mean_leader_confidence:    float = 0.0
    per_disease_resolution_rate: dict[str, float] = field(default_factory=dict)
    signal_variance:           dict[str, float]   = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 72,
            "COMPETITION SHARPENING REPORT",
            "=" * 72,
            f"  Resolved cases (gap>=0.20, cert>=0.40) : {self.n_resolved_cases}",
            f"  Tied cases (gap<0.10)                  : {self.n_tied_cases}",
            f"  Mean amplified gap                     : {self.mean_gap_amplified:.4f}",
            f"  Mean leader confidence                 : {self.mean_leader_confidence:.4f}",
            "-" * 72,
            "  SIGNAL VARIANCE (higher = more informative):",
        ]
        for sig, var in sorted(self.signal_variance.items(), key=lambda x: -x[1]):
            lines.append(f"    {sig:35s} var={var:.5f}")
        lines += [
            "-" * 72,
            "  RESOLUTION RATE BY DISEASE:",
        ]
        for dis, rate in sorted(self.per_disease_resolution_rate.items()):
            lines.append(f"    {dis:35s} {rate:.1%}")
        lines.append("=" * 72)
        return "\n".join(lines)


# ── Sharpener ─────────────────────────────────────────────────────────────────

class CompetitionSharpener:
    """
    Enriches symbolic feature vectors with amplified competition signals.

    Parameters
    ----------
    gap_gamma:
        Power exponent for gap amplification. Default 2.0.
    entropy_beta:
        Contrast exponent for entropy sharpening. Default 1.5.
    """

    def __init__(
        self,
        gap_gamma:    float = _GAP_AMPLIFICATION_GAMMA,
        entropy_beta: float = _ENTROPY_SHARPENING_BETA,
    ) -> None:
        self.gamma = gap_gamma
        self.beta  = entropy_beta

    # ── Public API ────────────────────────────────────────────────────────────

    def enrich(
        self,
        vectors: list[SymbolicFeatureVector],
    ) -> list[CompetitionEnrichedSignals]:
        """
        Derive competition enrichment signals for each vector.

        Parameters
        ----------
        vectors:
            Symbolic feature vectors from the pipeline.

        Returns
        -------
        List of CompetitionEnrichedSignals (one per vector).
        """
        return [self._enrich_one(v) for v in vectors]

    def build_enriched_matrix(
        self,
        X_clinical:       np.ndarray,
        vectors:          list[SymbolicFeatureVector],
        enriched_signals: list[CompetitionEnrichedSignals],
    ) -> tuple[np.ndarray, list[str]]:
        """
        Build combined feature matrix: clinical + symbolic + competition signals.

        Parameters
        ----------
        X_clinical:
            Clinical feature matrix (n × 12).
        vectors:
            Original symbolic vectors.
        enriched_signals:
            Competition enrichment signals.

        Returns
        -------
        (combined_matrix, feature_names)
        """
        # Base symbolic signals
        sym_keys = list(vectors[0].to_dict().keys())
        X_sym    = np.array([
            [float(v.to_dict()[k]) for k in sym_keys]
            for v in vectors
        ])

        # Competition enrichment signals
        comp_keys = CompetitionEnrichedSignals.signal_names()
        X_comp    = np.array([
            [float(e.to_dict()[k]) for k in comp_keys]
            for e in enriched_signals
        ])

        X_combined = np.hstack([X_clinical, X_sym, X_comp])
        all_names  = (
            [f"clin_{i}" for i in range(X_clinical.shape[1])]
            + sym_keys
            + comp_keys
        )
        return X_combined, all_names

    def build_analysis_report(
        self,
        vectors:          list[SymbolicFeatureVector],
        enriched_signals: list[CompetitionEnrichedSignals],
    ) -> CompetitionSharpeningReport:
        """Generate analysis report for the enriched competition signals."""
        if not enriched_signals:
            return CompetitionSharpeningReport()

        n_resolved = sum(1 for e in enriched_signals if e.competition_resolution >= 1.0)
        n_tied     = sum(1 for v in vectors if v.certainty_gap < 0.10)

        gaps        = np.array([e.gap_amplified      for e in enriched_signals])
        confidences = np.array([e.leader_confidence   for e in enriched_signals])

        # Per-disease resolution rate
        per_dis: dict[str, list[float]] = {}
        for v, e in zip(vectors, enriched_signals):
            per_dis.setdefault(v.disease_label, []).append(e.competition_resolution)
        per_dis_rate = {d: float(np.mean(v)) for d, v in per_dis.items()}

        # Signal variance
        sig_names = CompetitionEnrichedSignals.signal_names()
        X_comp    = np.array([[e.to_dict()[n] for n in sig_names] for e in enriched_signals])
        sig_var   = {n: float(np.var(X_comp[:, i])) for i, n in enumerate(sig_names)}

        return CompetitionSharpeningReport(
            n_resolved_cases=n_resolved,
            n_tied_cases=n_tied,
            mean_gap_amplified=float(np.mean(gaps)),
            mean_leader_confidence=float(np.mean(confidences)),
            per_disease_resolution_rate=per_dis_rate,
            signal_variance=sig_var,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _enrich_one(self, v: SymbolicFeatureVector) -> CompetitionEnrichedSignals:
        cert   = float(v.certainty)
        gap    = float(v.certainty_gap)
        conv   = float(v.convergence_index)
        n_ent  = float(v.normalised_entropy)
        lc     = int(v.leadership_changes_count)
        damped = bool(v.was_dampened)

        gap_amp    = gap ** self.gamma
        gap_cert   = cert * gap
        ent_cont   = 1.0 - (n_ent ** self.beta)
        leader_conf = (
            _LEADER_WEIGHT_CERT * cert
            + _LEADER_WEIGHT_GAP * gap
            + _LEADER_WEIGHT_CONV * conv
        )
        comp_res    = float(gap >= 0.20 and cert >= 0.40)
        gap_ratio   = gap / max(cert, 1e-6)
        conv_qual   = conv * leader_conf
        ent_cert_div = abs(n_ent - (1.0 - cert))
        lead_persist = max(0.0, 1.0 - 0.25 * lc)
        damp_resil   = 1.0 - (0.5 * float(damped) * (1.0 - conv))

        return CompetitionEnrichedSignals(
            gap_amplified=float(np.clip(gap_amp, 0.0, 1.0)),
            gap_certainty_product=float(np.clip(gap_cert, 0.0, 1.0)),
            entropy_contrast=float(np.clip(ent_cont, 0.0, 1.0)),
            leader_confidence=float(np.clip(leader_conf, 0.0, 1.0)),
            competition_resolution=comp_res,
            certainty_gap_ratio=float(np.clip(gap_ratio, 0.0, 5.0)),
            convergence_quality=float(np.clip(conv_qual, 0.0, 1.0)),
            entropy_certainty_divergence=float(np.clip(ent_cert_div, 0.0, 1.0)),
            leadership_persistence=float(np.clip(lead_persist, 0.0, 1.0)),
            dampening_resilience=float(np.clip(damp_resil, 0.0, 1.0)),
        )
