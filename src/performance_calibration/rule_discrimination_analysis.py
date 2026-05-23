"""
RuleDiscriminationAnalyzer — symbolic rule effectiveness audit.

Analyses which reasoning rules in the pipeline provide the strongest
discriminative signals between disease classes and which produce
noisy / shared evidence that fails to distinguish diseases.

This module does not directly access the rule engine internals —
instead, it performs a post-hoc analysis by correlating symbolic
feature signals with classification outcomes to infer which
reasoning dimensions are most and least discriminating.

Reasoning signal → rule dimension mapping
-----------------------------------------
  certainty               → overall rule support strength
  certainty_gap           → leading hypothesis margin (discrimination)
  contradiction_load      → conflicting evidence density
  ambiguity_index         → hypothesis entropy (distinguishability)
  convergence_index       → reasoning stability
  oscillation_count       → rule conflict dynamics
  peak_certainty          → maximum achievable certainty (rule ceiling)
  certainty_delta_total   → total certainty movement (rule responsiveness)
  normalised_entropy      → class distribution spread
  certainty_sufficiency   → whether rules produced sufficient support

Diagnostic questions
--------------------
  Q1. Which symbolic signals best separate correct from incorrect predictions?
  Q2. Which signals are disease-specific vs. uniformly activated?
  Q3. Are any signals redundant (high correlation with no added discrimination)?
  Q4. Which signals show the largest discriminability gap between diseases?
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.dataset_integration.symbolic_feature_adapter import SymbolicFeatureVector


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class RuleProfile:
    """
    Discriminability profile for a single symbolic reasoning signal.

    Attributes
    ----------
    signal_name:
        Name of the symbolic feature.
    between_class_variance:
        Variance of per-class mean signal value. Higher = more discriminating.
    within_class_variance:
        Average within-class variance (noise level).
    f_ratio:
        ANOVA F-ratio: between / within variance.
    correct_vs_wrong_delta:
        Mean signal value for correctly classified cases minus incorrectly
        classified cases. Positive = signal aids correct classification.
    disease_specificity:
        0–1 score of how disease-specific the signal is.
        1.0 = signal has completely different value per disease.
        0.0 = signal has same value for all diseases.
    rank:
        Discrimination rank (1 = most discriminating).
    interpretation:
        Brief clinical interpretation of what this signal measures.
    """

    signal_name:              str
    between_class_variance:   float = 0.0
    within_class_variance:    float = 0.0
    f_ratio:                  float = 0.0
    correct_vs_wrong_delta:   float = 0.0
    disease_specificity:      float = 0.0
    rank:                     int   = 0
    interpretation:           str   = ""

    def discrimination_tier(self) -> str:
        if self.f_ratio >= 5.0:
            return "strong"
        if self.f_ratio >= 2.0:
            return "moderate"
        if self.f_ratio >= 0.5:
            return "weak"
        return "negligible"


@dataclass
class SignalCorrelationPair:
    """Correlation between two symbolic signals (for redundancy detection)."""
    signal_a:    str
    signal_b:    str
    correlation: float   # Pearson r


@dataclass
class RuleDiscriminationReport:
    """
    Complete rule discrimination analysis output.

    Attributes
    ----------
    signal_profiles:
        Ranked profiles for all 22+ symbolic signals.
    top_discriminating_signals:
        Signal names with f_ratio ≥ 2.0 (moderate or strong).
    weak_signals:
        Signal names with f_ratio < 0.5.
    redundant_pairs:
        Signal pairs with |correlation| ≥ 0.85.
    per_disease_top_signals:
        For each disease, the 3 signals most active / specific.
    correct_prediction_signal_profile:
        Mean signal values for correctly classified cases.
    wrong_prediction_signal_profile:
        Mean signal values for incorrectly classified cases.
    n_records_analysed:
        Number of test records included in the analysis.
    """

    signal_profiles:                  list[RuleProfile]        = field(default_factory=list)
    top_discriminating_signals:       list[str]                = field(default_factory=list)
    weak_signals:                     list[str]                = field(default_factory=list)
    redundant_pairs:                  list[SignalCorrelationPair] = field(default_factory=list)
    per_disease_top_signals:          dict[str, list[str]]     = field(default_factory=dict)
    correct_prediction_signal_profile: dict[str, float]        = field(default_factory=dict)
    wrong_prediction_signal_profile:   dict[str, float]        = field(default_factory=dict)
    n_records_analysed:               int                      = 0

    def summary(self) -> str:
        lines = [
            "=" * 72,
            "RULE DISCRIMINATION ANALYSIS",
            "=" * 72,
            f"  Records analysed: {self.n_records_analysed}",
            f"  Strong signals (F≥5):    {sum(1 for p in self.signal_profiles if p.f_ratio >= 5.0)}",
            f"  Moderate signals (F≥2):  {sum(1 for p in self.signal_profiles if 2.0 <= p.f_ratio < 5.0)}",
            f"  Weak signals (F<0.5):    {len(self.weak_signals)}",
            f"  Redundant pairs (|r|≥0.85): {len(self.redundant_pairs)}",
            "-" * 72,
            "  TOP DISCRIMINATING SIGNALS:",
        ]
        for p in self.signal_profiles[:8]:
            lines.append(
                f"    #{p.rank:2d} [{p.discrimination_tier():10s}] "
                f"{p.signal_name:35s} F={p.f_ratio:6.2f} "
                f"Δ(correct-wrong)={p.correct_vs_wrong_delta:+.3f}"
            )
        lines += [
            "-" * 72,
            "  WEAK / NON-DISCRIMINATING SIGNALS:",
        ]
        for sig in self.weak_signals[:5]:
            lines.append(f"    · {sig}")
        lines.append("=" * 72)
        return "\n".join(lines)


# ── Analyser ──────────────────────────────────────────────────────────────────

_SIGNAL_INTERPRETATIONS: dict[str, str] = {
    "certainty":               "Overall rule support strength",
    "certainty_gap":           "Leading hypothesis margin (discrimination power)",
    "contradiction_load":      "Conflicting evidence density",
    "ambiguity_index":         "Hypothesis entropy (class overlap)",
    "convergence_index":       "Reasoning stability across pipeline stages",
    "oscillation_count":       "Rule conflict dynamics",
    "peak_certainty":          "Maximum achievable certainty given clinical input",
    "certainty_delta_total":   "Total certainty movement (rule responsiveness)",
    "normalised_entropy":      "Normalised class distribution spread",
    "certainty_sufficiency":   "Sufficiency of rule support for safe triage",
    "entropy_reduction":       "Information gained through reasoning stages",
    "stabilisation_stage":     "Pipeline stage at which certainty stabilised",
    "trajectory_length":       "Number of reasoning stages executed",
    "leadership_changes_count": "Frequency of leading hypothesis changes",
    "contradiction_emerged":   "Whether contradiction arose during reasoning",
    "leadership_changed":      "Whether leading hypothesis changed at any stage",
    "was_dampened":            "Whether certainty was dampened by contradiction",
    "fsm_state_encoded":       "Final finite state machine state (encoded)",
    "recommendation_encoded":  "Final clinical recommendation (encoded)",
    "leading_disease_encoded": "Leading hypothesis disease (encoded)",
    "requires_biopsy":         "Biopsy escalation decision",
    "is_safe_triage":          "Safe biopsy-avoidance decision",
}


class RuleDiscriminationAnalyzer:
    """
    Analyses symbolic signal discriminability across disease classes.

    Parameters
    ----------
    class_labels:
        Ordered canonical disease names.
    """

    def __init__(self, class_labels: list[str]) -> None:
        self.class_labels = class_labels

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse(
        self,
        symbolic_vectors: list[SymbolicFeatureVector],
        y_pred_model_b:   np.ndarray,
        y_true:           np.ndarray,
    ) -> RuleDiscriminationReport:
        """
        Run signal discrimination analysis.

        Parameters
        ----------
        symbolic_vectors:
            Test-set symbolic feature vectors.
        y_pred_model_b:
            Model B predicted labels (0-based).
        y_true:
            True labels (0-based).
        """
        if not symbolic_vectors:
            return RuleDiscriminationReport()

        # Build signal matrix
        signal_matrix, signal_names = self._build_signal_matrix(symbolic_vectors)
        disease_labels = np.array([v.disease_label for v in symbolic_vectors])
        correct_mask   = y_pred_model_b == y_true

        # Per-signal analysis
        profiles = self._compute_profiles(
            signal_matrix, signal_names, disease_labels, correct_mask, y_true
        )

        # Redundancy detection
        redundant = self._find_redundant_pairs(signal_matrix, signal_names)

        # Per-disease top signals
        per_dis = self._per_disease_top_signals(
            signal_matrix, signal_names, disease_labels
        )

        # Correct vs. wrong prediction profiles
        corr_prof = {
            n: float(np.mean(signal_matrix[correct_mask, i]))
            for i, n in enumerate(signal_names)
        }
        wrong_prof = {
            n: float(np.mean(signal_matrix[~correct_mask, i]))
            if np.any(~correct_mask) else 0.0
            for i, n in enumerate(signal_names)
        }

        top_discriminating = [
            p.signal_name for p in profiles if p.f_ratio >= 2.0
        ]
        weak_signals = [
            p.signal_name for p in profiles if p.f_ratio < 0.5
        ]

        return RuleDiscriminationReport(
            signal_profiles=profiles,
            top_discriminating_signals=top_discriminating,
            weak_signals=weak_signals,
            redundant_pairs=redundant,
            per_disease_top_signals=per_dis,
            correct_prediction_signal_profile=corr_prof,
            wrong_prediction_signal_profile=wrong_prof,
            n_records_analysed=len(symbolic_vectors),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_signal_matrix(
        self,
        vectors: list[SymbolicFeatureVector],
    ) -> tuple[np.ndarray, list[str]]:
        """Convert symbolic vectors to float matrix."""
        sample = vectors[0].to_dict()
        names  = list(sample.keys())
        matrix = np.array([
            [float(vectors[i].to_dict()[n]) for n in names]
            for i in range(len(vectors))
        ])
        return matrix, names

    def _compute_profiles(
        self,
        matrix:        np.ndarray,
        signal_names:  list[str],
        disease_labels: np.ndarray,
        correct_mask:  np.ndarray,
        y_true:        np.ndarray,
    ) -> list[RuleProfile]:
        """Compute discrimination profile for each signal."""
        profiles: list[RuleProfile] = []
        classes = np.unique(y_true)

        for fi, name in enumerate(signal_names):
            col    = matrix[:, fi]
            groups = [col[y_true == c] for c in classes if np.any(y_true == c)]

            grand_mean  = float(np.mean(col))
            n_classes   = len(groups)

            # ANOVA between-class variance
            between_var = sum(
                len(g) * (float(np.mean(g)) - grand_mean) ** 2
                for g in groups if len(g) > 0
            ) / max(n_classes - 1, 1)

            # Within-class variance
            within_var = sum(
                float(np.sum((g - np.mean(g)) ** 2))
                for g in groups if len(g) > 0
            ) / max(len(col) - n_classes, 1)

            f_ratio = float(between_var / max(within_var, 1e-12))

            # Correct vs wrong delta
            mean_corr  = float(np.mean(col[correct_mask])) if np.any(correct_mask) else 0.0
            mean_wrong = float(np.mean(col[~correct_mask])) if np.any(~correct_mask) else 0.0
            delta      = mean_corr - mean_wrong

            # Disease specificity: std of per-class means / grand std
            per_class_means = np.array([
                float(np.mean(col[y_true == c])) if np.any(y_true == c) else grand_mean
                for c in classes
            ])
            specificity = float(
                np.std(per_class_means) / max(np.std(col), 1e-9)
            )
            specificity = min(1.0, specificity)

            profiles.append(RuleProfile(
                signal_name=name,
                between_class_variance=float(between_var),
                within_class_variance=float(within_var),
                f_ratio=f_ratio,
                correct_vs_wrong_delta=delta,
                disease_specificity=specificity,
                rank=0,
                interpretation=_SIGNAL_INTERPRETATIONS.get(name, ""),
            ))

        # Rank by F-ratio
        profiles.sort(key=lambda p: p.f_ratio, reverse=True)
        for rank, p in enumerate(profiles, 1):
            p.rank = rank

        return profiles

    def _find_redundant_pairs(
        self,
        matrix:       np.ndarray,
        signal_names: list[str],
    ) -> list[SignalCorrelationPair]:
        """Find signal pairs with |Pearson r| ≥ 0.85."""
        n = len(signal_names)
        redundant: list[SignalCorrelationPair] = []
        for i in range(n):
            for j in range(i + 1, n):
                col_i = matrix[:, i]
                col_j = matrix[:, j]
                # Only compute if both have variance
                if np.std(col_i) < 1e-9 or np.std(col_j) < 1e-9:
                    continue
                r = float(np.corrcoef(col_i, col_j)[0, 1])
                if abs(r) >= 0.85:
                    redundant.append(SignalCorrelationPair(
                        signal_a=signal_names[i],
                        signal_b=signal_names[j],
                        correlation=r,
                    ))
        return sorted(redundant, key=lambda x: abs(x.correlation), reverse=True)

    def _per_disease_top_signals(
        self,
        matrix:        np.ndarray,
        signal_names:  list[str],
        disease_labels: np.ndarray,
    ) -> dict[str, list[str]]:
        """For each disease, identify the 3 signals with highest specificity."""
        result: dict[str, list[str]] = {}
        grand_means = np.mean(matrix, axis=0)

        for disease in np.unique(disease_labels):
            mask = disease_labels == disease
            if not np.any(mask):
                continue
            dis_means = np.mean(matrix[mask], axis=0)
            deviations = np.abs(dis_means - grand_means)
            top_indices = np.argsort(deviations)[::-1][:3]
            result[str(disease)] = [signal_names[i] for i in top_indices]

        return result
