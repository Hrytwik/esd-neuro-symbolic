/**
 * types/clinical-input.ts
 * =========================
 * Clinical feature input definitions for the diagnostic assessment form.
 * Each feature maps to a field in the UCI Dermatology feature vector.
 */

export type FeatureInputType = "slider" | "toggle";
export type FeatureCategory  = "primary" | "secondary" | "distribution" | "history" | "histological";

export interface ClinicalFeatureDef {
  key:         string;
  label:       string;
  description: string;
  inputType:   FeatureInputType;
  category:    FeatureCategory;
}

/** Severity score: 0 = absent, 1 = mild, 2 = moderate, 3 = severe. */
export type FeatureVector = Record<string, number>;

export const FEATURE_DEFINITIONS: ClinicalFeatureDef[] = [
  // Primary findings
  {
    key:         "erythema",
    label:       "Erythema",
    description: "Degree of skin redness and vascular dilation",
    inputType:   "slider",
    category:    "primary",
  },
  {
    key:         "scaling",
    label:       "Scaling",
    description: "Presence and severity of epidermal scaling",
    inputType:   "slider",
    category:    "primary",
  },
  {
    key:         "definite_borders",
    label:       "Definite Borders",
    description: "Sharpness and definition of lesion margins",
    inputType:   "slider",
    category:    "primary",
  },
  {
    key:         "itching",
    label:       "Pruritus",
    description: "Reported severity of itch",
    inputType:   "slider",
    category:    "primary",
  },
  // Secondary findings
  {
    key:         "koebner_phenomenon",
    label:       "Koebner Phenomenon",
    description: "New lesions appearing at sites of skin trauma or pressure",
    inputType:   "toggle",
    category:    "secondary",
  },
  {
    key:         "polygonal_papules",
    label:       "Polygonal Papules",
    description: "Flat-topped papules with angular, polygonal morphology",
    inputType:   "toggle",
    category:    "secondary",
  },
  {
    key:         "follicular_papules",
    label:       "Follicular Papules",
    description: "Papules centred on hair follicles",
    inputType:   "toggle",
    category:    "secondary",
  },
  {
    key:         "oral_involvement",
    label:       "Oral Involvement",
    description: "Mucosal lesions present inside the mouth or on the lips",
    inputType:   "toggle",
    category:    "secondary",
  },
  // Distribution
  {
    key:         "knee_elbow_involv",
    label:       "Knee / Elbow Involvement",
    description: "Lesions on extensor surfaces (knees and elbows)",
    inputType:   "toggle",
    category:    "distribution",
  },
  {
    key:         "scalp_involvement",
    label:       "Scalp Involvement",
    description: "Extension of lesions to the scalp",
    inputType:   "slider",
    category:    "distribution",
  },
  // History
  {
    key:         "family_history",
    label:       "Family History",
    description: "First-degree relative with a similar inflammatory skin condition",
    inputType:   "toggle",
    category:    "history",
  },
  // Histological (optional)
  {
    key:         "melanin_incontinence",
    label:       "Melanin Incontinence",
    description: "Histological finding — complete only if a prior biopsy is available",
    inputType:   "toggle",
    category:    "histological",
  },
];

export const FEATURE_CATEGORY_LABELS: Record<FeatureCategory, string> = {
  primary:      "Primary Findings",
  secondary:    "Secondary Findings",
  distribution: "Distribution Pattern",
  history:      "Clinical History",
  histological: "Histological Findings (optional)",
};

/** Default feature vector — all values absent (0). */
export function defaultFeatureVector(): FeatureVector {
  return Object.fromEntries(FEATURE_DEFINITIONS.map(f => [f.key, 0]));
}
