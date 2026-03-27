from __future__ import annotations

_STOPWORDS = {
    "about", "according", "after", "all", "and", "are", "based", "been", "for", "from",
    "guidance", "has", "have", "into", "most", "not", "that", "the", "their", "this", "treatment",
    "what", "with", "would", "patient", "variables", "management", "follow", "relevant", "document",
    "grounded", "available", "data", "should", "could", "regarding", "focus", "clinical",
}

_CARDIAC_KEYS = {
    "bnp", "nt_pro_bnp", "ef", "lvef", "nyha", "edema", "rales", "orthopnea", "jugularvein_01",
    "hepatomegaly", "qrsduur", "heart_rate", "heartrate", "bpsyst", "bpdiast", "sbp", "dbp",
    "afib_bl", "bundeltakblok", "pm_rhythm_ecg", "pacemaker_bl", "valvhd", "ischaemic", "hist_mi",
    "cabg", "angina01", "dosebb_prev", "rasdose_prev", "dosespiro_prev", "loop_dose_prev",
    "sglt2dose_prev", "arnidose_prev", "potassium", "sodium", "crea", "creatinine", "urea", "cysc",
    "cystatin_c", "hstnt", "hscrp", "hs_crp", "crp", "il6", "il_6",
}
_GLYCEMIC_KEYS = {"glucose", "hba1c", "diabetes"}
_RENAL_KEYS = {"crea", "creatinine", "urea", "cysc", "cystatin_c", "kidney_disease", "potassium", "sodium"}
_HEART_FAILURE_KEYWORDS = {
    "heart failure", "hfr ef", "hfref", "hfmr ef", "hfpef", "congestion", "decongestion",
    "gdmt", "euvolaemia", "euvolemia", "hyperkalaemia", "hyperkalemia", "hyponatraemia", "hyponatremia",
    "raas", "ace-i", "arb", "arni", "mra", "sglt2", "diuretic", "ivabradine", "nt-probnp", "bnp",
}

_QUESTION_LITERAL_TERMS = {"table", "figure", "supplementary", "appendix", "section", "described", "listed"}
_GENERIC_NON_ANSWER_PHRASES = {
    "clinically relevant pattern",
    "interpreted as a whole rather than marker by marker",
    "prioritize the most abnormal findings first",
    "use the retrieved guidance to prioritize",
}

_HF_FOCUS_CLUSTER_ORDER = (
    "HF severity and congestion",
    "Cardio-renal and electrolyte safety",
    "Rhythm and conduction",
    "Anemia and iron status",
    "Inflammation and injury",
    "Glycemic and cardiometabolic risk",
)
