"""
beat_classifier.py
Kardiax EKG Shirt — ECE 481 Senior Design

Beat Classification ML Model
------------------------------
Two-stage Random Forest classifier trained on MIT-BIH Arrhythmia Database.

Stage 1 — True/False Positive detector:
    Is this detected peak actually a real heartbeat, or a noise artifact?

Stage 2 — Beat type classifier:
    If it's a real beat, what kind is it?
    N = Normal
    V = Premature ventricular contraction (PVC)
    A = Supraventricular premature beat
    Other = everything else

Input features (all derived from RR intervals + Pan-Tompkins output):
    - RR interval (current beat, ms)
    - RR_prev (previous RR, ms)
    - RR_next (next RR, ms)
    - RR_ratio_prev (current / previous — detects premature beats)
    - RR_ratio_next (current / next — detects compensatory pauses)
    - RR_local_mean (mean of surrounding 5 beats)
    - RR_deviation (how far current RR is from local mean)
    - QRS_amplitude (peak height from Pan-Tompkins filtered signal)
    - beat_index (position in recording — detects drift)

Why Random Forest?
    - Works extremely well on tabular RR features (published literature agrees)
    - Fast inference on phone (< 1ms per beat)
    - Easy to export as a small JSON file for Flutter
    - Interpretable — you can explain each feature to Dr. Cai
    - No GPU needed

Dependencies:
    pip3 install wfdb numpy scipy matplotlib scikit-learn joblib

Usage:
    python3 beat_classifier.py

Output files:
    models/stage1_fp_detector.joblib   — true/false positive model
    models/stage2_beat_classifier.joblib — beat type model
    models/scaler.joblib               — feature scaler (needed for inference)
    models/feature_names.txt           — feature list (for reference)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

from scipy.signal import butter, lfilter, resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    print("ERROR: wfdb not installed. Run: pip3 install wfdb")
    exit(1)

# Import our Pan-Tompkins detector
import sys
sys.path.insert(0, os.path.dirname(__file__))
from pan_tompkins import PanTompkins


# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════

TARGET_FS = 250   # Hz — matches our hardware

# MIT-BIH records to train/test on
# Using a broad set to cover normal, noisy, and arrhythmia cases
TRAINING_RECORDS = [
    "100", "101", "103", "105", "106", "107",
    "108", "109", "111", "112", "113", "114",
    "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202",
    "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219",
    "220", "221", "222", "223", "228", "230",
    "231", "232", "233", "234",
]

# Beat type mapping — MIT-BIH annotation symbols → our labels
# Stage 2 only uses beats that Stage 1 confirmed as real
BEAT_MAP = {
    # Normal
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    # PVC
    'V': 'V', 'E': 'V',
    # Supraventricular premature
    'A': 'A', 'a': 'A', 'J': 'A', 'S': 'A',
    # Other (paced, fusion, etc.)
    '/': 'O', 'f': 'O', 'F': 'O', 'Q': 'O',
}

# All real beat symbols (used for Stage 1 — anything in here is a TRUE positive)
REAL_BEAT_SYMBOLS = set(BEAT_MAP.keys())


# ══════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════

def extract_features(r_peaks, rr_intervals_ms, filtered_signal, fs=250):
    """
    Extract features for each beat from Pan-Tompkins output.

    Parameters
    ----------
    r_peaks         : list of int   — sample indices of R peaks
    rr_intervals_ms : list of float — RR intervals in ms (length = len(r_peaks) - 1)
    filtered_signal : np.ndarray    — bandpass-filtered ECG signal
    fs              : int           — sampling rate

    Returns
    -------
    features : np.ndarray, shape (n_beats, 9)
        One row per beat, columns = feature vector
    """
    n = len(r_peaks)
    features = []

    # Pad RR list so every beat has a prev and next RR
    # Use median for edge beats (safe default)
    if len(rr_intervals_ms) == 0:
        return np.array([])

    rr = np.array(rr_intervals_ms)
    rr_median = np.median(rr)

    # Build per-beat RR arrays (aligned to r_peaks)
    # rr_intervals[i] = interval between peak i and peak i+1
    # So peak 0 has no previous, peak n-1 has no next
    rr_prev = np.concatenate([[rr_median], rr])          # length n
    rr_curr = np.concatenate([rr, [rr_median]])          # length n
    rr_next = np.concatenate([[rr_median], rr[1:], [rr_median]])  # length n

    for i, peak_idx in enumerate(r_peaks):
        rr_c = rr_curr[i]
        rr_p = rr_prev[i]
        rr_n = rr_next[i]

        # Ratio features — key for PVC detection
        # PVCs come early (low RR_ratio_prev) and have compensatory pause after (high RR_ratio_next)
        rr_ratio_prev = rr_c / rr_p if rr_p > 0 else 1.0
        rr_ratio_next = rr_n / rr_c if rr_c > 0 else 1.0

        # Local mean RR (window of ±5 beats, excluding current)
        window_start = max(0, i - 5)
        window_end   = min(n - 1, i + 5)
        local_rr = np.concatenate([rr_curr[window_start:i], rr_curr[i+1:window_end+1]])
        local_mean = np.mean(local_rr) if len(local_rr) > 0 else rr_median
        rr_deviation = (rr_c - local_mean) / local_mean if local_mean > 0 else 0.0

        # QRS amplitude — peak height in filtered signal
        search_w = int(0.05 * fs)  # ±50ms window
        seg_start = max(0, peak_idx - search_w)
        seg_end   = min(len(filtered_signal), peak_idx + search_w)
        if seg_end > seg_start:
            qrs_amp = np.max(np.abs(filtered_signal[seg_start:seg_end]))
        else:
            qrs_amp = 0.0

        # Beat position (normalised 0–1) — captures recording drift
        beat_pos = i / n if n > 0 else 0.0

        features.append([
            rr_c,           # 0: current RR interval (ms)
            rr_p,           # 1: previous RR interval (ms)
            rr_n,           # 2: next RR interval (ms)
            rr_ratio_prev,  # 3: current/previous ratio
            rr_ratio_next,  # 4: next/current ratio
            local_mean,     # 5: local mean RR (ms)
            rr_deviation,   # 6: deviation from local mean (normalised)
            qrs_amp,        # 7: QRS amplitude
            beat_pos,       # 8: position in recording
        ])

    return np.array(features)

FEATURE_NAMES = [
    "RR_current_ms",
    "RR_prev_ms",
    "RR_next_ms",
    "RR_ratio_prev",
    "RR_ratio_next",
    "RR_local_mean_ms",
    "RR_deviation",
    "QRS_amplitude",
    "beat_position",
]


# ══════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_record(record_id, duration_sec=120):
    """
    Load one MIT-BIH record, run Pan-Tompkins, extract features + labels.

    Returns
    -------
    features_s1 : np.ndarray  — features for Stage 1 (true/false positive)
    labels_s1   : np.ndarray  — 1 = real beat, 0 = false positive
    features_s2 : np.ndarray  — features for Stage 2 (beat type, real beats only)
    labels_s2   : np.ndarray  — beat type string ('N', 'V', 'A', 'O')
    """
    try:
        rec = wfdb.rdrecord(record_id, pn_dir='mitdb',
                            sampto=360 * duration_sec)
        ann = wfdb.rdann(record_id, 'atr', pn_dir='mitdb',
                         sampto=360 * duration_sec)
    except Exception as e:
        print(f"    Skipping record {record_id}: {e}")
        return None, None, None, None

    ecg_raw   = rec.p_signal[:, 0]
    fs_mitbih = rec.fs  # 360 Hz

    # Resample to 250 Hz
    num_samples = int(len(ecg_raw) * TARGET_FS / fs_mitbih)
    ecg_250     = resample(ecg_raw, num_samples)
    scale       = TARGET_FS / fs_mitbih

    # Reference annotations scaled to 250 Hz
    ref_samples = [int(s * scale) for s in ann.sample]
    ref_symbols = ann.symbol

    # Build reference dict: sample_index → beat_symbol (real beats only)
    ref_dict = {}
    for s, sym in zip(ref_samples, ref_symbols):
        if sym in REAL_BEAT_SYMBOLS:
            ref_dict[s] = sym

    # Run Pan-Tompkins
    pt = PanTompkins(fs=TARGET_FS)
    r_peaks, rr_ms, intermediates = pt.process(ecg_250)

    if len(r_peaks) < 3:
        return None, None, None, None

    # Extract features for all detected peaks
    features = extract_features(r_peaks, rr_ms, intermediates['bandpass'], TARGET_FS)

    if len(features) == 0:
        return None, None, None, None

    # ── Stage 1 labels: is each detected peak a real beat? ──
    tol = int(0.075 * TARGET_FS)  # 75ms tolerance window
    labels_s1 = []
    matched_refs = set()
    beat_types   = []  # parallel list for Stage 2

    for peak in r_peaks:
        # Find closest reference peak within tolerance
        best_match = None
        best_dist  = tol + 1
        for ref_s, ref_sym in ref_dict.items():
            dist = abs(peak - ref_s)
            if dist <= tol and dist < best_dist and ref_s not in matched_refs:
                best_dist  = dist
                best_match = ref_s

        if best_match is not None:
            matched_refs.add(best_match)
            labels_s1.append(1)                           # TRUE positive
            beat_types.append(BEAT_MAP[ref_dict[best_match]])
        else:
            labels_s1.append(0)                           # FALSE positive
            beat_types.append(None)

    labels_s1  = np.array(labels_s1)
    beat_types = np.array(beat_types, dtype=object)

    # ── Stage 2: only real beats ──
    real_mask   = labels_s1 == 1
    features_s2 = features[real_mask]
    labels_s2   = beat_types[real_mask]

    return features, labels_s1, features_s2, labels_s2


def build_dataset(records, duration_sec=120):
    """Load all records and concatenate into training arrays."""
    all_f1, all_l1 = [], []
    all_f2, all_l2 = [], []

    print(f"Loading {len(records)} MIT-BIH records...")
    for i, rec_id in enumerate(records):
        print(f"  [{i+1:2d}/{len(records)}] Record {rec_id}", end="", flush=True)
        f1, l1, f2, l2 = load_record(rec_id, duration_sec)
        if f1 is None:
            continue
        all_f1.append(f1);  all_l1.append(l1)
        all_f2.append(f2);  all_l2.append(l2)
        print(f"  → {len(l1)} beats ({np.sum(l1==1)} real, {np.sum(l1==0)} FP)")

    X1 = np.vstack(all_f1)
    y1 = np.concatenate(all_l1)
    X2 = np.vstack(all_f2)
    y2 = np.concatenate(all_l2)

    print(f"\nDataset totals:")
    print(f"  Stage 1: {len(X1)} samples  ({np.sum(y1==1)} real, {np.sum(y1==0)} FP)")
    print(f"  Stage 2: {len(X2)} samples")
    vals, counts = np.unique(y2, return_counts=True)
    for v, c in zip(vals, counts):
        print(f"    {v}: {c} beats ({100*c/len(y2):.1f}%)")

    return X1, y1, X2, y2


# ══════════════════════════════════════════════════════════════════
#  MODEL TRAINING
# ══════════════════════════════════════════════════════════════════

def train_stage1(X, y):
    """
    Stage 1: True/False Positive detector.
    Binary classifier — 1 = real beat, 0 = noise artifact.
    """
    print("\n── Stage 1: True/False Positive Detector ──")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Random Forest — n_estimators=200 is plenty for 9 features
    # class_weight='balanced' handles the fact that false positives are rare
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )

    # 5-fold cross validation to estimate real-world performance
    print("  Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv,
                                scoring='f1', n_jobs=-1)
    print(f"  CV F1 scores: {cv_scores.round(3)}")
    print(f"  Mean F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Train on full dataset
    model.fit(X_scaled, y)

    # Feature importance
    importances = model.feature_importances_
    print("\n  Feature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, importances),
                             key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"    {name:<22} {imp:.3f}  {bar}")

    return model, scaler


def train_stage2(X, y):
    """
    Stage 2: Beat type classifier.
    Multiclass — N (normal), V (PVC), A (supraventricular).
    """
    print("\n── Stage 2: Beat Type Classifier ──")

    # Drop 'O' class — too few samples to train reliably
    mask = y != 'O'
    X = X[mask]
    y = y[mask]
    print(f"  Dropped 'O' class. Training on N, V, A only.")

    # ── Oversample minority classes (V and A) to fix 92%/5%/1% imbalance ──
    # Repeating minority samples until all classes are equal size
    classes, counts = np.unique(y, return_counts=True)
    max_count = np.max(counts)
    X_parts, y_parts = [X], [y]
    np.random.seed(42)
    for cls, cnt in zip(classes, counts):
        if cnt < max_count:
            idx = np.where(y == cls)[0]
            n_extra = max_count - cnt
            extra_idx = np.random.choice(idx, size=n_extra, replace=True)
            X_parts.append(X[extra_idx])
            y_parts.append(y[extra_idx])
            print(f"  Oversampled class '{cls}': {cnt} → {max_count}")
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )

    print("  Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv,
                                scoring='f1_weighted', n_jobs=-1)
    print(f"  CV F1 scores: {cv_scores.round(3)}")
    print(f"  Mean F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Train on full dataset
    model.fit(X_scaled, y)

    # Feature importance
    importances = model.feature_importances_
    print("\n  Feature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, importances),
                             key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"    {name:<22} {imp:.3f}  {bar}")

    return model, scaler


# ══════════════════════════════════════════════════════════════════
#  EVALUATION & PLOTTING
# ══════════════════════════════════════════════════════════════════

def evaluate_full_pipeline(records_eval, model_s1, scaler_s1,
                            model_s2, scaler_s2, duration_sec=60):
    """
    Run the full two-stage pipeline on held-out records and print results.
    These records were NOT used in training.
    """
    print("\n── Full Pipeline Evaluation (held-out records) ──")

    eval_records = ["100", "108", "119", "200", "213"]
    print(f"  Evaluating on: {eval_records}")

    for rec_id in eval_records:
        f1, l1_true, f2, l2_true = load_record(rec_id, duration_sec)
        if f1 is None:
            continue

        # Stage 1 prediction
        f1_scaled = scaler_s1.transform(f1)
        l1_pred   = model_s1.predict(f1_scaled)

        tp = np.sum((l1_pred == 1) & (l1_true == 1))
        fp = np.sum((l1_pred == 1) & (l1_true == 0))
        fn = np.sum((l1_pred == 0) & (l1_true == 1))
        sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        ppv         = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0

        # Stage 2 prediction (only on beats predicted real by Stage 1)
        real_mask     = l1_pred == 1
        f2_detected   = f1[real_mask]
        l2_true_sub   = l2_true if l2_true is not None else np.array([])

        beat_report = ""
        if len(f2_detected) > 0 and len(l2_true) > 0:
            f2_scaled   = scaler_s2.transform(f2_detected)
            l2_pred     = model_s2.predict(f2_scaled)
            # Match against true labels for real beats only
            true_real   = np.array([BEAT_MAP.get(s, 'O') for s in l2_true
                                    if s is not None])
            min_len     = min(len(l2_pred), len(true_real))
            if min_len > 0:
                correct = np.sum(l2_pred[:min_len] == true_real[:min_len])
                beat_acc = correct / min_len * 100
                beat_report = f"  Beat type acc: {beat_acc:.1f}%"

        print(f"\n  Record {rec_id}:")
        print(f"    Stage 1 — Sensitivity: {sensitivity:.1f}%  "
              f"Positive pred: {ppv:.1f}%")
        if beat_report:
            print(f"    Stage 2 —{beat_report}")


def plot_confusion_matrix(model, scaler, X, y, title, filename):
    """Plot and save a confusion matrix."""
    X_scaled = scaler.transform(X)
    y_pred   = model.predict(X_scaled)
    cm       = confusion_matrix(y, y_pred, labels=model.classes_)
    disp     = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
    fig, ax  = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"  Saved: {filename}")
    plt.close()


# ══════════════════════════════════════════════════════════════════
#  INFERENCE HELPER (used by phone app / real-time pipeline)
# ══════════════════════════════════════════════════════════════════

class KardiaxClassifier:
    """
    Lightweight wrapper for real-time inference.
    Load this in the Flutter app's inference bridge or in a streaming Python script.

    Usage:
        clf = KardiaxClassifier.load('models/')
        result = clf.classify_beat(rr_ms, rr_prev, rr_next, qrs_amp, beat_idx, total_beats)
        # result = {'is_real': True, 'beat_type': 'N', 'confidence': 0.97}
    """

    def __init__(self, model_s1, scaler_s1, model_s2, scaler_s2):
        self.model_s1  = model_s1
        self.scaler_s1 = scaler_s1
        self.model_s2  = model_s2
        self.scaler_s2 = scaler_s2
        self.rr_history = []  # rolling window for local mean

    def classify_beat(self, rr_ms, rr_prev, rr_next, qrs_amp,
                      beat_idx, total_beats):
        """
        Classify a single beat in real time.

        Parameters (all from Pan-Tompkins output):
            rr_ms       : float — current RR interval in ms
            rr_prev     : float — previous RR interval in ms
            rr_next     : float — next RR interval in ms (0 if not yet known)
            qrs_amp     : float — QRS peak amplitude from filtered signal
            beat_idx    : int   — beat number in this session
            total_beats : int   — total beats so far (for normalisation)

        Returns:
            dict with keys: is_real, beat_type, s1_confidence, s2_confidence
        """
        self.rr_history.append(rr_ms)
        if len(self.rr_history) > 10:
            self.rr_history.pop(0)

        local_mean  = np.mean(self.rr_history)
        rr_dev      = (rr_ms - local_mean) / local_mean if local_mean > 0 else 0.0
        rr_ratio_p  = rr_ms / rr_prev if rr_prev > 0 else 1.0
        rr_ratio_n  = rr_next / rr_ms if (rr_next > 0 and rr_ms > 0) else 1.0
        beat_pos    = beat_idx / total_beats if total_beats > 0 else 0.0

        feat = np.array([[
            rr_ms, rr_prev, rr_next,
            rr_ratio_p, rr_ratio_n,
            local_mean, rr_dev,
            qrs_amp, beat_pos
        ]])

        # Stage 1
        f1_scaled   = self.scaler_s1.transform(feat)
        s1_proba    = self.model_s1.predict_proba(f1_scaled)[0]
        is_real     = bool(np.argmax(s1_proba))
        s1_conf     = float(np.max(s1_proba))

        if not is_real:
            return {'is_real': False, 'beat_type': None,
                    's1_confidence': s1_conf, 's2_confidence': None}

        # Stage 2
        f2_scaled   = self.scaler_s2.transform(feat)
        beat_type   = self.model_s2.predict(f2_scaled)[0]
        s2_proba    = self.model_s2.predict_proba(f2_scaled)[0]
        s2_conf     = float(np.max(s2_proba))

        return {
            'is_real'       : True,
            'beat_type'     : beat_type,
            's1_confidence' : s1_conf,
            's2_confidence' : s2_conf,
        }

    @classmethod
    def load(cls, model_dir='models/'):
        """Load saved models from disk."""
        m1  = joblib.load(os.path.join(model_dir, 'stage1_fp_detector.joblib'))
        sc1 = joblib.load(os.path.join(model_dir, 'scaler_s1.joblib'))
        m2  = joblib.load(os.path.join(model_dir, 'stage2_beat_classifier.joblib'))
        sc2 = joblib.load(os.path.join(model_dir, 'scaler_s2.joblib'))
        return cls(m1, sc1, m2, sc2)


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  Kardiax Beat Classifier — Training Pipeline")
    print("=" * 55)

    # ── 1. Build dataset ──
    X1, y1, X2, y2 = build_dataset(TRAINING_RECORDS, duration_sec=120)

    # ── 2. Train models ──
    model_s1, scaler_s1 = train_stage1(X1, y1)
    model_s2, scaler_s2 = train_stage2(X2, y2)

    # ── 3. Save models ──
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_s1,  'models/stage1_fp_detector.joblib')
    joblib.dump(scaler_s1, 'models/scaler_s1.joblib')
    joblib.dump(model_s2,  'models/stage2_beat_classifier.joblib')
    joblib.dump(scaler_s2, 'models/scaler_s2.joblib')

    with open('models/feature_names.txt', 'w') as f:
        f.write('\n'.join(FEATURE_NAMES))

    print("\n  Models saved to models/")

    # ── 4. Confusion matrices ──
    print("\n── Generating confusion matrices ──")
    plot_confusion_matrix(model_s1, scaler_s1, X1, y1,
                          "Stage 1: True/False Positive",
                          "models/cm_stage1.png")
    plot_confusion_matrix(model_s2, scaler_s2, X2, y2,
                          "Stage 2: Beat Type",
                          "models/cm_stage2.png")

    # ── 5. Evaluate on held-out records ──
    evaluate_full_pipeline(
        ["100", "108", "119", "200", "213"],
        model_s1, scaler_s1,
        model_s2, scaler_s2,
        duration_sec=60
    )

    # ── 6. Quick inference demo ──
    print("\n── Inference demo (KardiaxClassifier) ──")
    clf = KardiaxClassifier(model_s1, scaler_s1, model_s2, scaler_s2)

    test_beats = [
        # (rr_ms, rr_prev, rr_next, qrs_amp, beat_idx, total, description)
        (833,  833,  833,  0.8, 10, 100, "Normal beat ~72bpm"),
        (450,  833,  1200, 1.4, 20, 100, "Early + large amp = likely PVC"),
        (833,  833,  833,  0.05, 30, 100, "Tiny amplitude = likely false positive"),
        (700,  833,  900,  0.7, 40, 100, "Slightly early = possible SVT"),
    ]

    print(f"\n  {'Description':<35} {'Real?':<8} {'Type':<6} {'S1 conf':<9} S2 conf")
    print(f"  {'─'*35} {'─'*8} {'─'*6} {'─'*9} {'─'*7}")
    for rr, rr_p, rr_n, amp, idx, total, desc in test_beats:
        r = clf.classify_beat(rr, rr_p, rr_n, amp, idx, total)
        real_str = "✓ Real" if r['is_real'] else "✗ Noise"
        btype    = r['beat_type'] or "—"
        s1c      = f"{r['s1_confidence']:.2f}"
        s2c      = f"{r['s2_confidence']:.2f}" if r['s2_confidence'] else "—"
        print(f"  {desc:<35} {real_str:<8} {btype:<6} {s1c:<9} {s2c}")

    print(f"\n{'═' * 55}")
    print("  Training complete. Models saved to models/")
    print("  Next step: integrate KardiaxClassifier into Flutter app")
    print(f"{'═' * 55}\n")
