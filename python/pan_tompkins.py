"""
pan_tompkins.py
Kardiax EKG Shirt - ECE 481 Senior Design
 
Pan-Tompkins QRS Detection Algorithm
-------------------------------------
Use this file on your PC to:
  1. Test the algorithm against MIT-BIH database records
  2. Validate RR interval output before deploying to Arduino
  3. Tune filter/threshold parameters
 
Dependencies:
    pip install wfdb numpy scipy matplotlib
 
MIT-BIH data download (automatic via wfdb):
    The wfdb library will pull records directly from PhysioNet.
    No manual download needed.
 
Usage:
    python pan_tompkins.py
"""
 
import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi
import matplotlib.pyplot as plt
 
# ─────────────────────────────────────────────
# OPTIONAL: wfdb for MIT-BIH testing
# Comment out if you don't have wfdb installed yet
# ─────────────────────────────────────────────
try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    print("wfdb not installed. Run: pip install wfdb")
    print("Running with synthetic test signal instead.\n")
 
 
# ══════════════════════════════════════════════════════════════════
#  CORE PAN-TOMPKINS CLASS
#  This same logic gets ported to Arduino (see pan_tompkins.ino)
# ══════════════════════════════════════════════════════════════════
 
class PanTompkins:
    """
    Implements the Pan-Tompkins QRS detection algorithm.
 
    Pipeline:
        Raw ECG → Bandpass Filter → Derivative → Squaring
                → Moving Window Integration → Adaptive Thresholding → R peaks
 
    Parameters
    ----------
    fs : int
        Sampling frequency in Hz. Use 250 for this project (per ER1).
    """
 
    def __init__(self, fs=250):
        self.fs = fs
 
        # ── Bandpass filter: 5–15 Hz (preserves QRS, kills baseline wander & EMG) ──
        # IEC 60601-2-47 recommends ≥0.05 Hz high-pass; 5 Hz is more aggressive
        # but greatly reduces motion artifact for wearables.
        low  = 5.0  / (fs / 2.0)
        high = 15.0 / (fs / 2.0)
        self.b_bp, self.a_bp = butter(2, [low, high], btype='band')
 
        # Moving window size: 150 ms (standard Pan-Tompkins)
        self.mwi_window = int(0.150 * fs)   # 37 samples @ 250 Hz
 
        # Refractory period: 200 ms (human heart can't beat faster than ~300 bpm)
        self.refractory  = int(0.200 * fs)  # 50 samples @ 250 Hz
 
        # ── Adaptive threshold state ──
        # SPKI = signal peak estimate, NPKI = noise peak estimate
        # Thresholds are updated after each beat detection
        self.SPKI  = 0.0
        self.NPKI  = 0.0
        self.THRESHOLD_I1 = 0.0   # primary threshold
        self.THRESHOLD_I2 = 0.0   # secondary (searchback) threshold
 
        # Internal state
        self.last_r_sample = -self.refractory  # last confirmed R-peak sample index
        self.rr_intervals  = []                # output: list of RR intervals in ms
        self.r_peaks       = []                # output: sample indices of R peaks
 
    # ─────────────────────────────────────────
    #  STEP 1 — Bandpass filter
    # ─────────────────────────────────────────
    def bandpass(self, signal):
        """
        Zero-phase bandpass filter (5–15 Hz).
        Removes baseline wander (<5 Hz) and high-freq muscle noise (>15 Hz).
        """
        return lfilter(self.b_bp, self.a_bp, signal)
 
    # ─────────────────────────────────────────
    #  STEP 2 — Derivative
    # ─────────────────────────────────────────
    def derivative(self, signal):
        """
        5-point derivative operator from the original Pan-Tompkins paper.
        Approximates dy/dt and emphasizes the steep QRS slopes.
        Coefficients: (1/8T) * (-2x[n-2] - x[n-1] + x[n+1] + 2x[n+2])
        """
        deriv = np.zeros_like(signal)
        for i in range(2, len(signal) - 2):
            deriv[i] = (1 / (8 / self.fs)) * (
                -2 * signal[i-2]
                -    signal[i-1]
                +    signal[i+1]
                + 2 * signal[i+2]
            )
        return deriv
 
    # ─────────────────────────────────────────
    #  STEP 3 — Squaring
    # ─────────────────────────────────────────
    def squaring(self, signal):
        """
        Element-wise squaring.
        - Makes all values positive (rectification)
        - Amplifies large slopes (QRS) relative to small ones (noise)
        """
        return signal ** 2
 
    # ─────────────────────────────────────────
    #  STEP 4 — Moving Window Integration
    # ─────────────────────────────────────────
    def moving_window_integrate(self, signal):
        """
        Sliding window of 150 ms (37 samples @ 250 Hz).
        Produces a smooth envelope — the integrated waveform (MWI).
        Peaks in MWI correspond to QRS complexes.
        """
        window = self.mwi_window
        integrated = np.zeros_like(signal)
        for i in range(window, len(signal)):
            integrated[i] = np.sum(signal[i - window:i]) / window
        return integrated
 
    # ─────────────────────────────────────────
    #  STEP 5 — Adaptive Thresholding & R-peak detection
    # ─────────────────────────────────────────
    def find_peaks(self, mwi, filtered):
        """
        Scans the MWI for peaks above an adaptive threshold.
        Updates SPKI/NPKI after each decision (signal peak vs noise peak).
        Returns R-peak sample indices and RR intervals (ms).
 
        The threshold adapts over time so the detector works even if
        the ECG amplitude drifts (e.g., electrode contact changes).
        """
        # Initialise thresholds from first 2 seconds of data
        init_end = min(2 * self.fs, len(mwi))
        self.SPKI  = np.max(mwi[:init_end]) * 0.10
        self.NPKI  = np.mean(mwi[:init_end]) * 0.5
        self.THRESHOLD_I1 = self.NPKI + 0.25 * (self.SPKI - self.NPKI)
        self.THRESHOLD_I2 = 0.5 * self.THRESHOLD_I1
 
        r_peaks     = []
        rr_intervals = []
 
        i = 1
        while i < len(mwi) - 1:
            # Local peak detection (simple: greater than both neighbours)
            if mwi[i] > mwi[i-1] and mwi[i] > mwi[i+1]:
                peak_val = mwi[i]
 
                # ── Refractory check: ignore peaks too close to last R ──
                if (i - self.last_r_sample) < self.refractory:
                    i += 1
                    continue
 
                # ── Above primary threshold → likely QRS ──
                if peak_val > self.THRESHOLD_I1:
                    # Refine: find the actual R peak in the *filtered* signal
                    # within ±80 ms of the MWI peak
                    search_back  = max(0, i - int(0.08 * self.fs))
                    search_fwd   = min(len(filtered), i + int(0.08 * self.fs))
                    r_idx = search_back + np.argmax(np.abs(filtered[search_back:search_fwd]))
 
                    # Record
                    r_peaks.append(r_idx)
                    if len(r_peaks) > 1:
                        rr_ms = ((r_peaks[-1] - r_peaks[-2]) / self.fs) * 1000
                        rr_intervals.append(rr_ms)
 
                    # Update signal peak estimate (running average)
                    self.SPKI = 0.250 * peak_val + 0.750 * self.SPKI
                    self.last_r_sample = i
 
                else:
                    # Below threshold → noise
                    self.NPKI = 0.250 * peak_val + 0.750 * self.NPKI
 
                # Recalculate thresholds
                self.THRESHOLD_I1 = self.NPKI + 0.25 * (self.SPKI - self.NPKI)
                self.THRESHOLD_I2 = 0.5 * self.THRESHOLD_I1
 
            i += 1
 
        self.r_peaks      = r_peaks
        self.rr_intervals = rr_intervals
        return r_peaks, rr_intervals
 
    # ─────────────────────────────────────────
    #  Full pipeline (convenience method)
    # ─────────────────────────────────────────
    def process(self, raw_ecg):
        """
        Run the complete Pan-Tompkins pipeline on a raw ECG array.
 
        Parameters
        ----------
        raw_ecg : np.ndarray
            Raw ECG signal sampled at self.fs Hz.
 
        Returns
        -------
        r_peaks      : list of int   — sample indices of detected R peaks
        rr_intervals : list of float — RR intervals in milliseconds
        intermediates: dict          — intermediate signals for debugging/plotting
        """
        bp       = self.bandpass(raw_ecg)
        deriv    = self.derivative(bp)
        squared  = self.squaring(deriv)
        mwi      = self.moving_window_integrate(squared)
        r_peaks, rr_intervals = self.find_peaks(mwi, bp)
 
        intermediates = {
            "bandpass" : bp,
            "derivative": deriv,
            "squared"  : squared,
            "mwi"      : mwi,
        }
        return r_peaks, rr_intervals, intermediates
 
 
# ══════════════════════════════════════════════════════════════════
#  TESTING / VALIDATION
# ══════════════════════════════════════════════════════════════════
 
def test_with_synthetic():
    """
    Quick sanity check using a synthetic ECG-like signal.
    No external data needed.
    """
    print("Running synthetic signal test...")
    fs  = 250
    dur = 10  # seconds
    t   = np.linspace(0, dur, fs * dur)
 
    # Simulated heartbeat: narrow Gaussian pulses at 1 Hz (60 bpm)
    bpm      = 72
    r_times  = np.arange(0, dur, 60.0 / bpm)
    ecg      = np.zeros_like(t)
    for rt in r_times:
        ecg += 1.5 * np.exp(-((t - rt) ** 2) / (2 * (0.01) ** 2))  # R peak
        ecg += 0.3 * np.exp(-((t - rt - 0.04) ** 2) / (2 * (0.02) ** 2))  # S wave
        ecg += 0.2 * np.exp(-((t - rt - 0.15) ** 2) / (2 * (0.04) ** 2))  # T wave
 
    # Add noise
    ecg += 0.05 * np.random.randn(len(t))
 
    pt = PanTompkins(fs=fs)
    r_peaks, rr_ms, intermediates = pt.process(ecg)
 
    print(f"  Detected R peaks : {len(r_peaks)}")
    print(f"  Expected R peaks : {len(r_times)}")
    print(f"  Mean RR interval : {np.mean(rr_ms):.1f} ms  (expected ~{60000/bpm:.0f} ms)")
    print(f"  Mean heart rate  : {60000/np.mean(rr_ms):.1f} bpm\n")
 
    plot_results(t, ecg, r_peaks, intermediates, title="Synthetic ECG")
    return r_peaks, rr_ms
 
 
def test_with_mitbih(record="100", duration_sec=30):
    """
    Validate against MIT-BIH Arrhythmia Database.
    Record 100 is a clean normal sinus rhythm — great starting point.
 
    Benchmarking records to try:
      100, 101 — normal sinus
      108, 119 — arrhythmias (harder)
    """
    if not WFDB_AVAILABLE:
        print("Skipping MIT-BIH test (wfdb not installed).")
        return
 
    print(f"Loading MIT-BIH record {record} from PhysioNet...")
    rec = wfdb.rdrecord(record, pn_dir='mitdb', sampto=360*duration_sec)
    ann = wfdb.rdann(record, 'atr', pn_dir='mitdb', sampto=360*duration_sec)
 
    ecg_raw = rec.p_signal[:, 0]  # Channel 0 (MLII lead)
    fs_mitbih = rec.fs             # 360 Hz for MIT-BIH
 
    # Resample to 250 Hz to match our hardware
    from scipy.signal import resample
    num_samples = int(len(ecg_raw) * 250 / fs_mitbih)
    ecg_250     = resample(ecg_raw, num_samples)
    scale       = 250 / fs_mitbih
 
    # Reference R peaks (annotations), scaled to 250 Hz
    ref_peaks = [int(s * scale) for s, sym in zip(ann.sample, ann.symbol)
                 if sym in ('N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'F', 'e', 'j')]
 
    # Run detector
    pt = PanTompkins(fs=250)
    detected_peaks, rr_ms, intermediates = pt.process(ecg_250)
 
    # ── Sensitivity & specificity within ±75 ms window ──
    tol        = int(0.075 * 250)  # 75 ms tolerance
    tp, fp, fn = 0, 0, 0
    matched_ref = set()
 
    for dp in detected_peaks:
        match = [i for i, rp in enumerate(ref_peaks)
                 if abs(dp - rp) <= tol and i not in matched_ref]
        if match:
            tp += 1
            matched_ref.add(match[0])
        else:
            fp += 1
 
    fn = len(ref_peaks) - tp
    sensitivity   = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    specificity   = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    f1            = 2*tp / (2*tp + fp + fn) * 100 if (2*tp+fp+fn) > 0 else 0
 
    print(f"\nMIT-BIH Record {record} results ({duration_sec}s):")
    print(f"  Reference peaks  : {len(ref_peaks)}")
    print(f"  Detected peaks   : {len(detected_peaks)}")
    print(f"  True positives   : {tp}")
    print(f"  False positives  : {fp}")
    print(f"  False negatives  : {fn}")
    print(f"  Sensitivity      : {sensitivity:.1f}%  (target ≥ 95%)")
    print(f"  Positive pred.   : {specificity:.1f}%  (target ≥ 95%)")
    print(f"  F1 score         : {f1:.1f}%")
 
    t = np.arange(len(ecg_250)) / 250
    plot_results(t, ecg_250, detected_peaks, intermediates,
                 ref_peaks=ref_peaks, title=f"MIT-BIH Record {record}")
    return detected_peaks, rr_ms
 
 
def plot_results(t, ecg, detected_peaks, intermediates, ref_peaks=None, title="Pan-Tompkins"):
    """Plot the full pipeline for visual inspection."""
    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=14)
 
    signals = [
        (ecg,                          "Raw ECG",         "steelblue"),
        (intermediates["bandpass"],    "Bandpass (5-15Hz)","darkorange"),
        (intermediates["derivative"],  "Derivative",      "green"),
        (intermediates["squared"],     "Squared",         "purple"),
        (intermediates["mwi"],         "MWI (integrated)","red"),
    ]
 
    for ax, (sig, label, color) in zip(axes, signals):
        ax.plot(t, sig, color=color, linewidth=0.7, label=label)
        ax.set_ylabel(label, fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
 
    # Mark detected R peaks on raw ECG
    for rp in detected_peaks:
        if rp < len(t):
            axes[0].axvline(t[rp], color='red', alpha=0.5, linewidth=0.8)
 
    # Mark reference peaks in green if available
    if ref_peaks:
        for rp in ref_peaks:
            if rp < len(t):
                axes[0].axvline(t[rp], color='lime', alpha=0.3, linewidth=0.8)
        axes[0].legend(["ECG", "Detected (red)", "Reference (green)"], fontsize=7)
 
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("pan_tompkins_output.png", dpi=150)
    print("Plot saved to pan_tompkins_output.png")
    plt.show()
 
 
# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  Kardiax Pan-Tompkins Validator")
    print("=" * 55)

    test_with_synthetic()

    RECORDS = [
        ("100", "Normal sinus - clean"),
        ("101", "Normal sinus - baseline wander"),
        ("103", "Normal sinus - good quality"),
        ("105", "Noisy - muscle artifact"),
        ("108", "Noisy - severe baseline wander"),
        ("109", "Bundle branch block - left"),
        ("111", "Bundle branch block - right"),
        ("119", "Arrhythmia - frequent PVCs"),
        ("200", "Arrhythmia - mixed PVCs"),
        ("212", "Complex - amplitude variation"),
        ("213", "Complex - multiple arrhythmias"),
    ]

    results_summary = []

    for record_id, description in RECORDS:
        print(f"\n{'─' * 55}")
        print(f"  Record {record_id}: {description}")
        print(f"{'─' * 55}")
        try:
            test_with_mitbih(record=record_id, duration_sec=60)
            results_summary.append((record_id, description, "✓ Complete"))
        except Exception as e:
            print(f"  ERROR on record {record_id}: {e}")
            results_summary.append((record_id, description, f"✗ Error"))

    print(f"\n{'═' * 55}")
    print("  VALIDATION SUMMARY")
    print(f"{'═' * 55}")
    print(f"  {'Record':<8} {'Status':<12} Description")
    print(f"  {'─'*8} {'─'*12} {'─'*30}")
    for record_id, description, status in results_summary:
        print(f"  {record_id:<8} {status:<12} {description}")
    print(f"{'═' * 55}")
    print("  Target: ≥95% sensitivity AND positive pred. on all records")
    print(f"{'═' * 55}\n")
