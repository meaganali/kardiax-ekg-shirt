"""
cardiac_monitor.py
Kardiax EKG Shirt — ECE 481 Senior Design

Real-Time Cardiac Event Monitor
---------------------------------
This is the top-level module that sits above Pan-Tompkins and the beat
classifier. It watches the stream of classified beats over a rolling time
window and raises alerts when dangerous cardiac patterns are detected.

This is what actually saves lives. Pan-Tompkins finds beats. The ML model
labels them. This module decides: "is this person going into cardiac arrest?"

Alert levels:
    🔴 CRITICAL — Potentially life-threatening, immediate action required
    🟠 WARNING  — Concerning pattern, medical attention recommended
    🟡 CAUTION  — Abnormality detected, monitor closely

Conditions detected:
    CRITICAL:
        - Ventricular Fibrillation (VF): chaotic RR intervals, no organised rhythm
        - Ventricular Tachycardia (VT): HR >150 bpm sustained, V-type beats
        - Asystole: no beats for >3 seconds (leads-off OR true flat line)
    WARNING:
        - Sustained Bradycardia: HR <40 bpm for >10 seconds
        - Sustained Tachycardia: HR >150 bpm for >10 seconds (non-VT)
    CAUTION:
        - Frequent PVCs: >20% of beats in last 30s are V-type
        - Bigeminy: every other beat is a PVC (alternating N-V-N-V pattern)

How it integrates with the rest of the code:
    Arduino (pan_tompkins.ino)
        ↓ RR intervals over Serial/BLE
    Python / Flutter (pan_tompkins.py + beat_classifier.py)
        ↓ beat type + RR per beat
    THIS FILE (cardiac_monitor.py)
        ↓ alert level + message to Flutter UI

Dependencies:
    pip3 install numpy

Usage (standalone simulation):
    python3 cardiac_monitor.py

Usage (integrated with real data — see bottom of file):
    monitor = CardiacMonitor()
    monitor.load_classifier('models/')
    for each beat from pan_tompkins:
        alert = monitor.process_beat(rr_ms, beat_type, qrs_amp, timestamp)
        if alert:
            send_to_app(alert)
"""

import numpy as np
import time
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

# ══════════════════════════════════════════════════════════════════
#  ALERT TYPES
# ══════════════════════════════════════════════════════════════════

class AlertLevel(Enum):
    NONE     = 0
    CAUTION  = 1   # 🟡
    WARNING  = 2   # 🟠
    CRITICAL = 3   # 🔴

@dataclass
class CardiacAlert:
    level      : AlertLevel
    condition  : str          # short code, e.g. "VF", "VT", "BRADY"
    message    : str          # human-readable message for app UI
    hr_bpm     : float        # heart rate at time of alert
    timestamp  : float        # time.time() when alert was raised
    action     : str          # what the user should do

    def __str__(self):
        icons = {
            AlertLevel.CAUTION : "🟡",
            AlertLevel.WARNING : "🟠",
            AlertLevel.CRITICAL: "🔴",
        }
        icon = icons.get(self.level, "⚪")
        return (f"{icon} [{self.condition}] {self.message} "
                f"(HR: {self.hr_bpm:.0f} bpm) → {self.action}")

    def to_serial_string(self):
        """
        Format for BLE transmission to Flutter app.
        ALERT:<level>,<condition>,<hr>,<message>
        """
        return (f"ALERT:{self.level.name},"
                f"{self.condition},"
                f"{self.hr_bpm:.0f},"
                f"{self.message}")


# ══════════════════════════════════════════════════════════════════
#  BEAT RECORD
# ══════════════════════════════════════════════════════════════════

@dataclass
class BeatRecord:
    rr_ms      : float   # RR interval in ms
    beat_type  : str     # 'N', 'V', 'A', 'O', or 'FP' (false positive)
    hr_bpm     : float   # instantaneous heart rate
    qrs_amp    : float   # QRS amplitude
    timestamp  : float   # time.time()
    is_real    : bool    # True if Stage 1 confirmed real beat


# ══════════════════════════════════════════════════════════════════
#  CARDIAC MONITOR
# ══════════════════════════════════════════════════════════════════

class CardiacMonitor:
    """
    Real-time cardiac event detector.

    Maintains a rolling window of recent beats and continuously
    evaluates them for dangerous patterns.

    Parameters
    ----------
    window_beats  : int   — number of recent beats to analyse (default 30)
    window_sec    : float — time window for rate-based checks (default 30s)
    fs            : int   — ECG sampling rate, passed through to Pan-Tompkins
    """

    def __init__(self, window_beats=30, window_sec=30.0, fs=250):
        self.fs           = fs
        self.window_beats = window_beats
        self.window_sec   = window_sec

        # Rolling beat buffer
        self.beat_buffer  = deque(maxlen=window_beats)

        # Timing
        self.last_beat_time  = None   # time.time() of last confirmed beat
        self.session_start   = time.time()

        # Alert state — avoid spamming the same alert repeatedly
        self.active_alerts   = {}     # condition → CardiacAlert
        self.alert_cooldown  = 5.0    # seconds between repeat alerts

        # Asystole check
        self.asystole_threshold_sec = 3.0   # no beat for 3s = asystole

        # Classifier (loaded separately)
        self.classifier = None

    def load_classifier(self, model_dir='models/'):
        """Load the trained ML classifier for beat typing."""
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            from beat_classifier import KardiaxClassifier
            self.classifier = KardiaxClassifier.load(model_dir)
            print(f"  Classifier loaded from {model_dir}")
        except Exception as e:
            print(f"  Warning: Could not load classifier ({e})")
            print(f"  Running in RR-only mode (no beat type classification)")

    # ─────────────────────────────────────────────────────────────
    #  MAIN ENTRY POINT — call this for every new beat
    # ─────────────────────────────────────────────────────────────

    def process_beat(self, rr_ms: float, beat_type: str = 'N',
                     qrs_amp: float = 1.0,
                     timestamp: float = None) -> Optional[CardiacAlert]:
        """
        Process one incoming beat and check for cardiac events.

        Parameters
        ----------
        rr_ms      : float — RR interval in ms (from Pan-Tompkins)
        beat_type  : str   — beat classification ('N', 'V', 'A', 'FP')
        qrs_amp    : float — QRS amplitude from filtered signal
        timestamp  : float — time.time(), or None to use current time

        Returns
        -------
        CardiacAlert if a condition is detected, else None
        """
        if timestamp is None:
            timestamp = time.time()

        hr_bpm = 60000.0 / rr_ms if rr_ms > 0 else 0.0

        # Record this beat
        beat = BeatRecord(
            rr_ms     = rr_ms,
            beat_type = beat_type,
            hr_bpm    = hr_bpm,
            qrs_amp   = qrs_amp,
            timestamp = timestamp,
            is_real   = (beat_type != 'FP'),
        )
        if beat.is_real:
            self.beat_buffer.append(beat)
            self.last_beat_time = timestamp

        # Run all checks in priority order (highest first)
        alert = (
            self._check_asystole(timestamp) or
            self._check_vf() or
            self._check_vt() or
            self._check_bradycardia() or
            self._check_tachycardia() or
            self._check_frequent_pvcs() or
            self._check_bigeminy()
        )

        return alert

    def check_asystole_timeout(self) -> Optional[CardiacAlert]:
        """
        Call this periodically (e.g. every 500ms) even when no beat arrives.
        Detects asystole — the absence of beats — which process_beat alone can't catch
        because it only runs when a beat comes in.
        """
        return self._check_asystole(time.time())

    # ─────────────────────────────────────────────────────────────
    #  DETECTION RULES
    # ─────────────────────────────────────────────────────────────

    def _check_asystole(self, now: float) -> Optional[CardiacAlert]:
        """
        No beat detected for >3 seconds = cardiac arrest / leads off.
        Most urgent condition — check first.
        """
        if self.last_beat_time is None:
            return None  # Haven't started yet
        elapsed = now - self.last_beat_time
        if elapsed >= self.asystole_threshold_sec:
            return self._raise_alert(
                level     = AlertLevel.CRITICAL,
                condition = "ASYSTOLE",
                message   = f"No heartbeat detected for {elapsed:.1f} seconds",
                hr_bpm    = 0,
                timestamp = now,
                action    = "CALL 911 IMMEDIATELY. Begin CPR if unresponsive.",
            )
        return None

    def _check_vf(self) -> Optional[CardiacAlert]:
        """
        Ventricular Fibrillation: highly irregular RR intervals with no
        organised rhythm. Classic cardiac arrest pattern.

        Detection: coefficient of variation (CV) of RR intervals > 40%
        over the last 10 beats, with HR in the 150–350 range.
        """
        if len(self.beat_buffer) < 8:
            return None

        recent = list(self.beat_buffer)[-10:]
        rr_vals = np.array([b.rr_ms for b in recent])
        hr_vals = np.array([b.hr_bpm for b in recent])

        mean_rr = np.mean(rr_vals)
        std_rr  = np.std(rr_vals)
        cv      = std_rr / mean_rr if mean_rr > 0 else 0
        mean_hr = np.mean(hr_vals)

        # VF: chaotic (CV > 0.40) AND rapid (HR 150-400 range)
        if cv > 0.40 and 150 < mean_hr < 400:
            return self._raise_alert(
                level     = AlertLevel.CRITICAL,
                condition = "VF",
                message   = (f"Ventricular fibrillation suspected. "
                             f"Chaotic rhythm (CV={cv:.2f}), HR={mean_hr:.0f} bpm"),
                hr_bpm    = mean_hr,
                timestamp = time.time(),
                action    = "CALL 911 IMMEDIATELY. AED if available.",
            )
        return None

    def _check_vt(self) -> Optional[CardiacAlert]:
        """
        Ventricular Tachycardia: sustained rapid rhythm (>150 bpm)
        with a high proportion of V-type beats.

        Detection: last 6+ beats all HR >150, >50% classified as V.
        """
        if len(self.beat_buffer) < 6:
            return None

        recent  = list(self.beat_buffer)[-8:]
        hr_vals = np.array([b.hr_bpm for b in recent])
        v_frac  = sum(1 for b in recent if b.beat_type == 'V') / len(recent)
        mean_hr = np.mean(hr_vals)

        if mean_hr > 150 and v_frac >= 0.5:
            return self._raise_alert(
                level     = AlertLevel.CRITICAL,
                condition = "VT",
                message   = (f"Ventricular tachycardia suspected. "
                             f"HR={mean_hr:.0f} bpm, {v_frac*100:.0f}% V-beats"),
                hr_bpm    = mean_hr,
                timestamp = time.time(),
                action    = "CALL 911 IMMEDIATELY. Do not leave patient alone.",
            )
        return None

    def _check_bradycardia(self) -> Optional[CardiacAlert]:
        """
        Sustained Bradycardia: HR <40 bpm for at least 5 consecutive beats.
        Can indicate heart block or imminent arrest.
        """
        if len(self.beat_buffer) < 5:
            return None

        recent  = list(self.beat_buffer)[-5:]
        hr_vals = [b.hr_bpm for b in recent]

        if all(hr < 40 for hr in hr_vals):
            mean_hr = np.mean(hr_vals)
            return self._raise_alert(
                level     = AlertLevel.WARNING,
                condition = "BRADY",
                message   = f"Severe bradycardia. HR={mean_hr:.0f} bpm sustained",
                hr_bpm    = mean_hr,
                timestamp = time.time(),
                action    = "Seek immediate medical attention.",
            )
        return None

    def _check_tachycardia(self) -> Optional[CardiacAlert]:
        """
        Sustained Tachycardia: HR >150 bpm, but NOT classified as VT.
        Could be SVT, sinus tachycardia from exertion, etc.
        """
        if len(self.beat_buffer) < 6:
            return None

        recent  = list(self.beat_buffer)[-8:]
        hr_vals = [b.hr_bpm for b in recent]
        v_frac  = sum(1 for b in recent if b.beat_type == 'V') / len(recent)

        if all(hr > 150 for hr in hr_vals) and v_frac < 0.5:
            mean_hr = np.mean(hr_vals)
            return self._raise_alert(
                level     = AlertLevel.WARNING,
                condition = "TACHY",
                message   = f"Sustained tachycardia. HR={mean_hr:.0f} bpm",
                hr_bpm    = mean_hr,
                timestamp = time.time(),
                action    = "Rest immediately. Seek medical attention if persistent.",
            )
        return None

    def _check_frequent_pvcs(self) -> Optional[CardiacAlert]:
        """
        Frequent PVCs: >20% of beats in the recent window are V-type.
        Frequent PVCs can degenerate into VT or VF.
        """
        if len(self.beat_buffer) < 10:
            return None

        recent = list(self.beat_buffer)[-20:]
        v_frac = sum(1 for b in recent if b.beat_type == 'V') / len(recent)
        mean_hr = np.mean([b.hr_bpm for b in recent])

        if v_frac > 0.20:
            return self._raise_alert(
                level     = AlertLevel.CAUTION,
                condition = "FREQ_PVC",
                message   = f"Frequent PVCs detected ({v_frac*100:.0f}% of recent beats)",
                hr_bpm    = mean_hr,
                timestamp = time.time(),
                action    = "Monitor closely. Consult physician if persistent.",
            )
        return None

    def _check_bigeminy(self) -> Optional[CardiacAlert]:
        """
        Bigeminy: alternating normal and PVC beats (N-V-N-V-N-V pattern).
        A sign of cardiac irritability that can precede worse arrhythmias.

        Detection: in the last 8 beats, even-indexed are N and odd-indexed are V
        (or vice versa), with >75% pattern match.
        """
        if len(self.beat_buffer) < 8:
            return None

        recent = list(self.beat_buffer)[-8:]
        types  = [b.beat_type for b in recent]

        # Check N-V-N-V pattern
        pattern_nv = sum(1 for i, t in enumerate(types)
                         if (i % 2 == 0 and t == 'N') or
                            (i % 2 == 1 and t == 'V'))
        # Check V-N-V-N pattern
        pattern_vn = sum(1 for i, t in enumerate(types)
                         if (i % 2 == 0 and t == 'V') or
                            (i % 2 == 1 and t == 'N'))

        match_frac = max(pattern_nv, pattern_vn) / len(recent)
        mean_hr    = np.mean([b.hr_bpm for b in recent])

        if match_frac >= 0.75:
            return self._raise_alert(
                level     = AlertLevel.CAUTION,
                condition = "BIGEMINY",
                message   = f"Bigeminy pattern detected (alternating PVCs)",
                hr_bpm    = mean_hr,
                timestamp = time.time(),
                action    = "Consult physician. Avoid stimulants (caffeine, etc.).",
            )
        return None

    # ─────────────────────────────────────────────────────────────
    #  ALERT MANAGEMENT
    # ─────────────────────────────────────────────────────────────

    def _raise_alert(self, level, condition, message, hr_bpm,
                     timestamp, action) -> Optional[CardiacAlert]:
        """
        Create and return an alert, respecting cooldown to avoid spam.
        """
        now  = timestamp or time.time()
        last = self.active_alerts.get(condition)

        # Cooldown: don't repeat the same alert within cooldown period
        # EXCEPT for CRITICAL alerts — always re-raise those
        if last is not None and level != AlertLevel.CRITICAL:
            if (now - last.timestamp) < self.alert_cooldown:
                return None

        alert = CardiacAlert(
            level     = level,
            condition = condition,
            message   = message,
            hr_bpm    = hr_bpm,
            timestamp = now,
            action    = action,
        )
        self.active_alerts[condition] = alert
        return alert

    def clear_alert(self, condition: str):
        """Call this when the app dismisses an alert."""
        self.active_alerts.pop(condition, None)

    # ─────────────────────────────────────────────────────────────
    #  STATUS / REPORTING
    # ─────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """
        Returns current cardiac status summary.
        Call this to update the Flutter app dashboard.
        """
        if len(self.beat_buffer) == 0:
            return {'status': 'NO_DATA', 'hr_bpm': 0, 'rhythm': 'Unknown'}

        recent  = list(self.beat_buffer)[-10:]
        hr_vals = [b.hr_bpm for b in recent]
        mean_hr = np.mean(hr_vals)
        v_frac  = sum(1 for b in recent if b.beat_type == 'V') / len(recent)

        # Simple rhythm classification for display
        if mean_hr < 60:
            rhythm = "Bradycardia" if mean_hr < 50 else "Sinus bradycardia"
        elif mean_hr > 100:
            rhythm = "Tachycardia" if mean_hr > 150 else "Sinus tachycardia"
        else:
            rhythm = "Normal sinus rhythm"

        if v_frac > 0.1:
            rhythm += f" with PVCs ({v_frac*100:.0f}%)"

        return {
            'status'     : 'OK' if not self.active_alerts else 'ALERT',
            'hr_bpm'     : round(mean_hr, 1),
            'rhythm'     : rhythm,
            'beat_count' : len(self.beat_buffer),
            'v_fraction' : round(v_frac, 3),
            'alerts'     : [str(a) for a in self.active_alerts.values()],
        }


# ══════════════════════════════════════════════════════════════════
#  SIMULATION / TESTING
# ══════════════════════════════════════════════════════════════════

def simulate_scenario(monitor, scenario_name, beats):
    """
    Feed a sequence of simulated beats into the monitor and collect alerts.

    beats: list of (rr_ms, beat_type, qrs_amp) tuples
    """
    print(f"\n  ── {scenario_name} ──")
    alerts_raised = []
    t = time.time()

    for i, (rr_ms, beat_type, qrs_amp) in enumerate(beats):
        t += rr_ms / 1000.0  # advance simulated time
        alert = monitor.process_beat(rr_ms, beat_type, qrs_amp, timestamp=t)
        hr = 60000 / rr_ms
        if alert:
            print(f"    Beat {i+1:2d} (HR={hr:.0f}): {alert}")
            alerts_raised.append(alert)
        else:
            status = monitor.get_status()
            print(f"    Beat {i+1:2d} (HR={hr:.0f} bpm, type={beat_type}): "
                  f"No alert — {status['rhythm']}")

    # Reset monitor state between scenarios
    monitor.beat_buffer.clear()
    monitor.active_alerts.clear()
    monitor.last_beat_time = None
    return alerts_raised


if __name__ == "__main__":
    print("=" * 60)
    print("  Kardiax Cardiac Monitor — Simulation Test")
    print("=" * 60)

    monitor = CardiacMonitor()

    # ── Try to load classifier if models exist ──
    if os.path.exists('models/stage1_fp_detector.joblib'):
        monitor.load_classifier('models/')
    else:
        print("  No models found — run beat_classifier.py first")
        print("  Running in simulation mode with manual beat types\n")

    # ════════════════════════════════════════════════════════════
    #  SCENARIO 1: Normal sinus rhythm — no alert expected
    # ════════════════════════════════════════════════════════════
    normal_beats = [(833, 'N', 0.8)] * 15  # 72 bpm, all normal
    simulate_scenario(monitor, "Normal sinus rhythm (72 bpm)", normal_beats)

    # ════════════════════════════════════════════════════════════
    #  SCENARIO 2: Occasional PVCs — caution expected
    # ════════════════════════════════════════════════════════════
    pvcs = ([(833, 'N', 0.8)] * 4 +   # 4 normal
            [(500, 'V', 1.4)] +        # PVC (early, large)
            [(1100, 'N', 0.8)] +       # compensatory pause
            [(833, 'N', 0.8)] * 3 +   # normal
            [(480, 'V', 1.5)] +        # another PVC
            [(1150, 'N', 0.8)] +       # compensatory pause
            [(833, 'N', 0.8)] * 5 +   # normal
            [(460, 'V', 1.6)] * 3 +   # run of PVCs
            [(833, 'N', 0.8)] * 3)
    simulate_scenario(monitor, "Frequent PVCs", pvcs)

    # ════════════════════════════════════════════════════════════
    #  SCENARIO 3: Bigeminy — alternating N-V-N-V
    # ════════════════════════════════════════════════════════════
    bigeminy = []
    for _ in range(10):
        bigeminy.append((900, 'N', 0.8))   # normal
        bigeminy.append((500, 'V', 1.4))   # PVC
    simulate_scenario(monitor, "Bigeminy (N-V-N-V pattern)", bigeminy)

    # ════════════════════════════════════════════════════════════
    #  SCENARIO 4: Ventricular Tachycardia — CRITICAL expected
    # ════════════════════════════════════════════════════════════
    vt = ([(833, 'N', 0.8)] * 5 +        # starts normal
          [(350, 'V', 1.6)] * 12)         # VT: 171 bpm, all V-type
    simulate_scenario(monitor, "Ventricular Tachycardia", vt)

    # ════════════════════════════════════════════════════════════
    #  SCENARIO 5: Ventricular Fibrillation — CRITICAL expected
    # ════════════════════════════════════════════════════════════
    np.random.seed(42)
    vf_rr = np.random.uniform(150, 450, 12).tolist()  # chaotic RR
    vf = [(rr, 'V', np.random.uniform(0.2, 1.2)) for rr in vf_rr]
    simulate_scenario(monitor, "Ventricular Fibrillation", vf)

    # ════════════════════════════════════════════════════════════
    #  SCENARIO 6: Bradycardia — WARNING expected
    # ════════════════════════════════════════════════════════════
    brady = [(1700, 'N', 0.8)] * 10   # 35 bpm
    simulate_scenario(monitor, "Severe Bradycardia (35 bpm)", brady)

    # ════════════════════════════════════════════════════════════
    #  SCENARIO 7: Asystole — CRITICAL expected
    # ════════════════════════════════════════════════════════════
    print("\n  ── Asystole (no beats for 4 seconds) ──")
    monitor.last_beat_time = time.time() - 4.0  # simulate 4s since last beat
    alert = monitor.check_asystole_timeout()
    if alert:
        print(f"    {alert}")
    monitor.beat_buffer.clear()
    monitor.active_alerts.clear()
    monitor.last_beat_time = None

    print(f"\n{'═' * 60}")
    print("  All scenarios complete.")
    print("  Next step: connect process_beat() to BLE serial output")
    print("  from pan_tompkins.ino in the Flutter app.")
    print(f"{'═' * 60}\n")

    # ════════════════════════════════════════════════════════════
    #  INTEGRATION EXAMPLE (pseudocode comment)
    # ════════════════════════════════════════════════════════════
    """
    HOW TO WIRE THIS INTO THE FULL PIPELINE:
    
    # In your Flutter BLE receiver or Python serial reader:
    
    from pan_tompkins import PanTompkins
    from beat_classifier import KardiaxClassifier
    from cardiac_monitor import CardiacMonitor, AlertLevel
    
    monitor    = CardiacMonitor()
    classifier = KardiaxClassifier.load('models/')
    
    rr_history = []
    beat_idx   = 0
    
    # Called every time a new RR interval arrives from Arduino:
    def on_new_beat(rr_ms, qrs_amp):
        global beat_idx
        
        rr_prev = rr_history[-1] if rr_history else rr_ms
        rr_next = 0  # unknown until next beat
        
        result = classifier.classify_beat(
            rr_ms, rr_prev, rr_next, qrs_amp,
            beat_idx, beat_idx + 1
        )
        
        beat_type = result['beat_type'] if result['is_real'] else 'FP'
        
        alert = monitor.process_beat(rr_ms, beat_type, qrs_amp)
        
        if alert:
            if alert.level == AlertLevel.CRITICAL:
                app.show_emergency_screen(alert)
                app.vibrate_intense()
                app.play_alarm()
            elif alert.level == AlertLevel.WARNING:
                app.show_warning_banner(alert)
                app.vibrate_gentle()
            elif alert.level == AlertLevel.CAUTION:
                app.show_caution_notification(alert)
        
        rr_history.append(rr_ms)
        beat_idx += 1
        
        # Also update the dashboard every beat
        status = monitor.get_status()
        app.update_hr_display(status['hr_bpm'])
        app.update_rhythm_label(status['rhythm'])
    
    # Call this on a timer every 500ms to catch asystole:
    def periodic_check():
        alert = monitor.check_asystole_timeout()
        if alert:
            app.show_emergency_screen(alert)
    """
