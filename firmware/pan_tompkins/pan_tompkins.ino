/*
 * pan_tompkins.ino
 * Kardiax EKG Shirt — ECE 481 Senior Design
 * Author: Meagan Ali (firmware lead) / Deceli Soto (algorithm)
 *
 * Pan-Tompkins QRS Detection on Arduino Nano 33 IoT
 * --------------------------------------------------
 * Hardware:
 *   - Arduino Nano 33 IoT
 *   - SparkFun AD8232 ECG Amplifier
 *     AD8232 OUTPUT pin → Arduino A0
 *     AD8232 LO+        → Arduino D10
 *     AD8232 LO-        → Arduino D11
 *     AD8232 3.3V       → Arduino 3.3V
 *     AD8232 GND        → Arduino GND
 *
 * What this sketch does:
 *   1. Samples A0 at exactly 250 Hz using a hardware timer interrupt
 *   2. Runs Pan-Tompkins pipeline on each new sample (sample-by-sample)
 *   3. Detects R peaks → computes RR intervals (ms)
 *   4. Sends RR intervals + heart rate over Serial (for BLE pickup by Meagan's firmware)
 *
 * Serial output format (one line per beat):
 *   RR:<milliseconds>,HR:<bpm>
 *   Example:  RR:823,HR:72
 *
 * Note on fixed-point math:
 *   Arduino Nano 33 IoT has an FPU (ARM Cortex-M0+... actually M4 on SAMD21),
 *   so floats are fine here. If you port to a Uno/Nano classic (no FPU), 
 *   switch to integer arithmetic.
 */

#include <Arduino.h>

// ══════════════════════════════════════════════════════════════════
//  CONFIGURATION
// ══════════════════════════════════════════════════════════════════

#define FS            250      // Sampling rate (Hz) — matches ER1
#define ECG_PIN       A0       // AD8232 output
#define LEADS_OFF_POS 10       // AD8232 LO+
#define LEADS_OFF_NEG 11       // AD8232 LO-

// Bandpass filter cutoffs (5–15 Hz, 2nd-order Butterworth)
// Coefficients pre-computed for fs=250 Hz using scipy.signal.butter(2,[5,15],btype='band')
// b = [0.01335, 0, -0.02670, 0, 0.01335]
// a = [1.0, -3.5787, 4.9573, -3.2959, 0.8812]
// This is a 4th-order IIR (2nd-order bandpass = 4th-order in Direct Form II)
#define BP_B0   0.01335f
#define BP_B1   0.00000f
#define BP_B2  -0.02670f
#define BP_B3   0.00000f
#define BP_B4   0.01335f
#define BP_A1  -3.57870f
#define BP_A2   4.95730f
#define BP_A3  -3.29590f
#define BP_A4   0.88120f

// Moving window integration: 150 ms
#define MWI_WIN       38       // int(0.150 * 250) = 37 → 38 for safety

// Refractory period: 200 ms (can't beat faster than 300 bpm)
#define REFRACTORY    50       // int(0.200 * 250)

// Derivative delay: the 5-pt derivative needs a 4-sample buffer
#define DERIV_LEN     5

// ══════════════════════════════════════════════════════════════════
//  STATE VARIABLES
// ══════════════════════════════════════════════════════════════════

// ── Filter state (Direct Form II Transposed, 4th-order IIR) ──
float bp_w[4] = {0, 0, 0, 0};   // delay line for bandpass filter

// ── Derivative buffer (5-point running buffer) ──
float deriv_buf[DERIV_LEN] = {0, 0, 0, 0, 0};
int   deriv_idx = 0;

// ── MWI circular buffer ──
float mwi_buf[MWI_WIN];
int   mwi_idx = 0;
float mwi_sum = 0.0f;

// ── Adaptive thresholds ──
float SPKI  = 0.0f;      // signal peak estimate
float NPKI  = 0.0f;      // noise peak estimate
float THR1  = 0.0f;      // primary threshold
float THR2  = 0.0f;      // secondary threshold

// ── Peak tracking ──
float   prev_mwi  = 0.0f;     // previous MWI value (for slope tracking)
float   pprev_mwi = 0.0f;     // two-sample-ago MWI value
int     sample_count  = 0;    // total samples processed
int     last_r_sample = -100; // sample index of last confirmed R peak
long    last_r_time_us = 0;   // micros() at last R peak

// ── Initialisation window (first 2 sec = 500 samples) ──
bool    initialised = false;
float   init_buf[500];
int     init_count = 0;

// ── Interrupt flag ──
volatile bool new_sample_ready = false;
volatile int  raw_adc_value    = 0;

// ── Heartrate ──
float   rr_ms   = 0.0f;
float   hr_bpm  = 0.0f;

// ══════════════════════════════════════════════════════════════════
//  TIMER INTERRUPT (250 Hz)
//  The Nano 33 IoT uses SAMD21.  We use TC3 in 16-bit mode.
//  This fires every 4 ms (250 Hz).
// ══════════════════════════════════════════════════════════════════

void setupTimer250Hz() {
  // Enable TC3 clock
  GCLK->CLKCTRL.reg = GCLK_CLKCTRL_CLKEN |
                      GCLK_CLKCTRL_GEN_GCLK0 |
                      GCLK_CLKCTRL_ID_TCC2_TC3;
  while (GCLK->STATUS.bit.SYNCBUSY);

  TC3->COUNT16.CTRLA.reg &= ~TC_CTRLA_ENABLE;
  while (TC3->COUNT16.STATUS.bit.SYNCBUSY);

  TC3->COUNT16.CTRLA.reg = TC_CTRLA_MODE_COUNT16 |
                           TC_CTRLA_WAVEGEN_MFRQ  |
                           TC_CTRLA_PRESCALER_DIV64;
  while (TC3->COUNT16.STATUS.bit.SYNCBUSY);

  // 48 MHz / 64 / 250 Hz = 3000 - 1 = 2999
  TC3->COUNT16.CC[0].reg = 2999;
  while (TC3->COUNT16.STATUS.bit.SYNCBUSY);

  TC3->COUNT16.INTENSET.reg = TC_INTENSET_MC0;
  NVIC_EnableIRQ(TC3_IRQn);
  NVIC_SetPriority(TC3_IRQn, 0);

  TC3->COUNT16.CTRLA.bit.ENABLE = 1;
  while (TC3->COUNT16.STATUS.bit.SYNCBUSY);
}

// TC3 ISR — fires at 250 Hz
void TC3_Handler() {
  if (TC3->COUNT16.INTFLAG.bit.MC0) {
    TC3->COUNT16.INTFLAG.bit.MC0 = 1;   // clear flag
    raw_adc_value   = analogRead(ECG_PIN);
    new_sample_ready = true;
  }
}

// ══════════════════════════════════════════════════════════════════
//  PAN-TOMPKINS: STEP-BY-STEP SAMPLE PROCESSING
//  Called once per sample from loop()
// ══════════════════════════════════════════════════════════════════

// ── Step 1: Bandpass filter (5–15 Hz) ──
float bandpass_filter(float x) {
  // Direct Form II Transposed IIR
  float y = BP_B0 * x         + bp_w[0];
  bp_w[0] = BP_B1 * x - BP_A1 * y + bp_w[1];
  bp_w[1] = BP_B2 * x - BP_A2 * y + bp_w[2];
  bp_w[2] = BP_B3 * x - BP_A3 * y + bp_w[3];
  bp_w[3] = BP_B4 * x - BP_A4 * y;
  return y;
}

// ── Step 2: 5-point derivative ──
float derivative_filter(float x) {
  // Store new sample in circular buffer
  deriv_buf[deriv_idx] = x;
  deriv_idx = (deriv_idx + 1) % DERIV_LEN;

  // Access buffer: indices offset from current position
  // Buffer positions: [n-4], [n-3], [n-2], [n-1], [n]
  float xn2  = deriv_buf[(deriv_idx + 0) % DERIV_LEN]; // x[n-2] (oldest)
  float xn1  = deriv_buf[(deriv_idx + 1) % DERIV_LEN]; // x[n-1]
  // x[n]   = deriv_buf[(deriv_idx + 2) % DERIV_LEN]; // current (not used)
  float xp1  = deriv_buf[(deriv_idx + 3) % DERIV_LEN]; // x[n+1]
  float xp2  = deriv_buf[(deriv_idx + 4) % DERIV_LEN]; // x[n+2] (newest)

  // Pan-Tompkins 5-pt derivative: (1/8T)*(-2x[n-2] - x[n-1] + x[n+1] + 2x[n+2])
  float T = 1.0f / FS;
  return (1.0f / (8.0f * T)) * (-2.0f*xn2 - xn1 + xp1 + 2.0f*xp2);
}

// ── Step 3: Squaring ──
float squaring(float x) {
  return x * x;
}

// ── Step 4: Moving Window Integration ──
float moving_window_integrate(float x) {
  // Remove oldest value, add new value
  mwi_sum -= mwi_buf[mwi_idx];
  mwi_buf[mwi_idx] = x;
  mwi_sum += x;
  mwi_idx = (mwi_idx + 1) % MWI_WIN;
  return mwi_sum / MWI_WIN;
}

// ── Step 5: Peak detection & adaptive thresholding ──
void detect_peak(float mwi_val, int sample_idx) {
  // Local max detection: current > previous AND current > two-ago
  // (we check *after* the peak, i.e., on the downslope)
  bool is_peak = (pprev_mwi < prev_mwi) && (prev_mwi >= mwi_val);

  pprev_mwi = prev_mwi;
  prev_mwi  = mwi_val;

  if (!is_peak) return;

  float peak_val = prev_mwi;  // the peak we just passed

  // Refractory check
  if ((sample_idx - last_r_sample) < REFRACTORY) return;

  if (peak_val > THR1) {
    // ── Confirmed R peak ──
    long now_us = micros();

    if (last_r_sample > 0) {
      rr_ms  = (now_us - last_r_time_us) / 1000.0f;
      hr_bpm = 60000.0f / rr_ms;

      // Sanity check: 30–220 bpm is physiologically plausible
      if (hr_bpm >= 30.0f && hr_bpm <= 220.0f) {
        // Output RR interval on Serial for BLE firmware to pick up
        Serial.print("RR:");
        Serial.print((int)rr_ms);
        Serial.print(",HR:");
        Serial.println((int)hr_bpm);
      }
    }

    last_r_sample  = sample_idx;
    last_r_time_us = now_us;

    // Update signal peak running average
    SPKI = 0.125f * peak_val + 0.875f * SPKI;

  } else {
    // Noise peak
    NPKI = 0.125f * peak_val + 0.875f * NPKI;
  }

  // Recalculate adaptive thresholds
  THR1 = NPKI + 0.25f * (SPKI - NPKI);
  THR2 = 0.5f  * THR1;
}

// ── Initialise thresholds from first 2 seconds ──
void initialise_thresholds(float mwi_val) {
  if (init_count < 500) {
    init_buf[init_count++] = mwi_val;
    return;
  }

  // Find max and mean of first 2 seconds
  float mx = 0.0f, mn = 0.0f;
  for (int i = 0; i < 500; i++) {
    if (init_buf[i] > mx) mx = init_buf[i];
    mn += init_buf[i];
  }
  mn /= 500.0f;

  SPKI = mx  * 0.25f;
  NPKI = mn  * 0.50f;
  THR1 = NPKI + 0.25f * (SPKI - NPKI);
  THR2 = 0.5f * THR1;

  initialised = true;
  Serial.println("STATUS:INIT_COMPLETE");
}

// ── Process one ADC sample through the full pipeline ──
void process_sample(int adc_val) {
  // Convert ADC (0-4095 for 12-bit) to centred float (-1.0 to 1.0)
  float x = (adc_val - 2048.0f) / 2048.0f;

  float bp  = bandpass_filter(x);
  float drv = derivative_filter(bp);
  float sq  = squaring(drv);
  float mwi = moving_window_integrate(sq);

  if (!initialised) {
    initialise_thresholds(mwi);
    return;
  }

  detect_peak(mwi, sample_count);
}

// ══════════════════════════════════════════════════════════════════
//  SETUP & LOOP
// ══════════════════════════════════════════════════════════════════

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000);  // Wait up to 3s for Serial monitor

  Serial.println("STATUS:KARDIAX_START");

  // Configure AD8232 leads-off detection pins
  pinMode(LEADS_OFF_POS, INPUT);
  pinMode(LEADS_OFF_NEG, INPUT);

  // 12-bit ADC (Nano 33 IoT supports this natively)
  analogReadResolution(12);

  // Zero out MWI circular buffer
  memset(mwi_buf, 0, sizeof(mwi_buf));

  // Start 250 Hz timer
  setupTimer250Hz();

  Serial.println("STATUS:READY");
}

void loop() {
  if (!new_sample_ready) return;

  // Atomically grab the ISR value and clear the flag
  noInterrupts();
  int adc = raw_adc_value;
  new_sample_ready = false;
  interrupts();

  // Check leads-off condition before processing
  if (digitalRead(LEADS_OFF_POS) == 1 || digitalRead(LEADS_OFF_NEG) == 1) {
    Serial.println("STATUS:LEADS_OFF");
    return;
  }

  process_sample(adc);
  sample_count++;
}

/*
 * ──────────────────────────────────────────────────────────
 * SERIAL OUTPUT FORMAT (read by BLE firmware / Flutter app)
 * ──────────────────────────────────────────────────────────
 * RR:<ms>,HR:<bpm>          — each beat
 * STATUS:INIT_COMPLETE      — algorithm ready (after 2 sec)
 * STATUS:LEADS_OFF          — electrode not contacting skin
 * STATUS:KARDIAX_START      — boot message
 * STATUS:READY              — ready to acquire
 *
 * ──────────────────────────────────────────────────────────
 * NEXT STEPS (for Meagan — firmware integration)
 * ──────────────────────────────────────────────────────────
 * 1. Parse Serial lines in BLE firmware:
 *    if (line.startsWith("RR:")) { ... parse rr and hr ... }
 * 2. Package RR + HR into a BLE characteristic
 * 3. Add the impedance reading alongside RR in the serial line
 *    so the full BLE packet is: RR:<ms>,HR:<bpm>,IMP:<ohms>
 */
