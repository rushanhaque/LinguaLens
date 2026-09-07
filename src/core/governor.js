/**
 * governor.js — Adaptive detection-rate control.
 *
 * Inference cost varies enormously across devices: the same model can take 15 ms
 * on a laptop GPU and 250 ms on a mid-range phone. A fixed detection rate either
 * wastes headroom on fast hardware or pegs the CPU on slow hardware, and a
 * pegged CPU is what makes the *rendering* stutter — which is the part the user
 * actually sees.
 *
 * So instead of trusting the configured rate, we measure how long inference
 * really takes and spend a fixed fraction of wall-clock time on it.
 */

const DUTY = 0.55;          // target share of time spent inside the model
const SMOOTHING = 0.2;      // EMA weight for new latency samples
const MIN_HZ = 2;
const SETTLE_SAMPLES = 5;   // ignore the first few passes (warm-up is atypical)

export function createGovernor() {
  let latency = 0;          // smoothed inference time, ms
  let samples = 0;
  let worst = 0;

  /** Record one inference duration. */
  function sample(ms) {
    if (!(ms > 0) || ms > 5000) return;   // ignore nonsense / suspended tabs
    samples++;
    latency = samples === 1 ? ms : latency * (1 - SMOOTHING) + ms * SMOOTHING;
    if (samples > SETTLE_SAMPLES) worst = Math.max(worst, ms);
  }

  /**
   * The rate we should actually run at.
   * @param {number} requestedHz the user's configured ceiling
   */
  function targetHz(requestedHz) {
    if (samples <= SETTLE_SAMPLES || latency <= 0) return requestedHz;
    // Spending DUTY of each second inside the model allows 1000*DUTY/latency
    // passes per second. Never exceed what the user asked for.
    const affordable = (1000 * DUTY) / latency;
    return Math.max(MIN_HZ, Math.min(requestedHz, affordable));
  }

  /** Milliseconds to wait between detection passes. */
  function intervalMs(requestedHz) {
    return 1000 / targetHz(requestedHz);
  }

  /** True when the device cannot keep up with what was asked. */
  function isThrottling(requestedHz) {
    return samples > SETTLE_SAMPLES && targetHz(requestedHz) < requestedHz - 0.5;
  }

  function stats(requestedHz) {
    return {
      latency: Math.round(latency),
      worst: Math.round(worst),
      effectiveHz: Math.round(targetHz(requestedHz) * 10) / 10,
      throttling: isThrottling(requestedHz),
      samples
    };
  }

  function reset() { latency = 0; samples = 0; worst = 0; }

  return { sample, targetHz, intervalMs, isThrottling, stats, reset };
}
