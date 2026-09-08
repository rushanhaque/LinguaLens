/**
 * classifier.js — Whole-frame recognition via MobileNet, on top of COCO-SSD.
 *
 * The two models answer different questions and the app needs both.
 * COCO-SSD answers "where are things, and which of these eighty are they",
 * which is what the AR boxes need. MobileNet answers "what is this a picture
 * of", drawing on ImageNet's thousand classes — so it can name a pen, a
 * stapler, a globe, a fountain pen, a beagle, or a pretzel, none of which
 * COCO has ever heard of.
 *
 * Two design notes worth keeping:
 *
 * Centre crop. The classifier is asked about the middle square of the frame
 * rather than the whole thing. Pointing a camera *at* something is a centring
 * gesture, and cropping removes the desk, wall and floor that otherwise
 * dominate a wide shot and drag the prediction towards "studio couch".
 *
 * Probability is aggregated by mapped word, not taken from the top label.
 * ImageNet splits "dog" across a hundred and twenty breeds, so a clear photo
 * of a dog can land at 0.3 golden retriever, 0.2 labrador, 0.1 beagle — three
 * weak-looking predictions that are jointly overwhelming. Summing them after
 * mapping recovers the confidence the split threw away.
 */

import { mapImagenet } from '../data/imagenet.js';
import { DICT } from '../data/dictionary.js';

const INPUT = 224;            // MobileNet's native input size
const TOP_K = 8;              // predictions to aggregate over
const MIN_SCORE = 0.16;       // aggregated probability floor
const AGREE_FRAMES = 2;       // consecutive passes that must agree before showing
const HOLD_MS = 900;          // how long a result survives without reconfirmation

/* Self-pacing. The classifier is a guest on the same GPU as the box detector
   and the compositor, so it budgets itself a fixed share of wall-clock time
   rather than a fixed rate: measure how long a pass costs, then wait long
   enough afterwards that inference never exceeds DUTY of the time available.
   On a fast laptop that settles around four passes a second; on a struggling
   phone it backs off on its own instead of stealing frames from rendering. */
const DUTY = 0.22;
const MIN_GAP_MS = 220;       // never faster than this, however cheap it looks
const MAX_GAP_MS = 2500;      // never slower than this, however dear it looks
const GIVE_UP_MS = 1800;      // a pass this slow means the device cannot serve it
const LAT_SMOOTH = 0.3;

export function createClassifier() {
  const canvas = document.createElement('canvas');
  canvas.width = INPUT;
  canvas.height = INPUT;
  const ctx = canvas.getContext('2d', { willReadFrequently: false });

  let model = null;
  let loading = null;
  let busy = false;

  /* Hysteresis state: what we last saw, and what we are currently showing. */
  let candidate = null;
  let agree = 0;
  let shown = null;
  let shownAt = 0;

  /* Pacing state. */
  let latency = 0;            // smoothed cost of one pass, ms
  let lastRunAt = 0;
  let disabled = false;       // set when the device plainly cannot afford it

  /** Load MobileNet, then warm it up. Resolves true once it is usable. */
  function load() {
    if (model) return Promise.resolve(true);
    if (loading) return loading;
    if (typeof mobilenet === 'undefined') return Promise.resolve(false);
    loading = mobilenet
      // v2 is markedly more accurate than v1 at a similar size, and alpha 1.0
      // is worth the download for a model that is cached forever afterwards.
      .load({ version: 2, alpha: 1.0 })
      .then(async (m) => {
        model = m;
        // The first inference compiles shaders and is wildly unrepresentative
        // — it can take seconds where steady state takes tens of milliseconds.
        // Spending it here, on a blank frame, keeps it out of the live path
        // and out of the latency average that drives pacing.
        try {
          ctx.fillStyle = '#808080';
          ctx.fillRect(0, 0, INPUT, INPUT);
          await m.classify(canvas, 1);
        } catch { /* the real passes will tell us soon enough */ }
        lastRunAt = performance.now();
        return true;
      })
      .catch(() => false);
    return loading;
  }

  /** Milliseconds to leave between passes, from the measured cost of one. */
  function gapMs() {
    if (!latency) return MIN_GAP_MS;
    return Math.min(MAX_GAP_MS, Math.max(MIN_GAP_MS, latency / DUTY - latency));
  }

  /** Draw the centre square of the video into the 224x224 input canvas. */
  function cropCentre(video) {
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return false;
    // A slightly loose crop (80% of the short edge) keeps a little context,
    // which helps the model more than a tight crop does.
    const side = Math.min(vw, vh) * 0.8;
    const sx = (vw - side) / 2;
    const sy = (vh - side) / 2;
    ctx.drawImage(video, sx, sy, side, side, 0, 0, INPUT, INPUT);
    return true;
  }

  /**
   * Classify the current frame.
   * @returns {Promise<{key:string,score:number,label:string}|null>}
   *   The stable result to display, or null when nothing is confident enough.
   *   Returns the currently held result unchanged while a pass is in flight.
   */
  async function classify(video) {
    if (!model || busy || disabled || !video) return current();
    // Self-paced: a caller may ask every frame, and be told to wait.
    if (performance.now() - lastRunAt < gapMs()) return current();
    if (!cropCentre(video)) return current();

    busy = true;
    lastRunAt = performance.now();
    const t0 = performance.now();
    try {
      const preds = await model.classify(canvas, TOP_K);

      const ms = performance.now() - t0;
      latency = latency ? latency * (1 - LAT_SMOOTH) + ms * LAT_SMOOTH : ms;
      // A device this slow would spend its whole frame budget here, and the
      // box detector — which the AR overlay actually depends on — would
      // starve. Better to stand down and leave the app responsive.
      if (latency > GIVE_UP_MS) { disabled = true; shown = null; }

      /* Aggregate probability by the word each prediction maps to. */
      const byKey = new Map();
      for (const p of preds) {
        const key = mapImagenet(p.className);
        if (!key || !DICT[key]) continue;
        const prev = byKey.get(key);
        if (prev) prev.score += p.probability;
        else byKey.set(key, { key, score: p.probability, label: p.className.split(',')[0] });
      }

      let best = null;
      for (const v of byKey.values()) if (!best || v.score > best.score) best = v;

      if (!best || best.score < MIN_SCORE) {
        candidate = null;
        agree = 0;
      } else if (candidate === best.key) {
        agree++;
      } else {
        candidate = best.key;
        agree = 1;
      }

      // Promote to shown only once consecutive passes concur, which stops the
      // chip flickering between near-tied words while the camera settles.
      if (best && agree >= AGREE_FRAMES) {
        shown = best;
        shownAt = performance.now();
      }
    } catch {
      /* A failed pass is not worth reporting; the held result stands. */
    } finally {
      busy = false;
      lastRunAt = performance.now();      // the gap starts when the pass ends
    }
    return current();
  }

  /** The result that should be on screen right now, honouring the hold time. */
  function current() {
    if (!shown) return null;
    if (performance.now() - shownAt > HOLD_MS) { shown = null; return null; }
    return shown;
  }

  /**
   * Classify a still image, without the hysteresis the live path needs.
   *
   * A photo gets the whole frame rather than a centre crop — the user chose
   * the framing deliberately — and returns every word that clears the bar,
   * not just the winner, because a picture usually holds several things.
   *
   * @param {CanvasImageSource} source
   * @param {number} max how many words to return
   * @returns {Promise<string[]>} vocabulary keys, most confident first
   */
  async function classifyStill(source, max = 4) {
    if (!model || !source) return [];
    const preds = await model.classify(source, TOP_K);
    const byKey = new Map();
    for (const p of preds) {
      const key = mapImagenet(p.className);
      if (!key || !DICT[key]) continue;
      byKey.set(key, (byKey.get(key) || 0) + p.probability);
    }
    return [...byKey.entries()]
      .filter(([, score]) => score >= MIN_SCORE)
      .sort((a, b) => b[1] - a[1])
      .slice(0, max)
      .map(([key]) => key);
  }

  function reset() {
    candidate = null;
    agree = 0;
    shown = null;
    shownAt = 0;
  }

  return {
    load,
    classify,
    classifyStill,
    current,
    reset,
    /** Usable right now: loaded, and not stood down for being too slow. */
    get ready() { return !!model && !disabled; },
    get loaded() { return !!model; },
    get disabled() { return disabled; },
    get latency() { return Math.round(latency); },
    get gap() { return Math.round(gapMs()); }
  };
}
