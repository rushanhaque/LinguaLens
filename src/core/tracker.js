/**
 * tracker.js — Detection filtering and multi-object tracking.
 *
 * COCO-SSD emits an unordered, noisy list of boxes every frame. This module
 * turns that into stable, identity-carrying tracks:
 *
 *   1. Aspect-ratio scoring   penalise boxes shaped wrong for their class
 *   2. Size gating            reject boxes far outside a class's plausible area
 *   3. NMS                    drop overlapping duplicates
 *   4. Confusion resolution   pick one winner among look-alike classes
 *   5. Assignment             greedily match detections to existing tracks
 *   6. Confirmation & decay   require N hits to show, tolerate M misses
 *   7. Smoothing              exponential position/size smoothing per track
 *
 * The prototype keyed everything by class name, so two cats collapsed into one
 * label and a class flip teleported the box. Tracks fix both.
 */

import { DICT, SIZE_RANGES } from '../data/dictionary.js';

/* ── Tuning ───────────────────────────────────────────────────────────── */
const CONFIRM_HITS = 3;      // detection passes before a track is shown

/* Staleness is governed by three independent limits, and whichever trips first
   wins. Counting passes alone was the bug behind labels that outlived their
   object: at 4 Hz a twelve-pass grace period kept a label on screen for three
   seconds after the camera had moved on. Wall-clock is what a person actually
   perceives, so MAX_AGE_MS is the real guarantee and the pass counts merely
   catch the common case sooner. */
const MAX_MISSES = 3;        // passes a lost track survives before deletion
const SHOW_MISSES = 1;       // passes it keeps being drawn (always < MAX_MISSES)
const MAX_AGE_MS = 500;      // hard ceiling on time since the last real measurement

const NMS_IOU = 0.45;
const CONFUSE_IOU = 0.30;
const MATCH_IOU = 0.28;      // same-class detection↔track association floor
/* A detection of a *different* class needs far more overlap to capture an
   existing track. Without this a "laptop" track would happily absorb whatever
   box next appeared in the same corner of the frame and keep its old label. */
const MATCH_IOU_CROSS = 0.55;
const CLASS_LOCK_HITS = 14;  // hits after which a track resists relabelling
const CLASS_STEAL_RATIO = 1.25; // how much better a rival class must score
const SMOOTH_POS = 0.40;     // higher = snappier, lower = calmer
const SMOOTH_SIZE = 0.28;
const VEL_SMOOTH = 0.45;    // velocity EMA weight
const MAX_PREDICT_MS = 180; // never extrapolate further than this ahead

/* Classes the model routinely swaps. Only one of a group may occupy a spot. */
const CONFUSE_GROUPS = [
  ['cell phone', 'remote', 'mouse', 'hair drier', 'book'],
  ['cup', 'bowl', 'vase', 'bottle', 'wine glass'],
  ['knife', 'scissors', 'fork', 'spoon'],
  ['couch', 'bed', 'bench', 'chair'],
  ['skateboard', 'surfboard', 'snowboard', 'skis'],
  ['laptop', 'tv', 'keyboard'],
  ['backpack', 'handbag', 'suitcase'],
  ['car', 'truck', 'bus'],
  ['dining table', 'bench'],
  ['potted plant', 'broccoli'],
  ['orange', 'apple', 'sports ball'],
  ['donut', 'cake', 'pizza']
];

const confuseMap = new Map();
CONFUSE_GROUPS.forEach((g, i) => g.forEach((c) => {
  // A class can sit in several groups; keep a set of group ids per class.
  if (!confuseMap.has(c)) confuseMap.set(c, new Set());
  confuseMap.get(c).add(i);
}));

function sharesGroup(a, b) {
  const ga = confuseMap.get(a); const gb = confuseMap.get(b);
  if (!ga || !gb) return false;
  for (const id of ga) if (gb.has(id)) return true;
  return false;
}

/* Expected height/width ratios. `weight` scales the penalty for being outside. */
const ASPECT_HINTS = {
  'cell phone':  { min: 1.2, max: 2.9, weight: 0.25 },
  'remote':      { min: 2.0, max: 6.0, weight: 0.25 },
  'mouse':       { min: 0.5, max: 1.3, weight: 0.22 },
  'bottle':      { min: 1.7, max: 5.0, weight: 0.25 },
  'wine glass':  { min: 1.4, max: 3.2, weight: 0.20 },
  'cup':         { min: 0.6, max: 1.7, weight: 0.20 },
  'bowl':        { min: 0.3, max: 1.0, weight: 0.20 },
  'vase':        { min: 1.1, max: 3.5, weight: 0.16 },
  'laptop':      { min: 0.45, max: 1.2, weight: 0.20 },
  'tv':          { min: 0.4, max: 0.95, weight: 0.20 },
  'book':        { min: 0.6, max: 1.9, weight: 0.10 },
  'knife':       { min: 1.8, max: 8.0, weight: 0.20 },
  'scissors':    { min: 0.7, max: 2.6, weight: 0.14 },
  'keyboard':    { min: 0.12, max: 0.60, weight: 0.24 },
  'hair drier':  { min: 0.6, max: 1.6, weight: 0.14 },
  'fork':        { min: 2.5, max: 10.0, weight: 0.20 },
  'spoon':       { min: 2.2, max: 8.0, weight: 0.16 },
  'person':      { min: 0.8, max: 4.5, weight: 0.14 },
  'banana':      { min: 0.25, max: 2.2, weight: 0.10 },
  'dining table':{ min: 0.25, max: 1.2, weight: 0.12 },
  'toothbrush':  { min: 1.8, max: 9.0, weight: 0.18 }
};

/* ── Geometry ─────────────────────────────────────────────────────────── */

export function iou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[0] + a[2], b[0] + b[2]);
  const y2 = Math.min(a[1] + a[3], b[1] + b[3]);
  if (x2 <= x1 || y2 <= y1) return 0;
  const inter = (x2 - x1) * (y2 - y1);
  const union = a[2] * a[3] + b[2] * b[3] - inter;
  return union > 0 ? inter / union : 0;
}

function aspectScore(cls, bbox) {
  const hint = ASPECT_HINTS[cls];
  if (!hint) return 1;
  const [, , w, h] = bbox;
  if (w <= 0 || h <= 0) return 0.5;
  const ratio = h / w;
  if (ratio >= hint.min && ratio <= hint.max) return 1;
  const dist = ratio < hint.min ? (hint.min - ratio) / hint.min : (ratio - hint.max) / hint.max;
  return Math.max(0.30, 1 - dist * hint.weight * 3);
}

function sizeOk(cls, bbox, frameArea) {
  const entry = DICT[cls];
  if (!entry || !frameArea) return true;
  const range = SIZE_RANGES[entry.size];
  if (!range) return true;
  const ratio = (bbox[2] * bbox[3]) / frameArea;
  // Ranges are advisory — widen generously so unusual framing still passes.
  return ratio >= range.min * 0.4 && ratio <= range.max * 2.2;
}

function lerp(a, b, t) { return a + (b - a) * t; }

/* ── Tracker ──────────────────────────────────────────────────────────── */

export function createTracker() {
  /** @type {Map<number, object>} */
  const tracks = new Map();
  let nextId = 1;

  function reset() { tracks.clear(); }

  /**
   * @param {Array<{class:string,score:number,bbox:number[]}>} raw
   * @param {number} frameW
   * @param {number} frameH
   * @param {object} opts { minScore, maxOut }
   * @returns {Array} confirmed tracks, best-first
   */
  function update(raw, frameW, frameH, opts = {}) {
    const minScore = opts.minScore ?? 0.5;
    const maxOut = opts.maxOut ?? 6;
    const frameArea = frameW * frameH;

    /* 1–2. Score by shape, drop implausible sizes and weak boxes. */
    let dets = [];
    for (const p of raw) {
      if (!DICT[p.class]) continue;
      if (!sizeOk(p.class, p.bbox, frameArea)) continue;
      const a = aspectScore(p.class, p.bbox);
      const score = p.score * a;
      if (score < minScore) continue;
      dets.push({ cls: p.class, bbox: p.bbox, score, rawScore: p.score, aspect: a });
    }
    dets.sort((x, y) => y.score - x.score);

    /* 3. NMS. */
    const kept = [];
    for (const d of dets) {
      if (kept.some((k) => iou(d.bbox, k.bbox) > NMS_IOU)) continue;
      kept.push(d);
    }

    /* 4. Confusion resolution — one winner per overlapping look-alike cluster. */
    const resolved = [];
    for (const d of kept) {
      const rivalIdx = resolved.findIndex(
        (r) => sharesGroup(d.cls, r.cls) && iou(d.bbox, r.bbox) > CONFUSE_IOU
      );
      if (rivalIdx === -1) { resolved.push(d); continue; }
      const rival = resolved[rivalIdx];
      // Prefer the better shape fit; the raw model score breaks ties.
      if (d.aspect * d.rawScore > rival.aspect * rival.rawScore) resolved[rivalIdx] = d;
    }

    /* 5. Assignment — greedy, highest-overlap first.
       Same-class matches beat cross-class ones so identity survives a flicker. */
    const unmatched = new Set(tracks.keys());
    const pairs = [];
    for (const d of resolved) {
      for (const [id, t] of tracks) {
        const ov = iou(d.bbox, t.bbox);
        const same = t.cls === d.cls;
        if (ov < (same ? MATCH_IOU : MATCH_IOU_CROSS)) continue;
        pairs.push({ d, id, ov, same });
      }
    }
    pairs.sort((a, b) => (b.same - a.same) || (b.ov - a.ov));

    const usedDets = new Set();
    const usedTracks = new Set();
    for (const p of pairs) {
      if (usedDets.has(p.d) || usedTracks.has(p.id)) continue;
      usedDets.add(p.d);
      usedTracks.add(p.id);
      unmatched.delete(p.id);
      absorb(tracks.get(p.id), p.d);
    }

    /* New tracks for anything left over. */
    for (const d of resolved) {
      if (usedDets.has(d)) continue;
      const id = nextId++;
      tracks.set(id, {
        id,
        cls: d.cls,
        bbox: [...d.bbox],
        raw: [...d.bbox],
        score: d.score,
        hits: 1,
        misses: 0,
        confirmed: false,
        vx: 0,
        vy: 0,
        stamp: performance.now(),
        firstSeen: performance.now()
      });
    }

    /* 6. Decay everything we did not see this pass. */
    for (const id of unmatched) {
      const t = tracks.get(id);
      t.misses++;
      if (t.misses > MAX_MISSES) tracks.delete(id);
    }

    /* 7. Expire on wall-clock, then emit what is still worth drawing.
       A track past SHOW_MISSES stops being emitted while it is still alive, so
       it can recover its identity if the object comes back within the grace
       period without ever having flickered a stale label on screen. */
    const now = performance.now();
    const out = [];
    for (const [id, t] of tracks) {
      if (now - t.stamp > MAX_AGE_MS) { tracks.delete(id); continue; }
      if (t.hits >= CONFIRM_HITS) t.confirmed = true;
      if (!t.confirmed || t.misses > SHOW_MISSES) continue;
      out.push(t);
    }
    out.sort((a, b) => b.score - a.score);
    return out.slice(0, maxOut);
  }

  /**
   * Age every track by one pass without feeding it a detection. The camera
   * view calls this when the frame changes a lot, so tracks that belong to a
   * scene we have panned away from die immediately rather than coasting
   * through their grace period on stale measurements.
   */
  function decay(passes = 1) {
    for (const [id, t] of tracks) {
      t.misses += passes;
      if (t.misses > MAX_MISSES) tracks.delete(id);
    }
  }

  /** Fold a detection into an existing track, with class-stability guarding. */
  function absorb(t, d) {
    if (t.cls !== d.cls) {
      // An established track only changes class if the rival is clearly better.
      const incumbent = aspectScore(t.cls, d.bbox) * t.score;
      const challenger = d.aspect * d.rawScore;
      const locked = t.hits >= CLASS_LOCK_HITS;
      if (!locked || challenger > incumbent * CLASS_STEAL_RATIO) {
        t.cls = d.cls;
        t.hits = Math.max(1, Math.floor(t.hits * 0.5));  // re-earn some trust
      }
    }
    const now = performance.now();
    const prevX = t.bbox[0];
    const prevY = t.bbox[1];

    t.raw = [...d.bbox];
    t.bbox = [
      lerp(t.bbox[0], d.bbox[0], SMOOTH_POS),
      lerp(t.bbox[1], d.bbox[1], SMOOTH_POS),
      lerp(t.bbox[2], d.bbox[2], SMOOTH_SIZE),
      lerp(t.bbox[3], d.bbox[3], SMOOTH_SIZE)
    ];

    // Velocity in px/ms, smoothed. Detection runs several times a second while
    // rendering runs at display refresh, so without extrapolation a label
    // visibly lags its object during a pan.
    const dt = now - t.stamp;
    if (dt > 8 && dt < 500) {
      const vx = (t.bbox[0] - prevX) / dt;
      const vy = (t.bbox[1] - prevY) / dt;
      t.vx = t.vx * (1 - VEL_SMOOTH) + vx * VEL_SMOOTH;
      t.vy = t.vy * (1 - VEL_SMOOTH) + vy * VEL_SMOOTH;
    }
    t.stamp = now;
    t.score = t.score * 0.7 + d.score * 0.3;
    t.hits++;
    t.misses = 0;
  }

  /**
   * Opacity for a fading track. Live tracks are fully opaque; a missed one
   * drops away steeply (squared falloff) and is additionally faded by how long
   * it has gone unmeasured, so a label never lingers at readable opacity over
   * something the camera is no longer looking at.
   */
  function fadeOf(t, now = performance.now()) {
    const byMiss = t.misses === 0 ? 1 : Math.max(0, 1 - t.misses / (SHOW_MISSES + 1)) ** 2;
    const byAge = Math.max(0, 1 - (now - t.stamp) / MAX_AGE_MS);
    return Math.min(byMiss, byAge);
  }

  /**
   * Where the box is *now*, extrapolated from its last measured position.
   * Extrapolation is capped so a track that stops being updated (occluded,
   * or the tab was backgrounded) drifts off screen instead of freezing.
   */
  function predict(t, now = performance.now()) {
    const dt = Math.min(MAX_PREDICT_MS, Math.max(0, now - t.stamp));
    if (!dt) return t.bbox;
    return [t.bbox[0] + t.vx * dt, t.bbox[1] + t.vy * dt, t.bbox[2], t.bbox[3]];
  }

  return { update, reset, decay, fadeOf, predict, get size() { return tracks.size; } };
}
