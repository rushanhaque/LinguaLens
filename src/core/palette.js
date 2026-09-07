/**
 * palette.js — Dominant-colour extraction from a video frame region.
 *
 * For each tracked object we crop the middle of its bounding box, downsample it
 * hard, and take the modal colour bucket. Cropping to the centre matters: the
 * corners of a box are mostly background, and background is what makes naive
 * colour readings wrong.
 *
 * The result is deliberately conservative — when no bucket clearly wins, we
 * report nothing rather than teach the learner a colour that isn't there.
 */

import { COLOR_KEYS } from '../data/colors.js';

const SAMPLE = 24;          // crop is drawn into a SAMPLE x SAMPLE canvas
const INSET = 0.24;         // trim this fraction from each edge of the box
const MIN_SHARE = 0.30;     // winning bucket must hold this share of pixels
const MIN_MARGIN = 1.25;    // ...and beat the runner-up by this factor
const SMOOTH_FRAMES = 3;    // consecutive agreeing reads before we commit

/** sRGB → HSV, with h in degrees and s/v in 0..1. */
function rgbToHsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const d = max - min;
  let h = 0;
  if (d !== 0) {
    if (max === r) h = ((g - b) / d) % 6;
    else if (max === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h *= 60;
    if (h < 0) h += 360;
  }
  return { h, s: max === 0 ? 0 : d / max, v: max };
}

/**
 * Bucket one pixel into a colour name.
 * Achromatic checks come first — a very dark pixel is black whatever its hue.
 */
function bucketOf(r, g, b) {
  const { h, s, v } = rgbToHsv(r, g, b);

  if (v < 0.17) return 'black';
  if (s < 0.14) return v > 0.80 ? 'white' : v > 0.34 ? 'grey' : 'black';

  // Dark, desaturated warm hues read as brown rather than dim orange.
  if (h >= 8 && h < 48 && v < 0.62 && s > 0.20) return 'brown';

  if (h < 12 || h >= 344) return 'red';
  if (h < 42) return 'orange';
  if (h < 68) return 'yellow';
  if (h < 165) return 'green';
  if (h < 255) return 'blue';
  if (h < 295) return 'purple';
  return 'pink';
}

export function createPalette() {
  const canvas = document.createElement('canvas');
  canvas.width = SAMPLE;
  canvas.height = SAMPLE;
  // willReadFrequently keeps getImageData on the fast path.
  const ctx = canvas.getContext('2d', { willReadFrequently: true });

  /** trackId → { key, hits } — colours must agree across frames to stick. */
  const history = new Map();

  /**
   * Read the dominant colour inside a bounding box.
   * @param {HTMLVideoElement} source
   * @param {number[]} bbox  [x, y, w, h] in source pixels
   * @returns {{key:string, share:number}|null}
   */
  function read(source, bbox) {
    const [bx, by, bw, bh] = bbox;
    const sx = bx + bw * INSET;
    const sy = by + bh * INSET;
    const sw = bw * (1 - INSET * 2);
    const sh = bh * (1 - INSET * 2);
    if (sw < 4 || sh < 4) return null;

    try {
      ctx.clearRect(0, 0, SAMPLE, SAMPLE);
      ctx.drawImage(source, sx, sy, sw, sh, 0, 0, SAMPLE, SAMPLE);
    } catch {
      return null;           // frame not decodable yet, or tainted canvas
    }

    let data;
    try { data = ctx.getImageData(0, 0, SAMPLE, SAMPLE).data; }
    catch { return null; }

    const counts = Object.create(null);
    let total = 0;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i + 3] < 128) continue;
      const key = bucketOf(data[i], data[i + 1], data[i + 2]);
      counts[key] = (counts[key] || 0) + 1;
      total++;
    }
    if (!total) return null;

    let best = null, bestN = 0, secondN = 0;
    for (const key of COLOR_KEYS) {
      const n = counts[key] || 0;
      if (n > bestN) { secondN = bestN; best = key; bestN = n; }
      else if (n > secondN) secondN = n;
    }

    const share = bestN / total;
    if (!best || share < MIN_SHARE) return null;
    if (secondN > 0 && bestN / secondN < MIN_MARGIN) return null;
    return { key: best, share };
  }

  /**
   * Read a colour and only report it once consecutive frames agree, so a
   * label never flickers between "red" and "orange" as the object moves.
   */
  function readStable(source, bbox, trackId) {
    const now = read(source, bbox);
    const prev = history.get(trackId);

    if (!now) {
      if (prev) prev.hits = Math.max(0, prev.hits - 1);
      return prev && prev.hits >= SMOOTH_FRAMES ? prev.key : null;
    }
    if (prev && prev.key === now.key) {
      prev.hits = Math.min(SMOOTH_FRAMES + 2, prev.hits + 1);
    } else {
      history.set(trackId, { key: now.key, hits: 1 });
      return prev && prev.hits > SMOOTH_FRAMES ? prev.key : null;
    }
    return prev.hits >= SMOOTH_FRAMES ? prev.key : null;
  }

  /** Drop history for tracks that no longer exist. */
  function prune(liveIds) {
    for (const id of history.keys()) if (!liveIds.has(id)) history.delete(id);
  }

  function reset() { history.clear(); }

  return { read, readStable, prune, reset };
}
