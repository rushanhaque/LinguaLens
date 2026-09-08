/**
 * scene.js — Cheap whole-frame motion estimation.
 *
 * The tracker can only reason about the boxes the model hands it. When the
 * camera swings away from a desk, the model simply stops reporting the laptop
 * — and "stopped reporting" looks exactly like "briefly occluded", which the
 * tracker is deliberately tolerant of. The result was labels that outlived
 * their subject by a second or more.
 *
 * Watching the frame itself resolves the ambiguity. A pan moves every pixel;
 * an occlusion moves a few. So we downscale each frame to a thumbnail, compare
 * luma against the previous one, and let the camera view decide how hard to
 * age its tracks.
 *
 * The thumbnail is deliberately tiny: at 32x24 the whole comparison is 768
 * byte subtractions, which is free next to a single inference pass.
 */

const W = 32;
const H = 24;

/** Mean absolute luma difference above which the scene is a different place. */
export const SCENE_CUT = 0.17;
/** Above this the camera is moving enough that boxes are no longer trustworthy. */
export const SCENE_PAN = 0.085;

export function createSceneWatch() {
  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });

  let prev = null;
  let smoothed = 0;

  /**
   * Sample the current frame.
   * @returns {number} mean absolute luma change since the last sample, 0..1
   */
  function sample(video) {
    if (!video || !video.videoWidth) return 0;
    let data;
    try {
      ctx.drawImage(video, 0, 0, W, H);
      data = ctx.getImageData(0, 0, W, H).data;
    } catch {
      return 0;                       // tainted or not yet decodable
    }

    const cur = new Uint8Array(W * H);
    for (let i = 0, p = 0; p < cur.length; i += 4, p++) {
      // Integer luma (ITU-R BT.601), kept in fixed point to avoid float work.
      cur[p] = (data[i] * 77 + data[i + 1] * 150 + data[i + 2] * 29) >> 8;
    }

    if (!prev) { prev = cur; return 0; }

    let sum = 0;
    for (let p = 0; p < cur.length; p++) sum += Math.abs(cur[p] - prev[p]);
    prev = cur;

    const diff = sum / cur.length / 255;
    // A light EMA stops a single noisy frame from triggering a scene cut.
    smoothed = smoothed * 0.35 + diff * 0.65;
    return smoothed;
  }

  function reset() { prev = null; smoothed = 0; }

  return { sample, reset, get level() { return smoothed; } };
}
