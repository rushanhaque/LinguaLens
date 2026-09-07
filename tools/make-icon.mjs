/**
 * make-icon.mjs — Generate the Lemma app icon as a PNG, with no dependencies.
 *
 *   node tools/make-icon.mjs
 *
 * Draws the aperture mark into an RGBA buffer with 4x supersampling, then
 * writes a minimal PNG (IHDR/IDAT/IEND) using Node's built-in zlib. Kept in
 * the repo so the icon is reproducible rather than an opaque binary.
 *
 * The artwork is full-bleed on purpose: iOS applies its own squircle to
 * apple-touch-icon, and the PWA "maskable" purpose crops to a safe circle, so
 * transparent rounded corners would show through as notches on some platforms.
 */

import { deflateSync } from 'node:zlib';
import { writeFileSync } from 'node:fs';

const SIZE = 512;
const SS = 4;                      // supersampling factor per axis

const GROUND = [0x5e, 0x6e, 0x4c];   // sage
const MARK = [0xf4, 0xf1, 0xe8];     // paper cream

/* ── Geometry, in icon units (0..SIZE) ─────────────────────────────────────
   A lens ring over two lines of text: "look it up by looking". Radial blades
   were tried first and read as a wagon wheel at any size, so the mark keeps
   only shapes that stay legible when the icon is 32px on a home screen. */
const cx = SIZE / 2;
const cy = SIZE / 2;
const RING_R = 140;                 // lens ring radius
const RING_W = 24;                  // ring stroke width
const BAR_H = 24;                   // text-line thickness
// Left-aligned from a shared margin so the pair reads as lines of text
// rather than as an equals sign.
const BAR_X0 = cx - 80;
const BARS = [
  { dy: -30, x1: cx + 80 },         // full line
  { dy: 30, x1: cx + 12 }           // short last line
];

/** Distance from point p to the segment ab. */
function distToSegment(px, py, ax, ay, bx, by) {
  const dx = bx - ax;
  const dy = by - ay;
  const len2 = dx * dx + dy * dy;
  let t = len2 === 0 ? 0 : ((px - ax) * dx + (py - ay) * dy) / len2;
  t = Math.max(0, Math.min(1, t));
  const qx = ax + t * dx;
  const qy = ay + t * dy;
  return Math.hypot(px - qx, py - qy);
}

/** True when the sample point lies on the cream mark. */
function isMark(x, y) {
  const d = Math.hypot(x - cx, y - cy);
  if (Math.abs(d - RING_R) <= RING_W / 2) return true;   // lens ring
  for (const bar of BARS) {                              // text lines, round caps
    const y0 = cy + bar.dy;
    if (distToSegment(x, y, BAR_X0, y0, bar.x1, y0) <= BAR_H / 2) return true;
  }
  return false;
}

/* ── Rasterise ─────────────────────────────────────────────────────────── */
const raw = Buffer.alloc(SIZE * (SIZE * 4 + 1));   // +1 filter byte per row
let p = 0;
const step = 1 / SS;
const offset = step / 2;

for (let y = 0; y < SIZE; y++) {
  raw[p++] = 0;                                    // filter: None
  for (let x = 0; x < SIZE; x++) {
    let hits = 0;
    for (let sy = 0; sy < SS; sy++) {
      for (let sx = 0; sx < SS; sx++) {
        if (isMark(x + sx * step + offset, y + sy * step + offset)) hits++;
      }
    }
    const a = hits / (SS * SS);
    // Composite cream over sage; the icon is fully opaque everywhere.
    raw[p++] = Math.round(GROUND[0] + (MARK[0] - GROUND[0]) * a);
    raw[p++] = Math.round(GROUND[1] + (MARK[1] - GROUND[1]) * a);
    raw[p++] = Math.round(GROUND[2] + (MARK[2] - GROUND[2]) * a);
    raw[p++] = 255;
  }
}

/* ── PNG container ─────────────────────────────────────────────────────── */
const CRC_TABLE = (() => {
  const t = new Int32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c;
  }
  return t;
})();

function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length, 0);
  const body = Buffer.concat([Buffer.from(type, 'latin1'), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body), 0);
  return Buffer.concat([len, body, crc]);
}

const ihdr = Buffer.alloc(13);
ihdr.writeUInt32BE(SIZE, 0);
ihdr.writeUInt32BE(SIZE, 4);
ihdr[8] = 8;    // bit depth
ihdr[9] = 6;    // colour type: RGBA
ihdr[10] = 0;   // compression
ihdr[11] = 0;   // filter
ihdr[12] = 0;   // interlace

const png = Buffer.concat([
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
  chunk('IHDR', ihdr),
  chunk('IDAT', deflateSync(raw, { level: 9 })),
  chunk('IEND', Buffer.alloc(0))
]);

writeFileSync(new URL('../icon-512.png', import.meta.url), png);
console.log(`icon-512.png written — ${SIZE}x${SIZE}, ${(png.length / 1024).toFixed(1)} KB`);
