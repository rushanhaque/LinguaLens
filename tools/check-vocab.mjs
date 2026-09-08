/**
 * check-vocab.mjs — Validate the vocabulary tables and the ImageNet mapping.
 *
 *   node tools/check-vocab.mjs
 *
 * Catches the three mistakes that are easy to make when the dictionary grows:
 * an entry missing a language, a category that was never declared, and an
 * ImageNet rule pointing at a key that does not exist (which would silently
 * drop that whole family of detections).
 */

import { readFile } from 'node:fs/promises';
import { DICT, CATEGORIES, ALL_CLASSES, COCO_CLASSES } from '../src/data/dictionary.js';
import { EXTRA_DICT } from '../src/data/extras.js';
import { mapImagenet } from '../src/data/imagenet.js';

const LANGS = ['es', 'fr', 'de', 'it', 'pt', 'nl', 'ru', 'ja', 'ko', 'zh', 'hi', 'ar'];
const SIZES = ['tiny', 'small', 'medium', 'large', 'huge'];
let fail = 0;

console.log(`COCO classes   ${COCO_CLASSES.length}`);
console.log(`Extra classes  ${Object.keys(EXTRA_DICT).length}`);
console.log(`Total entries  ${ALL_CLASSES.length}`);
console.log(`Categories     ${Object.keys(CATEGORIES).length}`);
console.log(`Translations   ${ALL_CLASSES.length * LANGS.length}\n`);

/* 1. Every entry carries complete metadata and all twelve languages. */
for (const [key, e] of Object.entries(DICT)) {
  if (!e.em) { console.log(`no emoji: ${key}`); fail++; }
  if (!CATEGORIES[e.cat]) { console.log(`bad category: ${key} -> ${e.cat}`); fail++; }
  if (!SIZES.includes(e.size)) { console.log(`bad size: ${key} -> ${e.size}`); fail++; }
  if (![1, 2, 3].includes(e.lvl)) { console.log(`bad level: ${key} -> ${e.lvl}`); fail++; }
  for (const lang of LANGS) {
    const t = e.t?.[lang];
    if (!Array.isArray(t) || t.length !== 3 || !t[0] || !t[1]) {
      console.log(`bad translation: ${key}.${lang}`); fail++;
    }
  }
}

/* 2. Every key an ImageNet rule can produce must exist in DICT — otherwise
      that mapping is dead weight and the detection is silently discarded. */
const src = await readFile(new URL('../src/data/imagenet.js', import.meta.url), 'utf8');
const targets = new Set();
for (const m of src.matchAll(/:\s*'([a-z][a-z ]*)'/g)) targets.add(m[1]);
for (const m of src.matchAll(/\/,\s*'([a-z][a-z ]*)'\]/g)) targets.add(m[1]);
const unresolved = [...targets].filter((t) => !DICT[t]);
if (unresolved.length) {
  console.log(`\nUnresolved mapping targets (${unresolved.length}):`);
  for (const t of unresolved) console.log(`  ${t}`);
  fail += unresolved.length;
}

/* 3. Spot-check real ImageNet labels end to end. */
const SAMPLES = [
  'ballpoint, ballpoint pen, ballpen, Biro', 'notebook, notebook computer',
  'golden retriever', 'tabby, tabby cat', 'Granny Smith', 'coffee mug',
  'desk', 'rule, ruler', 'pencil sharpener', 'wall clock', 'sports car',
  'king penguin, Aptenodytes patagonica', 'binder, ring-binder', 'projector',
  'printer', 'analog clock', 'running shoe', 'sunglasses, dark glasses, shades',
  'rubber eraser, rubber, pencil eraser', 'bookcase', 'water bottle',
  'African elephant, Loxodonta africana', 'monarch, monarch butterfly',
  'space heater', 'electric guitar', 'mountain bike, all-terrain bike'
];
console.log('\nSample ImageNet mappings');
let mapped = 0;
for (const label of SAMPLES) {
  const key = mapImagenet(label);
  const status = key ? (DICT[key] ? 'ok  ' : 'MISS') : '--  ';
  if (key && DICT[key]) mapped++;
  if (key && !DICT[key]) fail++;
  console.log(`  ${status} ${label.slice(0, 40).padEnd(42)} ${key || ''}`);
}
console.log(`\n${mapped}/${SAMPLES.length} samples resolved`);

console.log(fail ? `\nFAILED — ${fail} problem(s)` : '\nAll vocabulary checks passed');
process.exit(fail ? 1 : 0);
