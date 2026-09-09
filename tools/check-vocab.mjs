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

/* 4. Coverage against the real ImageNet-1k label set.
      The list is extracted from the MobileNet bundle itself and cached beside
      this script, so the audit measures what the model can actually emit
      rather than what we remember it emitting. Run with --refresh to refetch. */
const CACHE = new URL('./imagenet-classes.json', import.meta.url);
const BUNDLE = 'https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@2.1.1/dist/mobilenet.min.js';

async function imagenetClasses() {
  if (!process.argv.includes('--refresh')) {
    try { return JSON.parse(await readFile(CACHE, 'utf8')); } catch { /* fetch it */ }
  }
  const js = await (await fetch(BUNDLE)).text();
  const byIndex = {};
  for (const m of js.matchAll(/(\d{1,3}):"((?:[^"\\]|\\.)*)"/g)) {
    const i = Number(m[1]);
    if (i >= 0 && i <= 999 && m[2].length > 1) byIndex[i] = m[2];
  }
  const list = Object.keys(byIndex).map(Number).sort((a, b) => a - b).map((i) => byIndex[i]);
  if (list.length === 1000) {
    await (await import('node:fs/promises')).writeFile(CACHE, JSON.stringify(list, null, 0));
  }
  return list;
}

try {
  const classes = await imagenetClasses();
  if (classes.length !== 1000) {
    console.log(`\nImageNet list looks wrong (${classes.length} entries) — skipping coverage`);
  } else {
    /* Deliberately unmapped. Firearms and instruments of execution have no
       place in a classroom vocabulary app, and the rest are words too
       ambiguous to name safely — ImageNet's "bow" is a weapon, a ribbon and
       a violin bow at once. Listing them here keeps the coverage report
       honest: these are choices, not gaps. */
    const EXCLUDED = new Set([
      'assault rifle', 'revolver', 'rifle', 'missile', 'projectile',
      'guillotine', 'nipple', 'bow', 'brass', 'pole', 'manhole cover'
    ]);
    const missed = classes.filter((c) => { const k = mapImagenet(c); return !k || !DICT[k]; });
    const unexpected = missed.filter((c) => !EXCLUDED.has(c.split(',')[0]));
    const pct = ((classes.length - missed.length) / classes.length) * 100;
    console.log(`\nImageNet coverage  ${classes.length - missed.length}/1000  (${pct.toFixed(1)}%)`);
    console.log(`  ${missed.length - unexpected.length} excluded by policy, ${unexpected.length} unaccounted for`);
    if (unexpected.length) {
      console.log('Unaccounted for:');
      console.log('  ' + unexpected.map((c) => c.split(',')[0]).join(', '));
    }
    // Coverage is a quality bar, not a correctness one: some ImageNet classes
    // are deliberately unmapped, so this warns rather than fails.
    if (pct < 90) console.log('\nWarning: coverage below 90%');
  }
} catch (e) {
  console.log(`\nCoverage check skipped (${e.message})`);
}

console.log(fail ? `\nFAILED — ${fail} problem(s)` : '\nAll vocabulary checks passed');
// Set the code rather than calling process.exit(): an outstanding fetch handle
// makes an immediate exit abort with a libuv assertion on Windows.
process.exitCode = fail ? 1 : 0;
