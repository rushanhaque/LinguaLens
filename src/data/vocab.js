/**
 * vocab.js — One vocabulary surface over three sources.
 *
 * The app teaches from three places: objects the camera can recognise
 * (dictionary.js), core words it cannot point at (lexicon.js), and colours,
 * which carry agreement forms and so live in their own table (colors.js).
 * Rather than teaching the rest of the app about all three, everything
 * downstream — review scheduling, progress, the word sheet — addresses a
 * single namespaced id:
 *
 *   obj:apple      a COCO class, discoverable through the camera
 *   lex:mother     a lexicon entry, studied directly
 *   col:red        a colour, readable off any object the camera sees
 *
 * That keeps one review queue, one progress store, and one set of stats.
 */

import { DICT, CATEGORIES, ALL_CLASSES } from './dictionary.js';
import { LEXICON, PACKS, LEXICON_KEYS } from './lexicon.js';
import { COLORS, COLOR_KEYS } from './colors.js';

export const SOURCE_OBJ = 'obj';
export const SOURCE_LEX = 'lex';
export const SOURCE_COL = 'col';

export const objId = (cls) => `${SOURCE_OBJ}:${cls}`;
export const lexId = (id) => `${SOURCE_LEX}:${id}`;
export const colId = (key) => `${SOURCE_COL}:${key}`;

/** Colours live in their own table because they carry agreement forms. */
export const COLOR_GROUP = 'colours';

/** Split an item id into its source and key. */
export function parseId(id) {
  const i = String(id).indexOf(':');
  if (i === -1) return { source: SOURCE_OBJ, key: String(id) };   // pre-v4 data
  return { source: id.slice(0, i), key: id.slice(i + 1) };
}

export const OBJECT_IDS = ALL_CLASSES.map(objId);
export const LEXICON_IDS = LEXICON_KEYS.map(lexId);
export const COLOR_IDS = COLOR_KEYS.map(colId);
export const ALL_ITEM_IDS = [...OBJECT_IDS, ...LEXICON_IDS, ...COLOR_IDS];

/**
 * Groups unify object categories and lexicon packs into one browsable set.
 * Category keys and pack keys are disjoint by construction — the assertion in
 * the test suite keeps them that way.
 */
export const GROUPS = {};
for (const [key, cat] of Object.entries(CATEGORIES)) {
  GROUPS[key] = { key, label: cat.label, em: cat.em, color: cat.color, source: SOURCE_OBJ, blurb: 'Point your camera at these' };
}
for (const [key, pack] of Object.entries(PACKS)) {
  GROUPS[key] = { key, label: pack.label, em: pack.em, color: pack.color, source: SOURCE_LEX, blurb: pack.blurb };
}
GROUPS[COLOR_GROUP] = {
  key: COLOR_GROUP, label: 'Colours', em: '🎨', color: '#A66F78',
  source: SOURCE_COL, blurb: 'Adjectives that agree'
};

export const GROUP_KEYS = Object.keys(GROUPS);

/** Static facts about an item, independent of language. */
export function itemMeta(id) {
  const { source, key } = parseId(id);
  if (source === SOURCE_COL) {
    const c = COLORS[key];
    if (!c) return null;
    return {
      id: colId(key), source, key, en: key, em: c.em,
      group: COLOR_GROUP, lvl: 1, camera: false, swatch: c.swatch
    };
  }
  if (source === SOURCE_LEX) {
    const e = LEXICON[key];
    if (!e) return null;
    return {
      id: lexId(key), source, key, en: e.en, em: e.em,
      group: e.pack, lvl: 1, camera: false
    };
  }
  const e = DICT[key];
  if (!e) return null;
  return {
    id: objId(key), source, key, en: key, em: e.em,
    group: e.cat, lvl: e.lvl, camera: true
  };
}

/** Translation of an item into one language, or null when unavailable. */
export function translateItem(id, lang) {
  const { source, key } = parseId(id);
  if (source === SOURCE_COL) {
    const c = COLORS[key];
    const t = c && c.t[lang];
    if (!t) return null;
    // Colours are taught in their citation form; agreement is demonstrated in
    // context by buildColorPhrase() rather than baked into the headword.
    return { word: t.cite, phonetic: t.ph, gender: '', em: c.em, en: key };
  }
  const entry = source === SOURCE_LEX ? LEXICON[key] : DICT[key];
  if (!entry) return null;
  const t = entry.t[lang];
  if (!t) return null;
  return {
    word: t[0],
    phonetic: t[1],
    gender: t[2],
    em: entry.em,
    en: source === SOURCE_LEX ? entry.en : key
  };
}

/** Every item id in a group. */
export function idsInGroup(group) {
  const g = GROUPS[group];
  if (!g) return [];
  if (g.source === SOURCE_COL) return COLOR_IDS;
  return g.source === SOURCE_LEX
    ? LEXICON_KEYS.filter((k) => LEXICON[k].pack === group).map(lexId)
    : ALL_CLASSES.filter((c) => DICT[c].cat === group).map(objId);
}

/** True when the item is something the camera can actually find. */
export function isCameraItem(id) {
  return parseId(id).source === SOURCE_OBJ;
}

/**
 * Case- and accent-insensitive search across English glosses and every
 * translation of the active language.
 */
export function searchItems(query, lang, limit = 40) {
  const q = normalise(query);
  if (!q) return [];
  const hits = [];
  for (const id of ALL_ITEM_IDS) {
    const meta = itemMeta(id);
    const tr = translateItem(id, lang);
    if (!meta || !tr) continue;
    const en = normalise(meta.en);
    const target = normalise(tr.word);
    const phon = normalise(tr.phonetic || '');
    let score = 0;
    if (en === q || target === q) score = 100;
    else if (en.startsWith(q) || target.startsWith(q)) score = 70;
    else if (en.includes(q) || target.includes(q)) score = 45;
    else if (phon.includes(q)) score = 25;
    if (score) hits.push({ id, score, meta, tr });
  }
  hits.sort((a, b) => b.score - a.score || a.meta.en.localeCompare(b.meta.en));
  return hits.slice(0, limit);
}

function normalise(s) {
  return String(s || '')
    .toLowerCase()
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .trim();
}

export const TOTAL_ITEMS = ALL_ITEM_IDS.length;
export const TOTAL_OBJECTS = OBJECT_IDS.length;
export const TOTAL_LEXEMES = LEXICON_IDS.length + COLOR_IDS.length;
