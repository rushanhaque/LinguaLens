/**
 * speech.js — Text-to-speech pronunciation.
 *
 * Voice lists load asynchronously and differ wildly across platforms, so the
 * picker scores candidates rather than assuming an exact locale match, and
 * reports when a language has no voice at all.
 */

import { LANGUAGES } from '../data/languages.js';
import { state } from './store.js';

let voices = [];
let ready = false;
const waiters = [];

function refresh() {
  if (!('speechSynthesis' in window)) return;
  voices = speechSynthesis.getVoices() || [];
  if (voices.length && !ready) {
    ready = true;
    waiters.splice(0).forEach((fn) => fn());
  }
}

if ('speechSynthesis' in window) {
  refresh();
  speechSynthesis.addEventListener('voiceschanged', refresh);
  // Chrome sometimes never fires voiceschanged; poll briefly as a safety net.
  let tries = 0;
  const poll = setInterval(() => {
    refresh();
    if (ready || ++tries > 20) clearInterval(poll);
  }, 250);
}

export function isSupported() {
  return 'speechSynthesis' in window && 'SpeechSynthesisUtterance' in window;
}

/** Resolve once the voice list is populated (or immediately if it already is). */
export function whenReady() {
  if (ready || !isSupported()) return Promise.resolve();
  return new Promise((res) => waiters.push(res));
}

/**
 * Best available voice for a language code.
 * Prefers an exact locale from the language's preference list, then any voice
 * whose lang starts with the base code, then a local (offline) voice.
 */
export function pickVoice(lang) {
  const meta = LANGUAGES[lang];
  if (!meta || !voices.length) return null;

  for (const locale of meta.tts) {
    const exact = voices.filter((v) => v.lang.replace('_', '-').toLowerCase() === locale.toLowerCase());
    if (exact.length) return exact.find((v) => v.localService) || exact[0];
  }
  const base = voices.filter((v) => v.lang.replace('_', '-').toLowerCase().startsWith(lang.toLowerCase()));
  if (base.length) return base.find((v) => v.localService) || base[0];
  return null;
}

/** Every installed voice for a language — powers the Settings voice picker. */
export function voicesFor(lang) {
  return voices.filter((v) => v.lang.replace('_', '-').toLowerCase().startsWith(lang.toLowerCase()));
}

export function hasVoice(lang) {
  return !!pickVoice(lang);
}

let current = null;

/**
 * Speak text in a language.
 * @returns {boolean} false when no voice exists for that language.
 */
export function speak(text, lang = state.settings.targetLang, opts = {}) {
  if (!isSupported() || !text) return false;
  try { speechSynthesis.cancel(); } catch { /* nothing queued */ }

  const u = new SpeechSynthesisUtterance(String(text));
  const voice = pickVoice(lang);
  const meta = LANGUAGES[lang];

  if (voice) u.voice = voice;
  u.lang = voice ? voice.lang : (meta ? meta.tts[0] : lang);
  u.rate = opts.rate ?? state.settings.speechRate;
  u.pitch = opts.pitch ?? 1;
  u.volume = opts.volume ?? 1;

  current = u;
  u.addEventListener('end', () => { if (current === u) current = null; });

  try { speechSynthesis.speak(u); } catch { return false; }
  return !!voice;
}

export function stop() {
  if (!isSupported()) return;
  try { speechSynthesis.cancel(); } catch { /* nothing queued */ }
  current = null;
}

/** Languages with no installed voice — surfaced as a hint in Settings. */
export function missingVoices() {
  return Object.keys(LANGUAGES).filter((l) => !hasVoice(l));
}
