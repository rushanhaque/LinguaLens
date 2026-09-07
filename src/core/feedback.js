/**
 * feedback.js — Haptics and UI sound.
 *
 * Both respect their settings toggle and degrade silently where the platform
 * does not support them (Vibration API is absent on iOS Safari, for example).
 */

import { state } from './store.js';

/* ── Haptics ──────────────────────────────────────────────────────────── */

const PATTERNS = {
  light: 8,
  medium: 16,
  heavy: 28,
  select: 5,
  success: [12, 40, 22],
  warning: [20, 60, 20],
  error: [28, 50, 28, 50, 28]
};

export function haptic(kind = 'light') {
  if (!state.settings.haptics) return;
  if (!('vibrate' in navigator)) return;
  try { navigator.vibrate(PATTERNS[kind] ?? PATTERNS.light); } catch { /* blocked */ }
}

/* ── Sound ────────────────────────────────────────────────────────────────
   Tones are synthesised rather than shipped as files: no network cost, no
   decode latency, and they scale to any pitch we want. */

let ctx = null;

function audio() {
  if (ctx) return ctx;
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) return null;
  try { ctx = new Ctx(); } catch { return null; }
  return ctx;
}

/** Browsers start the context suspended until a gesture; call this on first tap. */
export function unlockAudio() {
  const a = audio();
  if (a && a.state === 'suspended') a.resume().catch(() => {});
}

function tone(freq, duration = 0.09, type = 'sine', gain = 0.06, delay = 0) {
  const a = audio();
  if (!a) return;
  const t0 = a.currentTime + delay;
  const osc = a.createOscillator();
  const amp = a.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(freq, t0);
  // Short attack, exponential release — reads as a UI tick rather than a beep.
  amp.gain.setValueAtTime(0.0001, t0);
  amp.gain.exponentialRampToValueAtTime(gain, t0 + 0.012);
  amp.gain.exponentialRampToValueAtTime(0.0001, t0 + duration);
  osc.connect(amp).connect(a.destination);
  osc.start(t0);
  osc.stop(t0 + duration + 0.02);
}

const SOUNDS = {
  tap:      () => tone(880, 0.05, 'sine', 0.03),
  discover: () => { tone(659.25, 0.10); tone(987.77, 0.14, 'sine', 0.05, 0.07); },
  correct:  () => { tone(783.99, 0.09); tone(1174.66, 0.13, 'sine', 0.05, 0.06); },
  wrong:    () => { tone(220, 0.16, 'triangle', 0.05); },
  shutter:  () => { tone(1600, 0.03, 'square', 0.03); tone(900, 0.05, 'square', 0.025, 0.03); },
  levelup:  () => { [523.25, 659.25, 783.99, 1046.5].forEach((f, i) => tone(f, 0.16, 'sine', 0.05, i * 0.075)); },
  badge:    () => { [659.25, 830.61, 1046.5].forEach((f, i) => tone(f, 0.18, 'triangle', 0.045, i * 0.09)); }
};

export function sfx(name) {
  if (!state.settings.sound) return;
  const fn = SOUNDS[name];
  if (fn) try { fn(); } catch { /* audio unavailable */ }
}

/** Convenience: the two channels almost always fire together. */
export function cue(soundName, hapticKind) {
  sfx(soundName);
  haptic(hapticKind);
}
