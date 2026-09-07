/**
 * settings.js — Preferences, data management, and about.
 */

import { LANGUAGES } from '../data/languages.js';
import {
  state, setSetting, exportData, importData, resetProgress, resetEverything, getStats, emit
} from '../core/store.js';
import { speak, missingVoices, isSupported as ttsSupported } from '../core/speech.js';
import { haptic, cue } from '../core/feedback.js';
import { $, $$, el, esc, toast, confirmAction } from '../ui/kit.js';
import { icon } from '../ui/icons.js';
import { APP_VERSION } from '../version.js';

let root = null;

const ACCENTS = [
  { id: 'lens', name: 'Lens', color: '#A8C93A' },
  { id: 'blue', name: 'Blue', color: '#0A84FF' },
  { id: 'purple', name: 'Purple', color: '#BF5AF2' },
  { id: 'pink', name: 'Pink', color: '#FF375F' },
  { id: 'orange', name: 'Orange', color: '#FF9F0A' },
  { id: 'teal', name: 'Teal', color: '#40C8E0' }
];

const REVIEW_STYLES = [
  { id: 'mixed', label: 'Mixed' },
  { id: 'recall', label: 'Recall' },
  { id: 'choice', label: 'Choice' },
  { id: 'listen', label: 'Listen' },
  { id: 'type', label: 'Type' }
];

export function initSettings(container) {
  root = container;
  renderSettings();
}

export function renderSettings() {
  if (!root) return;
  const s = state.settings;
  const lang = LANGUAGES[s.targetLang];
  const stats = getStats();
  const noVoice = ttsSupported() ? missingVoices() : Object.keys(LANGUAGES);

  root.innerHTML = `
    <header class="view-header"><h1 class="large-title grow">Settings</h1></header>
    <div class="scroll grow view-scroll">

      <section>
        <div class="section-label">Language</div>
        <div class="group">
          <button class="list-row is-tappable" id="pick-lang">
            <div class="list-row-icon" style="background:transparent;font-size:20px">${lang.flag}</div>
            <div class="list-row-body">
              <div class="list-row-title">Learning</div>
              <div class="list-row-sub">${esc(lang.name)} · ${esc(lang.native)}</div>
            </div>
            <div class="list-row-chevron">${icon('chevronRight')}</div>
          </button>
          ${row('Phonetic guide', 'Show pronunciation hints', toggle('showPhonetics'))}
          ${row('Gender & articles', 'Show le / la / der / die …', toggle('showGender'))}
        </div>
      </section>

      <section>
        <div class="section-label">Appearance</div>
        <div class="group">
          <div class="list-row">
            <div class="list-row-icon">${icon('moon')}</div>
            <div class="list-row-body"><div class="list-row-title">Theme</div></div>
            <div style="width:190px">${segmented('theme', [
              { id: 'auto', label: 'Auto' }, { id: 'light', label: 'Light' }, { id: 'dark', label: 'Dark' }
            ])}</div>
          </div>
          <div class="list-row" style="flex-direction:column;align-items:stretch;gap:var(--s-2)">
            <div class="list-row-title">Accent colour</div>
            <div class="row wrap gap-3">${ACCENTS.map((a) => `
              <button class="swatch ${s.accent === a.id ? 'is-on' : ''}" data-accent="${a.id}"
                style="background:${a.color}" aria-label="${esc(a.name)} accent"
                aria-pressed="${s.accent === a.id}"></button>`).join('')}</div>
          </div>
        </div>
      </section>

      <section>
        <div class="section-label">Detection</div>
        <div class="group">
          ${sliderRow('confidence', 'Confidence threshold',
            'Higher means fewer, surer labels', 30, 90, Math.round(s.confidence * 100), '%')}
          ${sliderRow('maxDetections', 'Maximum labels', 'Objects tracked at once', 1, 10, s.maxDetections, '')}
          ${sliderRow('detectHz', 'Detection rate',
            'Lower saves battery on slower devices', 2, 15, s.detectHz, ' Hz')}
          <div class="list-row">
            <div class="list-row-body">
              <div class="list-row-title">Model</div>
              <div class="list-row-sub">Currently: ${esc(state.runtime.modelName || 'not loaded')}</div>
            </div>
            <div style="width:150px">${segmented('modelBase', [
              { id: 'auto', label: 'Auto' },
              { id: 'mobilenet_v2', label: 'Accurate' },
              { id: 'lite_mobilenet_v2', label: 'Fast' }
            ])}</div>
          </div>
          ${row('Confidence badge', 'Show model certainty on labels', toggle('showConfidence'))}
        </div>
        <p class="footnote" style="padding:var(--s-2) var(--s-1) 0">
          Changing the model takes effect after a reload.
        </p>
      </section>

      <section>
        <div class="section-label">Audio & feedback</div>
        <div class="group">
          ${row('Speak new words', 'Pronounce a word the first time you find it', toggle('speakOnDiscover'))}
          ${sliderRow('speechRate', 'Speech speed', '', 50, 130, Math.round(s.speechRate * 100), '%')}
          ${row('Sound effects', '', toggle('sound'))}
          ${row('Haptics', 'Vibration feedback where supported', toggle('haptics'))}
          <button class="list-row is-tappable" id="test-voice">
            <div class="list-row-icon">${icon('speaker')}</div>
            <div class="list-row-body">
              <div class="list-row-title">Test pronunciation</div>
              <div class="list-row-sub">Hear a sample in ${esc(lang.name)}</div>
            </div>
            <div class="list-row-chevron">${icon('chevronRight')}</div>
          </button>
        </div>
        ${noVoice.length ? `<p class="footnote" style="padding:var(--s-2) var(--s-1) 0">
          No installed speech voice for: ${noVoice.map((l) => esc(LANGUAGES[l].name)).join(', ')}.
          Text and phonetics still work; voices come from your operating system.
        </p>` : ''}
      </section>

      <section>
        <div class="section-label">Studying</div>
        <div class="group">
          <div class="list-row" style="flex-direction:column;align-items:stretch;gap:var(--s-2)">
            <div class="list-row-title">Review style</div>
            <div>${segmented('reviewStyle', REVIEW_STYLES)}</div>
          </div>
          ${sliderRow('dailyGoal', 'Daily goal', 'Reviews per day', 5, 60, s.dailyGoal, ' cards')}
        </div>
      </section>

      <section>
        <div class="section-label">Your data</div>
        <div class="group">
          <div class="list-row">
            <div class="list-row-icon">${icon('chart')}</div>
            <div class="list-row-body"><div class="list-row-title">Stored locally</div>
              <div class="list-row-sub">${stats.discovered} words · ${stats.reviews} reviews · ${stats.xp} XP</div></div>
          </div>
          <button class="list-row is-tappable" id="export">
            <div class="list-row-icon">${icon('download')}</div>
            <div class="list-row-body"><div class="list-row-title">Export backup</div>
              <div class="list-row-sub">Save a JSON file of all progress</div></div>
            <div class="list-row-chevron">${icon('chevronRight')}</div>
          </button>
          <button class="list-row is-tappable" id="import">
            <div class="list-row-icon">${icon('upload')}</div>
            <div class="list-row-body"><div class="list-row-title">Restore backup</div>
              <div class="list-row-sub">Replaces everything on this device</div></div>
            <div class="list-row-chevron">${icon('chevronRight')}</div>
          </button>
          <button class="list-row is-tappable" id="reset-lang">
            <div class="list-row-icon">${icon('refresh')}</div>
            <div class="list-row-body"><div class="list-row-title">Reset ${esc(lang.name)} progress</div>
              <div class="list-row-sub">Other languages are kept</div></div>
          </button>
          <button class="list-row is-tappable" id="reset-all">
            <div class="list-row-icon" style="color:var(--red)">${icon('trash')}</div>
            <div class="list-row-body"><div class="list-row-title" style="color:var(--red)">Erase all data</div>
              <div class="list-row-sub">Progress, settings and badges</div></div>
          </button>
        </div>
        <input type="file" id="import-file" accept="application/json,.json" hidden>
      </section>

      <section>
        <div class="section-label">About</div>
        <div class="card">
          <div class="about-block">
            <div class="mark">${icon('lens')}</div>
            <div class="title-3">LinguaLens</div>
            <p style="margin-top:var(--s-2)">
              Point your camera at the world and learn what everything is called.
              Object recognition runs entirely on your device — no images are ever
              uploaded, and nothing you learn leaves this browser.
            </p>
            <div class="ver">Version ${esc(APP_VERSION)} · ${esc(state.runtime.modelName || 'model idle')}</div>
            <a class="credit-link" href="https://www.rushanhaque.online" target="_blank" rel="noopener noreferrer">
              <span class="dot"></span>Developed by Rushan Haque
            </a>
          </div>
        </div>
        <button class="list-row is-tappable group" id="replay-tour" style="margin-top:var(--s-3)">
          <div class="list-row-icon">${icon('sparkle')}</div>
          <div class="list-row-body"><div class="list-row-title">Replay the welcome tour</div></div>
          <div class="list-row-chevron">${icon('chevronRight')}</div>
        </button>
      </section>
    </div>`;

  wire();
}

/* ── Markup helpers ───────────────────────────────────────────────────── */

function row(title, sub, control) {
  return `<div class="list-row">
    <div class="list-row-body"><div class="list-row-title">${esc(title)}</div>
    ${sub ? `<div class="list-row-sub">${esc(sub)}</div>` : ''}</div>${control}</div>`;
}

function toggle(key) {
  const on = !!state.settings[key];
  return `<button class="switch" role="switch" data-toggle="${key}"
    aria-checked="${on}" aria-label="${esc(key)}"></button>`;
}

function segmented(key, options) {
  const current = state.settings[key];
  return `<div class="segmented" data-segmented="${key}" role="tablist">
    ${options.map((o) => `<button role="tab" data-value="${esc(o.id)}"
      aria-selected="${o.id === current}">${esc(o.label)}</button>`).join('')}
  </div>`;
}

function sliderRow(key, title, sub, min, max, value, unit) {
  const fill = ((value - min) / (max - min)) * 100;
  return `<div class="list-row" style="flex-direction:column;align-items:stretch;gap:var(--s-1)">
    <div class="row between">
      <div><div class="list-row-title">${esc(title)}</div>
      ${sub ? `<div class="list-row-sub">${esc(sub)}</div>` : ''}</div>
      <div class="list-row-value tabular" data-out="${key}">${value}${esc(unit)}</div>
    </div>
    <input class="slider" type="range" data-slider="${key}" data-unit="${esc(unit)}"
      min="${min}" max="${max}" value="${value}" style="--fill-pct:${fill}%"
      aria-label="${esc(title)}">
  </div>`;
}

/* ── Wiring ───────────────────────────────────────────────────────────── */

function wire() {
  $$('[data-toggle]', root).forEach((btn) => btn.addEventListener('click', () => {
    const key = btn.dataset.toggle;
    const next = !state.settings[key];
    setSetting(key, next);
    btn.setAttribute('aria-checked', String(next));
    haptic('select');
  }));

  $$('[data-segmented]', root).forEach((seg) => {
    seg.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-value]');
      if (!btn) return;
      const key = seg.dataset.segmented;
      setSetting(key, btn.dataset.value);
      $$('[data-value]', seg).forEach((b) => b.setAttribute('aria-selected', String(b === btn)));
      haptic('select');
      if (key === 'theme') emit('theme');
    });
  });

  $$('[data-slider]', root).forEach((input) => {
    input.addEventListener('input', () => {
      const key = input.dataset.slider;
      const raw = Number(input.value);
      const value = (key === 'confidence' || key === 'speechRate') ? raw / 100 : raw;
      setSetting(key, value);
      const out = $(`[data-out="${key}"]`, root);
      if (out) out.textContent = raw + input.dataset.unit;
      const min = Number(input.min); const max = Number(input.max);
      input.style.setProperty('--fill-pct', `${((raw - min) / (max - min)) * 100}%`);
    });
    input.addEventListener('change', () => haptic('select'));
  });

  $$('[data-accent]', root).forEach((btn) => btn.addEventListener('click', () => {
    setSetting('accent', btn.dataset.accent);
    document.documentElement.dataset.accent = btn.dataset.accent;
    $$('[data-accent]', root).forEach((b) => {
      b.classList.toggle('is-on', b === btn);
      b.setAttribute('aria-pressed', String(b === btn));
    });
    cue('tap', 'select');
  }));

  $('#pick-lang')?.addEventListener('click', () =>
    import('./camera.js').then((m) => m.openLanguagePicker()));

  $('#test-voice')?.addEventListener('click', () => {
    const lang = state.settings.targetLang;
    const ok = speak(sampleFor(lang), lang);
    haptic('light');
    if (!ok) toast(`No ${LANGUAGES[lang].name} voice is installed on this device`);
  });

  $('#replay-tour')?.addEventListener('click', () => {
    setSetting('onboarded', false);
    emit('tour');
  });

  $('#export')?.addEventListener('click', doExport);
  $('#import')?.addEventListener('click', () => $('#import-file').click());
  $('#import-file')?.addEventListener('change', doImport);

  $('#reset-lang')?.addEventListener('click', async () => {
    const lang = state.settings.targetLang;
    if (await confirmAction({
      title: `Reset ${LANGUAGES[lang].name}?`,
      message: `Every discovered word and review schedule for ${LANGUAGES[lang].name} will be deleted. Other languages, badges and XP are kept.`,
      confirmLabel: 'Reset language', destructive: true
    })) {
      resetProgress(lang);
      toast(`${LANGUAGES[lang].name} progress reset`);
      renderSettings();
    }
  });

  $('#reset-all')?.addEventListener('click', async () => {
    if (await confirmAction({
      title: 'Erase everything?',
      message: 'All discovered words, review history, badges, XP and settings will be permanently deleted from this device. This cannot be undone.',
      confirmLabel: 'Erase all data', destructive: true
    })) {
      resetEverything();
      toast('All data erased');
      location.reload();
    }
  });
}

const SAMPLES = {
  es: 'Hola, esto es una manzana.', fr: "Bonjour, c'est une pomme.",
  de: 'Hallo, das ist ein Apfel.', it: 'Ciao, questa è una mela.',
  pt: 'Olá, isto é uma maçã.', nl: 'Hallo, dit is een appel.',
  ru: 'Привет, это яблоко.', ja: 'こんにちは、これはりんごです。',
  ko: '안녕하세요, 이것은 사과입니다.', zh: '你好，这是苹果。',
  hi: 'नमस्ते, यह सेब है।', ar: 'مرحبا، هذه تفاحة.'
};
function sampleFor(lang) { return SAMPLES[lang] || 'Hello'; }

function doExport() {
  try {
    const blob = new Blob([exportData()], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = el('a', { href: url, download: `lingualens-backup-${new Date().toISOString().slice(0, 10)}.json` });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
    toast('Backup saved', { emoji: '💾' });
  } catch { toast('Could not create the backup file'); }
}

async function doImport(e) {
  const file = e.target.files?.[0];
  e.target.value = '';
  if (!file) return;

  if (!await confirmAction({
    title: 'Restore backup?',
    message: 'This replaces all progress and settings currently on this device.',
    confirmLabel: 'Restore', destructive: true
  })) return;

  try {
    importData(await file.text());
    toast('Backup restored', { emoji: '✅' });
    setTimeout(() => location.reload(), 700);
  } catch (err) {
    toast(err.message || 'That file could not be read');
  }
}
