/**
 * app.js — Application shell: theming, tab routing, keyboard shortcuts,
 * install and update prompts, and boot sequencing.
 */

import { state, on, saveNow, refreshBadges, getStats } from './core/store.js';
import { levelFor } from './data/achievements.js';
import { LANGUAGES } from './data/languages.js';
import { haptic, cue, unlockAudio } from './core/feedback.js';
import { $, $$, toast, isSheetOpen, closeSheet, esc } from './ui/kit.js';
import { icon } from './ui/icons.js';
import { initCamera, paintCameraChrome, onCameraVisible, refreshCameraLanguage,
         toggleQuiz, capture, flipCamera, togglePause, openLanguagePicker } from './views/camera.js';
import { initLearn, refreshLearn, isSessionActive } from './views/learn.js';
import { initProgress, renderProgress } from './views/progress.js';
import { initSettings, renderSettings } from './views/settings.js';
import { initOnboarding, showOnboarding } from './views/onboarding.js';
import { dueCount } from './core/srs.js';

const TABS = [
  { id: 'camera', label: 'Camera', icon: 'camera' },
  { id: 'learn', label: 'Learn', icon: 'cards' },
  { id: 'progress', label: 'Progress', icon: 'chart' },
  { id: 'settings', label: 'Settings', icon: 'gear' }
];

let activeTab = 'camera';

/* ── Theme ────────────────────────────────────────────────────────────── */

function applyTheme() {
  const root = document.documentElement;
  const theme = state.settings.theme;
  if (theme === 'auto') root.removeAttribute('data-theme');
  else root.dataset.theme = theme;
  root.dataset.accent = state.settings.accent;

  // Keep the browser/PWA chrome in step with the app's own surface.
  const dark = theme === 'dark' ||
    (theme === 'auto' && matchMedia('(prefers-color-scheme: dark)').matches);
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute('content', dark ? '#000000' : '#F2F3EC');
}

/* ── Tab routing ──────────────────────────────────────────────────────── */

function buildTabBar() {
  const bar = $('#tabbar');
  bar.innerHTML = TABS.map((t) => `
    <button class="tab" role="tab" data-tab="${t.id}" aria-selected="${t.id === activeTab}"
            aria-controls="view-${t.id}" id="tab-${t.id}">
      ${icon(t.icon)}
      <span class="tab-label">${t.label}</span>
      ${t.id === 'learn' ? '<span class="tab-dot hidden" id="due-dot"></span>' : ''}
    </button>`).join('');

  $$('[data-tab]', bar).forEach((btn) =>
    btn.addEventListener('click', () => go(btn.dataset.tab)));
}

export function go(tab) {
  if (!TABS.some((t) => t.id === tab)) return;
  if (tab === activeTab) return;
  if (activeTab === 'learn' && isSessionActive() && tab !== 'learn') {
    // Leaving mid-session is allowed; graded cards are already persisted.
  }

  activeTab = tab;
  unlockAudio();
  haptic('select');

  $$('.view').forEach((v) => v.classList.toggle('is-active', v.id === `view-${tab}`));
  $$('[data-tab]').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.tab === tab)));

  onCameraVisible(tab === 'camera');
  if (tab === 'learn') refreshLearn();
  if (tab === 'progress') renderProgress();
  if (tab === 'settings') renderSettings();

  history.replaceState({ tab }, '', `#${tab}`);
  document.title = tab === 'camera'
    ? 'LinguaLens — Camera'
    : `LinguaLens — ${TABS.find((t) => t.id === tab).label}`;
}

function updateDueBadge() {
  const dot = $('#due-dot');
  if (!dot) return;
  dot.classList.toggle('hidden', dueCount() === 0);
}

/* ── Keyboard shortcuts ───────────────────────────────────────────────── */

const SHORTCUTS = [
  ['1–4', 'Switch tabs'],
  ['Space', 'Capture snapshot'],
  ['Q', 'Toggle quiz mode'],
  ['F', 'Flip camera'],
  ['P', 'Pause / resume detection'],
  ['L', 'Change language'],
  ['R', 'Start a review'],
  ['?', 'This list'],
  ['Esc', 'Close sheet']
];

function wireKeyboard() {
  document.addEventListener('keydown', (e) => {
    // Never hijack typing.
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName) || e.target.isContentEditable;
    if (typing || e.metaKey || e.ctrlKey || e.altKey) return;

    if (e.key === 'Escape' && isSheetOpen()) { closeSheet(); return; }

    const idx = ['1', '2', '3', '4'].indexOf(e.key);
    if (idx !== -1) { go(TABS[idx].id); e.preventDefault(); return; }

    const k = e.key.toLowerCase();
    if (k === '?') { showShortcuts(); e.preventDefault(); return; }
    if (k === 'r') { go('learn'); e.preventDefault(); return; }
    if (k === 'l') { openLanguagePicker(); e.preventDefault(); return; }

    if (activeTab !== 'camera') return;
    if (e.key === ' ') { capture(); e.preventDefault(); }
    else if (k === 'q') { toggleQuiz(); e.preventDefault(); }
    else if (k === 'f') { flipCamera(); e.preventDefault(); }
    else if (k === 'p') { togglePause(); e.preventDefault(); }
  });
}

function showShortcuts() {
  import('./ui/kit.js').then(({ openSheet }) => {
    const body = openSheet({ title: 'Keyboard shortcuts' });
    body.innerHTML = `<div class="group">${SHORTCUTS.map(([key, desc]) => `
      <div class="list-row">
        <div class="list-row-body"><div class="list-row-title">${esc(desc)}</div></div>
        <kbd class="chip mono">${esc(key)}</kbd>
      </div>`).join('')}</div>`;
  });
}

/* ── Celebrations ─────────────────────────────────────────────────────── */

function wireCelebrations() {
  on('levelup', (e) => {
    const lvl = levelFor(state.meta.xp);
    cue('levelup', 'success');
    toast(`Level ${e.detail.level} — ${lvl.title}!`, { emoji: '🎉', duration: 3200 });
  });

  on('badge', (e) => {
    cue('badge', 'success');
    toast(`${e.detail.title} unlocked`, { emoji: e.detail.em, duration: 3200 });
  });

  on('streak', (e) => {
    if (e.detail.current > 1) toast(`${e.detail.current}-day streak`, { emoji: '🔥' });
  });

  on('navigate', (e) => go(e.detail.tab));
  on('theme', applyTheme);
  on('tour', () => {
    initOnboarding($('#onboarding'), () => { refreshCameraLanguage(); refreshLearn(); });
    showOnboarding();
  });

  on('settings', (e) => {
    if (e.detail.key === 'targetLang') {
      refreshCameraLanguage();
      updateDueBadge();
      // Every view shows language-specific content, so redraw whichever is up.
      if (activeTab === 'learn') refreshLearn();
      else if (activeTab === 'progress') renderProgress();
      else if (activeTab === 'settings') renderSettings();
    }
    if (e.detail.key === 'accent' || e.detail.key === 'theme') applyTheme();
  });

  on('graded', updateDueBadge);
  on('discovered', updateDueBadge);
}

/* ── PWA: install prompt and update flow ──────────────────────────────── */

let installPrompt = null;

function wirePWA() {
  window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    installPrompt = e;
    // Offer it once the user has actually used the app, not on arrival.
    setTimeout(() => {
      if (!installPrompt || getStats().discovered < 3) return;
      const t = toast('Add LinguaLens to your home screen', { emoji: '📲', duration: 7000 });
      t.style.pointerEvents = 'auto';
      t.style.cursor = 'pointer';
      t.addEventListener('click', async () => {
        t.remove();
        installPrompt.prompt();
        await installPrompt.userChoice;
        installPrompt = null;
      });
    }, 60000);
  });

  if (!('serviceWorker' in navigator)) return;
  window.addEventListener('load', async () => {
    try {
      const reg = await navigator.serviceWorker.register('./sw.js', { scope: './' });
      reg.addEventListener('updatefound', () => {
        const sw = reg.installing;
        if (!sw) return;
        sw.addEventListener('statechange', () => {
          if (sw.state === 'installed' && navigator.serviceWorker.controller) {
            const t = toast('A new version is ready — tap to update', { emoji: '✨', duration: 9000 });
            t.style.pointerEvents = 'auto';
            t.style.cursor = 'pointer';
            t.addEventListener('click', () => {
              sw.postMessage({ type: 'SKIP_WAITING' });
              location.reload();
            });
          }
        });
      });
    } catch { /* offline support simply won't be available */ }
  });
}

/* ── Boot ─────────────────────────────────────────────────────────────── */

function setProgress(pct, message) {
  const fill = $('#launch-fill');
  const status = $('#launch-status');
  if (fill) fill.style.width = pct + '%';
  if (status) status.textContent = message;
}

async function boot() {
  applyTheme();
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);

  buildTabBar();
  wireKeyboard();
  wireCelebrations();
  wirePWA();

  initLearn($('#view-learn'));
  initProgress($('#view-progress'));
  initSettings($('#view-settings'));

  paintCameraChrome();
  updateDueBadge();

  // Restore the tab from the URL hash so a bookmarked view reopens correctly.
  const hash = location.hash.replace('#', '');
  if (TABS.some((t) => t.id === hash) && hash !== 'camera') {
    activeTab = 'camera';
    go(hash);
  }

  const firstRun = !state.settings.onboarded;

  // On a first run the tour comes first, so the slide explaining the camera
  // prompt is on screen *before* the browser shows it. Asking cold is the
  // single biggest reason people deny camera access and never come back.
  if (firstRun) {
    setProgress(100, 'Ready');
    setTimeout(() => $('#launch').classList.add('is-done'), 300);
    await new Promise((resolve) => {
      initOnboarding($('#onboarding'), resolve);
      showOnboarding();
    });
    refreshCameraLanguage();
    refreshLearn();
    await initCamera(setProgress);
  } else {
    initOnboarding($('#onboarding'), () => {
      refreshCameraLanguage();
      refreshLearn();
    });
    await initCamera(setProgress);
    // Give the launch screen a beat at 100% rather than snapping away.
    setTimeout(() => $('#launch').classList.add('is-done'), 420);
  }

  refreshBadges();
  paintCameraChrome();
}

/* Persist immediately when the page is being put away — 'visibilitychange'
   is the only event mobile browsers reliably fire before termination. */
document.addEventListener('visibilitychange', () => {
  if (document.hidden) { saveNow(); onCameraVisible(false); }
  else onCameraVisible(activeTab === 'camera');
});
window.addEventListener('pagehide', saveNow);

// Expose a tiny surface for debugging without leaking internals.
window.LinguaLens = {
  go,
  get stats() { return getStats(); },
  get language() { return LANGUAGES[state.settings.targetLang]; }
};

boot();
