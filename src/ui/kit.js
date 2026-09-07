/**
 * kit.js — Small DOM helpers, toasts, and the sheet controller.
 */

import { icon } from './icons.js';
import { haptic } from '../core/feedback.js';

/* ── DOM ──────────────────────────────────────────────────────────────── */

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'class') node.className = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k === 'text') node.textContent = v;
    else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === 'dataset') Object.assign(node.dataset, v);
    else if (v !== null && v !== undefined && v !== false) node.setAttribute(k, v === true ? '' : v);
  }
  for (const c of [].concat(children)) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return node;
}

/** Escape for safe interpolation into innerHTML. */
export function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

export function clamp(n, lo, hi) { return Math.max(lo, Math.min(hi, n)); }

/* ── Toast ────────────────────────────────────────────────────────────── */

let toastLayer = null;

export function toast(message, opts = {}) {
  if (!toastLayer) {
    toastLayer = $('#toast-layer') || el('div', { id: 'toast-layer' });
    if (!toastLayer.parentNode) document.body.append(toastLayer);
  }
  const node = el('div', {
    class: 'toast',
    role: 'status',
    'aria-live': 'polite'
  }, []);
  node.innerHTML = (opts.emoji ? `<span class="toast-em">${esc(opts.emoji)}</span>` : '') +
    `<span>${esc(message)}</span>`;

  toastLayer.append(node);
  // Cap the stack so a burst of discoveries doesn't wallpaper the screen.
  while (toastLayer.children.length > 3) toastLayer.firstElementChild.remove();

  setTimeout(() => {
    node.classList.add('is-leaving');
    setTimeout(() => node.remove(), 220);
  }, opts.duration ?? 2400);
  return node;
}

/* ── Sheet ────────────────────────────────────────────────────────────── */

let sheetEls = null;
let onCloseHook = null;
let lastFocused = null;

function ensureSheet() {
  if (sheetEls) return sheetEls;
  const scrim = el('div', { class: 'sheet-scrim', id: 'sheet-scrim' });
  const sheet = el('div', {
    class: 'sheet', id: 'sheet', role: 'dialog', 'aria-modal': 'true', 'aria-hidden': 'true'
  });
  sheet.innerHTML =
    '<div class="sheet-grabber"></div>' +
    '<div class="sheet-header">' +
      '<h2 class="sheet-title" id="sheet-title"></h2>' +
      `<button class="icon-btn" id="sheet-close" aria-label="Close">${icon('close')}</button>` +
    '</div>' +
    '<div class="sheet-body scroll" id="sheet-body"></div>';

  document.body.append(scrim, sheet);
  sheetEls = {
    scrim, sheet,
    title: $('#sheet-title', sheet),
    body: $('#sheet-body', sheet),
    close: $('#sheet-close', sheet)
  };

  scrim.addEventListener('click', closeSheet);
  sheetEls.close.addEventListener('click', closeSheet);
  sheet.setAttribute('aria-labelledby', 'sheet-title');

  // Drag-to-dismiss from the grabber area, the way a native sheet behaves.
  let startY = null;
  sheet.addEventListener('touchstart', (e) => {
    if (e.target.closest('.sheet-body')?.scrollTop > 0) return;
    startY = e.touches[0].clientY;
  }, { passive: true });
  sheet.addEventListener('touchmove', (e) => {
    if (startY === null) return;
    const dy = e.touches[0].clientY - startY;
    if (dy > 0) sheet.style.transform = `translateY(${dy}px)`;
  }, { passive: true });
  sheet.addEventListener('touchend', (e) => {
    if (startY === null) return;
    const dy = (e.changedTouches[0]?.clientY ?? startY) - startY;
    sheet.style.transform = '';
    if (dy > 110) closeSheet();
    startY = null;
  });

  return sheetEls;
}

/**
 * Open the shared sheet.
 * @param {object} opts { title, content (string|Node), onClose }
 */
export function openSheet({ title = '', content = '', onClose = null } = {}) {
  const s = ensureSheet();
  lastFocused = document.activeElement;
  s.title.textContent = title;
  s.body.innerHTML = '';
  if (typeof content === 'string') s.body.innerHTML = content;
  else if (content) s.body.append(content);

  s.sheet.setAttribute('aria-hidden', 'false');
  s.scrim.classList.add('is-open');
  // Force layout so the transition has a start value to animate from. A
  // requestAnimationFrame would be the usual trick, but browsers throttle rAF
  // in background tabs, which would leave the sheet stuck off-screen.
  void s.sheet.offsetHeight;
  s.sheet.classList.add('is-open');
  onCloseHook = onClose;
  haptic('light');

  document.addEventListener('keydown', onSheetKey);
  setTimeout(() => s.close.focus({ preventScroll: true }), 60);
  return s.body;
}

function onSheetKey(e) {
  if (e.key === 'Escape') { e.stopPropagation(); closeSheet(); return; }
  if (e.key !== 'Tab' || !sheetEls) return;
  // Trap focus inside the dialog.
  const focusables = $$('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])', sheetEls.sheet)
    .filter((n) => n.offsetParent !== null);
  if (!focusables.length) return;
  const first = focusables[0];
  const last = focusables[focusables.length - 1];
  if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
  else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
}

export function closeSheet() {
  if (!sheetEls) return;
  sheetEls.sheet.classList.remove('is-open');
  sheetEls.scrim.classList.remove('is-open');
  sheetEls.sheet.setAttribute('aria-hidden', 'true');
  document.removeEventListener('keydown', onSheetKey);
  const hook = onCloseHook;
  onCloseHook = null;
  setTimeout(() => { if (sheetEls && !sheetEls.sheet.classList.contains('is-open')) sheetEls.body.innerHTML = ''; }, 400);
  if (lastFocused && lastFocused.isConnected) lastFocused.focus({ preventScroll: true });
  if (hook) hook();
}

export function isSheetOpen() {
  return !!sheetEls && sheetEls.sheet.classList.contains('is-open');
}

/* ── Confirm dialog (sheet-based, so it matches the app) ──────────────── */

export function confirmAction({ title, message, confirmLabel = 'Confirm', destructive = false }) {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (v) => { if (!settled) { settled = true; resolve(v); } };

    const body = openSheet({ title, onClose: () => finish(false) });
    body.innerHTML = `<p class="subhead" style="line-height:1.55;margin-bottom:var(--s-5)">${esc(message)}</p>`;

    const confirm = el('button', {
      class: `btn btn-block ${destructive ? 'btn-destructive' : 'btn-primary'}`,
      text: confirmLabel
    });
    const cancel = el('button', { class: 'btn btn-block', text: 'Cancel', style: 'margin-top:var(--s-2)' });

    confirm.addEventListener('click', () => { finish(true); closeSheet(); });
    cancel.addEventListener('click', () => { finish(false); closeSheet(); });
    body.append(confirm, cancel);
    setTimeout(() => confirm.focus({ preventScroll: true }), 80);
  });
}

/* ── Formatting ───────────────────────────────────────────────────────── */

export function pct(n) { return `${Math.round(n * 100)}%`; }

export function relativeDay(ts) {
  const days = Math.floor((Date.now() - ts) / 86400000);
  if (days <= 0) return 'today';
  if (days === 1) return 'yesterday';
  if (days < 7) return `${days} days ago`;
  if (days < 30) return `${Math.floor(days / 7)}w ago`;
  return `${Math.floor(days / 30)}mo ago`;
}
