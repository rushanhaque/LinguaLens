/**
 * camera.js (view) — Live AR translation.
 *
 * Detection and rendering are deliberately decoupled: the model runs at
 * `detectHz` (a few times a second, which is all COCO-SSD can sustain), while
 * label positions interpolate at display refresh so motion stays smooth.
 */

import { DICT } from '../data/dictionary.js';
import { objId, translateItem } from '../data/vocab.js';
import { COLORS, colorForms } from '../data/colors.js';
import { LANGUAGES, genderLabel, citationForm, buildColorPhrase } from '../data/languages.js';
import { state, setSetting, recordSighting, refreshBadges, logSnapshot, emit } from '../core/store.js';
import { createTracker } from '../core/tracker.js';
import { createPalette } from '../core/palette.js';
import { createGovernor } from '../core/governor.js';
import { createCamera, CameraError } from '../core/camera.js';
import { createClassifier } from '../core/classifier.js';
import { createSceneWatch, SCENE_CUT, SCENE_PAN } from '../core/scene.js';
import { speak } from '../core/speech.js';
import { cue, haptic, unlockAudio } from '../core/feedback.js';
import { $, $$, el, esc, toast, openSheet, closeSheet, clamp } from '../ui/kit.js';
import { icon } from '../ui/icons.js';
import { openWordSheet } from './wordSheet.js';

let video, canvas, ctx, labelLayer, stage, statusPill, strip, focusChip;
let cam = null;
let tracker = createTracker();
const palette = createPalette();
const governor = createGovernor();
const classifier = createClassifier();
const scene = createSceneWatch();
/** trackId → colour key most recently read off that object. */
const trackColors = new Map();
let model = null;
let running = false;
let lastDetect = 0;
let frameTimes = [];
let liveTracks = [];
let labelNodes = new Map();     // track id → element
let stripCards = new Map();     // class → element
let cssZoom = 1;
let paused = false;

/* The classifier's current answer — what the camera is centred on. It runs on
   its own cadence, managed inside the classifier itself. */
let focusResult = null;

/* ── Boot ─────────────────────────────────────────────────────────────── */

export async function initCamera(setProgress) {
  video = $('#webcam');
  canvas = $('#overlay');
  ctx = canvas.getContext('2d', { alpha: true });
  labelLayer = $('#label-layer');
  stage = $('#stage');
  statusPill = $('#status-pill');
  strip = $('#detected-strip');
  focusChip = $('#focus-chip');

  wireControls();

  setProgress(15, 'Requesting camera access…');
  cam = createCamera(video);
  try {
    await cam.start({ facingMode: 'environment' });
    state.runtime.cameraFacing = cam.facing;
    applyMirror();
    hideError();
  } catch (err) {
    showError(err.code, err.message);
    setProgress(100, 'Camera unavailable');
    return false;
  }

  setProgress(40, 'Loading the vision model…');
  const ok = await loadModel(setProgress);
  if (!ok) {
    showError('model', 'The object-detection model could not be downloaded. Check your connection and reload.');
    return false;
  }

  setProgress(95, 'Warming up…');
  await warmUp();

  resize();
  window.addEventListener('resize', resize);
  const ro = new ResizeObserver(resize);
  ro.observe(stage);
  // The control deck grows when the focus chip appears, and label placement
  // reserves its height — so its size has to invalidate the cached guard too.
  const deck = $('#cam-bottom');
  if (deck) ro.observe(deck);

  running = true;
  state.runtime.ready = true;
  requestAnimationFrame(loop);
  setProgress(100, 'Ready');

  /* The classifier is the larger of the two models and only widens the
     vocabulary — it is not needed for the app to work. Loading it after the
     first frames are already on screen keeps startup fast, and detection
     simply gets better a few seconds in. */
  classifier.load().then((ok) => {
    state.runtime.classifierReady = ok;
    if (ok) toast('Extended recognition ready', { emoji: '🔎', duration: 2200 });
  });

  return true;
}

async function loadModel(setProgress) {
  if (typeof cocoSsd === 'undefined') return false;
  // WebGL is dramatically faster; fall back rather than fail if it is absent.
  try { if (window.tf?.setBackend) await tf.setBackend('webgl'); } catch { /* CPU fallback */ }
  try { await window.tf?.ready?.(); } catch { /* ignore */ }

  const pref = state.settings.modelBase;
  const order = pref === 'auto'
    ? (isLowPowerDevice() ? ['lite_mobilenet_v2', 'mobilenet_v2'] : ['mobilenet_v2', 'lite_mobilenet_v2'])
    : [pref, pref === 'mobilenet_v2' ? 'lite_mobilenet_v2' : 'mobilenet_v2'];

  for (const base of order) {
    try {
      setProgress(55, base === 'mobilenet_v2' ? 'Loading full accuracy model…' : 'Loading fast model…');
      model = await cocoSsd.load({ base });
      state.runtime.modelName = base;
      return true;
    } catch { /* try the next one */ }
  }
  return false;
}

function isLowPowerDevice() {
  const cores = navigator.hardwareConcurrency || 4;
  const mem = navigator.deviceMemory || 4;
  return cores <= 4 || mem <= 3;
}

/** One throwaway inference so the first real frame is not the slow one. */
async function warmUp() {
  try {
    const c = document.createElement('canvas');
    c.width = 320; c.height = 240;
    await model.detect(c);
  } catch { /* harmless */ }
}

/* ── Main loop ────────────────────────────────────────────────────────── */

function loop(now) {
  if (!running) return;
  requestAnimationFrame(loop);

  // FPS over a one-second sliding window.
  frameTimes.push(now);
  while (frameTimes.length && now - frameTimes[0] > 1000) frameTimes.shift();
  state.runtime.fps = frameTimes.length;

  // The governor spends a fixed share of wall-clock time on inference, so a
  // slow device drops its detection rate instead of dropping frames.
  const interval = governor.intervalMs(clamp(state.settings.detectHz, 2, 20));
  if (!paused && model && video.readyState >= 2 && now - lastDetect >= interval) {
    lastDetect = now;
    detect();                            // fire-and-forget; render never waits
  }

  render();
  updateStatus(now);
}

let detecting = false;

async function detect() {
  if (detecting) return;                 // never queue up inference calls
  detecting = true;
  const t0 = performance.now();
  try {
    /* How much has the picture changed since the last pass? A big jump means
       the camera has been moved somewhere else, and every existing track
       describes a scene that is no longer in front of the lens. Ageing them
       here is what stops a label outliving its object during a pan — the
       tracker on its own cannot tell "moved away" from "briefly hidden". */
    const motion = scene.sample(video);
    if (motion > SCENE_CUT) {
      tracker.reset();
      classifier.reset();
      liveTracks = [];
      focusResult = null;
      countedTracks.clear();
    } else if (motion > SCENE_PAN) {
      tracker.decay(2);
    }

    const raw = await model.detect(video, 20, 0.25);
    governor.sample(performance.now() - t0);

    const { w, h } = cam.videoSize;
    liveTracks = tracker.update(raw, w, h, {
      minScore: state.settings.confidence,
      maxOut: state.settings.maxDetections
    });
    state.runtime.live = liveTracks;
    readColors(liveTracks);
    handleDiscoveries(liveTracks);
    syncStrip(liveTracks);

    /* The classifier names whatever the camera is centred on, including the
       hundreds of everyday objects COCO has no label for. It paces itself
       against its own measured cost, so calling it on every pass is safe —
       it returns the held result and declines to run when it is too soon. */
    if (state.settings.focusMode && classifier.ready) {
      focusResult = await classifier.classify(video);
      if (focusResult) noteFocusDiscovery(focusResult.key);
    } else {
      focusResult = null;
    }
  } catch { /* a dropped frame is not worth reporting */ }
  detecting = false;
}

/**
 * Sample the dominant colour inside each track. Runs on the detection cadence
 * rather than per frame — colour barely changes between passes, and the crop
 * plus readback is not free.
 */
function readColors(tracks) {
  if (!state.settings.showColors) { trackColors.clear(); return; }
  const live = new Set();
  for (const t of tracks) {
    live.add(t.id);
    const key = palette.readStable(video, t.raw || t.bbox, t.id);
    if (key) trackColors.set(t.id, key);
    else if (!trackColors.has(t.id)) trackColors.delete(t.id);
  }
  for (const id of [...trackColors.keys()]) if (!live.has(id)) trackColors.delete(id);
  palette.prune(live);
}

/* A track stays alive for many frames, so counting a sighting per frame would
   inflate "seen 400x" within seconds. Count once per track instead — that
   matches what a person means by "I saw it again". */
let countedTracks = new Set();

function handleDiscoveries(tracks) {
  const live = new Set(tracks.map((t) => t.id));
  for (const t of tracks) {
    if (!DICT[t.cls] || countedTracks.has(t.id)) continue;
    countedTracks.add(t.id);
    const isNew = recordSighting(t.cls);
    if (!isNew) continue;
    const tr = translateItem(objId(t.cls), state.settings.targetLang);
    if (!tr) continue;
    cue('discover', 'success');
    toast(`${tr.word} — new word!`, { emoji: tr.em });
    if (state.settings.speakOnDiscover) {
      speak(citationForm(state.settings.targetLang, tr.word, tr.gender));
    }
    refreshBadges();
  }
  // Forget ids that have aged out so the set cannot grow without bound.
  if (countedTracks.size > live.size) {
    countedTracks = new Set([...countedTracks].filter((id) => live.has(id)));
  }
}

/* The classifier reports the same word on every pass while the camera is held
   still, so a sighting is counted only when the answer actually changes. */
let lastFocusKey = null;

function noteFocusDiscovery(cls) {
  if (cls === lastFocusKey) return;
  lastFocusKey = cls;
  if (!DICT[cls]) return;
  const isNew = recordSighting(cls);
  if (!isNew) return;
  const tr = translateItem(objId(cls), state.settings.targetLang);
  if (!tr) return;
  cue('discover', 'success');
  toast(`${tr.word} — new word!`, { emoji: tr.em });
  if (state.settings.speakOnDiscover) {
    speak(citationForm(state.settings.targetLang, tr.word, tr.gender));
  }
  refreshBadges();
}

/* ── Rendering ────────────────────────────────────────────────────────── */

function resize() {
  if (!stage) return;
  bottomGuardPx = 0;                       // re-measure the control deck once
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = stage.clientWidth;
  const h = stage.clientHeight;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

/**
 * Map a video-space box to display space, accounting for object-fit: cover,
 * front-camera mirroring, and the CSS zoom transform (all about the centre).
 */
function project(bbox) {
  const cw = stage.clientWidth;
  const ch = stage.clientHeight;
  const { w: vw, h: vh } = cam.videoSize;
  if (!vw || !vh) return null;

  const sc = Math.max(cw / vw, ch / vh);
  const ox = (cw - vw * sc) / 2;
  const oy = (ch - vh * sc) / 2;

  let x = bbox[0] * sc + ox;
  let y = bbox[1] * sc + oy;
  const w = bbox[2] * sc;
  const h = bbox[3] * sc;

  if (cam.isFront) x = cw - x - w;      // mirror about the vertical centre

  const z = cssZoom;
  x = cw / 2 + (x - cw / 2) * z;
  y = ch / 2 + (y - ch / 2) * z;

  return { x, y, w: w * z, h: h * z, cw, ch };
}

function render() {
  const cw = stage.clientWidth;
  const ch = stage.clientHeight;
  ctx.clearRect(0, 0, cw, ch);

  const seen = new Set();
  const accent = overlayAccent();

  const now = performance.now();
  for (const t of liveTracks) {
    // Extrapolate from the last measurement so labels keep up during a pan.
    const p = project(tracker.predict(t, now));
    if (!p) continue;
    const entry = DICT[t.cls];
    if (!entry) continue;
    seen.add(t.id);

    const fade = tracker.fadeOf(t);
    drawBox(p, fade, accent);
    positionLabel(t, p, fade);
  }

  // Retire labels whose tracks are gone.
  for (const [id, node] of labelNodes) {
    if (seen.has(id)) continue;
    if (!node.classList.contains('is-leaving')) {
      node.classList.add('is-leaving');
      setTimeout(() => { node.remove(); labelNodes.delete(id); }, 200);
    }
  }

  renderFocusChip();
}

/**
 * The classifier's answer, shown as a chip under the frame.
 *
 * It is suppressed when a tracked box already carries the same word, so the
 * user never sees the same thing named twice in two different places.
 */
function renderFocusChip() {
  if (!focusChip) return;
  const r = focusResult;
  const dup = r && liveTracks.some((t) => t.cls === r.key);

  if (!r || dup || !state.settings.focusMode) {
    if (focusChip.dataset.on === '1') {
      focusChip.dataset.on = '0';
      focusChip.classList.remove('is-shown');
    }
    return;
  }

  const tr = translateItem(objId(r.key), state.settings.targetLang);
  if (!tr) return;

  const sig = `${r.key}|${state.settings.targetLang}|${state.settings.showPhonetics}`;
  if (focusChip.dataset.sig !== sig) {
    focusChip.dataset.sig = sig;
    focusChip.dataset.cls = r.key;
    const rtl = LANGUAGES[state.settings.targetLang].rtl;
    focusChip.innerHTML =
      `<span class="em">${esc(tr.em)}</span>` +
      `<span class="body">` +
        `<span class="w" ${rtl ? 'dir="rtl"' : ''}>${esc(tr.word)}</span>` +
        (state.settings.showPhonetics && tr.phonetic
          ? `<span class="ph">${esc(tr.phonetic)}</span>` : '') +
        `<span class="e">${esc(r.key)}</span>` +
      `</span>`;
    focusChip.setAttribute('aria-label', `${tr.word}, ${r.key}. Tap for details.`);
  }
  focusChip.dataset.on = '1';
  focusChip.classList.add('is-shown');
}

/**
 * The stroke colour for the viewfinder brackets.
 *
 * It has to be read from the stage, not the document root: the app's accent is
 * tuned for dark ink on paper, which is invisible over video. `.on-camera`
 * publishes a light variant for exactly this. Cached because getComputedStyle
 * is not cheap and this runs every animation frame.
 */
let accentCache = null;
function overlayAccent() {
  if (accentCache) return accentCache;
  if (!stage) return '#BACB9C';
  const v = getComputedStyle(stage).getPropertyValue('--accent-bright').trim();
  accentCache = v || '#BACB9C';
  return accentCache;
}

/** Drop the cached colour when the theme or accent changes. */
export function invalidateOverlayAccent() { accentCache = null; }

function drawBox({ x, y, w, h }, fade, accent) {
  ctx.save();
  ctx.globalAlpha = fade * 0.38;
  ctx.strokeStyle = 'rgba(255,255,255,0.55)';
  ctx.lineWidth = 1.25;
  roundRect(x, y, w, h, Math.min(10, w * 0.09, h * 0.09));
  ctx.stroke();

  // Corner ticks read as a viewfinder without boxing the subject in.
  ctx.globalAlpha = fade * 0.95;
  ctx.strokeStyle = accent;
  ctx.lineWidth = 2.5;
  ctx.lineCap = 'round';
  const L = Math.min(20, w * 0.22, h * 0.22);
  corner(x, y, L, 1, 1);
  corner(x + w, y, L, -1, 1);
  corner(x + w, y + h, L, -1, -1);
  corner(x, y + h, L, 1, -1);
  ctx.restore();
}

function corner(x, y, L, sx, sy) {
  ctx.beginPath();
  ctx.moveTo(x, y + L * sy);
  ctx.lineTo(x, y);
  ctx.lineTo(x + L * sx, y);
  ctx.stroke();
}

function roundRect(x, y, w, h, r) {
  r = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function positionLabel(t, p, fade) {
  let node = labelNodes.get(t.id);
  const id = objId(t.cls);
  const tr = translateItem(id, state.settings.targetLang);
  if (!tr) return;
  const colorKey = trackColors.get(t.id) || null;

  if (!node) {
    node = el('button', { class: 'ar-label', type: 'button' });
    node.addEventListener('click', () => onLabelTap(node, t.id, t.cls, tr));
    labelLayer.append(node);
    labelNodes.set(t.id, node);
    node.dataset.cls = t.cls;
  }

  // Only touch the DOM when the rendered content actually changes.
  const sig = [t.cls, state.settings.targetLang, state.settings.showPhonetics,
    state.settings.showGender, state.settings.showConfidence,
    state.settings.showColors ? colorKey : '',
    state.settings.colorPhrases ? 'p' : '',
    state.settings.showConfidence ? Math.round(t.score * 20) : 0].join('|');
  if (node.dataset.sig !== sig) {
    node.dataset.sig = sig;
    node.dataset.cls = t.cls;
    node.innerHTML = labelMarkup(t, tr, colorKey);
    node.setAttribute('aria-label', `${tr.word}, ${t.cls}. Tap for details.`);
  }

  node.classList.toggle('is-quiz', state.settings.quizMode);
  if (!state.settings.quizMode) node.classList.remove('is-revealed');
  node.style.opacity = fade;

  /* Label size is measured only when its content changed. Reading offsetWidth
     forces the browser to flush layout, and doing that once per label per
     animation frame was the single largest source of jank in the overlay —
     the numbers cannot change unless the markup did, so cache them. */
  if (node.dataset.mw === undefined || node.dataset.msig !== node.dataset.sig) {
    node.dataset.msig = node.dataset.sig;
    node.dataset.mw = node.offsetWidth || 130;
    node.dataset.mh = node.offsetHeight || 62;
  }
  const lw = +node.dataset.mw;
  const lh = +node.dataset.mh;

  // Reserve the bands occupied by the top chrome and the bottom control deck
  // so a label never hides behind them.
  const topGuard = 96;
  const maxY = Math.max(topGuard, p.ch - bottomGuard() - lh);
  const lx = clamp(p.x, 8, Math.max(8, p.cw - lw - 8));
  let ly = p.y - lh - 8;
  if (ly < topGuard) ly = p.y + p.h + 8;
  ly = clamp(ly, topGuard, maxY);

  node.style.transform = `translate3d(${Math.round(lx)}px, ${Math.round(ly)}px, 0)`;
}

/* Height of the bottom control deck, measured once per layout rather than
   once per label per frame. Invalidated by resize(). */
let bottomGuardPx = 0;
function bottomGuard() {
  if (!bottomGuardPx) {
    bottomGuardPx = (document.getElementById('cam-bottom')?.offsetHeight || 120) + 12;
  }
  return bottomGuardPx;
}

function labelMarkup(t, tr, colorKey) {
  const lang = state.settings.targetLang;
  const g = state.settings.showGender ? genderLabel(lang, tr.gender) : '';
  const rtl = LANGUAGES[lang].rtl;

  /* Colour row: the phrase when we can build one that agrees correctly,
     otherwise just the colour word. Never a phrase we are unsure of. */
  let colorRow = '';
  if (colorKey && state.settings.showColors && COLORS[colorKey]) {
    const forms = colorForms(colorKey, lang);
    const phrase = state.settings.colorPhrases
      ? buildColorPhrase(lang, tr.word, tr.gender, forms)
      : null;
    const text = phrase || (forms ? forms.cite : '');
    if (text) {
      colorRow = `<div class="ar-color">
        <span class="sw" style="background:${COLORS[colorKey].swatch}"></span>
        <span class="phrase" ${rtl ? 'dir="rtl"' : ''}>${esc(text)}</span>
      </div>`;
    }
  }

  return `
    <div class="ar-word" ${rtl ? 'dir="rtl"' : ''}>
      <span class="w">${esc(tr.word)}</span>
      ${g ? `<span class="g">${esc(g)}</span>` : ''}
    </div>
    ${state.settings.showPhonetics && tr.phonetic ? `<div class="ar-phon">${esc(tr.phonetic)}</div>` : ''}
    ${colorRow}
    <div class="ar-en">
      <span class="em">${esc(tr.em)}</span>
      <span class="txt">${esc(t.cls)}</span>
      ${state.settings.showConfidence ? `<span class="conf">${Math.round(t.score * 100)}%</span>` : ''}
    </div>`;
}

function onLabelTap(node, trackId, cls, tr) {
  unlockAudio();
  if (state.settings.quizMode && !node.classList.contains('is-revealed')) {
    node.classList.add('is-revealed');
    cue('correct', 'medium');
    speak(citationForm(state.settings.targetLang, tr.word, tr.gender));
    return;
  }
  haptic('light');
  openWordSheet(objId(cls), trackColors.get(trackId) || null);
}

/* ── Detected strip ───────────────────────────────────────────────────── */

function syncStrip(tracks) {
  const present = new Set();
  for (const t of tracks) {
    if (present.has(t.cls)) continue;
    present.add(t.cls);
    const tr = translateItem(objId(t.cls), state.settings.targetLang);
    if (!tr) continue;

    let card = stripCards.get(t.cls);
    if (!card) {
      const cls = t.cls;
      const trackId = t.id;
      card = el('button', { class: 'detected-card', type: 'button' });
      card.addEventListener('click', () => {
        haptic('light');
        openWordSheet(objId(cls), trackColors.get(trackId) || null);
      });
      strip.append(card);
      stripCards.set(t.cls, card);
      // Keep the strip short so it never becomes a wall of cards.
      while (strip.children.length > 8) {
        const first = strip.firstElementChild;
        for (const [k, v] of stripCards) if (v === first) stripCards.delete(k);
        first.remove();
      }
    }
    const sig = t.cls + '|' + state.settings.targetLang;
    if (card.dataset.sig !== sig) {
      card.dataset.sig = sig;
      card.innerHTML =
        `<span class="em">${esc(tr.em)}</span>` +
        `<span class="body"><span class="w">${esc(tr.word)}</span>` +
        `<span class="e">${esc(t.cls)}</span></span>`;
    }
    card.classList.remove('is-stale');
  }
  for (const [cls, card] of stripCards) {
    if (!present.has(cls)) card.classList.add('is-stale');
  }
}

function clearStrip() {
  stripCards.clear();
  if (strip) strip.innerHTML = '';
}

/* ── Status pill ──────────────────────────────────────────────────────── */

/* The pill shows counters that change a few times a second at most, so
   rebuilding it at display refresh was pure waste. */
let lastStatusAt = 0;

function updateStatus(now = performance.now()) {
  if (!statusPill) return;
  if (now - lastStatusAt < 250) return;
  lastStatusAt = now;
  statusPill.classList.toggle('is-paused', paused);
  const g = governor.stats(clamp(state.settings.detectHz, 2, 20));
  state.runtime.detectStats = g;
  // Surfacing the throttle is honest: the user can see the device is the limit
  // rather than assuming detection is broken.
  const count = liveTracks.length + (focusResult && !liveTracks.some((t) => t.cls === focusResult.key) ? 1 : 0);
  const label = paused
    ? 'Paused'
    : `${count} live · ${state.runtime.fps} fps${g.throttling ? ` · ${g.effectiveHz}Hz` : ''}`;
  if (statusPill.dataset.txt !== label) {
    statusPill.dataset.txt = label;
    statusPill.innerHTML = `<span class="live-dot"></span><span>${esc(label)}</span>`;
  }
}

/* ── Controls ─────────────────────────────────────────────────────────── */

function wireControls() {
  $('#btn-flip')?.addEventListener('click', flipCamera);
  $('#btn-torch')?.addEventListener('click', toggleTorch);
  $('#btn-quiz')?.addEventListener('click', toggleQuiz);
  $('#btn-pause')?.addEventListener('click', togglePause);
  $('#btn-zoom')?.addEventListener('click', cycleZoom);
  $('#btn-shutter')?.addEventListener('click', capture);
  $('#btn-import')?.addEventListener('click', () => $('#photo-input')?.click());
  $('#photo-input')?.addEventListener('change', onPhotoPicked);
  $('#btn-lang')?.addEventListener('click', openLanguagePicker);
  $('#btn-help')?.addEventListener('click', openTips);

  focusChip?.addEventListener('click', () => {
    const cls = focusChip.dataset.cls;
    if (!cls) return;
    haptic('light');
    unlockAudio();
    openWordSheet(objId(cls));
  });

  stage?.addEventListener('click', onStageTap);
}

function onStageTap(e) {
  if (e.target.closest('.ar-label, .detected-card, button')) return;
  unlockAudio();
  const rect = stage.getBoundingClientRect();
  const ring = $('#focus-ring');
  if (ring) {
    ring.style.left = (e.clientX - rect.left) + 'px';
    ring.style.top = (e.clientY - rect.top) + 'px';
    ring.classList.remove('is-firing');
    void ring.offsetWidth;              // restart the animation
    ring.classList.add('is-firing');
  }
  haptic('select');
  cam?.refocus();
}

async function flipCamera() {
  if (!cam) return;
  haptic('medium');
  try {
    await cam.flip();
    state.runtime.cameraFacing = cam.facing;
    applyMirror();
    tracker.reset();
    palette.reset();
    trackColors.clear();
    governor.reset();
    classifier.reset();
    scene.reset();
    countedTracks.clear();
    lastFocusKey = null;
    liveTracks = [];
    focusResult = null;
    clearLabels();
    syncTorchButton();
    toast(cam.isFront ? 'Front camera' : 'Rear camera', { emoji: '🔄' });
  } catch {
    toast('Could not switch camera');
  }
}

function applyMirror() {
  video.classList.toggle('is-mirrored', cam.isFront);
}

async function toggleTorch() {
  if (!cam?.caps.torch) { toast('Torch is not available on this camera'); return; }
  const next = !state.runtime.torch;
  const ok = await cam.setTorch(next);
  if (!ok) { toast('Torch could not be toggled'); return; }
  state.runtime.torch = next;
  haptic('medium');
  syncTorchButton();
}

function syncTorchButton() {
  const b = $('#btn-torch');
  if (!b) return;
  b.hidden = !cam?.caps.torch;
  b.classList.toggle('is-on', state.runtime.torch);
  b.innerHTML = state.runtime.torch ? icon('bolt') : icon('boltOff');
  b.setAttribute('aria-pressed', String(state.runtime.torch));
}

function toggleQuiz() {
  const next = !state.settings.quizMode;
  setSetting('quizMode', next);
  $('#btn-quiz')?.classList.toggle('is-on', next);
  $('#btn-quiz')?.setAttribute('aria-pressed', String(next));
  labelNodes.forEach((n) => n.classList.remove('is-revealed'));
  haptic('medium');
  toast(next ? 'Quiz mode on — tap a label to reveal' : 'Quiz mode off', { emoji: next ? '🎯' : '👁️' });
}

function togglePause() {
  paused = !paused;
  state.runtime.detecting = !paused;
  const b = $('#btn-pause');
  if (b) {
    b.innerHTML = paused ? icon('play') : icon('pause');
    b.classList.toggle('is-on', paused);
    b.setAttribute('aria-label', paused ? 'Resume detection' : 'Pause detection');
  }
  haptic('medium');
}

/** Step through zoom levels; uses optical zoom where the device offers it. */
async function cycleZoom() {
  const steps = [1, 1.5, 2, 3];
  const idx = steps.indexOf(cssZoom);
  const next = steps[(idx + 1) % steps.length];
  cssZoom = next;
  state.runtime.zoom = next;

  const optical = cam?.caps.zoom;
  if (optical) {
    const range = optical.max - optical.min;
    const mapped = optical.min + range * ((next - 1) / 2);
    const ok = await cam.setZoom(Math.min(optical.max, mapped));
    // When optical zoom works the frame itself changes, so no CSS scaling.
    if (ok) cssZoom = 1;
  }
  stage.style.setProperty('--cam-zoom', cssZoom);
  const b = $('#btn-zoom');
  if (b) { b.textContent = next + '×'; b.classList.toggle('is-on', next !== 1); }
  haptic('light');
}

/* ── Capture ──────────────────────────────────────────────────────────── */

async function capture() {
  unlockAudio();
  cue('shutter', 'heavy');
  const flash = $('#flash');
  flash.classList.remove('is-firing');
  void flash.offsetWidth;
  flash.classList.add('is-firing');

  try {
    const blob = await composeSnapshot();
    const name = `lemma-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '')}.png`;
    const file = new File([blob], name, { type: 'image/png' });

    // Prefer the native share sheet on mobile; fall back to a download.
    if (navigator.canShare?.({ files: [file] })) {
      await navigator.share({ files: [file], title: 'Lemma' });
    } else {
      const url = URL.createObjectURL(blob);
      const a = el('a', { href: url, download: name });
      document.body.append(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 4000);
      toast('Snapshot saved', { emoji: '📸' });
    }
    logSnapshot();
    refreshBadges();
  } catch (err) {
    if (err?.name !== 'AbortError') toast('Could not save the snapshot');
  }
}

/** Compose the frame, the AR overlay, and a branded caption strip. */
function composeSnapshot() {
  const cw = stage.clientWidth;
  const ch = stage.clientHeight;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const footer = 78;

  const out = document.createElement('canvas');
  out.width = Math.round(cw * dpr);
  out.height = Math.round((ch + footer) * dpr);
  const g = out.getContext('2d');
  g.scale(dpr, dpr);

  // 1. The video frame, cropped exactly as the viewport shows it.
  const { w: vw, h: vh } = cam.videoSize;
  const sc = Math.max(cw / vw, ch / vh) * cssZoom;
  const dw = vw * sc;
  const dh = vh * sc;
  g.save();
  g.beginPath(); g.rect(0, 0, cw, ch); g.clip();
  if (cam.isFront) { g.translate(cw, 0); g.scale(-1, 1); }
  g.drawImage(video, (cw - dw) / 2, (ch - dh) / 2, dw, dh);
  g.restore();

  // 2. The boxes and corner ticks.
  g.drawImage(canvas, 0, 0, cw, ch);

  // 3. Word cards for whatever is on screen.
  const words = [...new Set(liveTracks.map((t) => t.cls))].slice(0, 4);
  const styles = getComputedStyle(document.documentElement);
  const accent = styles.getPropertyValue('--accent').trim() || '#A8C93A';

  g.fillStyle = '#F4F1E8';
  g.fillRect(0, ch, cw, footer);
  g.fillStyle = accent;
  g.fillRect(0, ch, cw, 2);

  g.font = '700 15px -apple-system, system-ui, sans-serif';
  g.fillStyle = '#24241D';
  g.textBaseline = 'top';

  const lang = LANGUAGES[state.settings.targetLang];
  g.fillText(`Lemma · ${lang.flag} ${lang.name}`, 16, ch + 14);

  g.font = '500 12px -apple-system, system-ui, sans-serif';
  g.fillStyle = 'rgba(36,36,29,0.66)';
  const line = words
    .map((c) => { const t = translateItem(objId(c), state.settings.targetLang); return t ? `${t.em} ${t.word}` : ''; })
    .filter(Boolean).join('   ');
  g.fillText(line || 'Point the camera at an object to begin', 16, ch + 38);

  g.font = '500 10px -apple-system, system-ui, sans-serif';
  g.fillStyle = 'rgba(36,36,29,0.38)';
  g.fillText('rushanhaque.online', 16, ch + 58);

  return new Promise((res, rej) =>
    out.toBlob((b) => (b ? res(b) : rej(new Error('Encoding failed'))), 'image/png'));
}

/* ── Photo import ─────────────────────────────────────────────────────── */

async function onPhotoPicked(e) {
  const file = e.target.files?.[0];
  e.target.value = '';
  if (!file || !model) return;

  toast('Analysing photo…', { emoji: '🔍' });
  try {
    const bitmap = await createImageBitmap(file);
    const c = document.createElement('canvas');
    c.width = bitmap.width; c.height = bitmap.height;
    c.getContext('2d').drawImage(bitmap, 0, 0);

    const raw = await model.detect(c, 20, 0.25);
    const found = [...new Set(
      raw.filter((p) => DICT[p.class] && p.score >= state.settings.confidence).map((p) => p.class)
    )];

    // Give the photo the same extended vocabulary the live view gets, so a
    // picture of a pen or a stapler is not reported as "nothing recognisable".
    if (classifier.ready) {
      try {
        const preds = await classifier.classifyStill(c);
        for (const key of preds) if (!found.includes(key)) found.push(key);
      } catch { /* the box detections still stand */ }
    }
    bitmap.close?.();

    if (!found.length) { toast('Nothing recognisable in that photo'); return; }
    found.forEach((cls) => recordSighting(cls));
    refreshBadges();
    showPhotoResults(found);
  } catch {
    toast('Could not read that image');
  }
}

function showPhotoResults(classes) {
  const lang = state.settings.targetLang;
  const body = openSheet({ title: `Found ${classes.length} object${classes.length > 1 ? 's' : ''}` });
  body.innerHTML = `<div class="group">${classes.map((cls) => {
    const t = translateItem(objId(cls), lang);
    if (!t) return '';
    const g = genderLabel(lang, t.gender);
    return `<button class="list-row is-tappable" data-cls="${esc(cls)}">
      <div class="list-row-icon">${esc(t.em)}</div>
      <div class="list-row-body">
        <div class="list-row-title">${esc(t.word)}
          ${g ? `<span class="gender-tag" style="margin-left:6px">${esc(g)}</span>` : ''}</div>
        <div class="list-row-sub">${esc(cls)}</div>
      </div>
      <div class="list-row-chevron">${icon('chevronRight')}</div>
    </button>`;
  }).join('')}</div>`;
  $$('[data-cls]', body).forEach((n) =>
    n.addEventListener('click', () => {
      closeSheet();
      setTimeout(() => openWordSheet(objId(n.dataset.cls)), 260);
    }));
}

/* ── Language picker ──────────────────────────────────────────────────── */

export function openLanguagePicker() {
  const body = openSheet({ title: 'Learning language' });
  body.innerHTML =
    '<p class="footnote" style="margin-bottom:var(--s-4)">Progress is tracked separately for each language.</p>' +
    `<div class="lang-grid">${Object.values(LANGUAGES).map((l) => `
      <button class="lang-card ${l.code === state.settings.targetLang ? 'is-on' : ''}" data-lang="${l.code}">
        <span class="flag">${l.flag}</span>
        <span class="grow"><span class="n">${esc(l.name)}</span><br>
        <span class="native" ${l.rtl ? 'dir="rtl"' : ''}>${esc(l.native)}</span></span>
      </button>`).join('')}</div>`;

  $$('[data-lang]', body).forEach((n) => n.addEventListener('click', () => {
    setLanguage(n.dataset.lang);
    closeSheet();
  }));
}

export function setLanguage(code) {
  if (!LANGUAGES[code]) return;
  setSetting('targetLang', code);
  clearLabels();
  clearStrip();
  syncLangPill();
  haptic('medium');
  toast(`Now learning ${LANGUAGES[code].name}`, { emoji: LANGUAGES[code].flag });
  emit('change');
}

export function syncLangPill() {
  const pill = $('#btn-lang');
  if (!pill) return;
  const l = LANGUAGES[state.settings.targetLang];
  pill.innerHTML = `<span class="flag">${l.flag}</span><span>${esc(l.name)}</span>${icon('chevronDown')}`;
  pill.setAttribute('aria-label', `Learning language: ${l.name}. Tap to change.`);
}

function clearLabels() {
  labelNodes.forEach((n) => n.remove());
  labelNodes.clear();
  trackColors.clear();
}

/* ── Errors ───────────────────────────────────────────────────────────── */

const ERROR_COPY = {
  [CameraError.DENIED]: {
    em: '🔒', title: 'Camera access blocked',
    body: 'Lemma needs the camera to recognise objects. Allow camera access in your browser’s site settings, then reload.'
  },
  [CameraError.NOT_FOUND]: {
    em: '🎥', title: 'No camera found',
    body: 'No usable camera is attached to this device. You can still study your saved vocabulary from the Learn tab.'
  },
  [CameraError.INSECURE]: {
    em: '🔐', title: 'Secure connection required',
    body: 'Browsers only grant camera access over HTTPS or on localhost. Open the app over https:// and try again.'
  },
  [CameraError.IN_USE]: {
    em: '📵', title: 'Camera is busy',
    body: 'Another app or tab is using the camera. Close it and reload this page.'
  },
  [CameraError.UNSUPPORTED]: {
    em: '🧭', title: 'Browser not supported',
    body: 'This browser does not expose a camera API. Try Safari, Chrome, or Edge.'
  },
  model: {
    em: '📡', title: 'Model download failed',
    body: 'The object-detection model could not be downloaded. Check your connection and reload — after one successful load it works offline.'
  }
};

function showError(code, detail) {
  const panel = $('#cam-error');
  if (!panel) return;
  const copy = ERROR_COPY[code] || {
    em: '⚠️', title: 'Something went wrong', body: detail || 'The camera could not be started.'
  };
  panel.innerHTML = `
    <div class="em">${copy.em}</div>
    <h2 class="title-2">${esc(copy.title)}</h2>
    <p class="subhead" style="max-width:34ch;line-height:1.55">${esc(copy.body)}</p>
    <div class="row gap-2" style="margin-top:var(--s-2)">
      <button class="btn btn-primary" id="btn-retry-camera">Reload</button>
      <button class="btn" id="btn-goto-learn">Go to Learn</button>
    </div>`;
  panel.classList.add('is-shown');
  $('#btn-retry-camera').addEventListener('click', () => location.reload());
  $('#btn-goto-learn').addEventListener('click', () => emit('navigate', { tab: 'learn' }));
}

function hideError() {
  $('#cam-error')?.classList.remove('is-shown');
}

/* ── Lifecycle ────────────────────────────────────────────────────────── */

export function onCameraVisible(visible) {
  // Detection is expensive; idle it whenever the camera is off screen.
  state.runtime.detecting = visible && !paused;
  if (!visible) {
    liveTracks = [];
    focusResult = null;
    clearLabels();
    // The scene will have changed by the time we come back, and stale
    // trackers would spend their grace period describing the old one.
    tracker.reset();
    classifier.reset();
    scene.reset();
  }
}

/** True once the camera view has been wired up. */
export function isCameraReady() { return !!stage; }

/**
 * Called whenever the target language changes — possibly from Settings or the
 * onboarding tour, before initCamera() has run. Every step here is written to
 * be safe on a view that has not been built yet.
 */
export function refreshCameraLanguage() {
  clearLabels();
  clearStrip();
  syncLangPill();
  palette.reset();
}

export { toggleQuiz, capture, flipCamera, togglePause };

/** Called after the shell is built so button glyphs and state line up. */
export function paintCameraChrome() {
  syncLangPill();
  syncTorchButton();
  const glyphs = {
    '#btn-flip': 'flip',
    '#btn-pause': 'pause',
    '#btn-import': 'image',
    '#btn-help': 'info',
    '#btn-quiz': 'target'
  };
  for (const [sel, name] of Object.entries(glyphs)) {
    const node = $(sel);
    if (node && !node.firstElementChild) node.innerHTML = icon(name);
  }
  $('#btn-quiz')?.classList.toggle('is-on', state.settings.quizMode);
}

/* ── Tips ─────────────────────────────────────────────────────────────── */

const TIPS = [
  ['🎯', 'Fill the frame', 'Get close enough that the object takes up a good part of the view — small, distant things are hard to recognise.'],
  ['🔎', 'Centre what you want named', 'Boxes appear around the things Lemma can locate. Anything else — a pen, a stapler, a globe — is named by pointing the middle of the frame straight at it.'],
  ['💡', 'Give it light', 'Detection accuracy drops sharply in dim rooms. Use the torch button if your device has one.'],
  ['🐢', 'Hold steady', 'A label appears only after the same object is seen for several frames, which keeps the overlay calm.'],
  ['👆', 'Tap a label', 'Open the full word card: gender, phonetics, example sentences, and the same word in every other language.'],
  ['🎓', 'Quiz yourself', 'Quiz mode blurs the translation so you can guess first, then tap to check.'],
  ['🖼️', 'Use a photo', 'No good subject nearby? Load a picture from your library and Lemma will read it the same way.'],
  ['🔒', 'Nothing is uploaded', 'The model runs in this browser. Your camera feed never leaves the device.']
];

function openTips() {
  const body = openSheet({ title: 'Getting the best results' });
  body.innerHTML = `<div class="group">${TIPS.map(([em, title, text]) => `
    <div class="list-row" style="align-items:flex-start">
      <div class="list-row-icon">${em}</div>
      <div class="list-row-body">
        <div class="list-row-title">${esc(title)}</div>
        <div class="list-row-sub" style="line-height:1.5;white-space:normal">${esc(text)}</div>
      </div>
    </div>`).join('')}</div>`;
}
