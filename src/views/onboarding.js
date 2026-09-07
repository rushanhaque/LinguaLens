/**
 * onboarding.js — First-run tour and language choice.
 *
 * Deliberately short: three slides, and the third is the language picker so
 * the tour ends with the user having made a real choice rather than tapping
 * "Done" on a marketing screen.
 */

import { LANGUAGES } from '../data/languages.js';
import { state, setSetting } from '../core/store.js';
import { haptic, unlockAudio } from '../core/feedback.js';
import { $, $$, esc } from '../ui/kit.js';

const SLIDES = [
  {
    em: '🔍',
    title: 'Point. Learn. Remember.',
    body: 'Aim your camera at anything around you. Lemma names it in the language you are learning, with pronunciation, gender and colour.'
  },
  {
    em: '🎨',
    title: 'It teaches the hard part',
    body: 'Adjectives that agree, articles that change, sentences you can actually say. Point at a red car and you get “la voiture rouge”, not a word list.'
  },
  {
    em: '🔒',
    title: 'Everything stays on your device',
    body: 'Recognition runs locally in your browser. No photo or video is ever uploaded, and your progress is stored only here.'
  },
  {
    em: '📷',
    title: 'Your browser will ask for the camera',
    body: 'That prompt is your browser, not us. Choosing Allow is what lets Lemma see what you point at — and you can still study every word without it.',
    primer: true
  },
  {
    em: '🌍',
    title: 'Pick a language',
    body: 'You can change this any time, and progress is tracked separately for each one.',
    picker: true
  }
];

let host = null;
let index = 0;
let onDone = null;

export function initOnboarding(container, done) {
  host = container;
  onDone = done;
}

export function showOnboarding() {
  index = 0;
  host.classList.add('is-shown');
  render();
}

export function hideOnboarding() {
  host.classList.remove('is-shown');
}

function render() {
  const slide = SLIDES[index];
  const last = index === SLIDES.length - 1;

  host.innerHTML = `
    <div class="ob-slide is-active">
      ${slide.picker ? '' : `<div class="em">${slide.em}</div>`}
      <h2>${esc(slide.title)}</h2>
      <p>${esc(slide.body)}</p>
      ${slide.picker ? `
        <div class="scroll" style="max-height:44dvh;width:100%;margin-top:var(--s-2)">
          <div class="lang-grid">${Object.values(LANGUAGES).map((l) => `
            <button class="lang-card ${l.code === state.settings.targetLang ? 'is-on' : ''}" data-lang="${l.code}">
              <span class="flag">${l.flag}</span>
              <span class="grow"><span class="n">${esc(l.name)}</span><br>
              <span class="native" ${l.rtl ? 'dir="rtl"' : ''}>${esc(l.native)}</span></span>
            </button>`).join('')}</div>
        </div>` : ''}
    </div>
    <div class="ob-foot">
      <div class="ob-dots">${SLIDES.map((_, i) =>
        `<i class="${i === index ? 'on' : ''}"></i>`).join('')}</div>
      <button class="btn btn-primary btn-lg" id="ob-next">
        ${last ? 'Start learning' : slide.primer ? 'Got it' : 'Continue'}
      </button>
      ${last ? '' : '<button class="btn-plain btn-sm" id="ob-skip">Skip</button>'}
    </div>`;

  $('#ob-next', host).addEventListener('click', () => {
    unlockAudio();
    haptic('medium');
    if (index < SLIDES.length - 1) { index++; render(); }
    else finish();
  });
  $('#ob-skip', host)?.addEventListener('click', finish);

  $$('[data-lang]', host).forEach((n) => n.addEventListener('click', () => {
    setSetting('targetLang', n.dataset.lang);
    $$('[data-lang]', host).forEach((b) => b.classList.toggle('is-on', b === n));
    haptic('select');
  }));
}

function finish() {
  setSetting('onboarded', true);
  hideOnboarding();
  onDone?.();
}
