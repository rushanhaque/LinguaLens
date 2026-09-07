/**
 * progress.js — Stats, streak calendar, per-language breakdown, badges.
 */

import { ACHIEVEMENTS } from '../data/achievements.js';
import { GROUPS, GROUP_KEYS, idsInGroup, SOURCE_OBJ, TOTAL_OBJECTS } from '../data/vocab.js';
import { LANGUAGES } from '../data/languages.js';
import { state, getStats, languageBreakdown, langProgress, todayKey } from '../core/store.js';
import { masteryLevel } from '../core/srs.js';
import { esc } from '../ui/kit.js';
import { icon } from '../ui/icons.js';

let root = null;

export function initProgress(container) {
  root = container;
  renderProgress();
}

export function renderProgress() {
  if (!root) return;
  const s = getStats();
  const lvl = s.level;
  const words = langProgress(s.lang);

  root.innerHTML = `
    <header class="view-header">
      <h1 class="large-title grow">Progress</h1>
    </header>

    <div class="scroll grow view-scroll stagger">
      ${levelCard(lvl, s)}

      <section class="stat-grid">
        ${tile('Words started', `${s.discovered}`, `of ${s.total} · ${s.objectsFound}/${TOTAL_OBJECTS} found by camera`)}
        ${tile('Mastered', `${s.mastered}`, 'long-term memory')}
        ${tile('Reviews', `${s.reviews}`, s.reviews ? `${Math.round(s.accuracy * 100)}% accurate` : 'none yet')}
        ${tile('Streak', `${s.streak}`, `best ${s.bestStreak} day${s.bestStreak === 1 ? '' : 's'}`)}
      </section>

      <section>
        <div class="section-label">Last 12 weeks</div>
        <div class="card card-pad">
          <div class="heatmap" id="heatmap">${heatmap()}</div>
          <div class="row between" style="margin-top:var(--s-3)">
            <span class="caption">Less</span>
            <div class="row gap-1">
              ${[0, 1, 2, 3, 4].map((l) => `<span class="heat-cell" data-level="${l}"></span>`).join('')}
            </div>
            <span class="caption">More</span>
          </div>
        </div>
      </section>

      <section>
        <div class="section-label">Camera categories · ${esc(LANGUAGES[s.lang].name)}</div>
        <div class="group">${groupRows(words, (g) => g.source === SOURCE_OBJ)}</div>
      </section>

      <section>
        <div class="section-label">Core vocabulary packs</div>
        <div class="group">${groupRows(words, (g) => g.source !== SOURCE_OBJ)}</div>
      </section>

      <section>
        <div class="section-label">Languages</div>
        <div class="group">${languageRows()}</div>
      </section>

      <section>
        <div class="section-label">Achievements · ${state.meta.badges.length}/${ACHIEVEMENTS.length}</div>
        <div class="badge-grid">${badgeTiles()}</div>
      </section>
    </div>`;
}

function tile(k, v, s) {
  return `<div class="stat-tile"><div class="k">${esc(k)}</div>
    <div class="v tabular">${esc(v)}</div><div class="s">${esc(s)}</div></div>`;
}

function levelCard(lvl, s) {
  const R = 34;
  const C = 2 * Math.PI * R;
  return `
    <section class="level-card">
      <div class="level-ring">
        <svg class="ring" width="84" height="84" viewBox="0 0 84 84">
          <circle class="ring-track" cx="42" cy="42" r="${R}" stroke-width="7"/>
          <circle class="ring-value" cx="42" cy="42" r="${R}" stroke-width="7"
            stroke-dasharray="${C}" stroke-dashoffset="${C * (1 - lvl.progress)}"/>
        </svg>
        <span class="num">${lvl.level}</span>
      </div>
      <div class="grow">
        <div class="title-3">${esc(lvl.title)}</div>
        <div class="footnote">${s.xp.toLocaleString()} XP${lvl.next
          ? ` · ${(lvl.next.xp - s.xp).toLocaleString()} to ${esc(lvl.next.title)}`
          : ' · max level'}</div>
        <div class="row gap-2" style="margin-top:var(--s-2)">
          <span class="chip">${icon('flame')} ${s.streak}-day streak</span>
          <span class="chip">${s.todayReviews}/${s.dailyGoal} today</span>
        </div>
      </div>
    </section>`;
}

/** 12 weeks of review activity, newest column last (GitHub/Fitness style). */
function heatmap() {
  const cells = [];
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const start = new Date(today);
  start.setDate(start.getDate() - (7 * 12 - 1));
  // Align the first column to a Sunday so rows read as weekdays.
  start.setDate(start.getDate() - start.getDay());

  const tKey = todayKey();
  const counts = state.meta.history || {};

  for (let d = new Date(start); d <= today; d.setDate(d.getDate() + 1)) {
    const key = todayKey(d);
    const n = counts[key] || 0;
    const level = n === 0 ? 0 : n < 5 ? 1 : n < 12 ? 2 : n < 25 ? 3 : 4;
    cells.push(
      `<span class="heat-cell ${key === tKey ? 'is-today' : ''}" data-level="${level}"
             title="${key}: ${n} review${n === 1 ? '' : 's'}"></span>`
    );
  }
  return cells.join('');
}

function groupRows(words, predicate) {
  return GROUP_KEYS.filter((key) => predicate(GROUPS[key])).map((key) => {
    const g = GROUPS[key];
    const ids = idsInGroup(key);
    const have = ids.filter((id) => words[id]).length;
    const mastered = ids.filter((id) => words[id] && masteryLevel(words[id]) >= 4).length;
    const pct = ids.length ? have / ids.length : 0;
    return `
      <div class="lang-progress-row">
        <span class="flag">${g.em}</span>
        <div class="body">
          <div class="n">${esc(g.label)}</div>
          <div class="bar"><div class="bar-fill" style="width:${pct * 100}%;background:${g.color}"></div></div>
        </div>
        <div class="pct">${have}/${ids.length}${mastered ? ` · ★${mastered}` : ''}</div>
      </div>`;
  }).join('');
}

function languageRows() {
  const rows = languageBreakdown().filter((l) => l.count > 0 || l.code === state.settings.targetLang);
  if (!rows.length) return '<div class="empty"><div class="empty-text">Start discovering words to see language progress.</div></div>';
  return rows.map((l) => {
    const meta = LANGUAGES[l.code];
    const pct = l.count / l.total;
    return `
      <div class="lang-progress-row">
        <span class="flag">${meta.flag}</span>
        <div class="body">
          <div class="n">${esc(meta.name)}${l.code === state.settings.targetLang
            ? ' <span class="gender-tag" style="margin-left:4px">active</span>' : ''}</div>
          <div class="bar"><div class="bar-fill" style="width:${pct * 100}%"></div></div>
        </div>
        <div class="pct">${l.count}/${l.total}</div>
      </div>`;
  }).join('');
}

function badgeTiles() {
  return ACHIEVEMENTS.map((a) => {
    const got = state.meta.badges.includes(a.id);
    return `
      <div class="badge-tile ${got ? '' : 'is-locked'}" title="${esc(a.desc)}">
        <span class="em">${a.em}</span>
        <span class="t">${esc(a.title)}</span>
        <span class="d">${esc(a.desc)}</span>
      </div>`;
  }).join('');
}
