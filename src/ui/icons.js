/**
 * icons.js — Inline SVG icon set.
 *
 * Stroke-based, 24×24, `currentColor` — so icons inherit text colour and
 * stay crisp at any size. Kept inline rather than as a sprite sheet so the
 * app has zero image requests and works fully offline from first paint.
 */

const wrap = (body, extra = '') =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"
        stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" ${extra}>${body}</svg>`;

export const icons = {
  /* Brand mark — an aperture blade ring around a focal dot. */
  lens: () => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
      stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="9"/>
      <path d="M12 3v6.2M20.8 8.4l-5.9 2M18.4 19.1l-3.6-5M5.6 19.1l3.6-5M3.2 8.4l5.9 2"/>
      <circle cx="12" cy="12" r="2.6" fill="currentColor" stroke="none"/>
    </svg>`,

  camera: () => wrap('<path d="M3 8.5A2.5 2.5 0 0 1 5.5 6h1.8l1.2-2h6.9l1.2 2h1.9A2.5 2.5 0 0 1 21 8.5v9A2.5 2.5 0 0 1 18.5 20h-13A2.5 2.5 0 0 1 3 17.5z"/><circle cx="12" cy="13" r="3.6"/>'),
  cards: () => wrap('<rect x="3" y="6.5" width="13" height="14" rx="2.6"/><path d="M7.5 3.5h10A3 3 0 0 1 20.5 6.5v10"/>'),
  chart: () => wrap('<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>'),
  gear: () => wrap('<circle cx="12" cy="12" r="3.2"/><path d="M19.4 14a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-1.8-.3 1.6 1.6 0 0 0-1 1.5V20a2 2 0 1 1-4 0v-.1A1.6 1.6 0 0 0 9 18.4a1.6 1.6 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.6 1.6 0 0 0 .3-1.8 1.6 1.6 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1A1.6 1.6 0 0 0 4.6 9a1.6 1.6 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 1.8.3H9a1.6 1.6 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 1 1.5 1.6 1.6 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0-.3 1.8V9a1.6 1.6 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1z"/>'),

  chevronRight: () => wrap('<path d="M9 5l7 7-7 7"/>'),
  chevronDown: () => wrap('<path d="M5 9l7 7 7-7"/>'),
  chevronLeft: () => wrap('<path d="M15 5l-7 7 7 7"/>'),
  close: () => wrap('<path d="M6 6l12 12M18 6L6 18"/>'),
  check: () => wrap('<path d="M4.5 12.5l5 5 10-11"/>'),
  plus: () => wrap('<path d="M12 5v14M5 12h14"/>'),

  bolt: () => wrap('<path d="M13.5 2L4 13.5h6.5L10 22l9.5-11.5H13z"/>'),
  boltOff: () => wrap('<path d="M13.5 2L8.6 8M10.2 13.5H4L7 9.9M10 22l4.6-5.6M19.5 10.5H13l1.4-2.4M3 3l18 18"/>'),
  flip: () => wrap('<path d="M3 11a9 9 0 0 1 14.6-7M21 13a9 9 0 0 1-14.6 7"/><path d="M17.5 3.2V7h-3.8M6.5 20.8V17h3.8"/>'),
  speaker: () => wrap('<path d="M11 5L6.5 9H3v6h3.5L11 19z"/><path d="M15.2 8.8a4.5 4.5 0 0 1 0 6.4M18 6a8.5 8.5 0 0 1 0 12"/>'),
  target: () => wrap('<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="1" fill="currentColor" stroke="none"/>'),
  sparkle: () => wrap('<path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z"/><path d="M18.5 15.5l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8z"/>'),
  flame: () => wrap('<path d="M12 22a6.5 6.5 0 0 0 6.5-6.5c0-4.5-4-6.5-4.5-11-2 2-3 3.8-3 6 0 1.4-1 2-1.7 1.3C8.5 11 8 10 8 8.8 6.4 10.4 5.5 12.6 5.5 15.5A6.5 6.5 0 0 0 12 22z"/>'),
  image: () => wrap('<rect x="3" y="4.5" width="18" height="15" rx="2.6"/><circle cx="8.6" cy="10" r="1.7"/><path d="M3.6 17.6l4.6-4.4a2 2 0 0 1 2.8 0l3.2 3.1a2 2 0 0 0 2.8 0l1.5-1.4a2 2 0 0 1 2.8 0l1 1"/>'),
  share: () => wrap('<path d="M12 3v13"/><path d="M8 6.6L12 2.8l4 3.8"/><path d="M5 13v6.5A1.5 1.5 0 0 0 6.5 21h11a1.5 1.5 0 0 0 1.5-1.5V13"/>'),
  download: () => wrap('<path d="M12 3v13"/><path d="M8 12.2l4 3.8 4-3.8"/><path d="M5 15v4.5A1.5 1.5 0 0 0 6.5 21h11a1.5 1.5 0 0 0 1.5-1.5V15"/>'),
  upload: () => wrap('<path d="M12 20V7"/><path d="M8 10.8L12 7l4 3.8"/><path d="M5 15v4.5A1.5 1.5 0 0 0 6.5 21h11a1.5 1.5 0 0 0 1.5-1.5V15"/>'),
  trash: () => wrap('<path d="M4 6.5h16M9.5 6.5V4.8A1.3 1.3 0 0 1 10.8 3.5h2.4a1.3 1.3 0 0 1 1.3 1.3v1.7"/><path d="M6.5 6.5l.9 12.7A1.8 1.8 0 0 0 9.2 21h5.6a1.8 1.8 0 0 0 1.8-1.8l.9-12.7"/><path d="M10.5 10.5v6.5M13.5 10.5v6.5"/>'),
  globe: () => wrap('<circle cx="12" cy="12" r="9"/><path d="M3.2 9.5h17.6M3.2 14.5h17.6"/><path d="M12 3a15 15 0 0 1 0 18 15 15 0 0 1 0-18z"/>'),
  eye: () => wrap('<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="3.2"/>'),
  pause: () => wrap('<path d="M9 4.5v15M15 4.5v15"/>'),
  play: () => wrap('<path d="M7 4.5l12 7.5-12 7.5z"/>'),
  book: () => wrap('<path d="M4 4.8A1.8 1.8 0 0 1 5.8 3H10a2.5 2.5 0 0 1 2 1 2.5 2.5 0 0 1 2-1h4.2A1.8 1.8 0 0 1 20 4.8v12.4a1.8 1.8 0 0 1-1.8 1.8H14a2.5 2.5 0 0 0-2 1 2.5 2.5 0 0 0-2-1H5.8A1.8 1.8 0 0 1 4 17.2z"/><path d="M12 4v16"/>'),
  trophy: () => wrap('<path d="M7 4h10v5a5 5 0 0 1-10 0z"/><path d="M7 5.5H4.5v1A3.5 3.5 0 0 0 8 10M17 5.5h2.5v1A3.5 3.5 0 0 1 16 10"/><path d="M12 14v3.5M8.5 21h7l-.7-3.5h-5.6z"/>'),
  info: () => wrap('<circle cx="12" cy="12" r="9"/><path d="M12 11v5.5"/><circle cx="12" cy="7.8" r="0.9" fill="currentColor" stroke="none"/>'),
  moon: () => wrap('<path d="M20 14.2A8.2 8.2 0 1 1 9.8 4a6.6 6.6 0 0 0 10.2 10.2z"/>'),
  sun: () => wrap('<circle cx="12" cy="12" r="4"/><path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M19.1 4.9l-1.8 1.8M6.7 17.3l-1.8 1.8"/>'),
  filter: () => wrap('<path d="M3 5.5h18M6.5 12h11M10 18.5h4"/>'),
  keyboard: () => wrap('<rect x="2.5" y="6" width="19" height="12" rx="2.2"/><path d="M6 9.5h.01M9.5 9.5h.01M13 9.5h.01M16.5 9.5h.01M6 13h.01M18 13h.01M9 16h6"/>'),
  ear: () => wrap('<path d="M7 9a5 5 0 0 1 10 0c0 2.5-2 3.5-3 5s-.5 3-2.5 3"/><path d="M9.5 18.5a2 2 0 0 1-4 0"/>'),
  refresh: () => wrap('<path d="M3.5 12a8.5 8.5 0 0 1 14.6-5.9L21 9"/><path d="M21 3.5V9h-5.5"/><path d="M20.5 12a8.5 8.5 0 0 1-14.6 5.9L3 15"/><path d="M3 20.5V15h5.5"/>'),
  grid: () => wrap('<rect x="3.5" y="3.5" width="7" height="7" rx="1.8"/><rect x="13.5" y="3.5" width="7" height="7" rx="1.8"/><rect x="3.5" y="13.5" width="7" height="7" rx="1.8"/><rect x="13.5" y="13.5" width="7" height="7" rx="1.8"/>'),
  zoomIn: () => wrap('<circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.4 15.4L21 21M10.5 8v5M8 10.5h5"/>')
};

/** Render an icon by name; unknown names render nothing rather than throwing. */
export function icon(name) {
  const fn = icons[name];
  return fn ? fn() : '';
}
