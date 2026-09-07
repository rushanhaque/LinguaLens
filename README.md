# LinguaLens

**Point your camera at the world and learn what everything is called.**

LinguaLens recognises everyday objects through your camera and labels them, in real
time, in the language you are learning — with pronunciation, grammatical gender,
example sentences, and a spaced-repetition system that turns what you spotted today
into vocabulary you still know next month.

Everything runs in the browser. No image or video ever leaves the device.

---

## What it does

**Camera** — Live AR labels tracked across frames, with a viewfinder overlay, quiz
mode that blurs the answer until you tap, front/rear switching, torch, zoom,
tap-to-focus, a photo-library import for when there is nothing good to point at,
and a branded snapshot export that shares through the native share sheet.

**Learn** — Category decks and a review session with four question types: free
recall with self-grading, multiple choice, listening comprehension, and typed
spelling. Scheduling is SM-2 with three grades; answers are compared
case-, accent- and punctuation-insensitively.

**Progress** — Level and XP ring, a twelve-week activity heatmap, per-category and
per-language completion, streaks, and twenty-one achievements.

**Settings** — Twelve languages, light/dark/auto theming, six accent colours,
detection tuning (confidence, label count, detection rate, model choice), speech
rate and voice testing, haptics and sound, daily goal, and full JSON
export/restore of everything you have learned.

## Languages

Spanish · French · German · Italian · Portuguese · Dutch · Russian · Japanese ·
Korean · Chinese · Hindi · Arabic

All 80 recognisable object classes are translated into all twelve — 960
translations, each with a phonetic guide or romanisation and a grammatical gender
where the language has one. Progress is tracked separately per language.

Articles and example sentences are **generated, not stored**: the grammar engine in
`src/data/languages.js` derives definite, indefinite and (for German) accusative
forms from one stored noun plus its gender, handles French and Italian elision
(*l'orange*, *un'arancia*), and suppresses articles for plural-only nouns
(*les ciseaux*). That is why every word gets five correct example sentences without
a single one being written by hand.

## Running it

It is a static site with **no build step**. Serve the folder over HTTP:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000`.

Camera access requires a **secure context** — `https://` or `localhost`. Opening
`index.html` directly from the filesystem will not work, because ES modules and
`getUserMedia` both require an origin.

## Deploying

Push the folder to any static host. `vercel.json` and `netlify.toml` are included
and set the headers that matter: `no-cache` on `sw.js` and `index.html` so updates
land, and a `Permissions-Policy` that grants the camera to this origin only.

- **Vercel** — `vercel deploy --prod`
- **Netlify** — drag the folder onto the dashboard, or `netlify deploy --prod`
- **GitHub Pages** — push and enable Pages on the branch root
- **Cloudflare Pages** — connect the repo, leave the build command empty

## Offline

The service worker caches the shell (stale-while-revalidate, so returning visits
paint instantly and update in the background) and caches TensorFlow.js plus the
model weights permanently, since they are versioned by URL and far too large to
re-fetch. After one successful online load the app works with no network at all.
When a new version is deployed, a tap-to-update toast appears.

## Architecture

```
index.html            App shell and launch screen
manifest.json         PWA metadata, icons, shortcuts
sw.js                 Offline caching and update flow

styles/
  tokens.css          Design system: type ramp, materials, themes, accents
  base.css            Reset, layout primitives, shared animations
  components.css      Buttons, lists, sheets, switches, sliders, tab bar
  views.css           Per-screen styling

src/
  app.js              Shell: theming, routing, shortcuts, PWA, boot
  version.js

  data/
    languages.js      Language metadata, TTS locales, grammar/sentence engine
    dictionary.js     80 classes x 12 languages, categories, size priors
    achievements.js   Badges, levels, XP table

  core/
    store.js          State, persistence, progress, streaks, XP, stats
    tracker.js        Detection filtering and multi-object tracking
    camera.js         getUserMedia lifecycle, torch, zoom, capabilities
    srs.js            SM-2 scheduling, session and distractor building
    speech.js         Voice selection and pronunciation
    feedback.js       Haptics and synthesised UI sound

  ui/
    icons.js          Inline SVG icon set
    kit.js            DOM helpers, toasts, sheet controller, confirm dialog

  views/
    camera.js         Live AR translation
    learn.js          Study hub and the four review modes
    progress.js       Stats, heatmap, badges
    settings.js       Preferences and data management
    wordSheet.js      Shared word detail sheet
    onboarding.js     First-run tour
```

### How detection is kept stable

COCO-SSD emits an unordered, noisy list of boxes every frame. `core/tracker.js`
turns that into tracks with identities:

1. **Aspect scoring** penalises boxes shaped wrong for their class — a wide, short
   box is a poor "bottle" no matter what the model says.
2. **Size gating** rejects boxes far outside a class's plausible share of the frame.
3. **NMS** drops overlapping duplicates.
4. **Confusion resolution** picks one winner among look-alike groups, so a phone
   and a remote cannot both claim the same rectangle.
5. **Greedy assignment** matches detections to existing tracks, preferring
   same-class matches so identity survives a one-frame flicker.
6. **Confirmation and decay** require several hits before a label appears and
   tolerate a grace period of misses before it leaves.
7. **Per-track smoothing** interpolates position and size.

An established track also resists relabelling: a rival class must score
substantially better before it can take over. Two cats side by side stay two
tracks with two labels — the earlier prototype keyed everything by class name and
collapsed them into one.

Detection runs at a configurable rate (8 Hz by default) while rendering runs at
display refresh, so labels move smoothly without paying for inference every frame.

## Privacy

Object recognition runs locally via TensorFlow.js. Camera frames are never
uploaded, never sent to a server, and never stored. All progress lives in this
browser's `localStorage` and can be exported or erased from Settings.

## Browser support

Chrome, Edge, Safari 16.4+, and Firefox on desktop and mobile. Speech
pronunciation depends on the voices your operating system has installed —
Settings lists any language without one, and text and phonetics still work.

---

Developed by [Rushan Haque](https://www.rushanhaque.online)
