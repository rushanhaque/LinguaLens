/**
 * languages.js — Language metadata, TTS locales, and grammar/sentence engines.
 *
 * Each language declares:
 *   code    ISO 639-1 code used as the key throughout the app
 *   name    English name
 *   native  Endonym (what speakers call it)
 *   flag    Regional indicator emoji
 *   tts     BCP-47 locale list, in preference order, for SpeechSynthesis
 *   rtl     Right-to-left script
 *   script  'latin' | 'cyrillic' | 'cjk' | 'kana' | 'hangul' | 'devanagari' | 'arabic'
 *   genders Grammatical genders present (used by the article resolver)
 *   phrases Sentence frames — see buildPhrase()
 */

export const LANGUAGES = {
  es: {
    code: 'es', name: 'Spanish', native: 'Español', flag: '🇪🇸',
    tts: ['es-ES', 'es-MX', 'es-US', 'es'], rtl: false, script: 'latin',
    genders: ['m', 'f'],
    phrases: {
      this_is: (w) => `Esto es ${w.indef}.`,
      i_see: (w) => `Veo ${w.indef}.`,
      where_is: (w) => `¿Dónde está ${w.def}?`,
      i_like: (w) => `Me gusta ${w.def}.`,
      how_much: (w) => `¿Cuánto cuesta ${w.def}?`
    }
  },
  fr: {
    code: 'fr', name: 'French', native: 'Français', flag: '🇫🇷',
    tts: ['fr-FR', 'fr-CA', 'fr'], rtl: false, script: 'latin',
    genders: ['m', 'f'],
    phrases: {
      this_is: (w) => `C'est ${w.indef}.`,
      i_see: (w) => `Je vois ${w.indef}.`,
      where_is: (w) => `Où est ${w.def} ?`,
      i_like: (w) => `J'aime ${w.def}.`,
      how_much: (w) => `Combien coûte ${w.def} ?`
    }
  },
  de: {
    code: 'de', name: 'German', native: 'Deutsch', flag: '🇩🇪',
    tts: ['de-DE', 'de-AT', 'de'], rtl: false, script: 'latin',
    genders: ['m', 'f', 'n'],
    phrases: {
      this_is: (w) => `Das ist ${w.indef}.`,
      i_see: (w) => `Ich sehe ${w.indefAcc}.`,
      where_is: (w) => `Wo ist ${w.def}?`,
      i_like: (w) => `Ich mag ${w.defAcc}.`,
      how_much: (w) => `Was kostet ${w.def}?`
    }
  },
  it: {
    code: 'it', name: 'Italian', native: 'Italiano', flag: '🇮🇹',
    tts: ['it-IT', 'it'], rtl: false, script: 'latin',
    genders: ['m', 'f'],
    phrases: {
      this_is: (w) => `Questo è ${w.indef}.`,
      i_see: (w) => `Vedo ${w.indef}.`,
      where_is: (w) => `Dov'è ${w.def}?`,
      i_like: (w) => `Mi piace ${w.def}.`,
      how_much: (w) => `Quanto costa ${w.def}?`
    }
  },
  pt: {
    code: 'pt', name: 'Portuguese', native: 'Português', flag: '🇵🇹',
    tts: ['pt-BR', 'pt-PT', 'pt'], rtl: false, script: 'latin',
    genders: ['m', 'f'],
    phrases: {
      this_is: (w) => `Isto é ${w.indef}.`,
      i_see: (w) => `Eu vejo ${w.indef}.`,
      where_is: (w) => `Onde está ${w.def}?`,
      i_like: (w) => `Eu gosto d${w.gender === 'f' ? 'a' : 'o'} ${w.word}.`,
      how_much: (w) => `Quanto custa ${w.def}?`
    }
  },
  nl: {
    code: 'nl', name: 'Dutch', native: 'Nederlands', flag: '🇳🇱',
    tts: ['nl-NL', 'nl-BE', 'nl'], rtl: false, script: 'latin',
    genders: ['c', 'n'],
    phrases: {
      this_is: (w) => `Dit is ${w.indef}.`,
      i_see: (w) => `Ik zie ${w.indef}.`,
      where_is: (w) => `Waar is ${w.def}?`,
      i_like: (w) => `Ik hou van ${w.def}.`,
      how_much: (w) => `Hoeveel kost ${w.def}?`
    }
  },
  ru: {
    code: 'ru', name: 'Russian', native: 'Русский', flag: '🇷🇺',
    tts: ['ru-RU', 'ru'], rtl: false, script: 'cyrillic',
    genders: ['m', 'f', 'n'],
    phrases: {
      this_is: (w) => `Это ${w.word}.`,
      i_see: (w) => `Я вижу ${w.word}.`,
      where_is: (w) => `Где ${w.word}?`,
      i_like: (w) => `Мне нравится ${w.word}.`,
      how_much: (w) => `Сколько стоит ${w.word}?`
    }
  },
  ja: {
    code: 'ja', name: 'Japanese', native: '日本語', flag: '🇯🇵',
    tts: ['ja-JP', 'ja'], rtl: false, script: 'kana',
    genders: [],
    phrases: {
      this_is: (w) => `これは${w.word}です。`,
      i_see: (w) => `${w.word}が見えます。`,
      where_is: (w) => `${w.word}はどこですか。`,
      i_like: (w) => `${w.word}が好きです。`,
      how_much: (w) => `${w.word}はいくらですか。`
    }
  },
  ko: {
    code: 'ko', name: 'Korean', native: '한국어', flag: '🇰🇷',
    tts: ['ko-KR', 'ko'], rtl: false, script: 'hangul',
    genders: [],
    phrases: {
      this_is: (w) => `이것은 ${w.word}입니다.`,
      i_see: (w) => `${w.word}이(가) 보여요.`,
      where_is: (w) => `${w.word}은(는) 어디에 있어요?`,
      i_like: (w) => `${w.word}을(를) 좋아해요.`,
      how_much: (w) => `${w.word}은(는) 얼마예요?`
    }
  },
  zh: {
    code: 'zh', name: 'Chinese', native: '中文', flag: '🇨🇳',
    tts: ['zh-CN', 'zh-Hans', 'zh'], rtl: false, script: 'cjk',
    genders: [],
    phrases: {
      this_is: (w) => `这是${w.word}。`,
      i_see: (w) => `我看到${w.word}。`,
      where_is: (w) => `${w.word}在哪里？`,
      i_like: (w) => `我喜欢${w.word}。`,
      how_much: (w) => `${w.word}多少钱？`
    }
  },
  hi: {
    code: 'hi', name: 'Hindi', native: 'हिन्दी', flag: '🇮🇳',
    tts: ['hi-IN', 'hi'], rtl: false, script: 'devanagari',
    genders: ['m', 'f'],
    phrases: {
      this_is: (w) => `यह ${w.word} है।`,
      i_see: (w) => `मुझे ${w.word} दिखता है।`,
      where_is: (w) => `${w.word} कहाँ है?`,
      i_like: (w) => `मुझे ${w.word} पसंद है।`,
      how_much: (w) => `${w.word} कितने का है?`
    }
  },
  ar: {
    code: 'ar', name: 'Arabic', native: 'العربية', flag: '🇸🇦',
    tts: ['ar-SA', 'ar-EG', 'ar'], rtl: true, script: 'arabic',
    genders: ['m', 'f'],
    phrases: {
      this_is: (w) => `${w.gender === 'f' ? 'هذه' : 'هذا'} ${w.word}.`,
      i_see: (w) => `أرى ${w.word}.`,
      where_is: (w) => `أين ${w.word}؟`,
      i_like: (w) => `أحب ${w.word}.`,
      how_much: (w) => `كم سعر ${w.word}؟`
    }
  }
};

export const LANG_CODES = Object.keys(LANGUAGES);

export const PHRASE_LABELS = {
  this_is: 'This is a …',
  i_see: 'I see a …',
  where_is: 'Where is the …?',
  i_like: 'I like the …',
  how_much: 'How much is the …?'
};

/**
 * Fill an English gloss template with a noun, fixing up the indefinite
 * article. "I see a orange" reads as a bug even though it is only the gloss.
 */
export function glossFor(frame, noun) {
  const template = PHRASE_LABELS[frame] || PHRASE_LABELS.this_is;
  const needsAn = /^[aeiou]/i.test(noun) && !/^(uni|use|eu|one)/i.test(noun);
  return template
    .replace(/a …/, (needsAn ? 'an ' : 'a ') + noun)
    .replace('…', noun);
}

/* ── Article resolution ───────────────────────────────────────────────────
   Dictionary entries store the bare noun plus a gender tag. Articles are
   derived here so a single stored form drives definite, indefinite and
   (for German) accusative variants. */

const ARTICLES = {
  es: { def: { m: 'el', f: 'la' }, indef: { m: 'un', f: 'una' } },
  fr: { def: { m: 'le', f: 'la' }, indef: { m: 'un', f: 'une' } },
  it: { def: { m: 'il', f: 'la' }, indef: { m: 'un', f: 'una' } },
  pt: { def: { m: 'o', f: 'a' }, indef: { m: 'um', f: 'uma' } },
  nl: { def: { c: 'de', n: 'het' }, indef: { c: 'een', n: 'een' } },
  de: {
    def: { m: 'der', f: 'die', n: 'das' },
    indef: { m: 'ein', f: 'eine', n: 'ein' },
    defAcc: { m: 'den', f: 'die', n: 'das' },
    indefAcc: { m: 'einen', f: 'eine', n: 'ein' }
  }
};

// French and Italian contract the article before a vowel sound.
function elide(lang, article, word) {
  const vowel = /^[aeiouàâäéèêëíìîïóòôöúùûü]/i.test(word);
  if (!vowel) return `${article} ${word}`;
  if (lang === 'fr' && (article === 'le' || article === 'la')) return `l'${word}`;
  if (lang === 'it' && (article === 'il' || article === 'la')) return `l'${word}`;
  if (lang === 'it' && article === 'una') return `un'${word}`;
  return `${article} ${word}`;
}

/**
 * Resolve every surface form of a noun for a language.
 * @returns {{word:string, gender:string, def:string, indef:string, defAcc:string, indefAcc:string}}
 */
export function resolveForms(lang, word, gender) {
  const table = ARTICLES[lang];
  // 'p' marks plural-only nouns (les ciseaux, las tijeras) — the singular
  // article tables do not apply, so the bare form is used everywhere.
  if (gender === 'p' || !table || !gender) {
    return {
      word, gender: gender || '', plural: gender === 'p',
      def: word, indef: word, defAcc: word, indefAcc: word,
      article: '', articleIndef: ''
    };
  }
  const g = table.def[gender] ? gender : Object.keys(table.def)[0];
  return {
    word,
    gender: g,
    plural: false,
    def: elide(lang, table.def[g], word),
    indef: elide(lang, table.indef[g], word),
    defAcc: elide(lang, (table.defAcc || table.def)[g], word),
    indefAcc: elide(lang, (table.indefAcc || table.indef)[g], word),
    article: table.def[g],
    articleIndef: table.indef[g]
  };
}

/** Build an example sentence for a word in a language. */
export function buildPhrase(lang, word, gender, frame = 'this_is') {
  const meta = LANGUAGES[lang];
  if (!meta) return word;
  const fn = meta.phrases[frame] || meta.phrases.this_is;
  try { return fn(resolveForms(lang, word, gender)); } catch { return word; }
}

/** The article-prefixed citation form learners should memorise. */
export function citationForm(lang, word, gender) {
  const forms = resolveForms(lang, word, gender);
  return forms.def === word ? word : forms.def;
}

/** Gender badge text shown in the UI, or '' when the language has no genders. */
export function genderLabel(lang, gender) {
  if (!gender) return '';
  if (gender === 'p') return 'pl.';
  const map = {
    de: { m: 'der', f: 'die', n: 'das' },
    es: { m: 'el', f: 'la' },
    fr: { m: 'le', f: 'la' },
    it: { m: 'il', f: 'la' },
    pt: { m: 'o', f: 'a' },
    nl: { c: 'de', n: 'het' },
    ru: { m: 'm.', f: 'f.', n: 'n.' },
    hi: { m: 'm.', f: 'f.' },
    ar: { m: 'm.', f: 'f.' }
  };
  return (map[lang] && map[lang][gender]) || '';
}

/* ── Adjective placement ──────────────────────────────────────────────────
   Colour is the attribute a camera can actually read, which makes it the
   natural way to teach adjective agreement. Each language owns its own rule
   rather than sharing a lowest-common-denominator one:

     Romance   noun follows the article, adjective follows the noun, and the
               adjective agrees in gender  ("la pomme rouge")
     Germanic  adjective sits between article and noun in its attributive
               form; in the nominative singular this is invariant, which is
               why only one `attr` form is stored  ("der rote Apfel")
     Slavic    adjective precedes and agrees in gender  ("красное яблоко")
     CJK       adjective precedes; Japanese uses i-adjectives or the の
               particle, Chinese 的, Korean the attributive form
     Arabic    adjective follows and agrees; two indefinites juxtaposed read
               as "a red apple"  ("تفاحة حمراء")

   n comes from resolveForms(), c from colorForms(). */

const COLOR_PHRASE = {
  es: (n, c) => `${n.article} ${n.word} ${c[n.gender] || c.m}`,
  fr: (n, c) => `${n.def} ${n.gender === 'f' ? c.f : c.m}`,
  it: (n, c) => `${n.def} ${n.gender === 'f' ? c.f : c.m}`,
  pt: (n, c) => `${n.article} ${n.word} ${n.gender === 'f' ? c.f : c.m}`,
  de: (n, c) => `${n.article} ${c.attr} ${n.word}`,
  nl: (n, c) => `${n.article} ${c.attr} ${n.word}`,
  ru: (n, c) => `${c[n.gender] || c.m} ${n.word}`,
  ja: (n, c) => `${c.attr}${n.word}`,
  ko: (n, c) => `${c.attr} ${n.word}`,
  zh: (n, c) => `${c.attr}${n.word}`,
  hi: (n, c) => `${c[n.gender] || c.m} ${n.word}`,
  ar: (n, c) => `${n.word} ${c[n.gender] || c.m}`
};

/**
 * Build an agreeing colour + noun phrase, e.g. "la voiture rouge".
 *
 * @param {string} lang
 * @param {string} word    bare noun
 * @param {string} gender  noun gender tag from the dictionary
 * @param {object} forms   colour forms from colorForms()
 * @returns {string|null}  null when the phrase cannot be built correctly
 */
export function buildColorPhrase(lang, word, gender, forms) {
  const fn = COLOR_PHRASE[lang];
  if (!fn || !forms) return null;
  const n = resolveForms(lang, word, gender);
  // Plural-only nouns would need plural adjective agreement, which is not
  // stored — showing a singular adjective there would teach the wrong form.
  if (n.plural) return null;
  try {
    return fn(n, forms).replace(/\s+/g, ' ').trim();
  } catch { return null; }
}

/** "The X is <colour>" — used when an attributive phrase would be ambiguous. */
export function colorIsPhrase(lang, word, gender, forms) {
  const phrase = buildColorPhrase(lang, word, gender, forms);
  return phrase || (forms ? forms.cite : null);
}
