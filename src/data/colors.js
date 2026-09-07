/**
 * colors.js — Colour vocabulary with adjective agreement.
 *
 * Colour is the one attribute a camera can read directly off an object, which
 * makes it the natural way to teach adjective agreement — the part of Romance,
 * Germanic and Semitic grammar learners get wrong longest.
 *
 * Per language each colour stores:
 *   cite  dictionary/citation form (what a learner memorises)
 *   ph    phonetic guide or romanisation
 *   m/f/n agreement forms, where the language inflects colour adjectives
 *   attr  attributive form used *before* the noun (de, nl, ja, ko, zh)
 *
 * Languages whose colour adjectives are invariable (French "rouge", Spanish
 * "azul") simply repeat the same string in m and f — that is the correct
 * answer, not a placeholder.
 */

export const COLORS = {
  red: {
    em: '🔴', swatch: '#C0483C',
    t: {
      es: { cite: 'rojo', ph: 'ROH-ho', m: 'rojo', f: 'roja' },
      fr: { cite: 'rouge', ph: 'roozh', m: 'rouge', f: 'rouge' },
      de: { cite: 'rot', ph: 'roht', attr: 'rote' },
      it: { cite: 'rosso', ph: 'ROS-so', m: 'rosso', f: 'rossa' },
      pt: { cite: 'vermelho', ph: 'ver-MEH-lyoo', m: 'vermelho', f: 'vermelha' },
      nl: { cite: 'rood', ph: 'roht', attr: 'rode' },
      ru: { cite: 'красный', ph: 'KRAS-nee', m: 'красный', f: 'красная', n: 'красное' },
      ja: { cite: '赤', ph: 'aka', attr: '赤い' },
      ko: { cite: '빨강', ph: 'ppalgang', attr: '빨간' },
      zh: { cite: '红色', ph: 'hóng sè', attr: '红色的' },
      hi: { cite: 'लाल', ph: 'laal', m: 'लाल', f: 'लाल' },
      ar: { cite: 'أحمر', ph: 'ahmar', m: 'أحمر', f: 'حمراء' }
    }
  },
  orange: {
    em: '🟠', swatch: '#CE8340',
    t: {
      es: { cite: 'naranja', ph: 'na-RAN-ha', m: 'naranja', f: 'naranja' },
      fr: { cite: 'orange', ph: 'o-RAHNZH', m: 'orange', f: 'orange' },
      de: { cite: 'orange', ph: 'o-RAHN-zheh', attr: 'orange' },
      it: { cite: 'arancione', ph: 'a-ran-CHO-neh', m: 'arancione', f: 'arancione' },
      pt: { cite: 'laranja', ph: 'la-RAN-zha', m: 'laranja', f: 'laranja' },
      nl: { cite: 'oranje', ph: 'o-RAHN-yeh', attr: 'oranje' },
      ru: { cite: 'оранжевый', ph: 'a-RAN-zhe-vee', m: 'оранжевый', f: 'оранжевая', n: 'оранжевое' },
      ja: { cite: 'オレンジ色', ph: 'orenji-iro', attr: 'オレンジ色の' },
      ko: { cite: '주황색', ph: 'juhwangsaek', attr: '주황색' },
      zh: { cite: '橙色', ph: 'chéng sè', attr: '橙色的' },
      hi: { cite: 'नारंगी', ph: 'naarangee', m: 'नारंगी', f: 'नारंगी' },
      ar: { cite: 'برتقالي', ph: 'burtuqaalee', m: 'برتقالي', f: 'برتقالية' }
    }
  },
  yellow: {
    em: '🟡', swatch: '#CFAE45',
    t: {
      es: { cite: 'amarillo', ph: 'a-ma-REE-yo', m: 'amarillo', f: 'amarilla' },
      fr: { cite: 'jaune', ph: 'zhohn', m: 'jaune', f: 'jaune' },
      de: { cite: 'gelb', ph: 'gelp', attr: 'gelbe' },
      it: { cite: 'giallo', ph: 'JAL-lo', m: 'giallo', f: 'gialla' },
      pt: { cite: 'amarelo', ph: 'a-ma-REH-loo', m: 'amarelo', f: 'amarela' },
      nl: { cite: 'geel', ph: 'khayl', attr: 'gele' },
      ru: { cite: 'жёлтый', ph: 'ZHOL-tee', m: 'жёлтый', f: 'жёлтая', n: 'жёлтое' },
      ja: { cite: '黄色', ph: 'kiiro', attr: '黄色い' },
      ko: { cite: '노랑', ph: 'norang', attr: '노란' },
      zh: { cite: '黄色', ph: 'huáng sè', attr: '黄色的' },
      hi: { cite: 'पीला', ph: 'peelaa', m: 'पीला', f: 'पीली' },
      ar: { cite: 'أصفر', ph: 'asfar', m: 'أصفر', f: 'صفراء' }
    }
  },
  green: {
    em: '🟢', swatch: '#6C8F55',
    t: {
      es: { cite: 'verde', ph: 'VER-deh', m: 'verde', f: 'verde' },
      fr: { cite: 'vert', ph: 'vehr', m: 'vert', f: 'verte' },
      de: { cite: 'grün', ph: 'gruun', attr: 'grüne' },
      it: { cite: 'verde', ph: 'VER-deh', m: 'verde', f: 'verde' },
      pt: { cite: 'verde', ph: 'VER-jee', m: 'verde', f: 'verde' },
      nl: { cite: 'groen', ph: 'khroon', attr: 'groene' },
      ru: { cite: 'зелёный', ph: 'zye-LYO-nee', m: 'зелёный', f: 'зелёная', n: 'зелёное' },
      ja: { cite: '緑', ph: 'midori', attr: '緑の' },
      ko: { cite: '초록색', ph: 'choroksaek', attr: '초록색' },
      zh: { cite: '绿色', ph: 'lǜ sè', attr: '绿色的' },
      hi: { cite: 'हरा', ph: 'haraa', m: 'हरा', f: 'हरी' },
      ar: { cite: 'أخضر', ph: 'akhdar', m: 'أخضر', f: 'خضراء' }
    }
  },
  blue: {
    em: '🔵', swatch: '#5478A6',
    t: {
      es: { cite: 'azul', ph: 'a-SOOL', m: 'azul', f: 'azul' },
      fr: { cite: 'bleu', ph: 'bluh', m: 'bleu', f: 'bleue' },
      de: { cite: 'blau', ph: 'blow', attr: 'blaue' },
      it: { cite: 'blu', ph: 'bloo', m: 'blu', f: 'blu' },
      pt: { cite: 'azul', ph: 'a-ZOOL', m: 'azul', f: 'azul' },
      nl: { cite: 'blauw', ph: 'blow', attr: 'blauwe' },
      ru: { cite: 'синий', ph: 'SEE-nee', m: 'синий', f: 'синяя', n: 'синее' },
      ja: { cite: '青', ph: 'ao', attr: '青い' },
      ko: { cite: '파랑', ph: 'parang', attr: '파란' },
      zh: { cite: '蓝色', ph: 'lán sè', attr: '蓝色的' },
      hi: { cite: 'नीला', ph: 'neelaa', m: 'नीला', f: 'नीली' },
      ar: { cite: 'أزرق', ph: 'azraq', m: 'أزرق', f: 'زرقاء' }
    }
  },
  purple: {
    em: '🟣', swatch: '#87699A',
    t: {
      es: { cite: 'morado', ph: 'mo-RA-do', m: 'morado', f: 'morada' },
      fr: { cite: 'violet', ph: 'vyo-LEH', m: 'violet', f: 'violette' },
      de: { cite: 'lila', ph: 'LEE-la', attr: 'lila' },
      it: { cite: 'viola', ph: 'vee-O-la', m: 'viola', f: 'viola' },
      pt: { cite: 'roxo', ph: 'HO-shoo', m: 'roxo', f: 'roxa' },
      nl: { cite: 'paars', ph: 'pahrs', attr: 'paarse' },
      ru: { cite: 'фиолетовый', ph: 'fee-a-LYE-ta-vee', m: 'фиолетовый', f: 'фиолетовая', n: 'фиолетовое' },
      ja: { cite: '紫', ph: 'murasaki', attr: '紫の' },
      ko: { cite: '보라색', ph: 'borasaek', attr: '보라색' },
      zh: { cite: '紫色', ph: 'zǐ sè', attr: '紫色的' },
      hi: { cite: 'बैंगनी', ph: 'bainganee', m: 'बैंगनी', f: 'बैंगनी' },
      ar: { cite: 'بنفسجي', ph: 'banafsajee', m: 'بنفسجي', f: 'بنفسجية' }
    }
  },
  pink: {
    em: '🩷', swatch: '#BE7C8E',
    t: {
      es: { cite: 'rosa', ph: 'RO-sa', m: 'rosa', f: 'rosa' },
      fr: { cite: 'rose', ph: 'rohz', m: 'rose', f: 'rose' },
      de: { cite: 'rosa', ph: 'ROH-za', attr: 'rosa' },
      it: { cite: 'rosa', ph: 'RO-za', m: 'rosa', f: 'rosa' },
      pt: { cite: 'rosa', ph: 'HO-za', m: 'rosa', f: 'rosa' },
      nl: { cite: 'roze', ph: 'ROH-zeh', attr: 'roze' },
      ru: { cite: 'розовый', ph: 'RO-za-vee', m: 'розовый', f: 'розовая', n: 'розовое' },
      ja: { cite: 'ピンク', ph: 'pinku', attr: 'ピンクの' },
      ko: { cite: '분홍색', ph: 'bunhongsaek', attr: '분홍색' },
      zh: { cite: '粉色', ph: 'fěn sè', attr: '粉色的' },
      hi: { cite: 'गुलाबी', ph: 'gulaabee', m: 'गुलाबी', f: 'गुलाबी' },
      ar: { cite: 'وردي', ph: 'wardee', m: 'وردي', f: 'وردية' }
    }
  },
  brown: {
    em: '🟤', swatch: '#8A6A50',
    t: {
      es: { cite: 'marrón', ph: 'ma-RRON', m: 'marrón', f: 'marrón' },
      fr: { cite: 'marron', ph: 'ma-ROHN', m: 'marron', f: 'marron' },
      de: { cite: 'braun', ph: 'brown', attr: 'braune' },
      it: { cite: 'marrone', ph: 'mar-RO-neh', m: 'marrone', f: 'marrone' },
      pt: { cite: 'marrom', ph: 'ma-HOHM', m: 'marrom', f: 'marrom' },
      nl: { cite: 'bruin', ph: 'brown', attr: 'bruine' },
      ru: { cite: 'коричневый', ph: 'ka-REECH-nye-vee', m: 'коричневый', f: 'коричневая', n: 'коричневое' },
      ja: { cite: '茶色', ph: 'chairo', attr: '茶色い' },
      ko: { cite: '갈색', ph: 'galsaek', attr: '갈색' },
      zh: { cite: '棕色', ph: 'zōng sè', attr: '棕色的' },
      hi: { cite: 'भूरा', ph: 'bhooraa', m: 'भूरा', f: 'भूरी' },
      ar: { cite: 'بني', ph: 'bunnee', m: 'بني', f: 'بنية' }
    }
  },
  black: {
    em: '⚫', swatch: '#2E2D26',
    t: {
      es: { cite: 'negro', ph: 'NEH-gro', m: 'negro', f: 'negra' },
      fr: { cite: 'noir', ph: 'nwahr', m: 'noir', f: 'noire' },
      de: { cite: 'schwarz', ph: 'shvarts', attr: 'schwarze' },
      it: { cite: 'nero', ph: 'NEH-ro', m: 'nero', f: 'nera' },
      pt: { cite: 'preto', ph: 'PREH-too', m: 'preto', f: 'preta' },
      nl: { cite: 'zwart', ph: 'zvart', attr: 'zwarte' },
      ru: { cite: 'чёрный', ph: 'CHOR-nee', m: 'чёрный', f: 'чёрная', n: 'чёрное' },
      ja: { cite: '黒', ph: 'kuro', attr: '黒い' },
      ko: { cite: '검정', ph: 'geomjeong', attr: '검은' },
      zh: { cite: '黑色', ph: 'hēi sè', attr: '黑色的' },
      hi: { cite: 'काला', ph: 'kaalaa', m: 'काला', f: 'काली' },
      ar: { cite: 'أسود', ph: 'aswad', m: 'أسود', f: 'سوداء' }
    }
  },
  white: {
    em: '⚪', swatch: '#F2EFE4',
    t: {
      es: { cite: 'blanco', ph: 'BLAN-ko', m: 'blanco', f: 'blanca' },
      fr: { cite: 'blanc', ph: 'blahn', m: 'blanc', f: 'blanche' },
      de: { cite: 'weiß', ph: 'vice', attr: 'weiße' },
      it: { cite: 'bianco', ph: 'BYAN-ko', m: 'bianco', f: 'bianca' },
      pt: { cite: 'branco', ph: 'BRAN-koo', m: 'branco', f: 'branca' },
      nl: { cite: 'wit', ph: 'vit', attr: 'witte' },
      ru: { cite: 'белый', ph: 'BYE-lee', m: 'белый', f: 'белая', n: 'белое' },
      ja: { cite: '白', ph: 'shiro', attr: '白い' },
      ko: { cite: '하양', ph: 'hayang', attr: '하얀' },
      zh: { cite: '白色', ph: 'bái sè', attr: '白色的' },
      hi: { cite: 'सफ़ेद', ph: 'safed', m: 'सफ़ेद', f: 'सफ़ेद' },
      ar: { cite: 'أبيض', ph: 'abyad', m: 'أبيض', f: 'بيضاء' }
    }
  },
  grey: {
    em: '🩶', swatch: '#98948A',
    t: {
      es: { cite: 'gris', ph: 'grees', m: 'gris', f: 'gris' },
      fr: { cite: 'gris', ph: 'gree', m: 'gris', f: 'grise' },
      de: { cite: 'grau', ph: 'grow', attr: 'graue' },
      it: { cite: 'grigio', ph: 'GREE-jo', m: 'grigio', f: 'grigia' },
      pt: { cite: 'cinzento', ph: 'seen-ZEN-too', m: 'cinzento', f: 'cinzenta' },
      nl: { cite: 'grijs', ph: 'khrice', attr: 'grijze' },
      ru: { cite: 'серый', ph: 'SYE-ree', m: 'серый', f: 'серая', n: 'серое' },
      ja: { cite: '灰色', ph: 'haiiro', attr: '灰色の' },
      ko: { cite: '회색', ph: 'hoesaek', attr: '회색' },
      zh: { cite: '灰色', ph: 'huī sè', attr: '灰色的' },
      hi: { cite: 'स्लेटी', ph: 'sletee', m: 'स्लेटी', f: 'स्लेटी' },
      ar: { cite: 'رمادي', ph: 'ramaadee', m: 'رمادي', f: 'رمادية' }
    }
  }
};

export const COLOR_KEYS = Object.keys(COLORS);

/** Agreement forms for one colour in one language, or null if unknown. */
export function colorForms(key, lang) {
  const c = COLORS[key];
  if (!c) return null;
  const t = c.t[lang];
  if (!t) return null;
  return { key, em: c.em, swatch: c.swatch, ...t };
}

/** Citation form — what the learner memorises and what TTS should say. */
export function colorWord(key, lang) {
  const f = colorForms(key, lang);
  return f ? f.cite : key;
}
