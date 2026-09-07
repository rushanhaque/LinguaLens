/**
 * lexicon.js — Core vocabulary beyond what a camera can point at.
 *
 * The camera teaches concrete nouns. This is the glue a learner actually needs
 * to say anything with them: numbers, greetings, family, places, time, verbs,
 * adjectives and question words.
 *
 * Entries share the dictionary's translation shape — [word, phonetic, gender] —
 * so the same review, speech and grammar code drives both.
 *
 * Gender is '' where the language does not mark it on that part of speech, and
 * for verbs and question words everywhere.
 */

export const PACKS = {
  numbers: { label: 'Numbers', em: '🔢', color: '#0A84FF', blurb: 'Zero to a hundred' },
  greetings: { label: 'Greetings', em: '👋', color: '#FF9F0A', blurb: 'Hello, please, thank you' },
  family: { label: 'Family', em: '👨‍👩‍👧', color: '#FF375F', blurb: 'People closest to you' },
  places: { label: 'Places', em: '🏙️', color: '#30D158', blurb: 'Getting around town' },
  time: { label: 'Time', em: '🕰️', color: '#5E5CE6', blurb: 'Days, hours, when' },
  verbs: { label: 'Verbs', em: '🏃', color: '#BF5AF2', blurb: 'The sixteen you need first' },
  adjectives: { label: 'Adjectives', em: '✨', color: '#64D2FF', blurb: 'Describing things' },
  questions: { label: 'Questions', em: '❓', color: '#FFD60A', blurb: 'Who, what, where, why' }
};

export const LEXICON = {
  /* ── Numbers ────────────────────────────────────────────────────────── */
  n0: {
    pack: 'numbers', en: 'zero', em: '0️⃣',
    t: {
      es: ['cero', 'SEH-ro', ''], fr: ['zéro', 'zay-ROH', ''], de: ['null', 'nool', ''],
      it: ['zero', 'DZEH-ro', ''], pt: ['zero', 'ZEH-roo', ''], nl: ['nul', 'nul', ''],
      ru: ['ноль', 'nol', ''], ja: ['ゼロ', 'zero', ''], ko: ['영', 'yeong', ''],
      zh: ['零', 'líng', ''], hi: ['शून्य', 'shoonya', ''], ar: ['صفر', 'sifr', '']
    }
  },
  n1: {
    pack: 'numbers', en: 'one', em: '1️⃣',
    t: {
      es: ['uno', 'OO-no', ''], fr: ['un', 'uhn', ''], de: ['eins', 'ines', ''],
      it: ['uno', 'OO-no', ''], pt: ['um', 'oong', ''], nl: ['één', 'ayn', ''],
      ru: ['один', 'a-DEEN', ''], ja: ['いち', 'ichi', ''], ko: ['하나', 'hana', ''],
      zh: ['一', 'yī', ''], hi: ['एक', 'ek', ''], ar: ['واحد', 'waahid', '']
    }
  },
  n2: {
    pack: 'numbers', en: 'two', em: '2️⃣',
    t: {
      es: ['dos', 'dohs', ''], fr: ['deux', 'duh', ''], de: ['zwei', 'tsvy', ''],
      it: ['due', 'DOO-eh', ''], pt: ['dois', 'doysh', ''], nl: ['twee', 'tvay', ''],
      ru: ['два', 'dva', ''], ja: ['に', 'ni', ''], ko: ['둘', 'dul', ''],
      zh: ['二', 'èr', ''], hi: ['दो', 'do', ''], ar: ['اثنان', 'ithnaan', '']
    }
  },
  n3: {
    pack: 'numbers', en: 'three', em: '3️⃣',
    t: {
      es: ['tres', 'trehs', ''], fr: ['trois', 'trwah', ''], de: ['drei', 'dry', ''],
      it: ['tre', 'treh', ''], pt: ['três', 'trehsh', ''], nl: ['drie', 'dree', ''],
      ru: ['три', 'tree', ''], ja: ['さん', 'san', ''], ko: ['셋', 'set', ''],
      zh: ['三', 'sān', ''], hi: ['तीन', 'teen', ''], ar: ['ثلاثة', 'thalaatha', '']
    }
  },
  n4: {
    pack: 'numbers', en: 'four', em: '4️⃣',
    t: {
      es: ['cuatro', 'KWA-tro', ''], fr: ['quatre', 'KATR', ''], de: ['vier', 'feer', ''],
      it: ['quattro', 'KWAT-tro', ''], pt: ['quatro', 'KWA-troo', ''], nl: ['vier', 'feer', ''],
      ru: ['четыре', 'chye-TI-rye', ''], ja: ['よん', 'yon', ''], ko: ['넷', 'net', ''],
      zh: ['四', 'sì', ''], hi: ['चार', 'chaar', ''], ar: ['أربعة', 'arbaa', '']
    }
  },
  n5: {
    pack: 'numbers', en: 'five', em: '5️⃣',
    t: {
      es: ['cinco', 'SEEN-ko', ''], fr: ['cinq', 'sank', ''], de: ['fünf', 'fuunf', ''],
      it: ['cinque', 'CHEEN-kweh', ''], pt: ['cinco', 'SEEN-koo', ''], nl: ['vijf', 'fife', ''],
      ru: ['пять', 'pyat', ''], ja: ['ご', 'go', ''], ko: ['다섯', 'daseot', ''],
      zh: ['五', 'wǔ', ''], hi: ['पाँच', 'paanch', ''], ar: ['خمسة', 'khamsa', '']
    }
  },
  n6: {
    pack: 'numbers', en: 'six', em: '6️⃣',
    t: {
      es: ['seis', 'sayss', ''], fr: ['six', 'seess', ''], de: ['sechs', 'zeks', ''],
      it: ['sei', 'say', ''], pt: ['seis', 'sayss', ''], nl: ['zes', 'zess', ''],
      ru: ['шесть', 'shest', ''], ja: ['ろく', 'roku', ''], ko: ['여섯', 'yeoseot', ''],
      zh: ['六', 'liù', ''], hi: ['छह', 'chhah', ''], ar: ['ستة', 'sitta', '']
    }
  },
  n7: {
    pack: 'numbers', en: 'seven', em: '7️⃣',
    t: {
      es: ['siete', 'SYEH-teh', ''], fr: ['sept', 'set', ''], de: ['sieben', 'ZEE-ben', ''],
      it: ['sette', 'SET-teh', ''], pt: ['sete', 'SEH-chee', ''], nl: ['zeven', 'ZAY-ven', ''],
      ru: ['семь', 'syem', ''], ja: ['なな', 'nana', ''], ko: ['일곱', 'ilgop', ''],
      zh: ['七', 'qī', ''], hi: ['सात', 'saat', ''], ar: ['سبعة', 'sabaa', '']
    }
  },
  n8: {
    pack: 'numbers', en: 'eight', em: '8️⃣',
    t: {
      es: ['ocho', 'O-cho', ''], fr: ['huit', 'weet', ''], de: ['acht', 'akht', ''],
      it: ['otto', 'OT-to', ''], pt: ['oito', 'OY-too', ''], nl: ['acht', 'akht', ''],
      ru: ['восемь', 'VO-syem', ''], ja: ['はち', 'hachi', ''], ko: ['여덟', 'yeodeol', ''],
      zh: ['八', 'bā', ''], hi: ['आठ', 'aath', ''], ar: ['ثمانية', 'thamaaniya', '']
    }
  },
  n9: {
    pack: 'numbers', en: 'nine', em: '9️⃣',
    t: {
      es: ['nueve', 'NWEH-veh', ''], fr: ['neuf', 'nuhf', ''], de: ['neun', 'noyn', ''],
      it: ['nove', 'NO-veh', ''], pt: ['nove', 'NO-vee', ''], nl: ['negen', 'NAY-khen', ''],
      ru: ['девять', 'DYE-vyat', ''], ja: ['きゅう', 'kyuu', ''], ko: ['아홉', 'ahop', ''],
      zh: ['九', 'jiǔ', ''], hi: ['नौ', 'nau', ''], ar: ['تسعة', 'tisaa', '']
    }
  },
  n10: {
    pack: 'numbers', en: 'ten', em: '🔟',
    t: {
      es: ['diez', 'dyess', ''], fr: ['dix', 'deess', ''], de: ['zehn', 'tsayn', ''],
      it: ['dieci', 'DYEH-chee', ''], pt: ['dez', 'dess', ''], nl: ['tien', 'teen', ''],
      ru: ['десять', 'DYE-syat', ''], ja: ['じゅう', 'juu', ''], ko: ['열', 'yeol', ''],
      zh: ['十', 'shí', ''], hi: ['दस', 'das', ''], ar: ['عشرة', 'ashara', '']
    }
  },
  n11: {
    pack: 'numbers', en: 'eleven', em: '1️⃣',
    t: {
      es: ['once', 'ON-seh', ''], fr: ['onze', 'ohnz', ''], de: ['elf', 'elf', ''],
      it: ['undici', 'OON-dee-chee', ''], pt: ['onze', 'ONZ', ''], nl: ['elf', 'elf', ''],
      ru: ['одиннадцать', 'a-DEE-nat-tsat', ''], ja: ['じゅういち', 'juuichi', ''],
      ko: ['열하나', 'yeolhana', ''], zh: ['十一', 'shí yī', ''],
      hi: ['ग्यारह', 'gyaarah', ''], ar: ['أحد عشر', 'ahada ashar', '']
    }
  },
  n12: {
    pack: 'numbers', en: 'twelve', em: '1️⃣',
    t: {
      es: ['doce', 'DO-seh', ''], fr: ['douze', 'dooz', ''], de: ['zwölf', 'tsvurlf', ''],
      it: ['dodici', 'DO-dee-chee', ''], pt: ['doze', 'DOZ', ''], nl: ['twaalf', 'tvahlf', ''],
      ru: ['двенадцать', 'dvye-NAT-tsat', ''], ja: ['じゅうに', 'juuni', ''],
      ko: ['열둘', 'yeoldul', ''], zh: ['十二', 'shí èr', ''],
      hi: ['बारह', 'baarah', ''], ar: ['اثنا عشر', 'ithnaa ashar', '']
    }
  },
  n13: {
    pack: 'numbers', en: 'thirteen', em: '1️⃣',
    t: {
      es: ['trece', 'TREH-seh', ''], fr: ['treize', 'trez', ''], de: ['dreizehn', 'DRY-tsayn', ''],
      it: ['tredici', 'TREH-dee-chee', ''], pt: ['treze', 'TREZ', ''], nl: ['dertien', 'DER-teen', ''],
      ru: ['тринадцать', 'tree-NAT-tsat', ''], ja: ['じゅうさん', 'juusan', ''],
      ko: ['열셋', 'yeolset', ''], zh: ['十三', 'shí sān', ''],
      hi: ['तेरह', 'terah', ''], ar: ['ثلاثة عشر', 'thalaathata ashar', '']
    }
  },
  n14: {
    pack: 'numbers', en: 'fourteen', em: '1️⃣',
    t: {
      es: ['catorce', 'ka-TOR-seh', ''], fr: ['quatorze', 'ka-TORZ', ''], de: ['vierzehn', 'FEER-tsayn', ''],
      it: ['quattordici', 'kwat-TOR-dee-chee', ''], pt: ['catorze', 'ka-TORZ', ''], nl: ['veertien', 'FAYR-teen', ''],
      ru: ['четырнадцать', 'chye-TIR-nat-tsat', ''], ja: ['じゅうよん', 'juuyon', ''],
      ko: ['열넷', 'yeolnet', ''], zh: ['十四', 'shí sì', ''],
      hi: ['चौदह', 'chaudah', ''], ar: ['أربعة عشر', 'arbaata ashar', '']
    }
  },
  n15: {
    pack: 'numbers', en: 'fifteen', em: '1️⃣',
    t: {
      es: ['quince', 'KEEN-seh', ''], fr: ['quinze', 'kanz', ''], de: ['fünfzehn', 'FUUNF-tsayn', ''],
      it: ['quindici', 'KWEEN-dee-chee', ''], pt: ['quinze', 'KEENZ', ''], nl: ['vijftien', 'FIFE-teen', ''],
      ru: ['пятнадцать', 'pyat-NAT-tsat', ''], ja: ['じゅうご', 'juugo', ''],
      ko: ['열다섯', 'yeoldaseot', ''], zh: ['十五', 'shí wǔ', ''],
      hi: ['पंद्रह', 'pandrah', ''], ar: ['خمسة عشر', 'khamsata ashar', '']
    }
  },
  n16: {
    pack: 'numbers', en: 'sixteen', em: '1️⃣',
    t: {
      es: ['dieciséis', 'dyeh-see-SAYSS', ''], fr: ['seize', 'sez', ''], de: ['sechzehn', 'ZEKH-tsayn', ''],
      it: ['sedici', 'SEH-dee-chee', ''], pt: ['dezesseis', 'deh-zeh-SAYSS', ''], nl: ['zestien', 'ZESS-teen', ''],
      ru: ['шестнадцать', 'shes-NAT-tsat', ''], ja: ['じゅうろく', 'juuroku', ''],
      ko: ['열여섯', 'yeollyeoseot', ''], zh: ['十六', 'shí liù', ''],
      hi: ['सोलह', 'solah', ''], ar: ['ستة عشر', 'sittata ashar', '']
    }
  },
  n17: {
    pack: 'numbers', en: 'seventeen', em: '1️⃣',
    t: {
      es: ['diecisiete', 'dyeh-see-SYEH-teh', ''], fr: ['dix-sept', 'dee-SET', ''], de: ['siebzehn', 'ZEEP-tsayn', ''],
      it: ['diciassette', 'dee-chas-SET-teh', ''], pt: ['dezessete', 'deh-zeh-SEH-chee', ''], nl: ['zeventien', 'ZAY-ven-teen', ''],
      ru: ['семнадцать', 'syem-NAT-tsat', ''], ja: ['じゅうなな', 'juunana', ''],
      ko: ['열일곱', 'yeollilgop', ''], zh: ['十七', 'shí qī', ''],
      hi: ['सत्रह', 'satrah', ''], ar: ['سبعة عشر', 'sabaata ashar', '']
    }
  },
  n18: {
    pack: 'numbers', en: 'eighteen', em: '1️⃣',
    t: {
      es: ['dieciocho', 'dyeh-see-O-cho', ''], fr: ['dix-huit', 'dee-ZWEET', ''], de: ['achtzehn', 'AKHT-tsayn', ''],
      it: ['diciotto', 'dee-CHOT-to', ''], pt: ['dezoito', 'deh-ZOY-too', ''], nl: ['achttien', 'AKHT-teen', ''],
      ru: ['восемнадцать', 'va-syem-NAT-tsat', ''], ja: ['じゅうはち', 'juuhachi', ''],
      ko: ['열여덟', 'yeollyeodeol', ''], zh: ['十八', 'shí bā', ''],
      hi: ['अठारह', 'athaarah', ''], ar: ['ثمانية عشر', 'thamaaniyata ashar', '']
    }
  },
  n19: {
    pack: 'numbers', en: 'nineteen', em: '1️⃣',
    t: {
      es: ['diecinueve', 'dyeh-see-NWEH-veh', ''], fr: ['dix-neuf', 'deez-NUHF', ''], de: ['neunzehn', 'NOYN-tsayn', ''],
      it: ['diciannove', 'dee-chan-NO-veh', ''], pt: ['dezenove', 'deh-zeh-NO-vee', ''], nl: ['negentien', 'NAY-khen-teen', ''],
      ru: ['девятнадцать', 'dye-vyat-NAT-tsat', ''], ja: ['じゅうきゅう', 'juukyuu', ''],
      ko: ['열아홉', 'yeorahop', ''], zh: ['十九', 'shí jiǔ', ''],
      hi: ['उन्नीस', 'unnees', ''], ar: ['تسعة عشر', 'tisaata ashar', '']
    }
  },
  n20: {
    pack: 'numbers', en: 'twenty', em: '2️⃣',
    t: {
      es: ['veinte', 'VAYN-teh', ''], fr: ['vingt', 'van', ''], de: ['zwanzig', 'TSVAN-tsikh', ''],
      it: ['venti', 'VEN-tee', ''], pt: ['vinte', 'VEEN-chee', ''], nl: ['twintig', 'TVIN-tikh', ''],
      ru: ['двадцать', 'DVAT-tsat', ''], ja: ['にじゅう', 'nijuu', ''],
      ko: ['스물', 'seumul', ''], zh: ['二十', 'èr shí', ''],
      hi: ['बीस', 'bees', ''], ar: ['عشرون', 'ishroon', '']
    }
  },
  n100: {
    pack: 'numbers', en: 'one hundred', em: '💯',
    t: {
      es: ['cien', 'syen', ''], fr: ['cent', 'sahn', ''], de: ['hundert', 'HOON-dert', ''],
      it: ['cento', 'CHEN-to', ''], pt: ['cem', 'seng', ''], nl: ['honderd', 'HON-dert', ''],
      ru: ['сто', 'sto', ''], ja: ['ひゃく', 'hyaku', ''], ko: ['백', 'baek', ''],
      zh: ['一百', 'yì bǎi', ''], hi: ['सौ', 'sau', ''], ar: ['مئة', 'mia', '']
    }
  },

  /* ── Greetings & courtesy ───────────────────────────────────────────── */
  hello: {
    pack: 'greetings', en: 'hello', em: '👋',
    t: {
      es: ['hola', 'O-la', ''], fr: ['bonjour', 'bohn-ZHOOR', ''], de: ['hallo', 'HA-lo', ''],
      it: ['ciao', 'chow', ''], pt: ['olá', 'o-LA', ''], nl: ['hallo', 'HA-lo', ''],
      ru: ['привет', 'pree-VYET', ''], ja: ['こんにちは', 'konnichiwa', ''],
      ko: ['안녕하세요', 'annyeonghaseyo', ''], zh: ['你好', 'nǐ hǎo', ''],
      hi: ['नमस्ते', 'namaste', ''], ar: ['مرحبا', 'marhaban', '']
    }
  },
  goodbye: {
    pack: 'greetings', en: 'goodbye', em: '👋',
    t: {
      es: ['adiós', 'a-DYOSS', ''], fr: ['au revoir', 'oh ruh-VWAHR', ''], de: ['auf Wiedersehen', 'owf VEE-der-zayn', ''],
      it: ['arrivederci', 'ar-ree-veh-DER-chee', ''], pt: ['tchau', 'chow', ''], nl: ['tot ziens', 'tot ZEENS', ''],
      ru: ['до свидания', 'da svee-DA-nee-ya', ''], ja: ['さようなら', 'sayounara', ''],
      ko: ['안녕히 가세요', 'annyeonghi gaseyo', ''], zh: ['再见', 'zài jiàn', ''],
      hi: ['अलविदा', 'alvidaa', ''], ar: ['مع السلامة', 'maa as-salaama', '']
    }
  },
  good_morning: {
    pack: 'greetings', en: 'good morning', em: '🌅',
    t: {
      es: ['buenos días', 'BWEH-nos DEE-as', ''], fr: ['bonjour', 'bohn-ZHOOR', ''], de: ['guten Morgen', 'GOO-ten MOR-gen', ''],
      it: ['buongiorno', 'bwon-JOR-no', ''], pt: ['bom dia', 'bong DEE-a', ''], nl: ['goedemorgen', 'khoo-deh-MOR-khen', ''],
      ru: ['доброе утро', 'DOB-ra-ye OO-tra', ''], ja: ['おはようございます', 'ohayou gozaimasu', ''],
      ko: ['좋은 아침', 'joeun achim', ''], zh: ['早上好', 'zǎo shang hǎo', ''],
      hi: ['सुप्रभात', 'suprabhaat', ''], ar: ['صباح الخير', 'sabaah al-khayr', '']
    }
  },
  good_night: {
    pack: 'greetings', en: 'good night', em: '🌙',
    t: {
      es: ['buenas noches', 'BWEH-nas NO-ches', ''], fr: ['bonne nuit', 'bun NWEE', ''], de: ['gute Nacht', 'GOO-teh nakht', ''],
      it: ['buonanotte', 'bwo-na-NOT-teh', ''], pt: ['boa noite', 'BO-a NOY-chee', ''], nl: ['welterusten', 'vel-teh-RUS-ten', ''],
      ru: ['спокойной ночи', 'spa-KOY-nay NO-chee', ''], ja: ['おやすみなさい', 'oyasuminasai', ''],
      ko: ['안녕히 주무세요', 'annyeonghi jumuseyo', ''], zh: ['晚安', 'wǎn ān', ''],
      hi: ['शुभ रात्रि', 'shubh raatri', ''], ar: ['تصبح على خير', 'tusbih ala khayr', '']
    }
  },
  please: {
    pack: 'greetings', en: 'please', em: '🙏',
    t: {
      es: ['por favor', 'por fa-VOR', ''], fr: ["s'il vous plaît", 'seel voo PLEH', ''], de: ['bitte', 'BIT-teh', ''],
      it: ['per favore', 'per fa-VO-reh', ''], pt: ['por favor', 'por fa-VOR', ''], nl: ['alsjeblieft', 'als-yeh-BLEEFT', ''],
      ru: ['пожалуйста', 'pa-ZHA-lus-ta', ''], ja: ['お願いします', 'onegaishimasu', ''],
      ko: ['제발', 'jebal', ''], zh: ['请', 'qǐng', ''],
      hi: ['कृपया', 'kripayaa', ''], ar: ['من فضلك', 'min fadlik', '']
    }
  },
  thank_you: {
    pack: 'greetings', en: 'thank you', em: '🙇',
    t: {
      es: ['gracias', 'GRA-syas', ''], fr: ['merci', 'mer-SEE', ''], de: ['danke', 'DAN-keh', ''],
      it: ['grazie', 'GRAT-tsyeh', ''], pt: ['obrigado', 'o-bree-GA-doo', ''], nl: ['dank je', 'dank yeh', ''],
      ru: ['спасибо', 'spa-SEE-ba', ''], ja: ['ありがとう', 'arigatou', ''],
      ko: ['감사합니다', 'gamsahamnida', ''], zh: ['谢谢', 'xiè xie', ''],
      hi: ['धन्यवाद', 'dhanyavaad', ''], ar: ['شكرا', 'shukran', '']
    }
  },
  youre_welcome: {
    pack: 'greetings', en: "you're welcome", em: '😊',
    t: {
      es: ['de nada', 'deh NA-da', ''], fr: ['de rien', 'duh RYAN', ''], de: ['bitte schön', 'BIT-teh shurn', ''],
      it: ['prego', 'PREH-go', ''], pt: ['de nada', 'jee NA-da', ''], nl: ['graag gedaan', 'khrahkh kheh-DAHN', ''],
      ru: ['пожалуйста', 'pa-ZHA-lus-ta', ''], ja: ['どういたしまして', 'dou itashimashite', ''],
      ko: ['천만에요', 'cheonmaneyo', ''], zh: ['不客气', 'bú kè qi', ''],
      hi: ['कोई बात नहीं', 'koee baat nahin', ''], ar: ['عفوا', 'afwan', '']
    }
  },
  excuse_me: {
    pack: 'greetings', en: 'excuse me', em: '🤚',
    t: {
      es: ['perdón', 'per-DON', ''], fr: ['excusez-moi', 'ex-kuu-zay MWAH', ''], de: ['Entschuldigung', 'ent-SHOOL-dee-goong', ''],
      it: ['scusi', 'SKOO-zee', ''], pt: ['com licença', 'kong lee-SEN-sa', ''], nl: ['pardon', 'par-DON', ''],
      ru: ['извините', 'eez-vee-NEE-tye', ''], ja: ['すみません', 'sumimasen', ''],
      ko: ['실례합니다', 'sillyehamnida', ''], zh: ['打扰一下', 'dǎ rǎo yí xià', ''],
      hi: ['क्षमा करें', 'kshamaa karen', ''], ar: ['لو سمحت', 'law samaht', '']
    }
  },
  sorry: {
    pack: 'greetings', en: 'sorry', em: '😔',
    t: {
      es: ['lo siento', 'lo SYEN-to', ''], fr: ['désolé', 'day-zo-LAY', ''], de: ['es tut mir leid', 'es toot meer LITE', ''],
      it: ['mi dispiace', 'mee dees-PYA-cheh', ''], pt: ['desculpe', 'des-KOOL-pee', ''], nl: ['sorry', 'SOR-ree', ''],
      ru: ['простите', 'pras-TEE-tye', ''], ja: ['ごめんなさい', 'gomen nasai', ''],
      ko: ['미안해요', 'mianhaeyo', ''], zh: ['对不起', 'duì bu qǐ', ''],
      hi: ['माफ़ कीजिए', 'maaf keejiye', ''], ar: ['آسف', 'aasif', '']
    }
  },
  yes: {
    pack: 'greetings', en: 'yes', em: '✅',
    t: {
      es: ['sí', 'see', ''], fr: ['oui', 'wee', ''], de: ['ja', 'yah', ''],
      it: ['sì', 'see', ''], pt: ['sim', 'seeng', ''], nl: ['ja', 'yah', ''],
      ru: ['да', 'da', ''], ja: ['はい', 'hai', ''], ko: ['네', 'ne', ''],
      zh: ['是', 'shì', ''], hi: ['हाँ', 'haan', ''], ar: ['نعم', 'naam', '']
    }
  },
  no: {
    pack: 'greetings', en: 'no', em: '❌',
    t: {
      es: ['no', 'no', ''], fr: ['non', 'nohn', ''], de: ['nein', 'nine', ''],
      it: ['no', 'no', ''], pt: ['não', 'nowng', ''], nl: ['nee', 'nay', ''],
      ru: ['нет', 'nyet', ''], ja: ['いいえ', 'iie', ''], ko: ['아니요', 'aniyo', ''],
      zh: ['不', 'bù', ''], hi: ['नहीं', 'nahin', ''], ar: ['لا', 'laa', '']
    }
  },
  how_are_you: {
    pack: 'greetings', en: 'how are you?', em: '💬',
    t: {
      es: ['¿cómo estás?', 'KO-mo es-TAS', ''], fr: ['comment ça va ?', 'ko-MAHN sa VA', ''], de: ["wie geht's?", 'vee GAYTS', ''],
      it: ['come stai?', 'KO-meh STAI', ''], pt: ['como está?', 'KO-moo es-TA', ''], nl: ['hoe gaat het?', 'hoo KHAHT het', ''],
      ru: ['как дела?', 'kak dye-LA', ''], ja: ['お元気ですか', 'ogenki desu ka', ''],
      ko: ['어떻게 지내세요?', 'eotteoke jinaeseyo', ''], zh: ['你好吗？', 'nǐ hǎo ma', ''],
      hi: ['आप कैसे हैं?', 'aap kaise hain', ''], ar: ['كيف حالك؟', 'kayfa haaluk', '']
    }
  },
  my_name_is: {
    pack: 'greetings', en: 'my name is', em: '🪪',
    t: {
      es: ['me llamo', 'meh YA-mo', ''], fr: ["je m'appelle", 'zhuh ma-PEL', ''], de: ['ich heiße', 'ikh HY-seh', ''],
      it: ['mi chiamo', 'mee KYA-mo', ''], pt: ['meu nome é', 'mew NO-mee eh', ''], nl: ['ik heet', 'ik HAYT', ''],
      ru: ['меня зовут', 'mye-NYA za-VOOT', ''], ja: ['私の名前は', 'watashi no namae wa', ''],
      ko: ['제 이름은', 'je ireumeun', ''], zh: ['我叫', 'wǒ jiào', ''],
      hi: ['मेरा नाम है', 'meraa naam hai', ''], ar: ['اسمي', 'ismee', '']
    }
  },
  dont_understand: {
    pack: 'greetings', en: "I don't understand", em: '🤷',
    t: {
      es: ['no entiendo', 'no en-TYEN-do', ''], fr: ['je ne comprends pas', 'zhuh nuh kom-PRAHN pa', ''],
      de: ['ich verstehe nicht', 'ikh fer-SHTAY-eh nikht', ''], it: ['non capisco', 'non ka-PEES-ko', ''],
      pt: ['não entendo', 'nowng en-TEN-doo', ''], nl: ['ik begrijp het niet', 'ik beh-KHRAYP het neet', ''],
      ru: ['я не понимаю', 'ya nye pa-nee-MA-yu', ''], ja: ['わかりません', 'wakarimasen', ''],
      ko: ['이해가 안 돼요', 'ihaega an dwaeyo', ''], zh: ['我不明白', 'wǒ bù míng bai', ''],
      hi: ['मुझे समझ नहीं आया', 'mujhe samajh nahin aayaa', ''], ar: ['لا أفهم', 'laa afham', '']
    }
  },

  /* ── Family ─────────────────────────────────────────────────────────── */
  mother: {
    pack: 'family', en: 'mother', em: '👩',
    t: {
      es: ['madre', 'MA-dreh', 'f'], fr: ['mère', 'mehr', 'f'], de: ['Mutter', 'MOO-ter', 'f'],
      it: ['madre', 'MA-dreh', 'f'], pt: ['mãe', 'mayng', 'f'], nl: ['moeder', 'MOO-der', 'c'],
      ru: ['мать', 'mat', 'f'], ja: ['母', 'haha', ''], ko: ['어머니', 'eomeoni', ''],
      zh: ['妈妈', 'mā ma', ''], hi: ['माँ', 'maan', 'f'], ar: ['أم', 'umm', 'f']
    }
  },
  father: {
    pack: 'family', en: 'father', em: '👨',
    t: {
      es: ['padre', 'PA-dreh', 'm'], fr: ['père', 'pehr', 'm'], de: ['Vater', 'FA-ter', 'm'],
      it: ['padre', 'PA-dreh', 'm'], pt: ['pai', 'pie', 'm'], nl: ['vader', 'FA-der', 'c'],
      ru: ['отец', 'a-TYETS', 'm'], ja: ['父', 'chichi', ''], ko: ['아버지', 'abeoji', ''],
      zh: ['爸爸', 'bà ba', ''], hi: ['पिता', 'pitaa', 'm'], ar: ['أب', 'ab', 'm']
    }
  },
  sister: {
    pack: 'family', en: 'sister', em: '👧',
    t: {
      es: ['hermana', 'er-MA-na', 'f'], fr: ['sœur', 'suhr', 'f'], de: ['Schwester', 'SHVES-ter', 'f'],
      it: ['sorella', 'so-REL-la', 'f'], pt: ['irmã', 'eer-MANG', 'f'], nl: ['zus', 'zuss', 'c'],
      ru: ['сестра', 'sees-TRA', 'f'], ja: ['姉', 'ane', ''], ko: ['자매', 'jamae', ''],
      zh: ['姐妹', 'jiě mèi', ''], hi: ['बहन', 'bahan', 'f'], ar: ['أخت', 'ukht', 'f']
    }
  },
  brother: {
    pack: 'family', en: 'brother', em: '👦',
    t: {
      es: ['hermano', 'er-MA-no', 'm'], fr: ['frère', 'frehr', 'm'], de: ['Bruder', 'BROO-der', 'm'],
      it: ['fratello', 'fra-TEL-lo', 'm'], pt: ['irmão', 'eer-MOWNG', 'm'], nl: ['broer', 'broor', 'c'],
      ru: ['брат', 'brat', 'm'], ja: ['兄', 'ani', ''], ko: ['형제', 'hyeongje', ''],
      zh: ['兄弟', 'xiōng dì', ''], hi: ['भाई', 'bhaaee', 'm'], ar: ['أخ', 'akh', 'm']
    }
  },
  son: {
    pack: 'family', en: 'son', em: '👶',
    t: {
      es: ['hijo', 'EE-ho', 'm'], fr: ['fils', 'feess', 'm'], de: ['Sohn', 'zohn', 'm'],
      it: ['figlio', 'FEE-lyo', 'm'], pt: ['filho', 'FEE-lyoo', 'm'], nl: ['zoon', 'zohn', 'c'],
      ru: ['сын', 'sin', 'm'], ja: ['息子', 'musuko', ''], ko: ['아들', 'adeul', ''],
      zh: ['儿子', 'ér zi', ''], hi: ['बेटा', 'betaa', 'm'], ar: ['ابن', 'ibn', 'm']
    }
  },
  daughter: {
    pack: 'family', en: 'daughter', em: '👧',
    t: {
      es: ['hija', 'EE-ha', 'f'], fr: ['fille', 'FEE-yeh', 'f'], de: ['Tochter', 'TOKH-ter', 'f'],
      it: ['figlia', 'FEE-lya', 'f'], pt: ['filha', 'FEE-lya', 'f'], nl: ['dochter', 'DOKH-ter', 'c'],
      ru: ['дочь', 'doch', 'f'], ja: ['娘', 'musume', ''], ko: ['딸', 'ttal', ''],
      zh: ['女儿', 'nǚ ér', ''], hi: ['बेटी', 'betee', 'f'], ar: ['بنت', 'bint', 'f']
    }
  },
  grandmother: {
    pack: 'family', en: 'grandmother', em: '👵',
    t: {
      es: ['abuela', 'a-BWEH-la', 'f'], fr: ['grand-mère', 'grahn-MEHR', 'f'], de: ['Großmutter', 'GROHS-moo-ter', 'f'],
      it: ['nonna', 'NON-na', 'f'], pt: ['avó', 'a-VO', 'f'], nl: ['oma', 'O-ma', 'c'],
      ru: ['бабушка', 'BA-boosh-ka', 'f'], ja: ['祖母', 'sobo', ''], ko: ['할머니', 'halmeoni', ''],
      zh: ['奶奶', 'nǎi nai', ''], hi: ['दादी', 'daadee', 'f'], ar: ['جدة', 'jadda', 'f']
    }
  },
  grandfather: {
    pack: 'family', en: 'grandfather', em: '👴',
    t: {
      es: ['abuelo', 'a-BWEH-lo', 'm'], fr: ['grand-père', 'grahn-PEHR', 'm'], de: ['Großvater', 'GROHS-fa-ter', 'm'],
      it: ['nonno', 'NON-no', 'm'], pt: ['avô', 'a-VOH', 'm'], nl: ['opa', 'O-pa', 'c'],
      ru: ['дедушка', 'DYE-doosh-ka', 'm'], ja: ['祖父', 'sofu', ''], ko: ['할아버지', 'harabeoji', ''],
      zh: ['爷爷', 'yé ye', ''], hi: ['दादा', 'daadaa', 'm'], ar: ['جد', 'jadd', 'm']
    }
  },
  friend: {
    pack: 'family', en: 'friend', em: '🤝',
    t: {
      es: ['amigo', 'a-MEE-go', 'm'], fr: ['ami', 'a-MEE', 'm'], de: ['Freund', 'froynt', 'm'],
      it: ['amico', 'a-MEE-ko', 'm'], pt: ['amigo', 'a-MEE-goo', 'm'], nl: ['vriend', 'freent', 'c'],
      ru: ['друг', 'drook', 'm'], ja: ['友達', 'tomodachi', ''], ko: ['친구', 'chingu', ''],
      zh: ['朋友', 'péng you', ''], hi: ['दोस्त', 'dost', 'm'], ar: ['صديق', 'sadeeq', 'm']
    }
  },
  family: {
    pack: 'family', en: 'family', em: '👨‍👩‍👧',
    t: {
      es: ['familia', 'fa-MEE-lya', 'f'], fr: ['famille', 'fa-MEE-yeh', 'f'], de: ['Familie', 'fa-MEE-lyeh', 'f'],
      it: ['famiglia', 'fa-MEE-lya', 'f'], pt: ['família', 'fa-MEE-lya', 'f'], nl: ['familie', 'fa-MEE-lee', 'c'],
      ru: ['семья', 'syem-YA', 'f'], ja: ['家族', 'kazoku', ''], ko: ['가족', 'gajok', ''],
      zh: ['家庭', 'jiā tíng', ''], hi: ['परिवार', 'parivaar', 'm'], ar: ['عائلة', 'aaila', 'f']
    }
  },

  /* ── Places ─────────────────────────────────────────────────────────── */
  house: {
    pack: 'places', en: 'house', em: '🏠',
    t: {
      es: ['casa', 'KA-sa', 'f'], fr: ['maison', 'meh-ZOHN', 'f'], de: ['Haus', 'howss', 'n'],
      it: ['casa', 'KA-za', 'f'], pt: ['casa', 'KA-za', 'f'], nl: ['huis', 'howss', 'n'],
      ru: ['дом', 'dom', 'm'], ja: ['家', 'ie', ''], ko: ['집', 'jip', ''],
      zh: ['房子', 'fáng zi', ''], hi: ['घर', 'ghar', 'm'], ar: ['بيت', 'bayt', 'm']
    }
  },
  school: {
    pack: 'places', en: 'school', em: '🏫',
    t: {
      es: ['escuela', 'es-KWEH-la', 'f'], fr: ['école', 'ay-KOL', 'f'], de: ['Schule', 'SHOO-leh', 'f'],
      it: ['scuola', 'SKWO-la', 'f'], pt: ['escola', 'es-KO-la', 'f'], nl: ['school', 'skhohl', 'c'],
      ru: ['школа', 'SHKO-la', 'f'], ja: ['学校', 'gakkou', ''], ko: ['학교', 'hakgyo', ''],
      zh: ['学校', 'xué xiào', ''], hi: ['स्कूल', 'skool', 'm'], ar: ['مدرسة', 'madrasa', 'f']
    }
  },
  hospital: {
    pack: 'places', en: 'hospital', em: '🏥',
    t: {
      es: ['hospital', 'os-pee-TAL', 'm'], fr: ['hôpital', 'oh-pee-TAL', 'm'], de: ['Krankenhaus', 'KRAN-ken-howss', 'n'],
      it: ['ospedale', 'os-peh-DA-leh', 'm'], pt: ['hospital', 'os-pee-TAL', 'm'], nl: ['ziekenhuis', 'ZEE-ken-howss', 'n'],
      ru: ['больница', 'bal-NEE-tsa', 'f'], ja: ['病院', 'byouin', ''], ko: ['병원', 'byeongwon', ''],
      zh: ['医院', 'yī yuàn', ''], hi: ['अस्पताल', 'aspataal', 'm'], ar: ['مستشفى', 'mustashfa', 'm']
    }
  },
  restaurant: {
    pack: 'places', en: 'restaurant', em: '🍽️',
    t: {
      es: ['restaurante', 'res-tow-RAN-teh', 'm'], fr: ['restaurant', 'res-toh-RAHN', 'm'], de: ['Restaurant', 'res-toh-RAHNT', 'n'],
      it: ['ristorante', 'rees-to-RAN-teh', 'm'], pt: ['restaurante', 'hes-tow-RAN-chee', 'm'], nl: ['restaurant', 'res-toh-RAHNT', 'n'],
      ru: ['ресторан', 'rees-ta-RAN', 'm'], ja: ['レストラン', 'resutoran', ''], ko: ['식당', 'sikdang', ''],
      zh: ['餐厅', 'cān tīng', ''], hi: ['रेस्तराँ', 'restaraan', 'm'], ar: ['مطعم', 'matam', 'm']
    }
  },
  shop: {
    pack: 'places', en: 'shop', em: '🏪',
    t: {
      es: ['tienda', 'TYEN-da', 'f'], fr: ['magasin', 'ma-ga-ZAN', 'm'], de: ['Geschäft', 'geh-SHEFT', 'n'],
      it: ['negozio', 'neh-GOT-tsyo', 'm'], pt: ['loja', 'LO-zha', 'f'], nl: ['winkel', 'VIN-kel', 'c'],
      ru: ['магазин', 'ma-ga-ZEEN', 'm'], ja: ['店', 'mise', ''], ko: ['가게', 'gage', ''],
      zh: ['商店', 'shāng diàn', ''], hi: ['दुकान', 'dukaan', 'f'], ar: ['متجر', 'matjar', 'm']
    }
  },
  station: {
    pack: 'places', en: 'station', em: '🚉',
    t: {
      es: ['estación', 'es-ta-SYON', 'f'], fr: ['gare', 'gahr', 'f'], de: ['Bahnhof', 'BAHN-hohf', 'm'],
      it: ['stazione', 'stat-TSYO-neh', 'f'], pt: ['estação', 'es-ta-SOWNG', 'f'], nl: ['station', 'sta-SYON', 'n'],
      ru: ['вокзал', 'vag-ZAL', 'm'], ja: ['駅', 'eki', ''], ko: ['역', 'yeok', ''],
      zh: ['车站', 'chē zhàn', ''], hi: ['स्टेशन', 'steshan', 'm'], ar: ['محطة', 'mahatta', 'f']
    }
  },
  airport: {
    pack: 'places', en: 'airport', em: '🛫',
    t: {
      es: ['aeropuerto', 'a-eh-ro-PWER-to', 'm'], fr: ['aéroport', 'a-ay-ro-POR', 'm'], de: ['Flughafen', 'FLOOK-ha-fen', 'm'],
      it: ['aeroporto', 'a-eh-ro-POR-to', 'm'], pt: ['aeroporto', 'a-eh-ro-POR-too', 'm'], nl: ['luchthaven', 'LUKHT-ha-ven', 'c'],
      ru: ['аэропорт', 'a-e-ra-PORT', 'm'], ja: ['空港', 'kuukou', ''], ko: ['공항', 'gonghang', ''],
      zh: ['机场', 'jī chǎng', ''], hi: ['हवाई अड्डा', 'hawaaee addaa', 'm'], ar: ['مطار', 'mataar', 'm']
    }
  },
  hotel: {
    pack: 'places', en: 'hotel', em: '🏨',
    t: {
      es: ['hotel', 'o-TEL', 'm'], fr: ['hôtel', 'oh-TEL', 'm'], de: ['Hotel', 'ho-TEL', 'n'],
      it: ['albergo', 'al-BER-go', 'm'], pt: ['hotel', 'o-TEL', 'm'], nl: ['hotel', 'ho-TEL', 'n'],
      ru: ['гостиница', 'gas-TEE-nee-tsa', 'f'], ja: ['ホテル', 'hoteru', ''], ko: ['호텔', 'hotel', ''],
      zh: ['酒店', 'jiǔ diàn', ''], hi: ['होटल', 'hotal', 'm'], ar: ['فندق', 'funduq', 'm']
    }
  },
  bank: {
    pack: 'places', en: 'bank', em: '🏦',
    t: {
      es: ['banco', 'BAN-ko', 'm'], fr: ['banque', 'bahnk', 'f'], de: ['Bank', 'bank', 'f'],
      it: ['banca', 'BAN-ka', 'f'], pt: ['banco', 'BAN-koo', 'm'], nl: ['bank', 'bank', 'c'],
      ru: ['банк', 'bank', 'm'], ja: ['銀行', 'ginkou', ''], ko: ['은행', 'eunhaeng', ''],
      zh: ['银行', 'yín háng', ''], hi: ['बैंक', 'baink', 'm'], ar: ['بنك', 'bank', 'm']
    }
  },
  park: {
    pack: 'places', en: 'park', em: '🌳',
    t: {
      es: ['parque', 'PAR-keh', 'm'], fr: ['parc', 'park', 'm'], de: ['Park', 'park', 'm'],
      it: ['parco', 'PAR-ko', 'm'], pt: ['parque', 'PAR-kee', 'm'], nl: ['park', 'park', 'n'],
      ru: ['парк', 'park', 'm'], ja: ['公園', 'kouen', ''], ko: ['공원', 'gongwon', ''],
      zh: ['公园', 'gōng yuán', ''], hi: ['पार्क', 'paark', 'm'], ar: ['حديقة', 'hadeeqa', 'f']
    }
  },
  street: {
    pack: 'places', en: 'street', em: '🛣️',
    t: {
      es: ['calle', 'KA-yeh', 'f'], fr: ['rue', 'ruu', 'f'], de: ['Straße', 'SHTRA-seh', 'f'],
      it: ['strada', 'STRA-da', 'f'], pt: ['rua', 'HOO-a', 'f'], nl: ['straat', 'straht', 'c'],
      ru: ['улица', 'OO-lee-tsa', 'f'], ja: ['通り', 'toori', ''], ko: ['거리', 'geori', ''],
      zh: ['街道', 'jiē dào', ''], hi: ['सड़क', 'sadak', 'f'], ar: ['شارع', 'shaari', 'm']
    }
  },
  city: {
    pack: 'places', en: 'city', em: '🏙️',
    t: {
      es: ['ciudad', 'syoo-DAD', 'f'], fr: ['ville', 'veel', 'f'], de: ['Stadt', 'shtat', 'f'],
      it: ['città', 'cheet-TA', 'f'], pt: ['cidade', 'see-DA-jee', 'f'], nl: ['stad', 'stat', 'c'],
      ru: ['город', 'GO-rat', 'm'], ja: ['都市', 'toshi', ''], ko: ['도시', 'dosi', ''],
      zh: ['城市', 'chéng shì', ''], hi: ['शहर', 'shahar', 'm'], ar: ['مدينة', 'madeena', 'f']
    }
  },

  /* ── Time ───────────────────────────────────────────────────────────── */
  today: {
    pack: 'time', en: 'today', em: '📅',
    t: {
      es: ['hoy', 'oy', ''], fr: ["aujourd'hui", 'oh-zhoor-DWEE', ''], de: ['heute', 'HOY-teh', ''],
      it: ['oggi', 'OD-jee', ''], pt: ['hoje', 'O-zhee', ''], nl: ['vandaag', 'van-DAHKH', ''],
      ru: ['сегодня', 'sye-VOD-nya', ''], ja: ['今日', 'kyou', ''], ko: ['오늘', 'oneul', ''],
      zh: ['今天', 'jīn tiān', ''], hi: ['आज', 'aaj', ''], ar: ['اليوم', 'al-yawm', '']
    }
  },
  tomorrow: {
    pack: 'time', en: 'tomorrow', em: '➡️',
    t: {
      es: ['mañana', 'ma-NYA-na', ''], fr: ['demain', 'duh-MAN', ''], de: ['morgen', 'MOR-gen', ''],
      it: ['domani', 'do-MA-nee', ''], pt: ['amanhã', 'a-ma-NYANG', ''], nl: ['morgen', 'MOR-khen', ''],
      ru: ['завтра', 'ZAF-tra', ''], ja: ['明日', 'ashita', ''], ko: ['내일', 'naeil', ''],
      zh: ['明天', 'míng tiān', ''], hi: ['कल', 'kal', ''], ar: ['غدا', 'ghadan', '']
    }
  },
  yesterday: {
    pack: 'time', en: 'yesterday', em: '⬅️',
    t: {
      es: ['ayer', 'a-YER', ''], fr: ['hier', 'yehr', ''], de: ['gestern', 'GES-tern', ''],
      it: ['ieri', 'YEH-ree', ''], pt: ['ontem', 'ON-teng', ''], nl: ['gisteren', 'KHIS-teh-ren', ''],
      ru: ['вчера', 'fchye-RA', ''], ja: ['昨日', 'kinou', ''], ko: ['어제', 'eoje', ''],
      zh: ['昨天', 'zuó tiān', ''], hi: ['कल', 'kal', ''], ar: ['أمس', 'ams', '']
    }
  },
  now: {
    pack: 'time', en: 'now', em: '⏱️',
    t: {
      es: ['ahora', 'a-O-ra', ''], fr: ['maintenant', 'mant-NAHN', ''], de: ['jetzt', 'yetst', ''],
      it: ['adesso', 'a-DES-so', ''], pt: ['agora', 'a-GO-ra', ''], nl: ['nu', 'nuu', ''],
      ru: ['сейчас', 'sye-CHAS', ''], ja: ['今', 'ima', ''], ko: ['지금', 'jigeum', ''],
      zh: ['现在', 'xiàn zài', ''], hi: ['अभी', 'abhee', ''], ar: ['الآن', 'al-aan', '']
    }
  },
  morning: {
    pack: 'time', en: 'morning', em: '🌅',
    t: {
      es: ['mañana', 'ma-NYA-na', 'f'], fr: ['matin', 'ma-TAN', 'm'], de: ['Morgen', 'MOR-gen', 'm'],
      it: ['mattina', 'mat-TEE-na', 'f'], pt: ['manhã', 'ma-NYANG', 'f'], nl: ['ochtend', 'OKH-tent', 'c'],
      ru: ['утро', 'OO-tra', 'n'], ja: ['朝', 'asa', ''], ko: ['아침', 'achim', ''],
      zh: ['早上', 'zǎo shang', ''], hi: ['सुबह', 'subah', 'f'], ar: ['صباح', 'sabaah', 'm']
    }
  },
  afternoon: {
    pack: 'time', en: 'afternoon', em: '☀️',
    t: {
      es: ['tarde', 'TAR-deh', 'f'], fr: ['après-midi', 'a-preh-mee-DEE', 'm'], de: ['Nachmittag', 'NAKH-mit-tahk', 'm'],
      it: ['pomeriggio', 'po-meh-REED-jo', 'm'], pt: ['tarde', 'TAR-jee', 'f'], nl: ['middag', 'MID-dakh', 'c'],
      ru: ['день', 'dyen', 'm'], ja: ['午後', 'gogo', ''], ko: ['오후', 'ohu', ''],
      zh: ['下午', 'xià wǔ', ''], hi: ['दोपहर', 'dopahar', 'f'], ar: ['بعد الظهر', 'bad az-zuhr', 'm']
    }
  },
  evening: {
    pack: 'time', en: 'evening', em: '🌆',
    t: {
      es: ['noche', 'NO-cheh', 'f'], fr: ['soir', 'swahr', 'm'], de: ['Abend', 'AH-bent', 'm'],
      it: ['sera', 'SEH-ra', 'f'], pt: ['noite', 'NOY-chee', 'f'], nl: ['avond', 'AH-vont', 'c'],
      ru: ['вечер', 'VYE-chyer', 'm'], ja: ['夕方', 'yuugata', ''], ko: ['저녁', 'jeonyeok', ''],
      zh: ['晚上', 'wǎn shang', ''], hi: ['शाम', 'shaam', 'f'], ar: ['مساء', 'masaa', 'm']
    }
  },
  night: {
    pack: 'time', en: 'night', em: '🌃',
    t: {
      es: ['noche', 'NO-cheh', 'f'], fr: ['nuit', 'nwee', 'f'], de: ['Nacht', 'nakht', 'f'],
      it: ['notte', 'NOT-teh', 'f'], pt: ['noite', 'NOY-chee', 'f'], nl: ['nacht', 'nakht', 'c'],
      ru: ['ночь', 'noch', 'f'], ja: ['夜', 'yoru', ''], ko: ['밤', 'bam', ''],
      zh: ['夜晚', 'yè wǎn', ''], hi: ['रात', 'raat', 'f'], ar: ['ليل', 'layl', 'm']
    }
  },
  day: {
    pack: 'time', en: 'day', em: '📆',
    t: {
      es: ['día', 'DEE-a', 'm'], fr: ['jour', 'zhoor', 'm'], de: ['Tag', 'tahk', 'm'],
      it: ['giorno', 'JOR-no', 'm'], pt: ['dia', 'DEE-a', 'm'], nl: ['dag', 'dakh', 'c'],
      ru: ['день', 'dyen', 'm'], ja: ['日', 'hi', ''], ko: ['날', 'nal', ''],
      zh: ['天', 'tiān', ''], hi: ['दिन', 'din', 'm'], ar: ['يوم', 'yawm', 'm']
    }
  },
  week: {
    pack: 'time', en: 'week', em: '🗓️',
    t: {
      es: ['semana', 'seh-MA-na', 'f'], fr: ['semaine', 'suh-MEN', 'f'], de: ['Woche', 'VO-kheh', 'f'],
      it: ['settimana', 'set-tee-MA-na', 'f'], pt: ['semana', 'seh-MA-na', 'f'], nl: ['week', 'vayk', 'c'],
      ru: ['неделя', 'nye-DYE-lya', 'f'], ja: ['週', 'shuu', ''], ko: ['주', 'ju', ''],
      zh: ['星期', 'xīng qī', ''], hi: ['सप्ताह', 'saptaah', 'm'], ar: ['أسبوع', 'usboo', 'm']
    }
  },
  month: {
    pack: 'time', en: 'month', em: '🈷️',
    t: {
      es: ['mes', 'mess', 'm'], fr: ['mois', 'mwah', 'm'], de: ['Monat', 'MO-naht', 'm'],
      it: ['mese', 'MEH-zeh', 'm'], pt: ['mês', 'mehss', 'm'], nl: ['maand', 'mahnt', 'c'],
      ru: ['месяц', 'MYE-syats', 'm'], ja: ['月', 'tsuki', ''], ko: ['달', 'dal', ''],
      zh: ['月', 'yuè', ''], hi: ['महीना', 'maheenaa', 'm'], ar: ['شهر', 'shahr', 'm']
    }
  },
  year: {
    pack: 'time', en: 'year', em: '🎊',
    t: {
      es: ['año', 'A-nyo', 'm'], fr: ['an', 'ahn', 'm'], de: ['Jahr', 'yahr', 'n'],
      it: ['anno', 'AN-no', 'm'], pt: ['ano', 'A-noo', 'm'], nl: ['jaar', 'yahr', 'n'],
      ru: ['год', 'got', 'm'], ja: ['年', 'toshi', ''], ko: ['해', 'hae', ''],
      zh: ['年', 'nián', ''], hi: ['साल', 'saal', 'm'], ar: ['سنة', 'sana', 'f']
    }
  },
  hour: {
    pack: 'time', en: 'hour', em: '🕐',
    t: {
      es: ['hora', 'O-ra', 'f'], fr: ['heure', 'uhr', 'f'], de: ['Stunde', 'SHTOON-deh', 'f'],
      it: ['ora', 'O-ra', 'f'], pt: ['hora', 'O-ra', 'f'], nl: ['uur', 'uur', 'n'],
      ru: ['час', 'chas', 'm'], ja: ['時間', 'jikan', ''], ko: ['시간', 'sigan', ''],
      zh: ['小时', 'xiǎo shí', ''], hi: ['घंटा', 'ghantaa', 'm'], ar: ['ساعة', 'saaa', 'f']
    }
  },
  minute: {
    pack: 'time', en: 'minute', em: '⏲️',
    t: {
      es: ['minuto', 'mee-NOO-to', 'm'], fr: ['minute', 'mee-NUUT', 'f'], de: ['Minute', 'mee-NOO-teh', 'f'],
      it: ['minuto', 'mee-NOO-to', 'm'], pt: ['minuto', 'mee-NOO-too', 'm'], nl: ['minuut', 'mee-NUUT', 'c'],
      ru: ['минута', 'mee-NOO-ta', 'f'], ja: ['分', 'fun', ''], ko: ['분', 'bun', ''],
      zh: ['分钟', 'fēn zhōng', ''], hi: ['मिनट', 'minat', 'm'], ar: ['دقيقة', 'daqeeqa', 'f']
    }
  },

  /* ── Verbs (infinitive / dictionary form) ───────────────────────────── */
  to_be: {
    pack: 'verbs', en: 'to be', em: '🧍',
    t: {
      es: ['ser', 'sehr', ''], fr: ['être', 'EH-truh', ''], de: ['sein', 'zine', ''],
      it: ['essere', 'ES-seh-reh', ''], pt: ['ser', 'sehr', ''], nl: ['zijn', 'zine', ''],
      ru: ['быть', 'bit', ''], ja: ['です', 'desu', ''], ko: ['이다', 'ida', ''],
      zh: ['是', 'shì', ''], hi: ['होना', 'honaa', ''], ar: ['يكون', 'yakoon', '']
    }
  },
  to_have: {
    pack: 'verbs', en: 'to have', em: '🤲',
    t: {
      es: ['tener', 'teh-NEHR', ''], fr: ['avoir', 'a-VWAHR', ''], de: ['haben', 'HA-ben', ''],
      it: ['avere', 'a-VEH-reh', ''], pt: ['ter', 'tehr', ''], nl: ['hebben', 'HEB-ben', ''],
      ru: ['иметь', 'ee-MYET', ''], ja: ['持つ', 'motsu', ''], ko: ['가지다', 'gajida', ''],
      zh: ['有', 'yǒu', ''], hi: ['पास होना', 'paas honaa', ''], ar: ['يملك', 'yamlik', '']
    }
  },
  to_go: {
    pack: 'verbs', en: 'to go', em: '🚶',
    t: {
      es: ['ir', 'eer', ''], fr: ['aller', 'a-LAY', ''], de: ['gehen', 'GAY-en', ''],
      it: ['andare', 'an-DA-reh', ''], pt: ['ir', 'eer', ''], nl: ['gaan', 'khahn', ''],
      ru: ['идти', 'eed-TEE', ''], ja: ['行く', 'iku', ''], ko: ['가다', 'gada', ''],
      zh: ['去', 'qù', ''], hi: ['जाना', 'jaanaa', ''], ar: ['يذهب', 'yadhhab', '']
    }
  },
  to_come: {
    pack: 'verbs', en: 'to come', em: '🏃',
    t: {
      es: ['venir', 'veh-NEER', ''], fr: ['venir', 'vuh-NEER', ''], de: ['kommen', 'KOM-men', ''],
      it: ['venire', 'veh-NEE-reh', ''], pt: ['vir', 'veer', ''], nl: ['komen', 'KO-men', ''],
      ru: ['приходить', 'pree-ha-DEET', ''], ja: ['来る', 'kuru', ''], ko: ['오다', 'oda', ''],
      zh: ['来', 'lái', ''], hi: ['आना', 'aanaa', ''], ar: ['يأتي', 'yatee', '']
    }
  },
  to_eat: {
    pack: 'verbs', en: 'to eat', em: '🍽️',
    t: {
      es: ['comer', 'ko-MEHR', ''], fr: ['manger', 'mahn-ZHAY', ''], de: ['essen', 'ES-sen', ''],
      it: ['mangiare', 'man-JA-reh', ''], pt: ['comer', 'ko-MEHR', ''], nl: ['eten', 'AY-ten', ''],
      ru: ['есть', 'yest', ''], ja: ['食べる', 'taberu', ''], ko: ['먹다', 'meokda', ''],
      zh: ['吃', 'chī', ''], hi: ['खाना', 'khaanaa', ''], ar: ['يأكل', 'yakul', '']
    }
  },
  to_drink: {
    pack: 'verbs', en: 'to drink', em: '🥤',
    t: {
      es: ['beber', 'beh-BEHR', ''], fr: ['boire', 'bwahr', ''], de: ['trinken', 'TRIN-ken', ''],
      it: ['bere', 'BEH-reh', ''], pt: ['beber', 'beh-BEHR', ''], nl: ['drinken', 'DRIN-ken', ''],
      ru: ['пить', 'peet', ''], ja: ['飲む', 'nomu', ''], ko: ['마시다', 'masida', ''],
      zh: ['喝', 'hē', ''], hi: ['पीना', 'peenaa', ''], ar: ['يشرب', 'yashrab', '']
    }
  },
  to_see: {
    pack: 'verbs', en: 'to see', em: '👀',
    t: {
      es: ['ver', 'vehr', ''], fr: ['voir', 'vwahr', ''], de: ['sehen', 'ZAY-en', ''],
      it: ['vedere', 'veh-DEH-reh', ''], pt: ['ver', 'vehr', ''], nl: ['zien', 'zeen', ''],
      ru: ['видеть', 'VEE-dyet', ''], ja: ['見る', 'miru', ''], ko: ['보다', 'boda', ''],
      zh: ['看', 'kàn', ''], hi: ['देखना', 'dekhnaa', ''], ar: ['يرى', 'yaraa', '']
    }
  },
  to_speak: {
    pack: 'verbs', en: 'to speak', em: '🗣️',
    t: {
      es: ['hablar', 'a-BLAR', ''], fr: ['parler', 'par-LAY', ''], de: ['sprechen', 'SHPREH-khen', ''],
      it: ['parlare', 'par-LA-reh', ''], pt: ['falar', 'fa-LAR', ''], nl: ['spreken', 'SPRAY-ken', ''],
      ru: ['говорить', 'ga-va-REET', ''], ja: ['話す', 'hanasu', ''], ko: ['말하다', 'malhada', ''],
      zh: ['说', 'shuō', ''], hi: ['बोलना', 'bolnaa', ''], ar: ['يتكلم', 'yatakallam', '']
    }
  },
  to_want: {
    pack: 'verbs', en: 'to want', em: '🙌',
    t: {
      es: ['querer', 'keh-REHR', ''], fr: ['vouloir', 'voo-LWAHR', ''], de: ['wollen', 'VOL-len', ''],
      it: ['volere', 'vo-LEH-reh', ''], pt: ['querer', 'keh-REHR', ''], nl: ['willen', 'VIL-len', ''],
      ru: ['хотеть', 'ha-TYET', ''], ja: ['欲しい', 'hoshii', ''], ko: ['원하다', 'wonhada', ''],
      zh: ['想要', 'xiǎng yào', ''], hi: ['चाहना', 'chaahnaa', ''], ar: ['يريد', 'yureed', '']
    }
  },
  to_need: {
    pack: 'verbs', en: 'to need', em: '❗',
    t: {
      es: ['necesitar', 'neh-seh-see-TAR', ''], fr: ['avoir besoin', 'a-vwahr buh-ZWAN', ''], de: ['brauchen', 'BROW-khen', ''],
      it: ['avere bisogno', 'a-veh-reh bee-ZO-nyo', ''], pt: ['precisar', 'preh-see-ZAR', ''], nl: ['nodig hebben', 'NO-dikh HEB-ben', ''],
      ru: ['нуждаться', 'nooz-DAT-tsa', ''], ja: ['必要とする', 'hitsuyou to suru', ''], ko: ['필요하다', 'piryohada', ''],
      zh: ['需要', 'xū yào', ''], hi: ['ज़रूरत होना', 'zaroorat honaa', ''], ar: ['يحتاج', 'yahtaaj', '']
    }
  },
  to_know: {
    pack: 'verbs', en: 'to know', em: '🧠',
    t: {
      es: ['saber', 'sa-BEHR', ''], fr: ['savoir', 'sa-VWAHR', ''], de: ['wissen', 'VIS-sen', ''],
      it: ['sapere', 'sa-PEH-reh', ''], pt: ['saber', 'sa-BEHR', ''], nl: ['weten', 'VAY-ten', ''],
      ru: ['знать', 'znat', ''], ja: ['知る', 'shiru', ''], ko: ['알다', 'alda', ''],
      zh: ['知道', 'zhī dào', ''], hi: ['जानना', 'jaannaa', ''], ar: ['يعرف', 'yarif', '']
    }
  },
  to_do: {
    pack: 'verbs', en: 'to do', em: '🛠️',
    t: {
      es: ['hacer', 'a-SEHR', ''], fr: ['faire', 'fehr', ''], de: ['machen', 'MA-khen', ''],
      it: ['fare', 'FA-reh', ''], pt: ['fazer', 'fa-ZEHR', ''], nl: ['doen', 'doon', ''],
      ru: ['делать', 'DYE-lat', ''], ja: ['する', 'suru', ''], ko: ['하다', 'hada', ''],
      zh: ['做', 'zuò', ''], hi: ['करना', 'karnaa', ''], ar: ['يفعل', 'yafal', '']
    }
  },
  to_buy: {
    pack: 'verbs', en: 'to buy', em: '🛒',
    t: {
      es: ['comprar', 'kom-PRAR', ''], fr: ['acheter', 'ash-TAY', ''], de: ['kaufen', 'KOW-fen', ''],
      it: ['comprare', 'kom-PRA-reh', ''], pt: ['comprar', 'kom-PRAR', ''], nl: ['kopen', 'KO-pen', ''],
      ru: ['покупать', 'pa-koo-PAT', ''], ja: ['買う', 'kau', ''], ko: ['사다', 'sada', ''],
      zh: ['买', 'mǎi', ''], hi: ['खरीदना', 'khareednaa', ''], ar: ['يشتري', 'yashtaree', '']
    }
  },
  to_read: {
    pack: 'verbs', en: 'to read', em: '📖',
    t: {
      es: ['leer', 'leh-EHR', ''], fr: ['lire', 'leer', ''], de: ['lesen', 'LAY-zen', ''],
      it: ['leggere', 'LED-jeh-reh', ''], pt: ['ler', 'lehr', ''], nl: ['lezen', 'LAY-zen', ''],
      ru: ['читать', 'chee-TAT', ''], ja: ['読む', 'yomu', ''], ko: ['읽다', 'ikda', ''],
      zh: ['读', 'dú', ''], hi: ['पढ़ना', 'padhnaa', ''], ar: ['يقرأ', 'yaqra', '']
    }
  },
  to_write: {
    pack: 'verbs', en: 'to write', em: '✍️',
    t: {
      es: ['escribir', 'es-kree-BEER', ''], fr: ['écrire', 'ay-KREER', ''], de: ['schreiben', 'SHRY-ben', ''],
      it: ['scrivere', 'SKREE-veh-reh', ''], pt: ['escrever', 'es-kreh-VEHR', ''], nl: ['schrijven', 'SKHRY-ven', ''],
      ru: ['писать', 'pee-SAT', ''], ja: ['書く', 'kaku', ''], ko: ['쓰다', 'sseuda', ''],
      zh: ['写', 'xiě', ''], hi: ['लिखना', 'likhnaa', ''], ar: ['يكتب', 'yaktub', '']
    }
  },
  to_sleep: {
    pack: 'verbs', en: 'to sleep', em: '😴',
    t: {
      es: ['dormir', 'dor-MEER', ''], fr: ['dormir', 'dor-MEER', ''], de: ['schlafen', 'SHLA-fen', ''],
      it: ['dormire', 'dor-MEE-reh', ''], pt: ['dormir', 'dor-MEER', ''], nl: ['slapen', 'SLA-pen', ''],
      ru: ['спать', 'spat', ''], ja: ['寝る', 'neru', ''], ko: ['자다', 'jada', ''],
      zh: ['睡觉', 'shuì jiào', ''], hi: ['सोना', 'sonaa', ''], ar: ['ينام', 'yanaam', '']
    }
  },

  /* ── Adjectives (base / masculine form) ─────────────────────────────── */
  big: {
    pack: 'adjectives', en: 'big', em: '🐘',
    t: {
      es: ['grande', 'GRAN-deh', ''], fr: ['grand', 'grahn', ''], de: ['groß', 'grohss', ''],
      it: ['grande', 'GRAN-deh', ''], pt: ['grande', 'GRAN-jee', ''], nl: ['groot', 'khroht', ''],
      ru: ['большой', 'bal-SHOY', ''], ja: ['大きい', 'ookii', ''], ko: ['큰', 'keun', ''],
      zh: ['大', 'dà', ''], hi: ['बड़ा', 'badaa', ''], ar: ['كبير', 'kabeer', '']
    }
  },
  small: {
    pack: 'adjectives', en: 'small', em: '🐜',
    t: {
      es: ['pequeño', 'peh-KEH-nyo', ''], fr: ['petit', 'puh-TEE', ''], de: ['klein', 'kline', ''],
      it: ['piccolo', 'PEEK-ko-lo', ''], pt: ['pequeno', 'peh-KEH-noo', ''], nl: ['klein', 'kline', ''],
      ru: ['маленький', 'MA-lyen-kee', ''], ja: ['小さい', 'chiisai', ''], ko: ['작은', 'jageun', ''],
      zh: ['小', 'xiǎo', ''], hi: ['छोटा', 'chhotaa', ''], ar: ['صغير', 'sagheer', '']
    }
  },
  good: {
    pack: 'adjectives', en: 'good', em: '👍',
    t: {
      es: ['bueno', 'BWEH-no', ''], fr: ['bon', 'bohn', ''], de: ['gut', 'goot', ''],
      it: ['buono', 'BWO-no', ''], pt: ['bom', 'bong', ''], nl: ['goed', 'khoot', ''],
      ru: ['хороший', 'ha-RO-shee', ''], ja: ['良い', 'yoi', ''], ko: ['좋은', 'joeun', ''],
      zh: ['好', 'hǎo', ''], hi: ['अच्छा', 'achchhaa', ''], ar: ['جيد', 'jayyid', '']
    }
  },
  bad: {
    pack: 'adjectives', en: 'bad', em: '👎',
    t: {
      es: ['malo', 'MA-lo', ''], fr: ['mauvais', 'mo-VEH', ''], de: ['schlecht', 'shlekht', ''],
      it: ['cattivo', 'kat-TEE-vo', ''], pt: ['mau', 'mow', ''], nl: ['slecht', 'slekht', ''],
      ru: ['плохой', 'pla-HOY', ''], ja: ['悪い', 'warui', ''], ko: ['나쁜', 'nappeun', ''],
      zh: ['坏', 'huài', ''], hi: ['बुरा', 'buraa', ''], ar: ['سيئ', 'sayyi', '']
    }
  },
  hot: {
    pack: 'adjectives', en: 'hot', em: '🔥',
    t: {
      es: ['caliente', 'ka-LYEN-teh', ''], fr: ['chaud', 'shoh', ''], de: ['heiß', 'hice', ''],
      it: ['caldo', 'KAL-do', ''], pt: ['quente', 'KEN-chee', ''], nl: ['heet', 'hayt', ''],
      ru: ['горячий', 'ga-RYA-chee', ''], ja: ['熱い', 'atsui', ''], ko: ['뜨거운', 'tteugeoun', ''],
      zh: ['热', 'rè', ''], hi: ['गरम', 'garam', ''], ar: ['حار', 'haar', '']
    }
  },
  cold: {
    pack: 'adjectives', en: 'cold', em: '🧊',
    t: {
      es: ['frío', 'FREE-o', ''], fr: ['froid', 'frwah', ''], de: ['kalt', 'kalt', ''],
      it: ['freddo', 'FRED-do', ''], pt: ['frio', 'FREE-oo', ''], nl: ['koud', 'kowt', ''],
      ru: ['холодный', 'ha-LOD-nee', ''], ja: ['冷たい', 'tsumetai', ''], ko: ['차가운', 'chagaun', ''],
      zh: ['冷', 'lěng', ''], hi: ['ठंडा', 'thandaa', ''], ar: ['بارد', 'baarid', '']
    }
  },
  new: {
    pack: 'adjectives', en: 'new', em: '🆕',
    t: {
      es: ['nuevo', 'NWEH-vo', ''], fr: ['nouveau', 'noo-VOH', ''], de: ['neu', 'noy', ''],
      it: ['nuovo', 'NWO-vo', ''], pt: ['novo', 'NO-voo', ''], nl: ['nieuw', 'nyoo', ''],
      ru: ['новый', 'NO-vee', ''], ja: ['新しい', 'atarashii', ''], ko: ['새로운', 'saeroun', ''],
      zh: ['新', 'xīn', ''], hi: ['नया', 'nayaa', ''], ar: ['جديد', 'jadeed', '']
    }
  },
  old: {
    pack: 'adjectives', en: 'old', em: '🏛️',
    t: {
      es: ['viejo', 'VYEH-ho', ''], fr: ['vieux', 'vyuh', ''], de: ['alt', 'alt', ''],
      it: ['vecchio', 'VEK-kyo', ''], pt: ['velho', 'VEH-lyoo', ''], nl: ['oud', 'owt', ''],
      ru: ['старый', 'STA-ree', ''], ja: ['古い', 'furui', ''], ko: ['오래된', 'oraedoen', ''],
      zh: ['旧', 'jiù', ''], hi: ['पुराना', 'puraanaa', ''], ar: ['قديم', 'qadeem', '']
    }
  },
  beautiful: {
    pack: 'adjectives', en: 'beautiful', em: '🌸',
    t: {
      es: ['bonito', 'bo-NEE-to', ''], fr: ['beau', 'boh', ''], de: ['schön', 'shurn', ''],
      it: ['bello', 'BEL-lo', ''], pt: ['bonito', 'bo-NEE-too', ''], nl: ['mooi', 'moy', ''],
      ru: ['красивый', 'kra-SEE-vee', ''], ja: ['美しい', 'utsukushii', ''], ko: ['아름다운', 'areumdaun', ''],
      zh: ['美丽', 'měi lì', ''], hi: ['सुंदर', 'sundar', ''], ar: ['جميل', 'jameel', '']
    }
  },
  expensive: {
    pack: 'adjectives', en: 'expensive', em: '💰',
    t: {
      es: ['caro', 'KA-ro', ''], fr: ['cher', 'shehr', ''], de: ['teuer', 'TOY-er', ''],
      it: ['caro', 'KA-ro', ''], pt: ['caro', 'KA-roo', ''], nl: ['duur', 'duur', ''],
      ru: ['дорогой', 'da-ra-GOY', ''], ja: ['高い', 'takai', ''], ko: ['비싼', 'bissan', ''],
      zh: ['贵', 'guì', ''], hi: ['महंगा', 'mahangaa', ''], ar: ['غالي', 'ghaalee', '']
    }
  },
  cheap: {
    pack: 'adjectives', en: 'cheap', em: '🏷️',
    t: {
      es: ['barato', 'ba-RA-to', ''], fr: ['bon marché', 'bohn mar-SHAY', ''], de: ['billig', 'BIL-likh', ''],
      it: ['economico', 'eh-ko-NO-mee-ko', ''], pt: ['barato', 'ba-RA-too', ''], nl: ['goedkoop', 'khoot-KOHP', ''],
      ru: ['дешёвый', 'dye-SHO-vee', ''], ja: ['安い', 'yasui', ''], ko: ['싼', 'ssan', ''],
      zh: ['便宜', 'pián yi', ''], hi: ['सस्ता', 'sastaa', ''], ar: ['رخيص', 'rakhees', '']
    }
  },
  fast: {
    pack: 'adjectives', en: 'fast', em: '⚡',
    t: {
      es: ['rápido', 'RA-pee-do', ''], fr: ['rapide', 'ra-PEED', ''], de: ['schnell', 'shnel', ''],
      it: ['veloce', 'veh-LO-cheh', ''], pt: ['rápido', 'HA-pee-doo', ''], nl: ['snel', 'snel', ''],
      ru: ['быстрый', 'BIS-tree', ''], ja: ['速い', 'hayai', ''], ko: ['빠른', 'ppareun', ''],
      zh: ['快', 'kuài', ''], hi: ['तेज़', 'tez', ''], ar: ['سريع', 'saree', '']
    }
  },
  slow: {
    pack: 'adjectives', en: 'slow', em: '🐢',
    t: {
      es: ['lento', 'LEN-to', ''], fr: ['lent', 'lahn', ''], de: ['langsam', 'LANG-zahm', ''],
      it: ['lento', 'LEN-to', ''], pt: ['lento', 'LEN-too', ''], nl: ['langzaam', 'LANG-zahm', ''],
      ru: ['медленный', 'MYED-lyen-nee', ''], ja: ['遅い', 'osoi', ''], ko: ['느린', 'neurin', ''],
      zh: ['慢', 'màn', ''], hi: ['धीमा', 'dheemaa', ''], ar: ['بطيء', 'batee', '']
    }
  },
  happy: {
    pack: 'adjectives', en: 'happy', em: '😄',
    t: {
      es: ['feliz', 'feh-LEES', ''], fr: ['heureux', 'uh-RUH', ''], de: ['glücklich', 'GLUUK-likh', ''],
      it: ['felice', 'feh-LEE-cheh', ''], pt: ['feliz', 'feh-LEES', ''], nl: ['blij', 'bly', ''],
      ru: ['счастливый', 'shas-LEE-vee', ''], ja: ['幸せ', 'shiawase', ''], ko: ['행복한', 'haengbokhan', ''],
      zh: ['快乐', 'kuài lè', ''], hi: ['खुश', 'khush', ''], ar: ['سعيد', 'saeed', '']
    }
  },

  /* ── Question words ─────────────────────────────────────────────────── */
  q_what: {
    pack: 'questions', en: 'what', em: '❓',
    t: {
      es: ['qué', 'keh', ''], fr: ['quoi', 'kwah', ''], de: ['was', 'vass', ''],
      it: ['che cosa', 'keh KO-za', ''], pt: ['o que', 'oo keh', ''], nl: ['wat', 'vat', ''],
      ru: ['что', 'shto', ''], ja: ['何', 'nani', ''], ko: ['무엇', 'mueot', ''],
      zh: ['什么', 'shén me', ''], hi: ['क्या', 'kyaa', ''], ar: ['ماذا', 'maadhaa', '']
    }
  },
  q_who: {
    pack: 'questions', en: 'who', em: '🙋',
    t: {
      es: ['quién', 'kyen', ''], fr: ['qui', 'kee', ''], de: ['wer', 'vehr', ''],
      it: ['chi', 'kee', ''], pt: ['quem', 'keng', ''], nl: ['wie', 'vee', ''],
      ru: ['кто', 'kto', ''], ja: ['誰', 'dare', ''], ko: ['누구', 'nugu', ''],
      zh: ['谁', 'shéi', ''], hi: ['कौन', 'kaun', ''], ar: ['من', 'man', '']
    }
  },
  q_where: {
    pack: 'questions', en: 'where', em: '📍',
    t: {
      es: ['dónde', 'DON-deh', ''], fr: ['où', 'oo', ''], de: ['wo', 'voh', ''],
      it: ['dove', 'DO-veh', ''], pt: ['onde', 'ON-jee', ''], nl: ['waar', 'vahr', ''],
      ru: ['где', 'gdye', ''], ja: ['どこ', 'doko', ''], ko: ['어디', 'eodi', ''],
      zh: ['哪里', 'nǎ li', ''], hi: ['कहाँ', 'kahaan', ''], ar: ['أين', 'ayna', '']
    }
  },
  q_when: {
    pack: 'questions', en: 'when', em: '⌚',
    t: {
      es: ['cuándo', 'KWAN-do', ''], fr: ['quand', 'kahn', ''], de: ['wann', 'van', ''],
      it: ['quando', 'KWAN-do', ''], pt: ['quando', 'KWAN-doo', ''], nl: ['wanneer', 'va-NAYR', ''],
      ru: ['когда', 'kag-DA', ''], ja: ['いつ', 'itsu', ''], ko: ['언제', 'eonje', ''],
      zh: ['什么时候', 'shén me shí hou', ''], hi: ['कब', 'kab', ''], ar: ['متى', 'mataa', '']
    }
  },
  q_why: {
    pack: 'questions', en: 'why', em: '🤔',
    t: {
      es: ['por qué', 'por KEH', ''], fr: ['pourquoi', 'poor-KWAH', ''], de: ['warum', 'va-ROOM', ''],
      it: ['perché', 'per-KEH', ''], pt: ['por que', 'por KEH', ''], nl: ['waarom', 'vah-ROM', ''],
      ru: ['почему', 'pa-chye-MOO', ''], ja: ['なぜ', 'naze', ''], ko: ['왜', 'wae', ''],
      zh: ['为什么', 'wèi shén me', ''], hi: ['क्यों', 'kyon', ''], ar: ['لماذا', 'limaadhaa', '']
    }
  },
  q_how: {
    pack: 'questions', en: 'how', em: '🔧',
    t: {
      es: ['cómo', 'KO-mo', ''], fr: ['comment', 'ko-MAHN', ''], de: ['wie', 'vee', ''],
      it: ['come', 'KO-meh', ''], pt: ['como', 'KO-moo', ''], nl: ['hoe', 'hoo', ''],
      ru: ['как', 'kak', ''], ja: ['どう', 'dou', ''], ko: ['어떻게', 'eotteoke', ''],
      zh: ['怎么', 'zěn me', ''], hi: ['कैसे', 'kaise', ''], ar: ['كيف', 'kayfa', '']
    }
  },
  q_how_much: {
    pack: 'questions', en: 'how much', em: '💵',
    t: {
      es: ['cuánto', 'KWAN-to', ''], fr: ['combien', 'kom-BYAN', ''], de: ['wie viel', 'vee FEEL', ''],
      it: ['quanto', 'KWAN-to', ''], pt: ['quanto', 'KWAN-too', ''], nl: ['hoeveel', 'hoo-VAYL', ''],
      ru: ['сколько', 'SKOL-ka', ''], ja: ['いくら', 'ikura', ''], ko: ['얼마', 'eolma', ''],
      zh: ['多少', 'duō shao', ''], hi: ['कितना', 'kitnaa', ''], ar: ['كم', 'kam', '']
    }
  },
  q_which: {
    pack: 'questions', en: 'which', em: '🔀',
    t: {
      es: ['cuál', 'kwal', ''], fr: ['quel', 'kel', ''], de: ['welcher', 'VEL-kher', ''],
      it: ['quale', 'KWA-leh', ''], pt: ['qual', 'kwal', ''], nl: ['welke', 'VEL-keh', ''],
      ru: ['какой', 'ka-KOY', ''], ja: ['どちら', 'dochira', ''], ko: ['어느', 'eoneu', ''],
      zh: ['哪个', 'nǎ ge', ''], hi: ['कौन सा', 'kaun saa', ''], ar: ['أي', 'ayy', '']
    }
  }
};

export const LEXICON_KEYS = Object.keys(LEXICON);
export const PACK_KEYS = Object.keys(PACKS);

/** Look up one lexicon translation, mirroring dictionary.translate(). */
export function translateLex(id, lang) {
  const entry = LEXICON[id];
  if (!entry) return null;
  const t = entry.t[lang];
  if (!t) return null;
  return { word: t[0], phonetic: t[1], gender: t[2], em: entry.em, en: entry.en, pack: entry.pack };
}

/** All lexicon ids in one pack, in declaration order. */
export function idsInPack(pack) {
  return LEXICON_KEYS.filter((id) => LEXICON[id].pack === pack);
}
