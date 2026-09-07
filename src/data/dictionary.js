/**
 * dictionary.js — Vocabulary for all 80 COCO-SSD classes across 12 languages.
 *
 * Entry shape:
 *   em    emoji glyph
 *   cat   category key (see CATEGORIES) — drives decks and filtering
 *   lvl   1 beginner · 2 intermediate · 3 advanced (drives study ordering)
 *   size  expected on-screen footprint, used by the detector's size gate
 *   t     translations, keyed by language code: [word, phonetic, gender]
 *
 * Gender codes: m masculine · f feminine · n neuter · c common (Dutch) ·
 *               p plural-only (article suppressed) · '' none
 */

export const CATEGORIES = {
  people: { label: 'People', em: '🧑', color: '#B57C46' },
  transport: { label: 'Transport', em: '🚗', color: '#4F6D91' },
  street: { label: 'Street', em: '🚦', color: '#A85548' },
  animal: { label: 'Animals', em: '🐾', color: '#7A6182' },
  accessory: { label: 'Accessories', em: '🎒', color: '#A66F78' },
  sport: { label: 'Sport', em: '⚽', color: '#5F7B4C' },
  kitchen: { label: 'Kitchen', em: '🍴', color: '#4E7D7B' },
  food: { label: 'Food', em: '🍎', color: '#A88C3C' },
  furniture: { label: 'Furniture', em: '🛋️', color: '#8A6A52' },
  electronics: { label: 'Electronics', em: '💻', color: '#5A6180' },
  appliance: { label: 'Appliances', em: '🔌', color: '#6B6659' },
  object: { label: 'Objects', em: '📦', color: '#9A7B5C' }
};

export const DICT = {
  person: {
    em: '🧑', cat: 'people', lvl: 1, size: 'large',
    t: {
      es: ['persona', 'per-SO-na', 'f'], fr: ['personne', 'pehr-SUN', 'f'],
      de: ['Person', 'per-ZOHN', 'f'], it: ['persona', 'per-SO-na', 'f'],
      pt: ['pessoa', 'peh-SO-ah', 'f'], nl: ['persoon', 'per-SOAN', 'c'],
      ru: ['человек', 'chuh-lah-VYEK', 'm'], ja: ['人', 'hito', ''],
      ko: ['사람', 'saram', ''], zh: ['人', 'rén', ''],
      hi: ['व्यक्ति', 'vyak-ti', 'm'], ar: ['شخص', 'shakhs', 'm']
    }
  },
  bicycle: {
    em: '🚲', cat: 'transport', lvl: 1, size: 'large',
    t: {
      es: ['bicicleta', 'bee-see-KLEH-ta', 'f'], fr: ['vélo', 'vay-LOH', 'm'],
      de: ['Fahrrad', 'FAAR-raat', 'n'], it: ['bicicletta', 'bee-chee-KLET-ta', 'f'],
      pt: ['bicicleta', 'bee-see-KLEH-ta', 'f'], nl: ['fiets', 'feets', 'c'],
      ru: ['велосипед', 'vye-la-see-PYET', 'm'], ja: ['自転車', 'jitensha', ''],
      ko: ['자전거', 'jajeon-geo', ''], zh: ['自行车', 'zì xíng chē', ''],
      hi: ['साइकिल', 'saai-kil', 'f'], ar: ['دراجة', 'darraaja', 'f']
    }
  },
  car: {
    em: '🚗', cat: 'transport', lvl: 1, size: 'huge',
    t: {
      es: ['coche', 'KO-cheh', 'm'], fr: ['voiture', 'vwa-TUUR', 'f'],
      de: ['Auto', 'OW-toh', 'n'], it: ['macchina', 'MAK-kee-na', 'f'],
      pt: ['carro', 'KAH-hoo', 'm'], nl: ['auto', 'OW-toh', 'c'],
      ru: ['машина', 'ma-SHEE-na', 'f'], ja: ['車', 'kuruma', ''],
      ko: ['자동차', 'jadongcha', ''], zh: ['汽车', 'qì chē', ''],
      hi: ['कार', 'kaar', 'f'], ar: ['سيارة', 'sayyaara', 'f']
    }
  },
  motorcycle: {
    em: '🏍️', cat: 'transport', lvl: 2, size: 'large',
    t: {
      es: ['motocicleta', 'mo-to-see-KLEH-ta', 'f'], fr: ['moto', 'mo-TOH', 'f'],
      de: ['Motorrad', 'mo-TOR-raat', 'n'], it: ['moto', 'MO-to', 'f'],
      pt: ['motocicleta', 'mo-to-see-KLEH-ta', 'f'], nl: ['motorfiets', 'MO-tor-feets', 'c'],
      ru: ['мотоцикл', 'ma-ta-TSIKL', 'm'], ja: ['バイク', 'baiku', ''],
      ko: ['오토바이', 'otobai', ''], zh: ['摩托车', 'mó tuō chē', ''],
      hi: ['मोटरसाइकिल', 'motar-saai-kil', 'f'], ar: ['دراجة نارية', 'darraaja naariyya', 'f']
    }
  },
  airplane: {
    em: '✈️', cat: 'transport', lvl: 1, size: 'huge',
    t: {
      es: ['avión', 'ah-vee-OHN', 'm'], fr: ['avion', 'ah-vee-OHN', 'm'],
      de: ['Flugzeug', 'FLOOK-tsoyk', 'n'], it: ['aereo', 'ah-EH-reh-o', 'm'],
      pt: ['avião', 'ah-vee-OWNG', 'm'], nl: ['vliegtuig', 'FLEEKH-toykh', 'n'],
      ru: ['самолёт', 'sa-ma-LYOT', 'm'], ja: ['飛行機', 'hikouki', ''],
      ko: ['비행기', 'bihaenggi', ''], zh: ['飞机', 'fēi jī', ''],
      hi: ['हवाई जहाज़', 'hawaai jahaaz', 'm'], ar: ['طائرة', 'taa-ira', 'f']
    }
  },
  bus: {
    em: '🚌', cat: 'transport', lvl: 1, size: 'huge',
    t: {
      es: ['autobús', 'ow-toh-BOOS', 'm'], fr: ['bus', 'boos', 'm'],
      de: ['Bus', 'boos', 'm'], it: ['autobus', 'OW-toh-boos', 'm'],
      pt: ['ônibus', 'OH-nee-boos', 'm'], nl: ['bus', 'bus', 'c'],
      ru: ['автобус', 'af-TOH-boos', 'm'], ja: ['バス', 'basu', ''],
      ko: ['버스', 'beoseu', ''], zh: ['公共汽车', 'gōng gòng qì chē', ''],
      hi: ['बस', 'bas', 'f'], ar: ['حافلة', 'haafila', 'f']
    }
  },
  train: {
    em: '🚆', cat: 'transport', lvl: 1, size: 'huge',
    t: {
      es: ['tren', 'trehn', 'm'], fr: ['train', 'trahn', 'm'],
      de: ['Zug', 'tsook', 'm'], it: ['treno', 'TREH-no', 'm'],
      pt: ['trem', 'trehng', 'm'], nl: ['trein', 'trine', 'c'],
      ru: ['поезд', 'PO-yezd', 'm'], ja: ['電車', 'densha', ''],
      ko: ['기차', 'gicha', ''], zh: ['火车', 'huǒ chē', ''],
      hi: ['रेलगाड़ी', 'rel-gaadi', 'f'], ar: ['قطار', 'qitaar', 'm']
    }
  },
  truck: {
    em: '🚛', cat: 'transport', lvl: 2, size: 'huge',
    t: {
      es: ['camión', 'ka-mee-OHN', 'm'], fr: ['camion', 'ka-mee-OHN', 'm'],
      de: ['Lastwagen', 'LAST-vaa-gen', 'm'], it: ['camion', 'KA-mee-on', 'm'],
      pt: ['caminhão', 'ka-mee-NYOWNG', 'm'], nl: ['vrachtwagen', 'FRAKHT-vaa-khen', 'c'],
      ru: ['грузовик', 'groo-za-VEEK', 'm'], ja: ['トラック', 'torakku', ''],
      ko: ['트럭', 'teureok', ''], zh: ['卡车', 'kǎ chē', ''],
      hi: ['ट्रक', 'trak', 'm'], ar: ['شاحنة', 'shaahina', 'f']
    }
  },
  boat: {
    em: '⛵', cat: 'transport', lvl: 1, size: 'huge',
    t: {
      es: ['barco', 'BAR-ko', 'm'], fr: ['bateau', 'ba-TOH', 'm'],
      de: ['Boot', 'boat', 'n'], it: ['barca', 'BAR-ka', 'f'],
      pt: ['barco', 'BAR-koo', 'm'], nl: ['boot', 'boat', 'c'],
      ru: ['лодка', 'LOT-ka', 'f'], ja: ['船', 'fune', ''],
      ko: ['배', 'bae', ''], zh: ['船', 'chuán', ''],
      hi: ['नाव', 'naav', 'f'], ar: ['قارب', 'qaarib', 'm']
    }
  },
  'traffic light': {
    em: '🚦', cat: 'street', lvl: 2, size: 'medium',
    t: {
      es: ['semáforo', 'seh-MAH-fo-ro', 'm'], fr: ['feu de circulation', 'fuh duh seer-koo-la-SYOHN', 'm'],
      de: ['Ampel', 'AM-pel', 'f'], it: ['semaforo', 'seh-MAH-fo-ro', 'm'],
      pt: ['semáforo', 'seh-MAH-fo-roo', 'm'], nl: ['stoplicht', 'STOP-likht', 'n'],
      ru: ['светофор', 'svye-ta-FOR', 'm'], ja: ['信号', 'shingou', ''],
      ko: ['신호등', 'sinhodeung', ''], zh: ['红绿灯', 'hóng lǜ dēng', ''],
      hi: ['ट्रैफ़िक लाइट', 'traifik laait', 'f'], ar: ['إشارة المرور', 'ishaarat al-muroor', 'f']
    }
  },
  'fire hydrant': {
    em: '🧯', cat: 'street', lvl: 3, size: 'medium',
    t: {
      es: ['hidrante', 'ee-DRAN-teh', 'm'], fr: ["bouche d'incendie", 'boosh dan-sahn-DEE', 'f'],
      de: ['Hydrant', 'hoo-DRANT', 'm'], it: ['idrante', 'ee-DRAN-teh', 'm'],
      pt: ['hidrante', 'ee-DRAN-chee', 'm'], nl: ['brandkraan', 'BRANT-kraan', 'c'],
      ru: ['гидрант', 'gee-DRANT', 'm'], ja: ['消火栓', 'shoukasen', ''],
      ko: ['소화전', 'sohwajeon', ''], zh: ['消防栓', 'xiāo fáng shuān', ''],
      hi: ['फ़ायर हाइड्रेंट', 'faayar haaidrent', 'm'], ar: ['صنبور إطفاء', 'sunboor itfaa', 'm']
    }
  },
  'stop sign': {
    em: '🛑', cat: 'street', lvl: 2, size: 'medium',
    t: {
      es: ['señal de stop', 'seh-NYAL deh stop', 'f'], fr: ['panneau stop', 'pa-NOH stop', 'm'],
      de: ['Stoppschild', 'SHTOP-shilt', 'n'], it: ['segnale di stop', 'seh-NYAH-leh dee stop', 'm'],
      pt: ['placa de pare', 'PLA-ka jee PA-ree', 'f'], nl: ['stopbord', 'STOP-bort', 'n'],
      ru: ['знак стоп', 'znak stop', 'm'], ja: ['停止標識', 'teishi hyoushiki', ''],
      ko: ['정지 표지판', 'jeongji pyojipan', ''], zh: ['停车标志', 'tíng chē biāo zhì', ''],
      hi: ['रुकने का चिह्न', 'rukne kaa chihn', 'm'], ar: ['علامة توقف', 'alaamat tawaqquf', 'f']
    }
  },
  'parking meter': {
    em: '🅿️', cat: 'street', lvl: 3, size: 'medium',
    t: {
      es: ['parquímetro', 'par-KEE-meh-tro', 'm'], fr: ['parcmètre', 'park-METR', 'm'],
      de: ['Parkuhr', 'PARK-oor', 'f'], it: ['parchimetro', 'par-KEE-meh-tro', 'm'],
      pt: ['parquímetro', 'par-KEE-meh-troo', 'm'], nl: ['parkeermeter', 'par-KEER-may-ter', 'c'],
      ru: ['паркомат', 'par-ka-MAT', 'm'], ja: ['パーキングメーター', 'paakingu meetaa', ''],
      ko: ['주차 미터기', 'jucha miteogi', ''], zh: ['停车计时器', 'tíng chē jì shí qì', ''],
      hi: ['पार्किंग मीटर', 'paarking meetar', 'm'], ar: ['عداد وقوف', 'addaad wuqoof', 'm']
    }
  },
  bench: {
    em: '🪑', cat: 'furniture', lvl: 2, size: 'large',
    t: {
      es: ['banco', 'BAN-ko', 'm'], fr: ['banc', 'bahn', 'm'],
      de: ['Bank', 'bank', 'f'], it: ['panchina', 'pan-KEE-na', 'f'],
      pt: ['banco', 'BAN-koo', 'm'], nl: ['bank', 'bahnk', 'c'],
      ru: ['скамейка', 'ska-MYAY-ka', 'f'], ja: ['ベンチ', 'benchi', ''],
      ko: ['벤치', 'benchi', ''], zh: ['长椅', 'cháng yǐ', ''],
      hi: ['बेंच', 'bench', 'f'], ar: ['مقعد', 'maqad', 'm']
    }
  },
  bird: {
    em: '🐦', cat: 'animal', lvl: 1, size: 'small',
    t: {
      es: ['pájaro', 'PA-ha-ro', 'm'], fr: ['oiseau', 'wa-ZOH', 'm'],
      de: ['Vogel', 'FOH-gel', 'm'], it: ['uccello', 'oo-CHEL-lo', 'm'],
      pt: ['pássaro', 'PA-sa-roo', 'm'], nl: ['vogel', 'FOH-khel', 'c'],
      ru: ['птица', 'PTEE-tsa', 'f'], ja: ['鳥', 'tori', ''],
      ko: ['새', 'sae', ''], zh: ['鸟', 'niǎo', ''],
      hi: ['पक्षी', 'pakshee', 'm'], ar: ['طائر', 'taa-ir', 'm']
    }
  },
  cat: {
    em: '🐱', cat: 'animal', lvl: 1, size: 'medium',
    t: {
      es: ['gato', 'GA-to', 'm'], fr: ['chat', 'shah', 'm'],
      de: ['Katze', 'KAT-tseh', 'f'], it: ['gatto', 'GAT-to', 'm'],
      pt: ['gato', 'GA-too', 'm'], nl: ['kat', 'kaht', 'c'],
      ru: ['кошка', 'KOSH-ka', 'f'], ja: ['猫', 'neko', ''],
      ko: ['고양이', 'goyangi', ''], zh: ['猫', 'māo', ''],
      hi: ['बिल्ली', 'billee', 'f'], ar: ['قطة', 'qitta', 'f']
    }
  },
  dog: {
    em: '🐕', cat: 'animal', lvl: 1, size: 'medium',
    t: {
      es: ['perro', 'PEH-rro', 'm'], fr: ['chien', 'shyan', 'm'],
      de: ['Hund', 'hoont', 'm'], it: ['cane', 'KAH-neh', 'm'],
      pt: ['cachorro', 'ka-SHO-hoo', 'm'], nl: ['hond', 'hont', 'c'],
      ru: ['собака', 'sa-BA-ka', 'f'], ja: ['犬', 'inu', ''],
      ko: ['개', 'gae', ''], zh: ['狗', 'gǒu', ''],
      hi: ['कुत्ता', 'kuttaa', 'm'], ar: ['كلب', 'kalb', 'm']
    }
  },
  horse: {
    em: '🐴', cat: 'animal', lvl: 1, size: 'large',
    t: {
      es: ['caballo', 'ka-BA-yo', 'm'], fr: ['cheval', 'shuh-VAL', 'm'],
      de: ['Pferd', 'pfairt', 'n'], it: ['cavallo', 'ka-VAL-lo', 'm'],
      pt: ['cavalo', 'ka-VA-loo', 'm'], nl: ['paard', 'paart', 'n'],
      ru: ['лошадь', 'LO-shat', 'f'], ja: ['馬', 'uma', ''],
      ko: ['말', 'mal', ''], zh: ['马', 'mǎ', ''],
      hi: ['घोड़ा', 'ghodaa', 'm'], ar: ['حصان', 'hisaan', 'm']
    }
  },
  sheep: {
    em: '🐑', cat: 'animal', lvl: 2, size: 'medium',
    t: {
      es: ['oveja', 'o-VEH-ha', 'f'], fr: ['mouton', 'moo-TOHN', 'm'],
      de: ['Schaf', 'shaaf', 'n'], it: ['pecora', 'PEH-ko-ra', 'f'],
      pt: ['ovelha', 'o-VEH-lya', 'f'], nl: ['schaap', 'skhaap', 'n'],
      ru: ['овца', 'af-TSA', 'f'], ja: ['羊', 'hitsuji', ''],
      ko: ['양', 'yang', ''], zh: ['羊', 'yáng', ''],
      hi: ['भेड़', 'bhed', 'f'], ar: ['خروف', 'kharoof', 'm']
    }
  },
  cow: {
    em: '🐄', cat: 'animal', lvl: 1, size: 'large',
    t: {
      es: ['vaca', 'VA-ka', 'f'], fr: ['vache', 'vahsh', 'f'],
      de: ['Kuh', 'koo', 'f'], it: ['mucca', 'MOOK-ka', 'f'],
      pt: ['vaca', 'VA-ka', 'f'], nl: ['koe', 'koo', 'c'],
      ru: ['корова', 'ka-RO-va', 'f'], ja: ['牛', 'ushi', ''],
      ko: ['소', 'so', ''], zh: ['牛', 'niú', ''],
      hi: ['गाय', 'gaay', 'f'], ar: ['بقرة', 'baqara', 'f']
    }
  },
  elephant: {
    em: '🐘', cat: 'animal', lvl: 2, size: 'huge',
    t: {
      es: ['elefante', 'eh-leh-FAN-teh', 'm'], fr: ['éléphant', 'ay-lay-FAHN', 'm'],
      de: ['Elefant', 'eh-leh-FANT', 'm'], it: ['elefante', 'eh-leh-FAN-teh', 'm'],
      pt: ['elefante', 'eh-leh-FAN-chee', 'm'], nl: ['olifant', 'OH-lee-fant', 'c'],
      ru: ['слон', 'slon', 'm'], ja: ['象', 'zou', ''],
      ko: ['코끼리', 'kokkiri', ''], zh: ['大象', 'dà xiàng', ''],
      hi: ['हाथी', 'haathee', 'm'], ar: ['فيل', 'feel', 'm']
    }
  },
  bear: {
    em: '🐻', cat: 'animal', lvl: 2, size: 'large',
    t: {
      es: ['oso', 'O-so', 'm'], fr: ['ours', 'oorss', 'm'],
      de: ['Bär', 'bair', 'm'], it: ['orso', 'OR-so', 'm'],
      pt: ['urso', 'OOR-soo', 'm'], nl: ['beer', 'bair', 'c'],
      ru: ['медведь', 'myed-VYET', 'm'], ja: ['熊', 'kuma', ''],
      ko: ['곰', 'gom', ''], zh: ['熊', 'xióng', ''],
      hi: ['भालू', 'bhaaloo', 'm'], ar: ['دب', 'dubb', 'm']
    }
  },
  zebra: {
    em: '🦓', cat: 'animal', lvl: 2, size: 'large',
    t: {
      es: ['cebra', 'SEH-bra', 'f'], fr: ['zèbre', 'ZEHBR', 'm'],
      de: ['Zebra', 'TSEH-bra', 'n'], it: ['zebra', 'DZEH-bra', 'f'],
      pt: ['zebra', 'ZEH-bra', 'f'], nl: ['zebra', 'ZEH-bra', 'c'],
      ru: ['зебра', 'ZYEB-ra', 'f'], ja: ['シマウマ', 'shimauma', ''],
      ko: ['얼룩말', 'eollungmal', ''], zh: ['斑马', 'bān mǎ', ''],
      hi: ['ज़ेबरा', 'zebraa', 'm'], ar: ['حمار وحشي', 'himaar wahshee', 'm']
    }
  },
  giraffe: {
    em: '🦒', cat: 'animal', lvl: 2, size: 'huge',
    t: {
      es: ['jirafa', 'hee-RA-fa', 'f'], fr: ['girafe', 'zhee-RAF', 'f'],
      de: ['Giraffe', 'gee-RAF-feh', 'f'], it: ['giraffa', 'jee-RAF-fa', 'f'],
      pt: ['girafa', 'zhee-RA-fa', 'f'], nl: ['giraf', 'zhee-RAF', 'c'],
      ru: ['жираф', 'zhee-RAF', 'm'], ja: ['キリン', 'kirin', ''],
      ko: ['기린', 'girin', ''], zh: ['长颈鹿', 'cháng jǐng lù', ''],
      hi: ['जिराफ़', 'jiraaf', 'm'], ar: ['زرافة', 'zaraafa', 'f']
    }
  },
  backpack: {
    em: '🎒', cat: 'accessory', lvl: 1, size: 'medium',
    t: {
      es: ['mochila', 'mo-CHEE-la', 'f'], fr: ['sac à dos', 'sak a DOH', 'm'],
      de: ['Rucksack', 'ROOK-zak', 'm'], it: ['zaino', 'DZAI-no', 'm'],
      pt: ['mochila', 'mo-SHEE-la', 'f'], nl: ['rugzak', 'RUKH-zak', 'c'],
      ru: ['рюкзак', 'ryook-ZAK', 'm'], ja: ['リュック', 'ryukku', ''],
      ko: ['배낭', 'baenang', ''], zh: ['背包', 'bèi bāo', ''],
      hi: ['बैग', 'baig', 'm'], ar: ['حقيبة ظهر', 'haqeebat zahr', 'f']
    }
  },
  umbrella: {
    em: '☂️', cat: 'accessory', lvl: 1, size: 'medium',
    t: {
      es: ['paraguas', 'pa-RA-gwas', 'm'], fr: ['parapluie', 'pa-ra-PLWEE', 'm'],
      de: ['Regenschirm', 'RAY-gen-sheerm', 'm'], it: ['ombrello', 'om-BREL-lo', 'm'],
      pt: ['guarda-chuva', 'GWAR-da SHOO-va', 'm'], nl: ['paraplu', 'pa-ra-PLOO', 'c'],
      ru: ['зонт', 'zont', 'm'], ja: ['傘', 'kasa', ''],
      ko: ['우산', 'usan', ''], zh: ['雨伞', 'yǔ sǎn', ''],
      hi: ['छाता', 'chhaataa', 'm'], ar: ['مظلة', 'mizalla', 'f']
    }
  },
  handbag: {
    em: '👜', cat: 'accessory', lvl: 2, size: 'small',
    t: {
      es: ['bolso', 'BOL-so', 'm'], fr: ['sac à main', 'sak a MAN', 'm'],
      de: ['Handtasche', 'HANT-ta-sheh', 'f'], it: ['borsa', 'BOR-sa', 'f'],
      pt: ['bolsa', 'BOL-sa', 'f'], nl: ['handtas', 'HANT-tas', 'c'],
      ru: ['сумка', 'SOOM-ka', 'f'], ja: ['ハンドバッグ', 'handobaggu', ''],
      ko: ['핸드백', 'haendeubaek', ''], zh: ['手提包', 'shǒu tí bāo', ''],
      hi: ['हैंडबैग', 'haindbaig', 'm'], ar: ['حقيبة يد', 'haqeebat yad', 'f']
    }
  },
  tie: {
    em: '👔', cat: 'accessory', lvl: 2, size: 'small',
    t: {
      es: ['corbata', 'kor-BA-ta', 'f'], fr: ['cravate', 'kra-VAT', 'f'],
      de: ['Krawatte', 'kra-VAT-teh', 'f'], it: ['cravatta', 'kra-VAT-ta', 'f'],
      pt: ['gravata', 'gra-VA-ta', 'f'], nl: ['stropdas', 'STROP-das', 'c'],
      ru: ['галстук', 'GAL-stook', 'm'], ja: ['ネクタイ', 'nekutai', ''],
      ko: ['넥타이', 'nektai', ''], zh: ['领带', 'lǐng dài', ''],
      hi: ['टाई', 'taai', 'f'], ar: ['ربطة عنق', 'rabtat unuq', 'f']
    }
  },
  suitcase: {
    em: '🧳', cat: 'accessory', lvl: 2, size: 'medium',
    t: {
      es: ['maleta', 'ma-LEH-ta', 'f'], fr: ['valise', 'va-LEEZ', 'f'],
      de: ['Koffer', 'KOF-fer', 'm'], it: ['valigia', 'va-LEE-ja', 'f'],
      pt: ['mala', 'MA-la', 'f'], nl: ['koffer', 'KOF-fer', 'c'],
      ru: ['чемодан', 'chye-ma-DAN', 'm'], ja: ['スーツケース', 'suutsukeesu', ''],
      ko: ['여행 가방', 'yeohaeng gabang', ''], zh: ['行李箱', 'xíng lǐ xiāng', ''],
      hi: ['सूटकेस', 'sootkes', 'm'], ar: ['حقيبة سفر', 'haqeebat safar', 'f']
    }
  },
  frisbee: {
    em: '🥏', cat: 'sport', lvl: 3, size: 'small',
    t: {
      es: ['frisbee', 'FREES-bee', 'm'], fr: ['frisbee', 'freez-BEE', 'm'],
      de: ['Frisbee', 'FRIS-bee', 'n'], it: ['frisbee', 'FREEZ-bee', 'm'],
      pt: ['frisbee', 'FREEZ-bee', 'm'], nl: ['frisbee', 'FRIS-bee', 'c'],
      ru: ['фрисби', 'FREES-bee', 'm'], ja: ['フリスビー', 'furisubii', ''],
      ko: ['프리스비', 'peuriseubi', ''], zh: ['飞盘', 'fēi pán', ''],
      hi: ['फ्रिसबी', 'frisbee', 'm'], ar: ['قرص طائر', 'qurs taa-ir', 'm']
    }
  },
  skis: {
    em: '🎿', cat: 'sport', lvl: 3, size: 'large',
    t: {
      es: ['esquís', 'es-KEES', 'p'], fr: ['skis', 'skee', 'p'],
      de: ['Ski', 'shee', 'm'], it: ['sci', 'shee', 'p'],
      pt: ['esquis', 'es-KEES', 'p'], nl: ["ski's", 'skees', 'p'],
      ru: ['лыжи', 'LI-zhee', 'p'], ja: ['スキー', 'sukii', ''],
      ko: ['스키', 'seuki', ''], zh: ['滑雪板', 'huá xuě bǎn', ''],
      hi: ['स्की', 'skee', 'f'], ar: ['زلاجات', 'zallaajaat', 'p']
    }
  },
  snowboard: {
    em: '🏂', cat: 'sport', lvl: 3, size: 'large',
    t: {
      es: ['tabla de snowboard', 'TA-bla deh SNOH-bord', 'f'], fr: ['planche à neige', 'plansh a NEHZH', 'f'],
      de: ['Snowboard', 'SNOH-bord', 'n'], it: ['snowboard', 'SNOH-bord', 'm'],
      pt: ['prancha de snowboard', 'PRAN-sha jee SNOH-bord', 'f'], nl: ['snowboard', 'SNOH-bort', 'n'],
      ru: ['сноуборд', 'sna-oo-BORD', 'm'], ja: ['スノーボード', 'sunooboodo', ''],
      ko: ['스노보드', 'seunobodeu', ''], zh: ['单板滑雪板', 'dān bǎn huá xuě bǎn', ''],
      hi: ['स्नोबोर्ड', 'snobord', 'm'], ar: ['لوح تزلج', 'lawh tazalluj', 'm']
    }
  },
  'sports ball': {
    em: '⚽', cat: 'sport', lvl: 1, size: 'small',
    t: {
      es: ['pelota', 'peh-LO-ta', 'f'], fr: ['ballon', 'ba-LOHN', 'm'],
      de: ['Ball', 'bal', 'm'], it: ['palla', 'PAL-la', 'f'],
      pt: ['bola', 'BO-la', 'f'], nl: ['bal', 'bahl', 'c'],
      ru: ['мяч', 'myach', 'm'], ja: ['ボール', 'booru', ''],
      ko: ['공', 'gong', ''], zh: ['球', 'qiú', ''],
      hi: ['गेंद', 'gend', 'f'], ar: ['كرة', 'kura', 'f']
    }
  },
  kite: {
    em: '🪁', cat: 'sport', lvl: 3, size: 'medium',
    t: {
      es: ['cometa', 'ko-MEH-ta', 'f'], fr: ['cerf-volant', 'sehr-vo-LAHN', 'm'],
      de: ['Drachen', 'DRA-khen', 'm'], it: ['aquilone', 'a-kwee-LO-neh', 'm'],
      pt: ['pipa', 'PEE-pa', 'f'], nl: ['vlieger', 'FLEE-kher', 'c'],
      ru: ['воздушный змей', 'vaz-DOOSH-nee zmyay', 'm'], ja: ['凧', 'tako', ''],
      ko: ['연', 'yeon', ''], zh: ['风筝', 'fēng zheng', ''],
      hi: ['पतंग', 'patang', 'f'], ar: ['طائرة ورقية', 'taa-ira waraqiyya', 'f']
    }
  },
  'baseball bat': {
    em: '🏏', cat: 'sport', lvl: 3, size: 'medium',
    t: {
      es: ['bate de béisbol', 'BA-teh deh BAYS-bol', 'm'], fr: ['batte de baseball', 'bat duh bays-BOL', 'f'],
      de: ['Baseballschläger', 'BAYS-bal-shlay-ger', 'm'], it: ['mazza da baseball', 'MAT-tsa da bays-BOL', 'f'],
      pt: ['taco de beisebol', 'TA-koo jee bay-zee-BOL', 'm'], nl: ['honkbalknuppel', 'HONK-bal-knup-pel', 'c'],
      ru: ['бейсбольная бита', 'bays-BOL-na-ya BEE-ta', 'f'], ja: ['バット', 'batto', ''],
      ko: ['야구 방망이', 'yagu bangmangi', ''], zh: ['棒球棒', 'bàng qiú bàng', ''],
      hi: ['बल्ला', 'ballaa', 'm'], ar: ['مضرب بيسبول', 'midrab baysbool', 'm']
    }
  },
  'baseball glove': {
    em: '🧤', cat: 'sport', lvl: 3, size: 'small',
    t: {
      es: ['guante de béisbol', 'GWAN-teh deh BAYS-bol', 'm'], fr: ['gant de baseball', 'gahn duh bays-BOL', 'm'],
      de: ['Baseballhandschuh', 'BAYS-bal-hant-shoo', 'm'], it: ['guantone', 'gwan-TO-neh', 'm'],
      pt: ['luva de beisebol', 'LOO-va jee bay-zee-BOL', 'f'], nl: ['honkbalhandschoen', 'HONK-bal-hant-skhoon', 'c'],
      ru: ['бейсбольная перчатка', 'bays-BOL-na-ya per-CHAT-ka', 'f'], ja: ['グローブ', 'guroobu', ''],
      ko: ['야구 글러브', 'yagu geulleobeu', ''], zh: ['棒球手套', 'bàng qiú shǒu tào', ''],
      hi: ['दस्ताना', 'dastaanaa', 'm'], ar: ['قفاز بيسبول', 'qiffaaz baysbool', 'm']
    }
  },
  skateboard: {
    em: '🛹', cat: 'sport', lvl: 2, size: 'medium',
    t: {
      es: ['monopatín', 'mo-no-pa-TEEN', 'm'], fr: ['skateboard', 'skayt-BORD', 'm'],
      de: ['Skateboard', 'SKAYT-bord', 'n'], it: ['skateboard', 'skayt-BORD', 'm'],
      pt: ['skate', 'ees-KAY-chee', 'm'], nl: ['skateboard', 'SKAYT-bort', 'n'],
      ru: ['скейтборд', 'skayt-BORD', 'm'], ja: ['スケートボード', 'sukeetoboodo', ''],
      ko: ['스케이트보드', 'seukeiteubodeu', ''], zh: ['滑板', 'huá bǎn', ''],
      hi: ['स्केटबोर्ड', 'sketbord', 'm'], ar: ['لوح تزلج', 'lawh tazalluj', 'm']
    }
  },
  surfboard: {
    em: '🏄', cat: 'sport', lvl: 3, size: 'large',
    t: {
      es: ['tabla de surf', 'TA-bla deh soorf', 'f'], fr: ['planche de surf', 'plansh duh soorf', 'f'],
      de: ['Surfbrett', 'SURF-bret', 'n'], it: ['tavola da surf', 'TA-vo-la da soorf', 'f'],
      pt: ['prancha de surf', 'PRAN-sha jee soorf', 'f'], nl: ['surfplank', 'SURF-plank', 'c'],
      ru: ['доска для сёрфинга', 'das-KA dlya SYOR-feen-ga', 'f'], ja: ['サーフボード', 'saafuboodo', ''],
      ko: ['서프보드', 'seopeubodeu', ''], zh: ['冲浪板', 'chōng làng bǎn', ''],
      hi: ['सर्फ़बोर्ड', 'sarfbord', 'm'], ar: ['لوح ركمجة', 'lawh rakmaja', 'm']
    }
  },
  'tennis racket': {
    em: '🎾', cat: 'sport', lvl: 2, size: 'medium',
    t: {
      es: ['raqueta', 'ra-KEH-ta', 'f'], fr: ['raquette', 'ra-KET', 'f'],
      de: ['Tennisschläger', 'TEN-nis-shlay-ger', 'm'], it: ['racchetta', 'rak-KET-ta', 'f'],
      pt: ['raquete', 'ha-KEH-chee', 'f'], nl: ['tennisracket', 'TEN-nis-ra-ket', 'n'],
      ru: ['ракетка', 'ra-KYET-ka', 'f'], ja: ['ラケット', 'raketto', ''],
      ko: ['테니스 라켓', 'teniseu raket', ''], zh: ['网球拍', 'wǎng qiú pāi', ''],
      hi: ['रैकेट', 'raiket', 'm'], ar: ['مضرب تنس', 'midrab tinis', 'm']
    }
  },
  bottle: {
    em: '🍾', cat: 'kitchen', lvl: 1, size: 'small',
    t: {
      es: ['botella', 'bo-TEH-ya', 'f'], fr: ['bouteille', 'boo-TAY', 'f'],
      de: ['Flasche', 'FLA-sheh', 'f'], it: ['bottiglia', 'bot-TEE-lya', 'f'],
      pt: ['garrafa', 'ga-HA-fa', 'f'], nl: ['fles', 'fless', 'c'],
      ru: ['бутылка', 'boo-TIL-ka', 'f'], ja: ['瓶', 'bin', ''],
      ko: ['병', 'byeong', ''], zh: ['瓶子', 'píng zi', ''],
      hi: ['बोतल', 'botal', 'f'], ar: ['زجاجة', 'zujaaja', 'f']
    }
  },
  'wine glass': {
    em: '🍷', cat: 'kitchen', lvl: 2, size: 'small',
    t: {
      es: ['copa de vino', 'KO-pa deh VEE-no', 'f'], fr: ['verre à vin', 'vehr a VAN', 'm'],
      de: ['Weinglas', 'VINE-glas', 'n'], it: ['bicchiere da vino', 'beek-KYEH-reh da VEE-no', 'm'],
      pt: ['taça de vinho', 'TA-sa jee VEE-nyoo', 'f'], nl: ['wijnglas', 'VINE-khlas', 'n'],
      ru: ['бокал', 'ba-KAL', 'm'], ja: ['ワイングラス', 'wain gurasu', ''],
      ko: ['와인 잔', 'wain jan', ''], zh: ['酒杯', 'jiǔ bēi', ''],
      hi: ['वाइन ग्लास', 'vaain glaas', 'm'], ar: ['كأس نبيذ', 'kas nabeedh', 'm']
    }
  },
  cup: {
    em: '☕', cat: 'kitchen', lvl: 1, size: 'small',
    t: {
      es: ['taza', 'TA-sa', 'f'], fr: ['tasse', 'tahss', 'f'],
      de: ['Tasse', 'TAS-seh', 'f'], it: ['tazza', 'TAT-tsa', 'f'],
      pt: ['xícara', 'SHEE-ka-ra', 'f'], nl: ['kopje', 'KOP-yeh', 'n'],
      ru: ['чашка', 'CHASH-ka', 'f'], ja: ['カップ', 'kappu', ''],
      ko: ['컵', 'keop', ''], zh: ['杯子', 'bēi zi', ''],
      hi: ['कप', 'kap', 'm'], ar: ['فنجان', 'finjaan', 'm']
    }
  },
  fork: {
    em: '🍴', cat: 'kitchen', lvl: 1, size: 'tiny',
    t: {
      es: ['tenedor', 'teh-neh-DOR', 'm'], fr: ['fourchette', 'foor-SHET', 'f'],
      de: ['Gabel', 'GAA-bel', 'f'], it: ['forchetta', 'for-KET-ta', 'f'],
      pt: ['garfo', 'GAR-foo', 'm'], nl: ['vork', 'fork', 'c'],
      ru: ['вилка', 'VEEL-ka', 'f'], ja: ['フォーク', 'fooku', ''],
      ko: ['포크', 'pokeu', ''], zh: ['叉子', 'chā zi', ''],
      hi: ['काँटा', 'kaantaa', 'm'], ar: ['شوكة', 'shawka', 'f']
    }
  },
  knife: {
    em: '🔪', cat: 'kitchen', lvl: 1, size: 'tiny',
    t: {
      es: ['cuchillo', 'koo-CHEE-yo', 'm'], fr: ['couteau', 'koo-TOH', 'm'],
      de: ['Messer', 'MES-ser', 'n'], it: ['coltello', 'kol-TEL-lo', 'm'],
      pt: ['faca', 'FA-ka', 'f'], nl: ['mes', 'mess', 'n'],
      ru: ['нож', 'nosh', 'm'], ja: ['ナイフ', 'naifu', ''],
      ko: ['칼', 'kal', ''], zh: ['刀', 'dāo', ''],
      hi: ['चाकू', 'chaakoo', 'm'], ar: ['سكين', 'sikkeen', 'f']
    }
  },
  spoon: {
    em: '🥄', cat: 'kitchen', lvl: 1, size: 'tiny',
    t: {
      es: ['cuchara', 'koo-CHA-ra', 'f'], fr: ['cuillère', 'kwee-YEHR', 'f'],
      de: ['Löffel', 'LUR-fel', 'm'], it: ['cucchiaio', 'kook-KYAI-o', 'm'],
      pt: ['colher', 'ko-LYEHR', 'f'], nl: ['lepel', 'LAY-pel', 'c'],
      ru: ['ложка', 'LOSH-ka', 'f'], ja: ['スプーン', 'supuun', ''],
      ko: ['숟가락', 'sutgarak', ''], zh: ['勺子', 'sháo zi', ''],
      hi: ['चम्मच', 'chammach', 'm'], ar: ['ملعقة', 'milaqa', 'f']
    }
  },
  bowl: {
    em: '🥣', cat: 'kitchen', lvl: 1, size: 'small',
    t: {
      es: ['cuenco', 'KWEN-ko', 'm'], fr: ['bol', 'bohl', 'm'],
      de: ['Schüssel', 'SHUS-sel', 'f'], it: ['ciotola', 'CHO-to-la', 'f'],
      pt: ['tigela', 'chee-ZHEH-la', 'f'], nl: ['kom', 'kom', 'c'],
      ru: ['миска', 'MEES-ka', 'f'], ja: ['ボウル', 'bouru', ''],
      ko: ['그릇', 'geureut', ''], zh: ['碗', 'wǎn', ''],
      hi: ['कटोरा', 'katoraa', 'm'], ar: ['وعاء', 'wi-aa', 'm']
    }
  },
  banana: {
    em: '🍌', cat: 'food', lvl: 1, size: 'small',
    t: {
      es: ['plátano', 'PLA-ta-no', 'm'], fr: ['banane', 'ba-NAN', 'f'],
      de: ['Banane', 'ba-NAA-neh', 'f'], it: ['banana', 'ba-NA-na', 'f'],
      pt: ['banana', 'ba-NA-na', 'f'], nl: ['banaan', 'ba-NAAN', 'c'],
      ru: ['банан', 'ba-NAN', 'm'], ja: ['バナナ', 'banana', ''],
      ko: ['바나나', 'banana', ''], zh: ['香蕉', 'xiāng jiāo', ''],
      hi: ['केला', 'kelaa', 'm'], ar: ['موزة', 'mawza', 'f']
    }
  },
  apple: {
    em: '🍎', cat: 'food', lvl: 1, size: 'small',
    t: {
      es: ['manzana', 'man-SA-na', 'f'], fr: ['pomme', 'pom', 'f'],
      de: ['Apfel', 'AP-fel', 'm'], it: ['mela', 'MEH-la', 'f'],
      pt: ['maçã', 'ma-SANG', 'f'], nl: ['appel', 'AP-pel', 'c'],
      ru: ['яблоко', 'YAB-la-ka', 'n'], ja: ['りんご', 'ringo', ''],
      ko: ['사과', 'sagwa', ''], zh: ['苹果', 'píng guǒ', ''],
      hi: ['सेब', 'seb', 'm'], ar: ['تفاحة', 'tuffaaha', 'f']
    }
  },
  sandwich: {
    em: '🥪', cat: 'food', lvl: 1, size: 'small',
    t: {
      es: ['sándwich', 'SAND-weech', 'm'], fr: ['sandwich', 'sand-WEECH', 'm'],
      de: ['Sandwich', 'SEND-vitch', 'n'], it: ['panino', 'pa-NEE-no', 'm'],
      pt: ['sanduíche', 'san-DWEE-shee', 'm'], nl: ['broodje', 'BROHT-yeh', 'n'],
      ru: ['бутерброд', 'boo-ter-BROT', 'm'], ja: ['サンドイッチ', 'sandoitchi', ''],
      ko: ['샌드위치', 'saendeuwichi', ''], zh: ['三明治', 'sān míng zhì', ''],
      hi: ['सैंडविच', 'saindvich', 'm'], ar: ['شطيرة', 'shateera', 'f']
    }
  },
  orange: {
    em: '🍊', cat: 'food', lvl: 1, size: 'small',
    t: {
      es: ['naranja', 'na-RAN-ha', 'f'], fr: ['orange', 'o-RAHNZH', 'f'],
      de: ['Orange', 'o-RAHN-zheh', 'f'], it: ['arancia', 'a-RAN-cha', 'f'],
      pt: ['laranja', 'la-RAN-zha', 'f'], nl: ['sinaasappel', 'SEE-naas-ap-pel', 'c'],
      ru: ['апельсин', 'a-pyel-SEEN', 'm'], ja: ['オレンジ', 'orenji', ''],
      ko: ['오렌지', 'orenji', ''], zh: ['橙子', 'chéng zi', ''],
      hi: ['संतरा', 'santaraa', 'm'], ar: ['برتقالة', 'burtuqaala', 'f']
    }
  },
  broccoli: {
    em: '🥦', cat: 'food', lvl: 2, size: 'small',
    t: {
      es: ['brócoli', 'BRO-ko-lee', 'm'], fr: ['brocoli', 'bro-ko-LEE', 'm'],
      de: ['Brokkoli', 'BROK-ko-lee', 'm'], it: ['broccolo', 'BROK-ko-lo', 'm'],
      pt: ['brócolis', 'BRO-ko-lees', 'm'], nl: ['broccoli', 'BRO-ko-lee', 'c'],
      ru: ['брокколи', 'BROK-ka-lee', 'f'], ja: ['ブロッコリー', 'burokkorii', ''],
      ko: ['브로콜리', 'beurokolli', ''], zh: ['西兰花', 'xī lán huā', ''],
      hi: ['ब्रोकली', 'brokalee', 'f'], ar: ['بروكلي', 'brookolee', 'm']
    }
  },
  carrot: {
    em: '🥕', cat: 'food', lvl: 1, size: 'small',
    t: {
      es: ['zanahoria', 'sa-na-O-ree-a', 'f'], fr: ['carotte', 'ka-ROT', 'f'],
      de: ['Karotte', 'ka-ROT-teh', 'f'], it: ['carota', 'ka-RO-ta', 'f'],
      pt: ['cenoura', 'seh-NOH-ra', 'f'], nl: ['wortel', 'VOR-tel', 'c'],
      ru: ['морковь', 'mar-KOF', 'f'], ja: ['にんじん', 'ninjin', ''],
      ko: ['당근', 'danggeun', ''], zh: ['胡萝卜', 'hú luó bo', ''],
      hi: ['गाजर', 'gaajar', 'f'], ar: ['جزرة', 'jazara', 'f']
    }
  },
  'hot dog': {
    em: '🌭', cat: 'food', lvl: 2, size: 'small',
    t: {
      es: ['perrito caliente', 'peh-RREE-to ka-LYEN-teh', 'm'], fr: ['hot-dog', 'ot-DOG', 'm'],
      de: ['Hotdog', 'HOT-dog', 'm'], it: ['hot dog', 'ot DOG', 'm'],
      pt: ['cachorro-quente', 'ka-SHO-hoo KEN-chee', 'm'], nl: ['hotdog', 'HOT-dog', 'c'],
      ru: ['хот-дог', 'hot-DOG', 'm'], ja: ['ホットドッグ', 'hotto doggu', ''],
      ko: ['핫도그', 'hatdogeu', ''], zh: ['热狗', 'rè gǒu', ''],
      hi: ['हॉट डॉग', 'hot dog', 'm'], ar: ['هوت دوغ', 'hoot doog', 'm']
    }
  },
  pizza: {
    em: '🍕', cat: 'food', lvl: 1, size: 'medium',
    t: {
      es: ['pizza', 'PEET-sa', 'f'], fr: ['pizza', 'peed-ZA', 'f'],
      de: ['Pizza', 'PIT-tsa', 'f'], it: ['pizza', 'PEET-tsa', 'f'],
      pt: ['pizza', 'PEE-tsa', 'f'], nl: ['pizza', 'PEET-sa', 'c'],
      ru: ['пицца', 'PEET-tsa', 'f'], ja: ['ピザ', 'piza', ''],
      ko: ['피자', 'pija', ''], zh: ['披萨', 'pī sà', ''],
      hi: ['पिज़्ज़ा', 'pizzaa', 'm'], ar: ['بيتزا', 'beetzaa', 'f']
    }
  },
  donut: {
    em: '🍩', cat: 'food', lvl: 2, size: 'small',
    t: {
      es: ['rosquilla', 'ros-KEE-ya', 'f'], fr: ['beignet', 'beh-NYEH', 'm'],
      de: ['Donut', 'DOH-nat', 'm'], it: ['ciambella', 'cham-BEL-la', 'f'],
      pt: ['rosquinha', 'hos-KEE-nya', 'f'], nl: ['donut', 'DOH-nut', 'c'],
      ru: ['пончик', 'PON-cheek', 'm'], ja: ['ドーナツ', 'doonatsu', ''],
      ko: ['도넛', 'doneot', ''], zh: ['甜甜圈', 'tián tián quān', ''],
      hi: ['डोनट', 'donat', 'm'], ar: ['دونات', 'doonaat', 'f']
    }
  },
  cake: {
    em: '🎂', cat: 'food', lvl: 1, size: 'medium',
    t: {
      es: ['pastel', 'pas-TEL', 'm'], fr: ['gâteau', 'ga-TOH', 'm'],
      de: ['Kuchen', 'KOO-khen', 'm'], it: ['torta', 'TOR-ta', 'f'],
      pt: ['bolo', 'BO-loo', 'm'], nl: ['taart', 'taart', 'c'],
      ru: ['торт', 'tort', 'm'], ja: ['ケーキ', 'keeki', ''],
      ko: ['케이크', 'keikeu', ''], zh: ['蛋糕', 'dàn gāo', ''],
      hi: ['केक', 'kek', 'm'], ar: ['كعكة', 'kaka', 'f']
    }
  },
  chair: {
    em: '🪑', cat: 'furniture', lvl: 1, size: 'medium',
    t: {
      es: ['silla', 'SEE-ya', 'f'], fr: ['chaise', 'shez', 'f'],
      de: ['Stuhl', 'shtool', 'm'], it: ['sedia', 'SEH-dya', 'f'],
      pt: ['cadeira', 'ka-DAY-ra', 'f'], nl: ['stoel', 'stool', 'c'],
      ru: ['стул', 'stool', 'm'], ja: ['椅子', 'isu', ''],
      ko: ['의자', 'uija', ''], zh: ['椅子', 'yǐ zi', ''],
      hi: ['कुर्सी', 'kursee', 'f'], ar: ['كرسي', 'kursee', 'm']
    }
  },
  couch: {
    em: '🛋️', cat: 'furniture', lvl: 1, size: 'large',
    t: {
      es: ['sofá', 'so-FA', 'm'], fr: ['canapé', 'ka-na-PAY', 'm'],
      de: ['Sofa', 'ZOH-fa', 'n'], it: ['divano', 'dee-VA-no', 'm'],
      pt: ['sofá', 'so-FA', 'm'], nl: ['bank', 'bahnk', 'c'],
      ru: ['диван', 'dee-VAN', 'm'], ja: ['ソファ', 'sofa', ''],
      ko: ['소파', 'sopa', ''], zh: ['沙发', 'shā fā', ''],
      hi: ['सोफ़ा', 'sofaa', 'm'], ar: ['أريكة', 'areeka', 'f']
    }
  },
  'potted plant': {
    em: '🪴', cat: 'furniture', lvl: 2, size: 'medium',
    t: {
      es: ['planta en maceta', 'PLAN-ta en ma-SEH-ta', 'f'], fr: ['plante en pot', 'plant ahn POH', 'f'],
      de: ['Topfpflanze', 'TOPF-pflan-tseh', 'f'], it: ['pianta in vaso', 'PYAN-ta een VA-zo', 'f'],
      pt: ['planta em vaso', 'PLAN-ta eng VA-zoo', 'f'], nl: ['kamerplant', 'KAA-mer-plant', 'c'],
      ru: ['комнатное растение', 'KOM-nat-na-ye ras-TYE-nee-ye', 'n'], ja: ['鉢植え', 'hachiue', ''],
      ko: ['화분', 'hwabun', ''], zh: ['盆栽', 'pén zāi', ''],
      hi: ['गमले का पौधा', 'gamle kaa paudhaa', 'm'], ar: ['نبتة في أصيص', 'nabta fee aseeS', 'f']
    }
  },
  bed: {
    em: '🛏️', cat: 'furniture', lvl: 1, size: 'huge',
    t: {
      es: ['cama', 'KA-ma', 'f'], fr: ['lit', 'lee', 'm'],
      de: ['Bett', 'bet', 'n'], it: ['letto', 'LET-to', 'm'],
      pt: ['cama', 'KA-ma', 'f'], nl: ['bed', 'bet', 'n'],
      ru: ['кровать', 'kra-VAT', 'f'], ja: ['ベッド', 'beddo', ''],
      ko: ['침대', 'chimdae', ''], zh: ['床', 'chuáng', ''],
      hi: ['बिस्तर', 'bistar', 'm'], ar: ['سرير', 'sareer', 'm']
    }
  },
  'dining table': {
    em: '🍽️', cat: 'furniture', lvl: 1, size: 'large',
    t: {
      es: ['mesa', 'MEH-sa', 'f'], fr: ['table', 'TABL', 'f'],
      de: ['Tisch', 'tish', 'm'], it: ['tavolo', 'TA-vo-lo', 'm'],
      pt: ['mesa', 'MEH-za', 'f'], nl: ['tafel', 'TAA-fel', 'c'],
      ru: ['стол', 'stol', 'm'], ja: ['テーブル', 'teeburu', ''],
      ko: ['식탁', 'siktak', ''], zh: ['餐桌', 'cān zhuō', ''],
      hi: ['मेज़', 'mez', 'f'], ar: ['طاولة', 'taawila', 'f']
    }
  },
  toilet: {
    em: '🚽', cat: 'furniture', lvl: 2, size: 'medium',
    t: {
      es: ['inodoro', 'ee-no-DO-ro', 'm'], fr: ['toilettes', 'twa-LET', 'p'],
      de: ['Toilette', 'toy-LET-teh', 'f'], it: ['toilette', 'twa-LET', 'f'],
      pt: ['vaso sanitário', 'VA-zoo sa-nee-TA-ryoo', 'm'], nl: ['toilet', 'twa-LET', 'n'],
      ru: ['туалет', 'too-a-LYET', 'm'], ja: ['トイレ', 'toire', ''],
      ko: ['변기', 'byeongi', ''], zh: ['马桶', 'mǎ tǒng', ''],
      hi: ['शौचालय', 'shauchaalay', 'm'], ar: ['مرحاض', 'mirhaad', 'm']
    }
  },
  tv: {
    em: '📺', cat: 'electronics', lvl: 1, size: 'large',
    t: {
      es: ['televisión', 'teh-leh-vee-SYON', 'f'], fr: ['télévision', 'tay-lay-vee-ZYON', 'f'],
      de: ['Fernseher', 'FERN-zay-er', 'm'], it: ['televisione', 'teh-leh-vee-ZYO-neh', 'f'],
      pt: ['televisão', 'teh-leh-vee-ZOWNG', 'f'], nl: ['televisie', 'tay-leh-VEE-see', 'c'],
      ru: ['телевизор', 'tye-lye-VEE-zar', 'm'], ja: ['テレビ', 'terebi', ''],
      ko: ['텔레비전', 'tellebijeon', ''], zh: ['电视', 'diàn shì', ''],
      hi: ['टेलीविज़न', 'teleevizan', 'm'], ar: ['تلفاز', 'tilfaaz', 'm']
    }
  },
  laptop: {
    em: '💻', cat: 'electronics', lvl: 1, size: 'medium',
    t: {
      es: ['portátil', 'por-TA-teel', 'm'], fr: ['ordinateur portable', 'or-dee-na-TUR por-TABL', 'm'],
      de: ['Laptop', 'LEP-top', 'm'], it: ['portatile', 'por-TA-tee-leh', 'm'],
      pt: ['notebook', 'NO-chee-book', 'm'], nl: ['laptop', 'LEP-top', 'c'],
      ru: ['ноутбук', 'no-oot-BOOK', 'm'], ja: ['ノートパソコン', 'nooto pasokon', ''],
      ko: ['노트북', 'noteubuk', ''], zh: ['笔记本电脑', 'bǐ jì běn diàn nǎo', ''],
      hi: ['लैपटॉप', 'laiptop', 'm'], ar: ['حاسوب محمول', 'haasoob mahmool', 'm']
    }
  },
  mouse: {
    em: '🖱️', cat: 'electronics', lvl: 2, size: 'tiny',
    t: {
      es: ['ratón', 'ra-TON', 'm'], fr: ['souris', 'soo-REE', 'f'],
      de: ['Maus', 'mowss', 'f'], it: ['mouse', 'MOWSS', 'm'],
      pt: ['mouse', 'MOW-see', 'm'], nl: ['muis', 'mowss', 'c'],
      ru: ['мышь', 'mish', 'f'], ja: ['マウス', 'mausu', ''],
      ko: ['마우스', 'mauseu', ''], zh: ['鼠标', 'shǔ biāo', ''],
      hi: ['माउस', 'maaus', 'm'], ar: ['فأرة', 'fara', 'f']
    }
  },
  remote: {
    em: '📱', cat: 'electronics', lvl: 2, size: 'small',
    t: {
      es: ['mando', 'MAN-do', 'm'], fr: ['télécommande', 'tay-lay-ko-MAND', 'f'],
      de: ['Fernbedienung', 'FERN-beh-dee-noong', 'f'], it: ['telecomando', 'teh-leh-ko-MAN-do', 'm'],
      pt: ['controle remoto', 'kon-TRO-lee heh-MO-too', 'm'], nl: ['afstandsbediening', 'AF-stants-beh-dee-ning', 'c'],
      ru: ['пульт', 'poolt', 'm'], ja: ['リモコン', 'rimokon', ''],
      ko: ['리모컨', 'rimokeon', ''], zh: ['遥控器', 'yáo kòng qì', ''],
      hi: ['रिमोट', 'rimot', 'm'], ar: ['جهاز التحكم', 'jihaaz at-tahakkum', 'm']
    }
  },
  keyboard: {
    em: '⌨️', cat: 'electronics', lvl: 1, size: 'medium',
    t: {
      es: ['teclado', 'teh-KLA-do', 'm'], fr: ['clavier', 'kla-VYAY', 'm'],
      de: ['Tastatur', 'tas-ta-TOOR', 'f'], it: ['tastiera', 'tas-TYEH-ra', 'f'],
      pt: ['teclado', 'teh-KLA-doo', 'm'], nl: ['toetsenbord', 'TOOT-sen-bort', 'n'],
      ru: ['клавиатура', 'kla-vee-a-TOO-ra', 'f'], ja: ['キーボード', 'kiiboodo', ''],
      ko: ['키보드', 'kibodeu', ''], zh: ['键盘', 'jiàn pán', ''],
      hi: ['कीबोर्ड', 'keebord', 'm'], ar: ['لوحة مفاتيح', 'lawhat mafaateeh', 'f']
    }
  },
  'cell phone': {
    em: '📱', cat: 'electronics', lvl: 1, size: 'tiny',
    t: {
      es: ['móvil', 'MO-veel', 'm'], fr: ['téléphone', 'tay-lay-FON', 'm'],
      de: ['Handy', 'HEN-dee', 'n'], it: ['cellulare', 'chel-loo-LA-reh', 'm'],
      pt: ['celular', 'seh-loo-LAR', 'm'], nl: ['telefoon', 'tay-leh-FOAN', 'c'],
      ru: ['телефон', 'tye-lye-FON', 'm'], ja: ['携帯電話', 'keitai denwa', ''],
      ko: ['휴대폰', 'hyudaepon', ''], zh: ['手机', 'shǒu jī', ''],
      hi: ['मोबाइल', 'mobaail', 'm'], ar: ['هاتف محمول', 'haatif mahmool', 'm']
    }
  },
  microwave: {
    em: '📡', cat: 'appliance', lvl: 2, size: 'medium',
    t: {
      es: ['microondas', 'mee-kro-ON-das', 'm'], fr: ['micro-ondes', 'mee-kro-OND', 'm'],
      de: ['Mikrowelle', 'MEE-kro-vel-leh', 'f'], it: ['microonde', 'mee-kro-ON-deh', 'm'],
      pt: ['micro-ondas', 'MEE-kro ON-das', 'm'], nl: ['magnetron', 'MAKH-neh-tron', 'c'],
      ru: ['микроволновка', 'mee-kra-val-NOF-ka', 'f'], ja: ['電子レンジ', 'denshi renji', ''],
      ko: ['전자레인지', 'jeonjareinji', ''], zh: ['微波炉', 'wēi bō lú', ''],
      hi: ['माइक्रोवेव', 'maaikrovev', 'm'], ar: ['ميكروويف', 'maykrooweef', 'm']
    }
  },
  oven: {
    em: '🔥', cat: 'appliance', lvl: 1, size: 'large',
    t: {
      es: ['horno', 'OR-no', 'm'], fr: ['four', 'foor', 'm'],
      de: ['Ofen', 'OH-fen', 'm'], it: ['forno', 'FOR-no', 'm'],
      pt: ['forno', 'FOR-noo', 'm'], nl: ['oven', 'OH-ven', 'c'],
      ru: ['духовка', 'doo-HOF-ka', 'f'], ja: ['オーブン', 'oobun', ''],
      ko: ['오븐', 'obeun', ''], zh: ['烤箱', 'kǎo xiāng', ''],
      hi: ['ओवन', 'ovan', 'm'], ar: ['فرن', 'furn', 'm']
    }
  },
  toaster: {
    em: '🍞', cat: 'appliance', lvl: 2, size: 'small',
    t: {
      es: ['tostadora', 'tos-ta-DO-ra', 'f'], fr: ['grille-pain', 'gree-PAN', 'm'],
      de: ['Toaster', 'TOHS-ter', 'm'], it: ['tostapane', 'tos-ta-PA-neh', 'm'],
      pt: ['torradeira', 'to-ha-DAY-ra', 'f'], nl: ['broodrooster', 'BROAT-roh-ster', 'c'],
      ru: ['тостер', 'TOS-ter', 'm'], ja: ['トースター', 'toosutaa', ''],
      ko: ['토스터', 'toseuteo', ''], zh: ['烤面包机', 'kǎo miàn bāo jī', ''],
      hi: ['टोस्टर', 'tostar', 'm'], ar: ['محمصة', 'muhammasa', 'f']
    }
  },
  sink: {
    em: '🚰', cat: 'appliance', lvl: 2, size: 'medium',
    t: {
      es: ['fregadero', 'freh-ga-DEH-ro', 'm'], fr: ['évier', 'ay-VYAY', 'm'],
      de: ['Spüle', 'SHPUU-leh', 'f'], it: ['lavandino', 'la-van-DEE-no', 'm'],
      pt: ['pia', 'PEE-a', 'f'], nl: ['gootsteen', 'KHOAT-stayn', 'c'],
      ru: ['раковина', 'RA-ka-vee-na', 'f'], ja: ['流し', 'nagashi', ''],
      ko: ['싱크대', 'singkeudae', ''], zh: ['水槽', 'shuǐ cáo', ''],
      hi: ['सिंक', 'sink', 'm'], ar: ['حوض', 'hawd', 'm']
    }
  },
  refrigerator: {
    em: '🧊', cat: 'appliance', lvl: 1, size: 'large',
    t: {
      es: ['nevera', 'neh-VEH-ra', 'f'], fr: ['réfrigérateur', 'ray-free-zhay-ra-TUR', 'm'],
      de: ['Kühlschrank', 'KUUL-shrank', 'm'], it: ['frigorifero', 'free-go-REE-feh-ro', 'm'],
      pt: ['geladeira', 'zheh-la-DAY-ra', 'f'], nl: ['koelkast', 'KOOL-kast', 'c'],
      ru: ['холодильник', 'ha-la-DEEL-neek', 'm'], ja: ['冷蔵庫', 'reizouko', ''],
      ko: ['냉장고', 'naengjanggo', ''], zh: ['冰箱', 'bīng xiāng', ''],
      hi: ['फ़्रिज', 'frij', 'm'], ar: ['ثلاجة', 'thallaaja', 'f']
    }
  },
  book: {
    em: '📖', cat: 'object', lvl: 1, size: 'small',
    t: {
      es: ['libro', 'LEE-bro', 'm'], fr: ['livre', 'LEEVR', 'm'],
      de: ['Buch', 'bookh', 'n'], it: ['libro', 'LEE-bro', 'm'],
      pt: ['livro', 'LEE-vroo', 'm'], nl: ['boek', 'book', 'n'],
      ru: ['книга', 'KNEE-ga', 'f'], ja: ['本', 'hon', ''],
      ko: ['책', 'chaek', ''], zh: ['书', 'shū', ''],
      hi: ['किताब', 'kitaab', 'f'], ar: ['كتاب', 'kitaab', 'm']
    }
  },
  clock: {
    em: '🕐', cat: 'object', lvl: 1, size: 'small',
    t: {
      es: ['reloj', 'reh-LOH', 'm'], fr: ['horloge', 'or-LOZH', 'f'],
      de: ['Uhr', 'oor', 'f'], it: ['orologio', 'o-ro-LO-jo', 'm'],
      pt: ['relógio', 'heh-LO-zhyoo', 'm'], nl: ['klok', 'klok', 'c'],
      ru: ['часы', 'chi-SI', 'p'], ja: ['時計', 'tokei', ''],
      ko: ['시계', 'sigye', ''], zh: ['钟', 'zhōng', ''],
      hi: ['घड़ी', 'ghadee', 'f'], ar: ['ساعة', 'saa-a', 'f']
    }
  },
  vase: {
    em: '🏺', cat: 'object', lvl: 2, size: 'small',
    t: {
      es: ['jarrón', 'ha-RRON', 'm'], fr: ['vase', 'vahz', 'm'],
      de: ['Vase', 'VAA-zeh', 'f'], it: ['vaso', 'VA-zo', 'm'],
      pt: ['vaso', 'VA-zoo', 'm'], nl: ['vaas', 'vaas', 'c'],
      ru: ['ваза', 'VA-za', 'f'], ja: ['花瓶', 'kabin', ''],
      ko: ['꽃병', 'kkotbyeong', ''], zh: ['花瓶', 'huā píng', ''],
      hi: ['फूलदान', 'phooldaan', 'm'], ar: ['مزهرية', 'mizhariyya', 'f']
    }
  },
  scissors: {
    em: '✂️', cat: 'object', lvl: 2, size: 'tiny',
    t: {
      es: ['tijeras', 'tee-HEH-ras', 'p'], fr: ['ciseaux', 'see-ZOH', 'p'],
      de: ['Schere', 'SHAY-reh', 'f'], it: ['forbici', 'FOR-bee-chee', 'p'],
      pt: ['tesoura', 'teh-ZOH-ra', 'f'], nl: ['schaar', 'skhaar', 'c'],
      ru: ['ножницы', 'NOZH-nee-tsi', 'p'], ja: ['はさみ', 'hasami', ''],
      ko: ['가위', 'gawi', ''], zh: ['剪刀', 'jiǎn dāo', ''],
      hi: ['कैंची', 'kainchee', 'f'], ar: ['مقص', 'miqass', 'm']
    }
  },
  'teddy bear': {
    em: '🧸', cat: 'object', lvl: 2, size: 'small',
    t: {
      es: ['osito de peluche', 'o-SEE-to deh peh-LOO-cheh', 'm'], fr: ['ours en peluche', 'oorss ahn puh-LOOSH', 'm'],
      de: ['Teddybär', 'TED-dee-bair', 'm'], it: ['orsacchiotto', 'or-sak-KYOT-to', 'm'],
      pt: ['ursinho de pelúcia', 'oor-SEE-nyoo jee peh-LOO-sya', 'm'], nl: ['teddybeer', 'TED-dee-bair', 'c'],
      ru: ['плюшевый мишка', 'PLYU-she-vee MEESH-ka', 'm'], ja: ['テディベア', 'tedi bea', ''],
      ko: ['곰 인형', 'gom inhyeong', ''], zh: ['泰迪熊', 'tài dí xióng', ''],
      hi: ['टेडी बियर', 'tedee biyar', 'm'], ar: ['دمية دب', 'dumyat dubb', 'f']
    }
  },
  'hair drier': {
    em: '💇', cat: 'appliance', lvl: 3, size: 'small',
    t: {
      es: ['secador', 'seh-ka-DOR', 'm'], fr: ['sèche-cheveux', 'sesh-shuh-VUH', 'm'],
      de: ['Föhn', 'furn', 'm'], it: ['asciugacapelli', 'a-shoo-ga-ka-PEL-lee', 'm'],
      pt: ['secador', 'seh-ka-DOR', 'm'], nl: ['haardroger', 'HAAR-droh-kher', 'c'],
      ru: ['фен', 'fyen', 'm'], ja: ['ドライヤー', 'doraiyaa', ''],
      ko: ['헤어드라이어', 'heeodeuraieo', ''], zh: ['吹风机', 'chuī fēng jī', ''],
      hi: ['हेयर ड्रायर', 'heyar draayar', 'm'], ar: ['مجفف شعر', 'mujaffif shar', 'm']
    }
  },
  toothbrush: {
    em: '🪥', cat: 'object', lvl: 1, size: 'tiny',
    t: {
      es: ['cepillo de dientes', 'seh-PEE-yo deh DYEN-tes', 'm'], fr: ['brosse à dents', 'bross a DAHN', 'f'],
      de: ['Zahnbürste', 'TSAAN-buurs-teh', 'f'], it: ['spazzolino', 'spat-tso-LEE-no', 'm'],
      pt: ['escova de dentes', 'es-KO-va jee DEN-chees', 'f'], nl: ['tandenborstel', 'TAN-den-bor-stel', 'c'],
      ru: ['зубная щётка', 'zoob-NA-ya SHOT-ka', 'f'], ja: ['歯ブラシ', 'haburashi', ''],
      ko: ['칫솔', 'chitsol', ''], zh: ['牙刷', 'yá shuā', ''],
      hi: ['टूथब्रश', 'toothbrash', 'm'], ar: ['فرشاة أسنان', 'furshaat asnaan', 'f']
    }
  }
};

/** Expected on-screen area as a fraction of the frame, per size class. */
export const SIZE_RANGES = {
  tiny: { min: 0.0005, max: 0.04 },
  small: { min: 0.002, max: 0.12 },
  medium: { min: 0.008, max: 0.35 },
  large: { min: 0.02, max: 0.60 },
  huge: { min: 0.05, max: 1.0 }
};

export const ALL_CLASSES = Object.keys(DICT);

/** Look up one translation. Returns null when the pair is missing. */
export function translate(cls, lang) {
  const entry = DICT[cls];
  if (!entry) return null;
  const t = entry.t[lang];
  if (!t) return null;
  return { word: t[0], phonetic: t[1], gender: t[2], em: entry.em, cat: entry.cat, lvl: entry.lvl };
}

/** All classes in a category. */
export function classesInCategory(cat) {
  return ALL_CLASSES.filter((c) => DICT[c].cat === cat);
}
