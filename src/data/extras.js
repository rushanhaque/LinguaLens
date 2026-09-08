/**
 * extras.js — Vocabulary beyond the 80 COCO-SSD classes.
 *
 * COCO's label set is tiny and skewed towards street scenes and pets: of its
 * eighty classes, barely twenty are things you can point at indoors, and it has
 * no word for a pen. That made the app awkward in exactly the setting it is
 * most useful in — a room, a desk, a classroom.
 *
 * These entries are reachable through the MobileNet image classifier, which
 * recognises the full ImageNet label set. See imagenet.js for the mapping from
 * ImageNet's own (verbose, synonym-laden) labels onto these keys.
 *
 * The entry shape is identical to dictionary.js, and both tables are merged
 * into a single DICT so nothing downstream needs to know an object's origin.
 */

export const EXTRA_CATEGORIES = {
  school: { label: 'School', em: '🎓', color: '#6B7F9E' },
  clothing: { label: 'Clothing', em: '👕', color: '#9C7189' },
  nature: { label: 'Nature', em: '🌿', color: '#6E8B5A' },
  place: { label: 'Places', em: '🏛️', color: '#8B7B65' },
  music: { label: 'Music', em: '🎵', color: '#8A6A8E' },
  tool: { label: 'Tools', em: '🔧', color: '#77706A' }
};

export const EXTRA_DICT = {

  /* ── School and desk ────────────────────────────────────────────────── */

  pen: {
    em: '🖊️', cat: 'school', lvl: 1, size: 'tiny',
    t: {
      es: ['bolígrafo', 'bo-LEE-gra-fo', 'm'], fr: ['stylo', 'stee-LOH', 'm'],
      de: ['Kugelschreiber', 'KOO-gel-shry-ber', 'm'], it: ['penna', 'PEN-na', 'f'],
      pt: ['caneta', 'ka-NEH-ta', 'f'], nl: ['pen', 'pen', 'c'],
      ru: ['ручка', 'ROOCH-ka', 'f'], ja: ['ペン', 'pen', ''],
      ko: ['펜', 'pen', ''], zh: ['笔', 'bǐ', ''],
      hi: ['कलम', 'ka-lam', 'f'], ar: ['قلم', 'qalam', 'm']
    }
  },
  pencil: {
    em: '✏️', cat: 'school', lvl: 1, size: 'tiny',
    t: {
      es: ['lápiz', 'LA-pees', 'm'], fr: ['crayon', 'kray-OHN', 'm'],
      de: ['Bleistift', 'BLY-shtift', 'm'], it: ['matita', 'ma-TEE-ta', 'f'],
      pt: ['lápis', 'LA-pees', 'm'], nl: ['potlood', 'POT-loat', 'n'],
      ru: ['карандаш', 'ka-ran-DASH', 'm'], ja: ['鉛筆', 'enpitsu', ''],
      ko: ['연필', 'yeonpil', ''], zh: ['铅笔', 'qiān bǐ', ''],
      hi: ['पेंसिल', 'pen-sil', 'f'], ar: ['قلم رصاص', 'qalam rasaas', 'm']
    }
  },
  eraser: {
    em: '🧽', cat: 'school', lvl: 1, size: 'tiny',
    t: {
      es: ['goma', 'GO-ma', 'f'], fr: ['gomme', 'gom', 'f'],
      de: ['Radiergummi', 'ra-DEER-goo-mee', 'm'], it: ['gomma', 'GOM-ma', 'f'],
      pt: ['borracha', 'bo-HA-sha', 'f'], nl: ['gum', 'khum', 'c'],
      ru: ['ластик', 'LAS-teek', 'm'], ja: ['消しゴム', 'keshigomu', ''],
      ko: ['지우개', 'jiugae', ''], zh: ['橡皮', 'xiàng pí', ''],
      hi: ['रबर', 'ra-bar', 'm'], ar: ['ممحاة', 'mimhaat', 'f']
    }
  },
  ruler: {
    em: '📏', cat: 'school', lvl: 1, size: 'small',
    t: {
      es: ['regla', 'REH-gla', 'f'], fr: ['règle', 'reh-gl', 'f'],
      de: ['Lineal', 'lee-neh-AHL', 'n'], it: ['righello', 'ree-GEL-lo', 'm'],
      pt: ['régua', 'HEH-gwa', 'f'], nl: ['liniaal', 'lee-nee-AHL', 'c'],
      ru: ['линейка', 'lee-NYAY-ka', 'f'], ja: ['定規', 'jougi', ''],
      ko: ['자', 'ja', ''], zh: ['尺子', 'chǐ zi', ''],
      hi: ['रूलर', 'roo-lar', 'm'], ar: ['مسطرة', 'mistara', 'f']
    }
  },
  notebook: {
    em: '📓', cat: 'school', lvl: 1, size: 'small',
    t: {
      es: ['cuaderno', 'kwa-DEHR-no', 'm'], fr: ['cahier', 'ka-YAY', 'm'],
      de: ['Heft', 'heft', 'n'], it: ['quaderno', 'kwa-DEHR-no', 'm'],
      pt: ['caderno', 'ka-DEHR-noo', 'm'], nl: ['schrift', 'skhrift', 'n'],
      ru: ['тетрадь', 'tee-TRAT', 'f'], ja: ['ノート', 'nooto', ''],
      ko: ['공책', 'gongchaek', ''], zh: ['笔记本', 'bǐ jì běn', ''],
      hi: ['कॉपी', 'ko-pee', 'f'], ar: ['دفتر', 'daftar', 'm']
    }
  },
  sharpener: {
    em: '🖇️', cat: 'school', lvl: 2, size: 'tiny',
    t: {
      es: ['sacapuntas', 'sa-ka-POON-tas', 'm'], fr: ['taille-crayon', 'tie-kray-OHN', 'm'],
      de: ['Anspitzer', 'AN-shpit-ser', 'm'], it: ['temperamatite', 'tem-peh-ra-ma-TEE-teh', 'm'],
      pt: ['apontador', 'a-pon-ta-DOR', 'm'], nl: ['puntenslijper', 'PUN-ten-sly-per', 'c'],
      ru: ['точилка', 'ta-CHEEL-ka', 'f'], ja: ['鉛筆削り', 'enpitsukezuri', ''],
      ko: ['연필깎이', 'yeonpilkkakki', ''], zh: ['卷笔刀', 'juǎn bǐ dāo', ''],
      hi: ['शार्पनर', 'shaar-pa-nar', 'm'], ar: ['مبراة', 'mibraat', 'f']
    }
  },
  folder: {
    em: '📁', cat: 'school', lvl: 2, size: 'small',
    t: {
      es: ['carpeta', 'kar-PEH-ta', 'f'], fr: ['classeur', 'kla-SUR', 'm'],
      de: ['Ordner', 'ORD-ner', 'm'], it: ['cartella', 'kar-TEL-la', 'f'],
      pt: ['pasta', 'PAS-ta', 'f'], nl: ['map', 'map', 'c'],
      ru: ['папка', 'PAP-ka', 'f'], ja: ['ファイル', 'fairu', ''],
      ko: ['폴더', 'poldeo', ''], zh: ['文件夹', 'wén jiàn jiā', ''],
      hi: ['फ़ोल्डर', 'fol-dar', 'm'], ar: ['ملف', 'malaff', 'm']
    }
  },
  calculator: {
    em: '🧮', cat: 'school', lvl: 2, size: 'small',
    t: {
      es: ['calculadora', 'kal-koo-la-DO-ra', 'f'], fr: ['calculatrice', 'kal-kew-la-TREES', 'f'],
      de: ['Taschenrechner', 'TA-shen-rekh-ner', 'm'], it: ['calcolatrice', 'kal-ko-la-TREE-cheh', 'f'],
      pt: ['calculadora', 'kal-koo-la-DO-ra', 'f'], nl: ['rekenmachine', 'RAY-ken-ma-shee-ne', 'c'],
      ru: ['калькулятор', 'kal-koo-LYA-tar', 'm'], ja: ['電卓', 'dentaku', ''],
      ko: ['계산기', 'gyesangi', ''], zh: ['计算器', 'jì suàn qì', ''],
      hi: ['कैलकुलेटर', 'kal-kyu-le-tar', 'm'], ar: ['آلة حاسبة', 'aala haasiba', 'f']
    }
  },
  desk: {
    em: '🪑', cat: 'school', lvl: 1, size: 'large',
    t: {
      es: ['escritorio', 'es-kree-TO-ryo', 'm'], fr: ['bureau', 'bew-ROH', 'm'],
      de: ['Schreibtisch', 'SHRYP-tish', 'm'], it: ['scrivania', 'skree-va-NEE-a', 'f'],
      pt: ['escrivaninha', 'es-kree-va-NEEN-ya', 'f'], nl: ['bureau', 'bew-ROH', 'n'],
      ru: ['парта', 'PAR-ta', 'f'], ja: ['机', 'tsukue', ''],
      ko: ['책상', 'chaeksang', ''], zh: ['书桌', 'shū zhuō', ''],
      hi: ['मेज़', 'mez', 'f'], ar: ['مكتب', 'maktab', 'm']
    }
  },
  blackboard: {
    em: '🧑‍🏫', cat: 'school', lvl: 1, size: 'huge',
    t: {
      es: ['pizarra', 'pee-SA-rra', 'f'], fr: ['tableau', 'ta-BLOH', 'm'],
      de: ['Tafel', 'TAA-fel', 'f'], it: ['lavagna', 'la-VA-nya', 'f'],
      pt: ['quadro', 'KWA-droo', 'm'], nl: ['schoolbord', 'SKHOAL-bort', 'n'],
      ru: ['доска', 'das-KA', 'f'], ja: ['黒板', 'kokuban', ''],
      ko: ['칠판', 'chilpan', ''], zh: ['黑板', 'hēi bǎn', ''],
      hi: ['श्यामपट', 'shyaam-pat', 'm'], ar: ['سبورة', 'sabbuura', 'f']
    }
  },
  chalk: {
    em: '🖍️', cat: 'school', lvl: 2, size: 'tiny',
    t: {
      es: ['tiza', 'TEE-sa', 'f'], fr: ['craie', 'kray', 'f'],
      de: ['Kreide', 'KRY-de', 'f'], it: ['gesso', 'JES-so', 'm'],
      pt: ['giz', 'zheez', 'm'], nl: ['krijt', 'kryt', 'n'],
      ru: ['мел', 'myel', 'm'], ja: ['チョーク', 'chooku', ''],
      ko: ['분필', 'bunpil', ''], zh: ['粉笔', 'fěn bǐ', ''],
      hi: ['चॉक', 'chok', 'm'], ar: ['طباشير', 'tabaasheer', 'm']
    }
  },
  marker: {
    em: '🖍️', cat: 'school', lvl: 2, size: 'tiny',
    t: {
      es: ['rotulador', 'ro-too-la-DOR', 'm'], fr: ['marqueur', 'mar-KUR', 'm'],
      de: ['Marker', 'MAR-ker', 'm'], it: ['pennarello', 'pen-na-REL-lo', 'm'],
      pt: ['marcador', 'mar-ka-DOR', 'm'], nl: ['stift', 'stift', 'c'],
      ru: ['маркер', 'MAR-kyer', 'm'], ja: ['マーカー', 'maakaa', ''],
      ko: ['마커', 'makeo', ''], zh: ['记号笔', 'jì hào bǐ', ''],
      hi: ['मार्कर', 'maar-kar', 'm'], ar: ['قلم تحديد', 'qalam tahdeed', 'm']
    }
  },
  glue: {
    em: '🧴', cat: 'school', lvl: 2, size: 'tiny',
    t: {
      es: ['pegamento', 'peh-ga-MEN-to', 'm'], fr: ['colle', 'kol', 'f'],
      de: ['Kleber', 'KLAY-ber', 'm'], it: ['colla', 'KOL-la', 'f'],
      pt: ['cola', 'KO-la', 'f'], nl: ['lijm', 'lym', 'c'],
      ru: ['клей', 'klyay', 'm'], ja: ['のり', 'nori', ''],
      ko: ['풀', 'pul', ''], zh: ['胶水', 'jiāo shuǐ', ''],
      hi: ['गोंद', 'gond', 'm'], ar: ['غراء', 'ghiraa', 'm']
    }
  },
  stapler: {
    em: '📎', cat: 'school', lvl: 2, size: 'tiny',
    t: {
      es: ['grapadora', 'gra-pa-DO-ra', 'f'], fr: ['agrafeuse', 'a-gra-FUZ', 'f'],
      de: ['Hefter', 'HEF-ter', 'm'], it: ['cucitrice', 'koo-chee-TREE-cheh', 'f'],
      pt: ['grampeador', 'gram-peh-a-DOR', 'm'], nl: ['nietmachine', 'NEET-ma-shee-ne', 'c'],
      ru: ['степлер', 'STEP-lyer', 'm'], ja: ['ホッチキス', 'hocchikisu', ''],
      ko: ['스테이플러', 'seuteipeulleo', ''], zh: ['订书机', 'dìng shū jī', ''],
      hi: ['स्टेपलर', 'ste-pa-lar', 'm'], ar: ['دباسة', 'dabbaasa', 'f']
    }
  },
  paper: {
    em: '📄', cat: 'school', lvl: 1, size: 'small',
    t: {
      es: ['papel', 'pa-PEL', 'm'], fr: ['papier', 'pa-PYAY', 'm'],
      de: ['Papier', 'pa-PEER', 'n'], it: ['carta', 'KAR-ta', 'f'],
      pt: ['papel', 'pa-PEL', 'm'], nl: ['papier', 'pa-PEER', 'n'],
      ru: ['бумага', 'boo-MA-ga', 'f'], ja: ['紙', 'kami', ''],
      ko: ['종이', 'jongi', ''], zh: ['纸', 'zhǐ', ''],
      hi: ['कागज़', 'kaa-gaz', 'm'], ar: ['ورق', 'waraq', 'm']
    }
  },
  globe: {
    em: '🌍', cat: 'school', lvl: 2, size: 'medium',
    t: {
      es: ['globo terráqueo', 'GLO-bo teh-RRA-keh-o', 'm'], fr: ['globe', 'glob', 'm'],
      de: ['Globus', 'GLOH-boos', 'm'], it: ['mappamondo', 'map-pa-MON-do', 'm'],
      pt: ['globo', 'GLO-boo', 'm'], nl: ['wereldbol', 'VAY-relt-bol', 'c'],
      ru: ['глобус', 'GLO-boos', 'm'], ja: ['地球儀', 'chikyuugi', ''],
      ko: ['지구본', 'jigubon', ''], zh: ['地球仪', 'dì qiú yí', ''],
      hi: ['ग्लोब', 'glob', 'm'], ar: ['كرة أرضية', 'kura ardiyya', 'f']
    }
  },
  map: {
    em: '🗺️', cat: 'school', lvl: 1, size: 'large',
    t: {
      es: ['mapa', 'MA-pa', 'm'], fr: ['carte', 'kart', 'f'],
      de: ['Landkarte', 'LANT-kar-te', 'f'], it: ['mappa', 'MAP-pa', 'f'],
      pt: ['mapa', 'MA-pa', 'm'], nl: ['kaart', 'kaart', 'c'],
      ru: ['карта', 'KAR-ta', 'f'], ja: ['地図', 'chizu', ''],
      ko: ['지도', 'jido', ''], zh: ['地图', 'dì tú', ''],
      hi: ['नक्शा', 'nak-shaa', 'm'], ar: ['خريطة', 'khareeta', 'f']
    }
  },
  projector: {
    em: '📽️', cat: 'electronics', lvl: 2, size: 'medium',
    t: {
      es: ['proyector', 'pro-yek-TOR', 'm'], fr: ['projecteur', 'pro-zhek-TUR', 'm'],
      de: ['Beamer', 'BEE-mer', 'm'], it: ['proiettore', 'pro-yet-TO-reh', 'm'],
      pt: ['projetor', 'pro-zhe-TOR', 'm'], nl: ['beamer', 'BEE-mer', 'c'],
      ru: ['проектор', 'pra-EK-tar', 'm'], ja: ['プロジェクター', 'purojekutaa', ''],
      ko: ['프로젝터', 'peurojekteo', ''], zh: ['投影仪', 'tóu yǐng yí', ''],
      hi: ['प्रोजेक्टर', 'pro-jek-tar', 'm'], ar: ['جهاز عرض', 'jihaaz ard', 'm']
    }
  },
  printer: {
    em: '🖨️', cat: 'electronics', lvl: 2, size: 'medium',
    t: {
      es: ['impresora', 'eem-preh-SO-ra', 'f'], fr: ['imprimante', 'am-pree-MAHNT', 'f'],
      de: ['Drucker', 'DROO-ker', 'm'], it: ['stampante', 'stam-PAN-teh', 'f'],
      pt: ['impressora', 'eem-preh-SO-ra', 'f'], nl: ['printer', 'PRIN-ter', 'c'],
      ru: ['принтер', 'PREEN-ter', 'm'], ja: ['プリンター', 'purintaa', ''],
      ko: ['프린터', 'peurinteo', ''], zh: ['打印机', 'dǎ yìn jī', ''],
      hi: ['प्रिंटर', 'prin-tar', 'm'], ar: ['طابعة', 'taabi-a', 'f']
    }
  },
  monitor: {
    em: '🖥️', cat: 'electronics', lvl: 2, size: 'large',
    t: {
      es: ['monitor', 'mo-nee-TOR', 'm'], fr: ['écran', 'ay-KRAHN', 'm'],
      de: ['Monitor', 'MO-nee-tor', 'm'], it: ['monitor', 'MO-nee-tor', 'm'],
      pt: ['monitor', 'mo-nee-TOR', 'm'], nl: ['monitor', 'MO-nee-tor', 'c'],
      ru: ['монитор', 'ma-nee-TOR', 'm'], ja: ['モニター', 'monitaa', ''],
      ko: ['모니터', 'moniteo', ''], zh: ['显示器', 'xiǎn shì qì', ''],
      hi: ['मॉनिटर', 'mo-ni-tar', 'm'], ar: ['شاشة', 'shaasha', 'f']
    }
  },
  computer: {
    em: '🖥️', cat: 'electronics', lvl: 1, size: 'large',
    t: {
      es: ['ordenador', 'or-deh-na-DOR', 'm'], fr: ['ordinateur', 'or-dee-na-TUR', 'm'],
      de: ['Computer', 'kom-PYOO-ter', 'm'], it: ['computer', 'kom-PYOO-ter', 'm'],
      pt: ['computador', 'kom-poo-ta-DOR', 'm'], nl: ['computer', 'kom-PYOO-ter', 'c'],
      ru: ['компьютер', 'kam-PYOO-ter', 'm'], ja: ['コンピューター', 'konpyuutaa', ''],
      ko: ['컴퓨터', 'keompyuteo', ''], zh: ['电脑', 'diàn nǎo', ''],
      hi: ['कंप्यूटर', 'kam-pyu-tar', 'm'], ar: ['حاسوب', 'haasuub', 'm']
    }
  },
  headphones: {
    em: '🎧', cat: 'electronics', lvl: 1, size: 'small',
    t: {
      es: ['auriculares', 'ow-ree-koo-LA-res', 'p'], fr: ['casque', 'kask', 'm'],
      de: ['Kopfhörer', 'KOPF-hur-rer', 'm'], it: ['cuffie', 'KOOF-fyeh', 'p'],
      pt: ['fones de ouvido', 'FO-nees jee oh-VEE-doo', 'p'], nl: ['koptelefoon', 'KOP-te-le-foan', 'c'],
      ru: ['наушники', 'na-OOSH-nee-kee', 'p'], ja: ['ヘッドホン', 'heddohon', ''],
      ko: ['헤드폰', 'hedeupon', ''], zh: ['耳机', 'ěr jī', ''],
      hi: ['हेडफ़ोन', 'hed-fon', 'm'], ar: ['سماعات', 'sammaa-aat', 'f']
    }
  },
  tablet: {
    em: '📱', cat: 'electronics', lvl: 2, size: 'small',
    t: {
      es: ['tableta', 'ta-BLEH-ta', 'f'], fr: ['tablette', 'ta-BLET', 'f'],
      de: ['Tablet', 'TAB-let', 'n'], it: ['tablet', 'TAB-let', 'm'],
      pt: ['tablet', 'TAB-let', 'm'], nl: ['tablet', 'TAB-let', 'c'],
      ru: ['планшет', 'plan-SHET', 'm'], ja: ['タブレット', 'taburetto', ''],
      ko: ['태블릿', 'taebeullit', ''], zh: ['平板电脑', 'píng bǎn diàn nǎo', ''],
      hi: ['टैबलेट', 'tab-let', 'm'], ar: ['لوح إلكتروني', 'lawh iliktrooni', 'm']
    }
  },
  camera: {
    em: '📷', cat: 'electronics', lvl: 1, size: 'small',
    t: {
      es: ['cámara', 'KA-ma-ra', 'f'], fr: ['appareil photo', 'a-pa-RAY fo-TOH', 'm'],
      de: ['Kamera', 'KA-me-ra', 'f'], it: ['macchina fotografica', 'MAK-kee-na fo-to-GRA-fee-ka', 'f'],
      pt: ['câmera', 'KA-me-ra', 'f'], nl: ['camera', 'KA-me-ra', 'c'],
      ru: ['фотоаппарат', 'fo-to-a-pa-RAT', 'm'], ja: ['カメラ', 'kamera', ''],
      ko: ['카메라', 'kamera', ''], zh: ['相机', 'xiàng jī', ''],
      hi: ['कैमरा', 'kai-ma-ra', 'm'], ar: ['كاميرا', 'kaamira', 'f']
    }
  },
  bookshelf: {
    em: '📚', cat: 'furniture', lvl: 2, size: 'huge',
    t: {
      es: ['estantería', 'es-tan-teh-REE-a', 'f'], fr: ['étagère', 'ay-ta-ZHEHR', 'f'],
      de: ['Bücherregal', 'BUU-kher-reh-gaal', 'n'], it: ['libreria', 'lee-breh-REE-a', 'f'],
      pt: ['estante', 'es-TAN-chee', 'f'], nl: ['boekenkast', 'BOO-ken-kast', 'c'],
      ru: ['книжная полка', 'KNEEZH-na-ya POL-ka', 'f'], ja: ['本棚', 'hondana', ''],
      ko: ['책장', 'chaekjang', ''], zh: ['书架', 'shū jià', ''],
      hi: ['किताबों की अलमारी', 'ki-taa-bon kee al-maa-ree', 'f'], ar: ['رف كتب', 'raff kutub', 'm']
    }
  },
  envelope: {
    em: '✉️', cat: 'school', lvl: 2, size: 'small',
    t: {
      es: ['sobre', 'SO-breh', 'm'], fr: ['enveloppe', 'ahn-VLOP', 'f'],
      de: ['Umschlag', 'OOM-shlaak', 'm'], it: ['busta', 'BOO-sta', 'f'],
      pt: ['envelope', 'en-veh-LO-pee', 'm'], nl: ['envelop', 'en-ve-LOP', 'c'],
      ru: ['конверт', 'kan-VYERT', 'm'], ja: ['封筒', 'fuutou', ''],
      ko: ['봉투', 'bongtu', ''], zh: ['信封', 'xìn fēng', ''],
      hi: ['लिफ़ाफ़ा', 'li-faa-faa', 'm'], ar: ['مظروف', 'mazroof', 'm']
    }
  },

  /* ── Home and everyday ──────────────────────────────────────────────── */

  lamp: {
    em: '💡', cat: 'furniture', lvl: 1, size: 'medium',
    t: {
      es: ['lámpara', 'LAM-pa-ra', 'f'], fr: ['lampe', 'lahmp', 'f'],
      de: ['Lampe', 'LAM-pe', 'f'], it: ['lampada', 'LAM-pa-da', 'f'],
      pt: ['lâmpada', 'LAM-pa-da', 'f'], nl: ['lamp', 'lamp', 'c'],
      ru: ['лампа', 'LAM-pa', 'f'], ja: ['ランプ', 'ranpu', ''],
      ko: ['램프', 'raempeu', ''], zh: ['灯', 'dēng', ''],
      hi: ['दीपक', 'dee-pak', 'm'], ar: ['مصباح', 'misbaah', 'm']
    }
  },
  door: {
    em: '🚪', cat: 'furniture', lvl: 1, size: 'huge',
    t: {
      es: ['puerta', 'PWEHR-ta', 'f'], fr: ['porte', 'port', 'f'],
      de: ['Tür', 'tuur', 'f'], it: ['porta', 'POR-ta', 'f'],
      pt: ['porta', 'POR-ta', 'f'], nl: ['deur', 'dur', 'c'],
      ru: ['дверь', 'dvyer', 'f'], ja: ['ドア', 'doa', ''],
      ko: ['문', 'mun', ''], zh: ['门', 'mén', ''],
      hi: ['दरवाज़ा', 'dar-vaa-zaa', 'm'], ar: ['باب', 'baab', 'm']
    }
  },
  window: {
    em: '🪟', cat: 'furniture', lvl: 1, size: 'huge',
    t: {
      es: ['ventana', 'ven-TA-na', 'f'], fr: ['fenêtre', 'fuh-NEH-truh', 'f'],
      de: ['Fenster', 'FEN-ster', 'n'], it: ['finestra', 'fee-NES-tra', 'f'],
      pt: ['janela', 'zha-NEH-la', 'f'], nl: ['raam', 'raam', 'n'],
      ru: ['окно', 'ak-NO', 'n'], ja: ['窓', 'mado', ''],
      ko: ['창문', 'changmun', ''], zh: ['窗户', 'chuāng hu', ''],
      hi: ['खिड़की', 'khir-kee', 'f'], ar: ['نافذة', 'naafidha', 'f']
    }
  },
  curtain: {
    em: '🪟', cat: 'furniture', lvl: 2, size: 'huge',
    t: {
      es: ['cortina', 'kor-TEE-na', 'f'], fr: ['rideau', 'ree-DOH', 'm'],
      de: ['Vorhang', 'FOR-hang', 'm'], it: ['tenda', 'TEN-da', 'f'],
      pt: ['cortina', 'kor-CHEE-na', 'f'], nl: ['gordijn', 'khor-DYN', 'n'],
      ru: ['штора', 'SHTO-ra', 'f'], ja: ['カーテン', 'kaaten', ''],
      ko: ['커튼', 'keoteun', ''], zh: ['窗帘', 'chuāng lián', ''],
      hi: ['पर्दा', 'par-daa', 'm'], ar: ['ستارة', 'sitaara', 'f']
    }
  },
  mirror: {
    em: '🪞', cat: 'furniture', lvl: 2, size: 'large',
    t: {
      es: ['espejo', 'es-PEH-kho', 'm'], fr: ['miroir', 'meer-WAR', 'm'],
      de: ['Spiegel', 'SHPEE-gel', 'm'], it: ['specchio', 'SPEK-kyo', 'm'],
      pt: ['espelho', 'es-PEH-lyoo', 'm'], nl: ['spiegel', 'SPEE-khel', 'c'],
      ru: ['зеркало', 'ZYER-ka-la', 'n'], ja: ['鏡', 'kagami', ''],
      ko: ['거울', 'geoul', ''], zh: ['镜子', 'jìng zi', ''],
      hi: ['आईना', 'aa-ee-naa', 'm'], ar: ['مرآة', 'miraat', 'f']
    }
  },
  pillow: {
    em: '🛏️', cat: 'furniture', lvl: 1, size: 'medium',
    t: {
      es: ['almohada', 'al-mo-A-da', 'f'], fr: ['oreiller', 'o-ray-YAY', 'm'],
      de: ['Kissen', 'KIS-sen', 'n'], it: ['cuscino', 'koo-SHEE-no', 'm'],
      pt: ['travesseiro', 'tra-veh-SAY-roo', 'm'], nl: ['kussen', 'KUS-sen', 'n'],
      ru: ['подушка', 'pa-DOOSH-ka', 'f'], ja: ['枕', 'makura', ''],
      ko: ['베개', 'begae', ''], zh: ['枕头', 'zhěn tou', ''],
      hi: ['तकिया', 'ta-ki-yaa', 'm'], ar: ['وسادة', 'wisaada', 'f']
    }
  },
  blanket: {
    em: '🛏️', cat: 'furniture', lvl: 2, size: 'large',
    t: {
      es: ['manta', 'MAN-ta', 'f'], fr: ['couverture', 'koo-vehr-TUUR', 'f'],
      de: ['Decke', 'DEK-ke', 'f'], it: ['coperta', 'ko-PEHR-ta', 'f'],
      pt: ['cobertor', 'ko-behr-TOR', 'm'], nl: ['deken', 'DAY-ken', 'c'],
      ru: ['одеяло', 'a-dee-YA-la', 'n'], ja: ['毛布', 'moufu', ''],
      ko: ['담요', 'damyo', ''], zh: ['毯子', 'tǎn zi', ''],
      hi: ['कंबल', 'kam-bal', 'm'], ar: ['بطانية', 'bataaniyya', 'f']
    }
  },
  towel: {
    em: '🧻', cat: 'object', lvl: 2, size: 'medium',
    t: {
      es: ['toalla', 'to-A-ya', 'f'], fr: ['serviette', 'sehr-VYET', 'f'],
      de: ['Handtuch', 'HANT-tookh', 'n'], it: ['asciugamano', 'a-shoo-ga-MA-no', 'm'],
      pt: ['toalha', 'to-A-lya', 'f'], nl: ['handdoek', 'HANT-dook', 'c'],
      ru: ['полотенце', 'pa-la-TYEN-tse', 'n'], ja: ['タオル', 'taoru', ''],
      ko: ['수건', 'sugeon', ''], zh: ['毛巾', 'máo jīn', ''],
      hi: ['तौलिया', 'tow-li-yaa', 'm'], ar: ['منشفة', 'minshafa', 'f']
    }
  },
  soap: {
    em: '🧼', cat: 'object', lvl: 2, size: 'tiny',
    t: {
      es: ['jabón', 'kha-BON', 'm'], fr: ['savon', 'sa-VOHN', 'm'],
      de: ['Seife', 'ZY-fe', 'f'], it: ['sapone', 'sa-PO-neh', 'm'],
      pt: ['sabonete', 'sa-bo-NEH-chee', 'm'], nl: ['zeep', 'zayp', 'c'],
      ru: ['мыло', 'MY-la', 'n'], ja: ['石鹸', 'sekken', ''],
      ko: ['비누', 'binu', ''], zh: ['肥皂', 'féi zào', ''],
      hi: ['साबुन', 'saa-bun', 'm'], ar: ['صابون', 'saaboon', 'm']
    }
  },
  candle: {
    em: '🕯️', cat: 'object', lvl: 2, size: 'small',
    t: {
      es: ['vela', 'VEH-la', 'f'], fr: ['bougie', 'boo-ZHEE', 'f'],
      de: ['Kerze', 'KEHR-tse', 'f'], it: ['candela', 'kan-DEH-la', 'f'],
      pt: ['vela', 'VEH-la', 'f'], nl: ['kaars', 'kaars', 'c'],
      ru: ['свеча', 'svee-CHA', 'f'], ja: ['ろうそく', 'rousoku', ''],
      ko: ['초', 'cho', ''], zh: ['蜡烛', 'là zhú', ''],
      hi: ['मोमबत्ती', 'mom-bat-tee', 'f'], ar: ['شمعة', 'shamaa', 'f']
    }
  },
  key: {
    em: '🔑', cat: 'object', lvl: 1, size: 'tiny',
    t: {
      es: ['llave', 'YA-veh', 'f'], fr: ['clé', 'klay', 'f'],
      de: ['Schlüssel', 'SHLUUS-sel', 'm'], it: ['chiave', 'KYA-veh', 'f'],
      pt: ['chave', 'SHA-vee', 'f'], nl: ['sleutel', 'SLU-tel', 'c'],
      ru: ['ключ', 'klyooch', 'm'], ja: ['鍵', 'kagi', ''],
      ko: ['열쇠', 'yeolsoe', ''], zh: ['钥匙', 'yào shi', ''],
      hi: ['चाबी', 'chaa-bee', 'f'], ar: ['مفتاح', 'miftaah', 'm']
    }
  },
  lock: {
    em: '🔒', cat: 'object', lvl: 2, size: 'tiny',
    t: {
      es: ['candado', 'kan-DA-do', 'm'], fr: ['cadenas', 'kad-NA', 'm'],
      de: ['Schloss', 'shloss', 'n'], it: ['lucchetto', 'look-KET-to', 'm'],
      pt: ['cadeado', 'ka-jee-A-doo', 'm'], nl: ['slot', 'slot', 'n'],
      ru: ['замок', 'za-MOK', 'm'], ja: ['錠', 'jou', ''],
      ko: ['자물쇠', 'jamulsoe', ''], zh: ['锁', 'suǒ', ''],
      hi: ['ताला', 'taa-laa', 'm'], ar: ['قفل', 'qufl', 'm']
    }
  },
  wallet: {
    em: '👛', cat: 'accessory', lvl: 1, size: 'tiny',
    t: {
      es: ['cartera', 'kar-TEH-ra', 'f'], fr: ['portefeuille', 'port-FUY', 'm'],
      de: ['Geldbörse', 'GELT-bur-se', 'f'], it: ['portafoglio', 'por-ta-FO-lyo', 'm'],
      pt: ['carteira', 'kar-TAY-ra', 'f'], nl: ['portemonnee', 'por-te-mo-NAY', 'c'],
      ru: ['кошелёк', 'ka-shee-LYOK', 'm'], ja: ['財布', 'saifu', ''],
      ko: ['지갑', 'jigap', ''], zh: ['钱包', 'qián bāo', ''],
      hi: ['बटुआ', 'ba-tu-aa', 'm'], ar: ['محفظة', 'mihfaza', 'f']
    }
  },
  glasses: {
    em: '👓', cat: 'accessory', lvl: 1, size: 'small',
    t: {
      es: ['gafas', 'GA-fas', 'p'], fr: ['lunettes', 'lew-NET', 'p'],
      de: ['Brille', 'BRIL-le', 'f'], it: ['occhiali', 'ok-KYA-lee', 'p'],
      pt: ['óculos', 'O-koo-loos', 'p'], nl: ['bril', 'bril', 'c'],
      ru: ['очки', 'ach-KEE', 'p'], ja: ['眼鏡', 'megane', ''],
      ko: ['안경', 'angyeong', ''], zh: ['眼镜', 'yǎn jìng', ''],
      hi: ['चश्मा', 'chash-maa', 'm'], ar: ['نظارة', 'nazzaara', 'f']
    }
  },
  watch: {
    em: '⌚', cat: 'accessory', lvl: 1, size: 'tiny',
    t: {
      es: ['reloj', 'reh-LOKH', 'm'], fr: ['montre', 'MOHN-truh', 'f'],
      de: ['Armbanduhr', 'ARM-bant-oor', 'f'], it: ['orologio', 'o-ro-LO-jo', 'm'],
      pt: ['relógio', 'heh-LO-zhyoo', 'm'], nl: ['horloge', 'hor-LO-zhe', 'n'],
      ru: ['часы', 'chee-SY', 'p'], ja: ['腕時計', 'udedokei', ''],
      ko: ['손목시계', 'sonmoksigye', ''], zh: ['手表', 'shǒu biǎo', ''],
      hi: ['घड़ी', 'gha-ree', 'f'], ar: ['ساعة', 'saa-a', 'f']
    }
  },
  ring: {
    em: '💍', cat: 'accessory', lvl: 2, size: 'tiny',
    t: {
      es: ['anillo', 'a-NEE-yo', 'm'], fr: ['bague', 'bag', 'f'],
      de: ['Ring', 'ring', 'm'], it: ['anello', 'a-NEL-lo', 'm'],
      pt: ['anel', 'a-NEL', 'm'], nl: ['ring', 'ring', 'c'],
      ru: ['кольцо', 'kal-TSO', 'n'], ja: ['指輪', 'yubiwa', ''],
      ko: ['반지', 'banji', ''], zh: ['戒指', 'jiè zhi', ''],
      hi: ['अंगूठी', 'an-goo-thee', 'f'], ar: ['خاتم', 'khaatam', 'm']
    }
  },
  necklace: {
    em: '📿', cat: 'accessory', lvl: 2, size: 'small',
    t: {
      es: ['collar', 'ko-YAR', 'm'], fr: ['collier', 'ko-LYAY', 'm'],
      de: ['Halskette', 'HALS-ket-te', 'f'], it: ['collana', 'kol-LA-na', 'f'],
      pt: ['colar', 'ko-LAR', 'm'], nl: ['ketting', 'KET-ting', 'c'],
      ru: ['ожерелье', 'a-zhee-RYEL-ye', 'n'], ja: ['ネックレス', 'nekkuresu', ''],
      ko: ['목걸이', 'mokgeori', ''], zh: ['项链', 'xiàng liàn', ''],
      hi: ['हार', 'haar', 'm'], ar: ['قلادة', 'qilaada', 'f']
    }
  },
  basket: {
    em: '🧺', cat: 'object', lvl: 2, size: 'medium',
    t: {
      es: ['cesta', 'SES-ta', 'f'], fr: ['panier', 'pa-NYAY', 'm'],
      de: ['Korb', 'korp', 'm'], it: ['cesto', 'CHES-to', 'm'],
      pt: ['cesta', 'SES-ta', 'f'], nl: ['mand', 'mant', 'c'],
      ru: ['корзина', 'kar-ZEE-na', 'f'], ja: ['かご', 'kago', ''],
      ko: ['바구니', 'baguni', ''], zh: ['篮子', 'lán zi', ''],
      hi: ['टोकरी', 'to-ka-ree', 'f'], ar: ['سلة', 'salla', 'f']
    }
  },
  box: {
    em: '📦', cat: 'object', lvl: 1, size: 'medium',
    t: {
      es: ['caja', 'KA-kha', 'f'], fr: ['boîte', 'bwat', 'f'],
      de: ['Schachtel', 'SHAKH-tel', 'f'], it: ['scatola', 'SKA-to-la', 'f'],
      pt: ['caixa', 'KY-sha', 'f'], nl: ['doos', 'doas', 'c'],
      ru: ['коробка', 'ka-ROP-ka', 'f'], ja: ['箱', 'hako', ''],
      ko: ['상자', 'sangja', ''], zh: ['盒子', 'hé zi', ''],
      hi: ['डिब्बा', 'dib-baa', 'm'], ar: ['صندوق', 'sunduuq', 'm']
    }
  },
  broom: {
    em: '🧹', cat: 'tool', lvl: 2, size: 'large',
    t: {
      es: ['escoba', 'es-KO-ba', 'f'], fr: ['balai', 'ba-LAY', 'm'],
      de: ['Besen', 'BAY-zen', 'm'], it: ['scopa', 'SKO-pa', 'f'],
      pt: ['vassoura', 'va-SO-ra', 'f'], nl: ['bezem', 'BAY-zem', 'c'],
      ru: ['метла', 'meet-LA', 'f'], ja: ['ほうき', 'houki', ''],
      ko: ['빗자루', 'bitjaru', ''], zh: ['扫帚', 'sào zhou', ''],
      hi: ['झाड़ू', 'jhaa-roo', 'f'], ar: ['مكنسة', 'miknasa', 'f']
    }
  },
  bucket: {
    em: '🪣', cat: 'object', lvl: 2, size: 'medium',
    t: {
      es: ['cubo', 'KOO-bo', 'm'], fr: ['seau', 'soh', 'm'],
      de: ['Eimer', 'AY-mer', 'm'], it: ['secchio', 'SEK-kyo', 'm'],
      pt: ['balde', 'BAL-jee', 'm'], nl: ['emmer', 'EM-mer', 'c'],
      ru: ['ведро', 'veed-RO', 'n'], ja: ['バケツ', 'baketsu', ''],
      ko: ['양동이', 'yangdongi', ''], zh: ['水桶', 'shuǐ tǒng', ''],
      hi: ['बाल्टी', 'baal-tee', 'f'], ar: ['دلو', 'dalw', 'm']
    }
  },
  plate: {
    em: '🍽️', cat: 'kitchen', lvl: 1, size: 'small',
    t: {
      es: ['plato', 'PLA-to', 'm'], fr: ['assiette', 'a-SYET', 'f'],
      de: ['Teller', 'TEL-ler', 'm'], it: ['piatto', 'PYAT-to', 'm'],
      pt: ['prato', 'PRA-too', 'm'], nl: ['bord', 'bort', 'n'],
      ru: ['тарелка', 'ta-RYEL-ka', 'f'], ja: ['皿', 'sara', ''],
      ko: ['접시', 'jeopsi', ''], zh: ['盘子', 'pán zi', ''],
      hi: ['थाली', 'thaa-lee', 'f'], ar: ['طبق', 'tabaq', 'm']
    }
  },
  mug: {
    em: '☕', cat: 'kitchen', lvl: 1, size: 'small',
    t: {
      es: ['taza', 'TA-sa', 'f'], fr: ['mug', 'muhg', 'm'],
      de: ['Tasse', 'TAS-se', 'f'], it: ['tazza', 'TAT-tsa', 'f'],
      pt: ['caneca', 'ka-NEH-ka', 'f'], nl: ['mok', 'mok', 'c'],
      ru: ['кружка', 'KROOSH-ka', 'f'], ja: ['マグカップ', 'magukappu', ''],
      ko: ['머그컵', 'meogeukeop', ''], zh: ['马克杯', 'mǎ kè bēi', ''],
      hi: ['मग', 'mag', 'm'], ar: ['كوب', 'kuub', 'm']
    }
  },
  teapot: {
    em: '🫖', cat: 'kitchen', lvl: 2, size: 'small',
    t: {
      es: ['tetera', 'teh-TEH-ra', 'f'], fr: ['théière', 'tay-YEHR', 'f'],
      de: ['Teekanne', 'TAY-kan-ne', 'f'], it: ['teiera', 'teh-YEH-ra', 'f'],
      pt: ['bule', 'BOO-lee', 'm'], nl: ['theepot', 'TAY-pot', 'c'],
      ru: ['чайник', 'CHY-neek', 'm'], ja: ['急須', 'kyuusu', ''],
      ko: ['주전자', 'jujeonja', ''], zh: ['茶壶', 'chá hú', ''],
      hi: ['चायदानी', 'chaay-daa-nee', 'f'], ar: ['إبريق شاي', 'ibreeq shaay', 'm']
    }
  },
  pan: {
    em: '🍳', cat: 'kitchen', lvl: 2, size: 'medium',
    t: {
      es: ['sartén', 'sar-TEN', 'f'], fr: ['poêle', 'pwal', 'f'],
      de: ['Pfanne', 'PFAN-ne', 'f'], it: ['padella', 'pa-DEL-la', 'f'],
      pt: ['frigideira', 'free-zhee-DAY-ra', 'f'], nl: ['pan', 'pan', 'c'],
      ru: ['сковорода', 'ska-va-ra-DA', 'f'], ja: ['フライパン', 'furaipan', ''],
      ko: ['프라이팬', 'peuraipaen', ''], zh: ['平底锅', 'píng dǐ guō', ''],
      hi: ['कड़ाही', 'ka-raa-hee', 'f'], ar: ['مقلاة', 'miqlaat', 'f']
    }
  },
  pot: {
    em: '🍲', cat: 'kitchen', lvl: 2, size: 'medium',
    t: {
      es: ['olla', 'O-ya', 'f'], fr: ['casserole', 'kas-ROL', 'f'],
      de: ['Topf', 'topf', 'm'], it: ['pentola', 'PEN-to-la', 'f'],
      pt: ['panela', 'pa-NEH-la', 'f'], nl: ['pot', 'pot', 'c'],
      ru: ['кастрюля', 'kas-TRYOO-lya', 'f'], ja: ['鍋', 'nabe', ''],
      ko: ['냄비', 'naembi', ''], zh: ['锅', 'guō', ''],
      hi: ['बर्तन', 'bar-tan', 'm'], ar: ['قدر', 'qidr', 'm']
    }
  },
  tray: {
    em: '🍽️', cat: 'kitchen', lvl: 2, size: 'medium',
    t: {
      es: ['bandeja', 'ban-DEH-kha', 'f'], fr: ['plateau', 'pla-TOH', 'm'],
      de: ['Tablett', 'ta-BLET', 'n'], it: ['vassoio', 'vas-SO-yo', 'm'],
      pt: ['bandeja', 'ban-DEH-zha', 'f'], nl: ['dienblad', 'DEEN-blat', 'n'],
      ru: ['поднос', 'pad-NOS', 'm'], ja: ['トレー', 'toree', ''],
      ko: ['쟁반', 'jaengban', ''], zh: ['托盘', 'tuō pán', ''],
      hi: ['ट्रे', 'tre', 'f'], ar: ['صينية', 'seeniyya', 'f']
    }
  },

  /* ── Clothing ───────────────────────────────────────────────────────── */

  shirt: {
    em: '👕', cat: 'clothing', lvl: 1, size: 'large',
    t: {
      es: ['camisa', 'ka-MEE-sa', 'f'], fr: ['chemise', 'shuh-MEEZ', 'f'],
      de: ['Hemd', 'hemt', 'n'], it: ['camicia', 'ka-MEE-cha', 'f'],
      pt: ['camisa', 'ka-MEE-za', 'f'], nl: ['hemd', 'hemt', 'n'],
      ru: ['рубашка', 'roo-BASH-ka', 'f'], ja: ['シャツ', 'shatsu', ''],
      ko: ['셔츠', 'syeocheu', ''], zh: ['衬衫', 'chèn shān', ''],
      hi: ['कमीज़', 'ka-meez', 'f'], ar: ['قميص', 'qamees', 'm']
    }
  },
  trousers: {
    em: '👖', cat: 'clothing', lvl: 1, size: 'large',
    t: {
      es: ['pantalones', 'pan-ta-LO-nes', 'p'], fr: ['pantalon', 'pahn-ta-LOHN', 'm'],
      de: ['Hose', 'HO-ze', 'f'], it: ['pantaloni', 'pan-ta-LO-nee', 'p'],
      pt: ['calça', 'KAL-sa', 'f'], nl: ['broek', 'brook', 'c'],
      ru: ['брюки', 'BRYOO-kee', 'p'], ja: ['ズボン', 'zubon', ''],
      ko: ['바지', 'baji', ''], zh: ['裤子', 'kù zi', ''],
      hi: ['पतलून', 'pat-loon', 'f'], ar: ['بنطال', 'bantaal', 'm']
    }
  },
  shoe: {
    em: '👟', cat: 'clothing', lvl: 1, size: 'small',
    t: {
      es: ['zapato', 'sa-PA-to', 'm'], fr: ['chaussure', 'shoh-SUUR', 'f'],
      de: ['Schuh', 'shoo', 'm'], it: ['scarpa', 'SKAR-pa', 'f'],
      pt: ['sapato', 'sa-PA-too', 'm'], nl: ['schoen', 'skhoon', 'c'],
      ru: ['ботинок', 'ba-TEE-nak', 'm'], ja: ['靴', 'kutsu', ''],
      ko: ['신발', 'sinbal', ''], zh: ['鞋', 'xié', ''],
      hi: ['जूता', 'joo-taa', 'm'], ar: ['حذاء', 'hidhaa', 'm']
    }
  },
  hat: {
    em: '🎩', cat: 'clothing', lvl: 1, size: 'small',
    t: {
      es: ['sombrero', 'som-BREH-ro', 'm'], fr: ['chapeau', 'sha-POH', 'm'],
      de: ['Hut', 'hoot', 'm'], it: ['cappello', 'kap-PEL-lo', 'm'],
      pt: ['chapéu', 'sha-PEH-oo', 'm'], nl: ['hoed', 'hoot', 'c'],
      ru: ['шляпа', 'SHLYA-pa', 'f'], ja: ['帽子', 'boushi', ''],
      ko: ['모자', 'moja', ''], zh: ['帽子', 'mào zi', ''],
      hi: ['टोपी', 'to-pee', 'f'], ar: ['قبعة', 'qubba-a', 'f']
    }
  },
  coat: {
    em: '🧥', cat: 'clothing', lvl: 1, size: 'large',
    t: {
      es: ['abrigo', 'a-BREE-go', 'm'], fr: ['manteau', 'mahn-TOH', 'm'],
      de: ['Mantel', 'MAN-tel', 'm'], it: ['cappotto', 'kap-POT-to', 'm'],
      pt: ['casaco', 'ka-ZA-koo', 'm'], nl: ['jas', 'yas', 'c'],
      ru: ['пальто', 'pal-TO', 'n'], ja: ['コート', 'kooto', ''],
      ko: ['코트', 'koteu', ''], zh: ['外套', 'wài tào', ''],
      hi: ['कोट', 'kot', 'm'], ar: ['معطف', 'mi-taf', 'm']
    }
  },
  sock: {
    em: '🧦', cat: 'clothing', lvl: 2, size: 'tiny',
    t: {
      es: ['calcetín', 'kal-seh-TEEN', 'm'], fr: ['chaussette', 'shoh-SET', 'f'],
      de: ['Socke', 'ZOK-ke', 'f'], it: ['calzino', 'kal-TSEE-no', 'm'],
      pt: ['meia', 'MAY-a', 'f'], nl: ['sok', 'sok', 'c'],
      ru: ['носок', 'na-SOK', 'm'], ja: ['靴下', 'kutsushita', ''],
      ko: ['양말', 'yangmal', ''], zh: ['袜子', 'wà zi', ''],
      hi: ['मोजा', 'mo-jaa', 'm'], ar: ['جورب', 'jawrab', 'm']
    }
  },
  dress: {
    em: '👗', cat: 'clothing', lvl: 2, size: 'large',
    t: {
      es: ['vestido', 'ves-TEE-do', 'm'], fr: ['robe', 'rob', 'f'],
      de: ['Kleid', 'klyt', 'n'], it: ['vestito', 'ves-TEE-to', 'm'],
      pt: ['vestido', 'ves-CHEE-doo', 'm'], nl: ['jurk', 'yurk', 'c'],
      ru: ['платье', 'PLAT-ye', 'n'], ja: ['ドレス', 'doresu', ''],
      ko: ['드레스', 'deureseu', ''], zh: ['连衣裙', 'lián yī qún', ''],
      hi: ['पोशाक', 'po-shaak', 'f'], ar: ['فستان', 'fustaan', 'm']
    }
  },
  glove: {
    em: '🧤', cat: 'clothing', lvl: 2, size: 'tiny',
    t: {
      es: ['guante', 'GWAN-teh', 'm'], fr: ['gant', 'gahn', 'm'],
      de: ['Handschuh', 'HANT-shoo', 'm'], it: ['guanto', 'GWAN-to', 'm'],
      pt: ['luva', 'LOO-va', 'f'], nl: ['handschoen', 'HANT-skhoon', 'c'],
      ru: ['перчатка', 'peer-CHAT-ka', 'f'], ja: ['手袋', 'tebukuro', ''],
      ko: ['장갑', 'janggap', ''], zh: ['手套', 'shǒu tào', ''],
      hi: ['दस्ताना', 'das-taa-naa', 'm'], ar: ['قفاز', 'quffaaz', 'm']
    }
  },
  helmet: {
    em: '⛑️', cat: 'clothing', lvl: 2, size: 'small',
    t: {
      es: ['casco', 'KAS-ko', 'm'], fr: ['casque', 'kask', 'm'],
      de: ['Helm', 'helm', 'm'], it: ['casco', 'KAS-ko', 'm'],
      pt: ['capacete', 'ka-pa-SEH-chee', 'm'], nl: ['helm', 'helm', 'c'],
      ru: ['шлем', 'shlyem', 'm'], ja: ['ヘルメット', 'herumetto', ''],
      ko: ['헬멧', 'helmet', ''], zh: ['头盔', 'tóu kuī', ''],
      hi: ['हेलमेट', 'hel-met', 'm'], ar: ['خوذة', 'khoodha', 'f']
    }
  },

  /* ── Music ──────────────────────────────────────────────────────────── */

  guitar: {
    em: '🎸', cat: 'music', lvl: 1, size: 'large',
    t: {
      es: ['guitarra', 'gee-TA-rra', 'f'], fr: ['guitare', 'gee-TAR', 'f'],
      de: ['Gitarre', 'gee-TAR-re', 'f'], it: ['chitarra', 'kee-TAR-ra', 'f'],
      pt: ['violão', 'vee-o-LOWN', 'm'], nl: ['gitaar', 'khee-TAAR', 'c'],
      ru: ['гитара', 'gee-TA-ra', 'f'], ja: ['ギター', 'gitaa', ''],
      ko: ['기타', 'gita', ''], zh: ['吉他', 'jí tā', ''],
      hi: ['गिटार', 'gi-taar', 'm'], ar: ['جيتار', 'jeetaar', 'm']
    }
  },
  piano: {
    em: '🎹', cat: 'music', lvl: 1, size: 'huge',
    t: {
      es: ['piano', 'PYA-no', 'm'], fr: ['piano', 'pya-NOH', 'm'],
      de: ['Klavier', 'kla-VEER', 'n'], it: ['pianoforte', 'pya-no-FOR-teh', 'm'],
      pt: ['piano', 'PYA-noo', 'm'], nl: ['piano', 'pee-YA-no', 'c'],
      ru: ['пианино', 'pya-NEE-na', 'n'], ja: ['ピアノ', 'piano', ''],
      ko: ['피아노', 'piano', ''], zh: ['钢琴', 'gāng qín', ''],
      hi: ['पियानो', 'pi-yaa-no', 'm'], ar: ['بيانو', 'biyaano', 'm']
    }
  },
  violin: {
    em: '🎻', cat: 'music', lvl: 2, size: 'medium',
    t: {
      es: ['violín', 'vee-o-LEEN', 'm'], fr: ['violon', 'vyo-LOHN', 'm'],
      de: ['Geige', 'GY-ge', 'f'], it: ['violino', 'vee-o-LEE-no', 'm'],
      pt: ['violino', 'vee-o-LEE-noo', 'm'], nl: ['viool', 'vee-YOAL', 'c'],
      ru: ['скрипка', 'SKREEP-ka', 'f'], ja: ['バイオリン', 'baiorin', ''],
      ko: ['바이올린', 'baiollin', ''], zh: ['小提琴', 'xiǎo tí qín', ''],
      hi: ['वायलिन', 'vaay-lin', 'm'], ar: ['كمان', 'kamaan', 'm']
    }
  },
  drum: {
    em: '🥁', cat: 'music', lvl: 1, size: 'medium',
    t: {
      es: ['tambor', 'tam-BOR', 'm'], fr: ['tambour', 'tahm-BOOR', 'm'],
      de: ['Trommel', 'TROM-mel', 'f'], it: ['tamburo', 'tam-BOO-ro', 'm'],
      pt: ['tambor', 'tam-BOR', 'm'], nl: ['trommel', 'TROM-mel', 'c'],
      ru: ['барабан', 'ba-ra-BAN', 'm'], ja: ['太鼓', 'taiko', ''],
      ko: ['드럼', 'deureom', ''], zh: ['鼓', 'gǔ', ''],
      hi: ['ढोल', 'dhol', 'm'], ar: ['طبل', 'tabl', 'm']
    }
  },
  flute: {
    em: '🪈', cat: 'music', lvl: 2, size: 'small',
    t: {
      es: ['flauta', 'FLOW-ta', 'f'], fr: ['flûte', 'flewt', 'f'],
      de: ['Flöte', 'FLUR-te', 'f'], it: ['flauto', 'FLOW-to', 'm'],
      pt: ['flauta', 'FLOW-ta', 'f'], nl: ['fluit', 'flowt', 'c'],
      ru: ['флейта', 'FLYAY-ta', 'f'], ja: ['フルート', 'furuuto', ''],
      ko: ['플루트', 'peulluteu', ''], zh: ['长笛', 'cháng dí', ''],
      hi: ['बांसुरी', 'baan-su-ree', 'f'], ar: ['ناي', 'naay', 'm']
    }
  },
  trumpet: {
    em: '🎺', cat: 'music', lvl: 2, size: 'medium',
    t: {
      es: ['trompeta', 'trom-PEH-ta', 'f'], fr: ['trompette', 'trohm-PET', 'f'],
      de: ['Trompete', 'trom-PAY-te', 'f'], it: ['tromba', 'TROM-ba', 'f'],
      pt: ['trompete', 'trom-PEH-chee', 'm'], nl: ['trompet', 'trom-PET', 'c'],
      ru: ['труба', 'troo-BA', 'f'], ja: ['トランペット', 'toranpetto', ''],
      ko: ['트럼펫', 'teureompet', ''], zh: ['小号', 'xiǎo hào', ''],
      hi: ['तुरही', 'tur-hee', 'f'], ar: ['بوق', 'buuq', 'm']
    }
  },

  /* ── Tools ──────────────────────────────────────────────────────────── */

  hammer: {
    em: '🔨', cat: 'tool', lvl: 2, size: 'small',
    t: {
      es: ['martillo', 'mar-TEE-yo', 'm'], fr: ['marteau', 'mar-TOH', 'm'],
      de: ['Hammer', 'HAM-mer', 'm'], it: ['martello', 'mar-TEL-lo', 'm'],
      pt: ['martelo', 'mar-TEH-loo', 'm'], nl: ['hamer', 'HAA-mer', 'c'],
      ru: ['молоток', 'ma-la-TOK', 'm'], ja: ['ハンマー', 'hanmaa', ''],
      ko: ['망치', 'mangchi', ''], zh: ['锤子', 'chuí zi', ''],
      hi: ['हथौड़ा', 'ha-thow-raa', 'm'], ar: ['مطرقة', 'mitraqa', 'f']
    }
  },
  screwdriver: {
    em: '🪛', cat: 'tool', lvl: 2, size: 'small',
    t: {
      es: ['destornillador', 'des-tor-nee-ya-DOR', 'm'], fr: ['tournevis', 'toor-nuh-VEES', 'm'],
      de: ['Schraubenzieher', 'SHROW-ben-tsee-er', 'm'], it: ['cacciavite', 'kach-cha-VEE-teh', 'm'],
      pt: ['chave de fenda', 'SHA-vee jee FEN-da', 'f'], nl: ['schroevendraaier', 'SKHROO-ven-draa-yer', 'c'],
      ru: ['отвёртка', 'at-VYORT-ka', 'f'], ja: ['ドライバー', 'doraibaa', ''],
      ko: ['드라이버', 'deuraibeo', ''], zh: ['螺丝刀', 'luó sī dāo', ''],
      hi: ['पेचकस', 'pech-kas', 'm'], ar: ['مفك', 'mifakk', 'm']
    }
  },
  paintbrush: {
    em: '🖌️', cat: 'tool', lvl: 2, size: 'tiny',
    t: {
      es: ['pincel', 'peen-SEL', 'm'], fr: ['pinceau', 'pan-SOH', 'm'],
      de: ['Pinsel', 'PIN-zel', 'm'], it: ['pennello', 'pen-NEL-lo', 'm'],
      pt: ['pincel', 'peen-SEL', 'm'], nl: ['penseel', 'pen-SAYL', 'n'],
      ru: ['кисть', 'keest', 'f'], ja: ['絵筆', 'efude', ''],
      ko: ['붓', 'but', ''], zh: ['画笔', 'huà bǐ', ''],
      hi: ['ब्रश', 'brash', 'm'], ar: ['فرشاة', 'furshaat', 'f']
    }
  },
  binoculars: {
    em: '🔭', cat: 'tool', lvl: 3, size: 'small',
    t: {
      es: ['binoculares', 'bee-no-koo-LA-res', 'p'], fr: ['jumelles', 'zhew-MEL', 'p'],
      de: ['Fernglas', 'FERN-glaas', 'n'], it: ['binocolo', 'bee-NO-ko-lo', 'm'],
      pt: ['binóculos', 'bee-NO-koo-loos', 'p'], nl: ['verrekijker', 'VER-re-ky-ker', 'c'],
      ru: ['бинокль', 'bee-NOKL', 'm'], ja: ['双眼鏡', 'sougankyou', ''],
      ko: ['쌍안경', 'ssangangyeong', ''], zh: ['望远镜', 'wàng yuǎn jìng', ''],
      hi: ['दूरबीन', 'door-been', 'f'], ar: ['منظار', 'minzaar', 'm']
    }
  },
  ladder: {
    em: '🪜', cat: 'tool', lvl: 2, size: 'huge',
    t: {
      es: ['escalera', 'es-ka-LEH-ra', 'f'], fr: ['échelle', 'ay-SHEL', 'f'],
      de: ['Leiter', 'LY-ter', 'f'], it: ['scala', 'SKA-la', 'f'],
      pt: ['escada', 'es-KA-da', 'f'], nl: ['ladder', 'LAD-der', 'c'],
      ru: ['лестница', 'LYES-neet-sa', 'f'], ja: ['はしご', 'hashigo', ''],
      ko: ['사다리', 'sadari', ''], zh: ['梯子', 'tī zi', ''],
      hi: ['सीढ़ी', 'see-rhee', 'f'], ar: ['سلم', 'sullam', 'm']
    }
  },

  /* ── Nature and animals ─────────────────────────────────────────────── */

  tree: {
    em: '🌳', cat: 'nature', lvl: 1, size: 'huge',
    t: {
      es: ['árbol', 'AR-bol', 'm'], fr: ['arbre', 'AR-bruh', 'm'],
      de: ['Baum', 'bowm', 'm'], it: ['albero', 'AL-beh-ro', 'm'],
      pt: ['árvore', 'AR-vo-ree', 'f'], nl: ['boom', 'boam', 'c'],
      ru: ['дерево', 'DYE-ree-va', 'n'], ja: ['木', 'ki', ''],
      ko: ['나무', 'namu', ''], zh: ['树', 'shù', ''],
      hi: ['पेड़', 'per', 'm'], ar: ['شجرة', 'shajara', 'f']
    }
  },
  flower: {
    em: '🌸', cat: 'nature', lvl: 1, size: 'small',
    t: {
      es: ['flor', 'flor', 'f'], fr: ['fleur', 'flur', 'f'],
      de: ['Blume', 'BLOO-me', 'f'], it: ['fiore', 'FYO-reh', 'm'],
      pt: ['flor', 'flor', 'f'], nl: ['bloem', 'bloom', 'c'],
      ru: ['цветок', 'tsvee-TOK', 'm'], ja: ['花', 'hana', ''],
      ko: ['꽃', 'kkot', ''], zh: ['花', 'huā', ''],
      hi: ['फूल', 'phool', 'm'], ar: ['زهرة', 'zahra', 'f']
    }
  },
  mushroom: {
    em: '🍄', cat: 'nature', lvl: 2, size: 'tiny',
    t: {
      es: ['seta', 'SEH-ta', 'f'], fr: ['champignon', 'shahm-pee-NYOHN', 'm'],
      de: ['Pilz', 'pilts', 'm'], it: ['fungo', 'FOON-go', 'm'],
      pt: ['cogumelo', 'ko-goo-MEH-loo', 'm'], nl: ['paddenstoel', 'PAD-den-stool', 'c'],
      ru: ['гриб', 'greep', 'm'], ja: ['きのこ', 'kinoko', ''],
      ko: ['버섯', 'beoseot', ''], zh: ['蘑菇', 'mó gu', ''],
      hi: ['मशरूम', 'mash-room', 'm'], ar: ['فطر', 'futr', 'm']
    }
  },
  butterfly: {
    em: '🦋', cat: 'animal', lvl: 1, size: 'tiny',
    t: {
      es: ['mariposa', 'ma-ree-PO-sa', 'f'], fr: ['papillon', 'pa-pee-YOHN', 'm'],
      de: ['Schmetterling', 'SHMET-ter-ling', 'm'], it: ['farfalla', 'far-FAL-la', 'f'],
      pt: ['borboleta', 'bor-bo-LEH-ta', 'f'], nl: ['vlinder', 'VLIN-der', 'c'],
      ru: ['бабочка', 'BA-bach-ka', 'f'], ja: ['蝶', 'chou', ''],
      ko: ['나비', 'nabi', ''], zh: ['蝴蝶', 'hú dié', ''],
      hi: ['तितली', 'tit-lee', 'f'], ar: ['فراشة', 'faraasha', 'f']
    }
  },
  fish: {
    em: '🐟', cat: 'animal', lvl: 1, size: 'small',
    t: {
      es: ['pez', 'pes', 'm'], fr: ['poisson', 'pwa-SOHN', 'm'],
      de: ['Fisch', 'fish', 'm'], it: ['pesce', 'PEH-sheh', 'm'],
      pt: ['peixe', 'PAY-shee', 'm'], nl: ['vis', 'vis', 'c'],
      ru: ['рыба', 'RY-ba', 'f'], ja: ['魚', 'sakana', ''],
      ko: ['물고기', 'mulgogi', ''], zh: ['鱼', 'yú', ''],
      hi: ['मछली', 'mach-lee', 'f'], ar: ['سمكة', 'samaka', 'f']
    }
  },
  lion: {
    em: '🦁', cat: 'animal', lvl: 1, size: 'large',
    t: {
      es: ['león', 'leh-ON', 'm'], fr: ['lion', 'lee-OHN', 'm'],
      de: ['Löwe', 'LUR-ve', 'm'], it: ['leone', 'leh-O-neh', 'm'],
      pt: ['leão', 'leh-OWN', 'm'], nl: ['leeuw', 'layw', 'c'],
      ru: ['лев', 'lyef', 'm'], ja: ['ライオン', 'raion', ''],
      ko: ['사자', 'saja', ''], zh: ['狮子', 'shī zi', ''],
      hi: ['शेर', 'sher', 'm'], ar: ['أسد', 'asad', 'm']
    }
  },
  tiger: {
    em: '🐯', cat: 'animal', lvl: 1, size: 'large',
    t: {
      es: ['tigre', 'TEE-greh', 'm'], fr: ['tigre', 'TEE-gruh', 'm'],
      de: ['Tiger', 'TEE-ger', 'm'], it: ['tigre', 'TEE-greh', 'f'],
      pt: ['tigre', 'TEE-gree', 'm'], nl: ['tijger', 'TY-kher', 'c'],
      ru: ['тигр', 'teegr', 'm'], ja: ['虎', 'tora', ''],
      ko: ['호랑이', 'horangi', ''], zh: ['老虎', 'lǎo hǔ', ''],
      hi: ['बाघ', 'baagh', 'm'], ar: ['نمر', 'namir', 'm']
    }
  },
  monkey: {
    em: '🐒', cat: 'animal', lvl: 1, size: 'medium',
    t: {
      es: ['mono', 'MO-no', 'm'], fr: ['singe', 'sanzh', 'm'],
      de: ['Affe', 'AF-fe', 'm'], it: ['scimmia', 'SHEEM-mya', 'f'],
      pt: ['macaco', 'ma-KA-koo', 'm'], nl: ['aap', 'aap', 'c'],
      ru: ['обезьяна', 'a-bee-ZYA-na', 'f'], ja: ['猿', 'saru', ''],
      ko: ['원숭이', 'wonsungi', ''], zh: ['猴子', 'hóu zi', ''],
      hi: ['बंदर', 'ban-dar', 'm'], ar: ['قرد', 'qird', 'm']
    }
  },
  rabbit: {
    em: '🐰', cat: 'animal', lvl: 1, size: 'small',
    t: {
      es: ['conejo', 'ko-NEH-kho', 'm'], fr: ['lapin', 'la-PAN', 'm'],
      de: ['Kaninchen', 'ka-NEEN-khen', 'n'], it: ['coniglio', 'ko-NEE-lyo', 'm'],
      pt: ['coelho', 'ko-EH-lyoo', 'm'], nl: ['konijn', 'ko-NYN', 'n'],
      ru: ['кролик', 'KRO-leek', 'm'], ja: ['うさぎ', 'usagi', ''],
      ko: ['토끼', 'tokki', ''], zh: ['兔子', 'tù zi', ''],
      hi: ['खरगोश', 'khar-gosh', 'm'], ar: ['أرنب', 'arnab', 'm']
    }
  },
  frog: {
    em: '🐸', cat: 'animal', lvl: 2, size: 'tiny',
    t: {
      es: ['rana', 'RA-na', 'f'], fr: ['grenouille', 'gruh-NOO-y', 'f'],
      de: ['Frosch', 'frosh', 'm'], it: ['rana', 'RA-na', 'f'],
      pt: ['sapo', 'SA-poo', 'm'], nl: ['kikker', 'KIK-ker', 'c'],
      ru: ['лягушка', 'lya-GOOSH-ka', 'f'], ja: ['カエル', 'kaeru', ''],
      ko: ['개구리', 'gaeguri', ''], zh: ['青蛙', 'qīng wā', ''],
      hi: ['मेंढक', 'men-dhak', 'm'], ar: ['ضفدع', 'difda', 'm']
    }
  },
  snake: {
    em: '🐍', cat: 'animal', lvl: 2, size: 'medium',
    t: {
      es: ['serpiente', 'ser-PYEN-teh', 'f'], fr: ['serpent', 'ser-PAHN', 'm'],
      de: ['Schlange', 'SHLANG-e', 'f'], it: ['serpente', 'ser-PEN-teh', 'm'],
      pt: ['cobra', 'KO-bra', 'f'], nl: ['slang', 'slang', 'c'],
      ru: ['змея', 'zmee-YA', 'f'], ja: ['蛇', 'hebi', ''],
      ko: ['뱀', 'baem', ''], zh: ['蛇', 'shé', ''],
      hi: ['साँप', 'saanp', 'm'], ar: ['ثعبان', 'thu-baan', 'm']
    }
  },
  turtle: {
    em: '🐢', cat: 'animal', lvl: 2, size: 'small',
    t: {
      es: ['tortuga', 'tor-TOO-ga', 'f'], fr: ['tortue', 'tor-TEW', 'f'],
      de: ['Schildkröte', 'SHILT-krur-te', 'f'], it: ['tartaruga', 'tar-ta-ROO-ga', 'f'],
      pt: ['tartaruga', 'tar-ta-ROO-ga', 'f'], nl: ['schildpad', 'SKHILT-pat', 'c'],
      ru: ['черепаха', 'chee-ree-PA-kha', 'f'], ja: ['亀', 'kame', ''],
      ko: ['거북', 'geobuk', ''], zh: ['乌龟', 'wū guī', ''],
      hi: ['कछुआ', 'ka-chu-aa', 'm'], ar: ['سلحفاة', 'sulahfaat', 'f']
    }
  },
  spider: {
    em: '🕷️', cat: 'animal', lvl: 2, size: 'tiny',
    t: {
      es: ['araña', 'a-RA-nya', 'f'], fr: ['araignée', 'a-ray-NYAY', 'f'],
      de: ['Spinne', 'SHPIN-ne', 'f'], it: ['ragno', 'RA-nyo', 'm'],
      pt: ['aranha', 'a-RA-nya', 'f'], nl: ['spin', 'spin', 'c'],
      ru: ['паук', 'pa-OOK', 'm'], ja: ['クモ', 'kumo', ''],
      ko: ['거미', 'geomi', ''], zh: ['蜘蛛', 'zhī zhū', ''],
      hi: ['मकड़ी', 'mak-ree', 'f'], ar: ['عنكبوت', 'ankabuut', 'm']
    }
  },
  'tropical fish': {
    em: '🐠', cat: 'animal', lvl: 3, size: 'tiny',
    t: {
      es: ['pez tropical', 'pes tro-pee-KAL', 'm'], fr: ['poisson tropical', 'pwa-SOHN tro-pee-KAL', 'm'],
      de: ['Tropenfisch', 'TRO-pen-fish', 'm'], it: ['pesce tropicale', 'PEH-sheh tro-pee-KA-leh', 'm'],
      pt: ['peixe tropical', 'PAY-shee tro-pee-KAL', 'm'], nl: ['tropische vis', 'TRO-pee-se vis', 'c'],
      ru: ['тропическая рыба', 'tra-PEE-chees-ka-ya RY-ba', 'f'], ja: ['熱帯魚', 'nettaigyo', ''],
      ko: ['열대어', 'yeoldaeeo', ''], zh: ['热带鱼', 'rè dài yú', ''],
      hi: ['उष्णकटिबंधीय मछली', 'ushn-ka-tib-andheey mach-lee', 'f'], ar: ['سمكة استوائية', 'samaka istiwaa-iyya', 'f']
    }
  },
  penguin: {
    em: '🐧', cat: 'animal', lvl: 2, size: 'medium',
    t: {
      es: ['pingüino', 'peen-GWEE-no', 'm'], fr: ['pingouin', 'pan-GWAN', 'm'],
      de: ['Pinguin', 'PIN-gu-een', 'm'], it: ['pinguino', 'peen-GWEE-no', 'm'],
      pt: ['pinguim', 'peen-GWEENG', 'm'], nl: ['pinguïn', 'PIN-gu-in', 'c'],
      ru: ['пингвин', 'peen-GVEEN', 'm'], ja: ['ペンギン', 'pengin', ''],
      ko: ['펭귄', 'penggwin', ''], zh: ['企鹅', 'qǐ é', ''],
      hi: ['पेंगुइन', 'pen-gu-in', 'm'], ar: ['بطريق', 'batreeq', 'm']
    }
  },
  owl: {
    em: '🦉', cat: 'animal', lvl: 2, size: 'small',
    t: {
      es: ['búho', 'BOO-o', 'm'], fr: ['hibou', 'ee-BOO', 'm'],
      de: ['Eule', 'OY-le', 'f'], it: ['gufo', 'GOO-fo', 'm'],
      pt: ['coruja', 'ko-ROO-zha', 'f'], nl: ['uil', 'owl', 'c'],
      ru: ['сова', 'sa-VA', 'f'], ja: ['フクロウ', 'fukurou', ''],
      ko: ['올빼미', 'olppaemi', ''], zh: ['猫头鹰', 'māo tóu yīng', ''],
      hi: ['उल्लू', 'ul-loo', 'm'], ar: ['بومة', 'buuma', 'f']
    }
  },
  peacock: {
    em: '🦚', cat: 'animal', lvl: 3, size: 'large',
    t: {
      es: ['pavo real', 'PA-vo reh-AL', 'm'], fr: ['paon', 'pahn', 'm'],
      de: ['Pfau', 'pfow', 'm'], it: ['pavone', 'pa-VO-neh', 'm'],
      pt: ['pavão', 'pa-VOWN', 'm'], nl: ['pauw', 'pow', 'c'],
      ru: ['павлин', 'pav-LEEN', 'm'], ja: ['クジャク', 'kujaku', ''],
      ko: ['공작', 'gongjak', ''], zh: ['孔雀', 'kǒng què', ''],
      hi: ['मोर', 'mor', 'm'], ar: ['طاووس', 'taawuus', 'm']
    }
  },

  /* ── Places ─────────────────────────────────────────────────────────── */

  house: {
    em: '🏠', cat: 'place', lvl: 1, size: 'huge',
    t: {
      es: ['casa', 'KA-sa', 'f'], fr: ['maison', 'may-ZOHN', 'f'],
      de: ['Haus', 'hows', 'n'], it: ['casa', 'KA-sa', 'f'],
      pt: ['casa', 'KA-za', 'f'], nl: ['huis', 'hows', 'n'],
      ru: ['дом', 'dom', 'm'], ja: ['家', 'ie', ''],
      ko: ['집', 'jip', ''], zh: ['房子', 'fáng zi', ''],
      hi: ['घर', 'ghar', 'm'], ar: ['بيت', 'bayt', 'm']
    }
  },
  church: {
    em: '⛪', cat: 'place', lvl: 2, size: 'huge',
    t: {
      es: ['iglesia', 'ee-GLEH-sya', 'f'], fr: ['église', 'ay-GLEEZ', 'f'],
      de: ['Kirche', 'KEER-khe', 'f'], it: ['chiesa', 'KYEH-sa', 'f'],
      pt: ['igreja', 'ee-GREH-zha', 'f'], nl: ['kerk', 'kerk', 'c'],
      ru: ['церковь', 'TSER-kaf', 'f'], ja: ['教会', 'kyoukai', ''],
      ko: ['교회', 'gyohoe', ''], zh: ['教堂', 'jiào táng', ''],
      hi: ['गिरजाघर', 'gir-jaa-ghar', 'm'], ar: ['كنيسة', 'kaneesa', 'f']
    }
  },
  castle: {
    em: '🏰', cat: 'place', lvl: 2, size: 'huge',
    t: {
      es: ['castillo', 'kas-TEE-yo', 'm'], fr: ['château', 'sha-TOH', 'm'],
      de: ['Schloss', 'shloss', 'n'], it: ['castello', 'kas-TEL-lo', 'm'],
      pt: ['castelo', 'kas-TEH-loo', 'm'], nl: ['kasteel', 'kas-TAYL', 'n'],
      ru: ['замок', 'ZA-mak', 'm'], ja: ['城', 'shiro', ''],
      ko: ['성', 'seong', ''], zh: ['城堡', 'chéng bǎo', ''],
      hi: ['किला', 'ki-laa', 'm'], ar: ['قلعة', 'qal-a', 'f']
    }
  },
  bridge: {
    em: '🌉', cat: 'place', lvl: 1, size: 'huge',
    t: {
      es: ['puente', 'PWEN-teh', 'm'], fr: ['pont', 'pohn', 'm'],
      de: ['Brücke', 'BRUUK-ke', 'f'], it: ['ponte', 'PON-teh', 'm'],
      pt: ['ponte', 'PON-chee', 'f'], nl: ['brug', 'brukh', 'c'],
      ru: ['мост', 'most', 'm'], ja: ['橋', 'hashi', ''],
      ko: ['다리', 'dari', ''], zh: ['桥', 'qiáo', ''],
      hi: ['पुल', 'pul', 'm'], ar: ['جسر', 'jisr', 'm']
    }
  },
  fountain: {
    em: '⛲', cat: 'place', lvl: 2, size: 'large',
    t: {
      es: ['fuente', 'FWEN-teh', 'f'], fr: ['fontaine', 'fohn-TEN', 'f'],
      de: ['Brunnen', 'BROON-nen', 'm'], it: ['fontana', 'fon-TA-na', 'f'],
      pt: ['fonte', 'FON-chee', 'f'], nl: ['fontein', 'fon-TAYN', 'c'],
      ru: ['фонтан', 'fan-TAN', 'm'], ja: ['噴水', 'funsui', ''],
      ko: ['분수', 'bunsu', ''], zh: ['喷泉', 'pēn quán', ''],
      hi: ['फव्वारा', 'phav-vaa-raa', 'm'], ar: ['نافورة', 'naafuura', 'f']
    }
  },
  mountain: {
    em: '⛰️', cat: 'nature', lvl: 1, size: 'huge',
    t: {
      es: ['montaña', 'mon-TA-nya', 'f'], fr: ['montagne', 'mohn-TA-nyuh', 'f'],
      de: ['Berg', 'berk', 'm'], it: ['montagna', 'mon-TA-nya', 'f'],
      pt: ['montanha', 'mon-TA-nya', 'f'], nl: ['berg', 'berkh', 'c'],
      ru: ['гора', 'ga-RA', 'f'], ja: ['山', 'yama', ''],
      ko: ['산', 'san', ''], zh: ['山', 'shān', ''],
      hi: ['पहाड़', 'pa-haar', 'm'], ar: ['جبل', 'jabal', 'm']
    }
  },
  beach: {
    em: '🏖️', cat: 'nature', lvl: 1, size: 'huge',
    t: {
      es: ['playa', 'PLA-ya', 'f'], fr: ['plage', 'plazh', 'f'],
      de: ['Strand', 'shtrant', 'm'], it: ['spiaggia', 'SPYAD-ja', 'f'],
      pt: ['praia', 'PRY-a', 'f'], nl: ['strand', 'strant', 'n'],
      ru: ['пляж', 'plyash', 'm'], ja: ['浜', 'hama', ''],
      ko: ['해변', 'haebyeon', ''], zh: ['海滩', 'hǎi tān', ''],
      hi: ['समुद्र तट', 'sa-mu-dra tat', 'm'], ar: ['شاطئ', 'shaati', 'm']
    }
  },
  volcano: {
    em: '🌋', cat: 'nature', lvl: 3, size: 'huge',
    t: {
      es: ['volcán', 'vol-KAN', 'm'], fr: ['volcan', 'vol-KAHN', 'm'],
      de: ['Vulkan', 'vool-KAAN', 'm'], it: ['vulcano', 'vool-KA-no', 'm'],
      pt: ['vulcão', 'vool-KOWN', 'm'], nl: ['vulkaan', 'vool-KAAN', 'c'],
      ru: ['вулкан', 'vool-KAN', 'm'], ja: ['火山', 'kazan', ''],
      ko: ['화산', 'hwasan', ''], zh: ['火山', 'huǒ shān', ''],
      hi: ['ज्वालामुखी', 'jvaa-laa-mu-khee', 'm'], ar: ['بركان', 'burkaan', 'm']
    }
  },
  fence: {
    em: '🚧', cat: 'place', lvl: 2, size: 'huge',
    t: {
      es: ['valla', 'VA-ya', 'f'], fr: ['clôture', 'kloh-TUUR', 'f'],
      de: ['Zaun', 'tsown', 'm'], it: ['recinto', 'reh-CHEEN-to', 'm'],
      pt: ['cerca', 'SEHR-ka', 'f'], nl: ['hek', 'hek', 'n'],
      ru: ['забор', 'za-BOR', 'm'], ja: ['柵', 'saku', ''],
      ko: ['울타리', 'ultari', ''], zh: ['栅栏', 'zhà lan', ''],
      hi: ['बाड़', 'baar', 'f'], ar: ['سياج', 'siyaaj', 'm']
    }
  },

  /* ── Food ───────────────────────────────────────────────────────────── */

  bread: {
    em: '🍞', cat: 'food', lvl: 1, size: 'small',
    t: {
      es: ['pan', 'pan', 'm'], fr: ['pain', 'pan', 'm'],
      de: ['Brot', 'broat', 'n'], it: ['pane', 'PA-neh', 'm'],
      pt: ['pão', 'powng', 'm'], nl: ['brood', 'broat', 'n'],
      ru: ['хлеб', 'khlyep', 'm'], ja: ['パン', 'pan', ''],
      ko: ['빵', 'ppang', ''], zh: ['面包', 'miàn bāo', ''],
      hi: ['रोटी', 'ro-tee', 'f'], ar: ['خبز', 'khubz', 'm']
    }
  },
  cheese: {
    em: '🧀', cat: 'food', lvl: 1, size: 'small',
    t: {
      es: ['queso', 'KEH-so', 'm'], fr: ['fromage', 'fro-MAZH', 'm'],
      de: ['Käse', 'KAY-ze', 'm'], it: ['formaggio', 'for-MAD-jo', 'm'],
      pt: ['queijo', 'KAY-zhoo', 'm'], nl: ['kaas', 'kaas', 'c'],
      ru: ['сыр', 'syr', 'm'], ja: ['チーズ', 'chiizu', ''],
      ko: ['치즈', 'chijeu', ''], zh: ['奶酪', 'nǎi lào', ''],
      hi: ['पनीर', 'pa-neer', 'm'], ar: ['جبن', 'jubn', 'm']
    }
  },
  egg: {
    em: '🥚', cat: 'food', lvl: 1, size: 'tiny',
    t: {
      es: ['huevo', 'WEH-vo', 'm'], fr: ['œuf', 'uf', 'm'],
      de: ['Ei', 'ay', 'n'], it: ['uovo', 'WO-vo', 'm'],
      pt: ['ovo', 'O-voo', 'm'], nl: ['ei', 'ay', 'n'],
      ru: ['яйцо', 'yay-TSO', 'n'], ja: ['卵', 'tamago', ''],
      ko: ['달걀', 'dalgyal', ''], zh: ['鸡蛋', 'jī dàn', ''],
      hi: ['अंडा', 'an-daa', 'm'], ar: ['بيضة', 'bayda', 'f']
    }
  },
  lemon: {
    em: '🍋', cat: 'food', lvl: 1, size: 'tiny',
    t: {
      es: ['limón', 'lee-MON', 'm'], fr: ['citron', 'see-TROHN', 'm'],
      de: ['Zitrone', 'tsee-TRO-ne', 'f'], it: ['limone', 'lee-MO-neh', 'm'],
      pt: ['limão', 'lee-MOWN', 'm'], nl: ['citroen', 'see-TROON', 'c'],
      ru: ['лимон', 'lee-MON', 'm'], ja: ['レモン', 'remon', ''],
      ko: ['레몬', 'remon', ''], zh: ['柠檬', 'níng méng', ''],
      hi: ['नींबू', 'neem-boo', 'm'], ar: ['ليمون', 'laymuun', 'm']
    }
  },
  strawberry: {
    em: '🍓', cat: 'food', lvl: 1, size: 'tiny',
    t: {
      es: ['fresa', 'FREH-sa', 'f'], fr: ['fraise', 'frez', 'f'],
      de: ['Erdbeere', 'AYRT-bay-re', 'f'], it: ['fragola', 'FRA-go-la', 'f'],
      pt: ['morango', 'mo-RAN-goo', 'm'], nl: ['aardbei', 'AART-bay', 'c'],
      ru: ['клубника', 'kloob-NEE-ka', 'f'], ja: ['いちご', 'ichigo', ''],
      ko: ['딸기', 'ttalgi', ''], zh: ['草莓', 'cǎo méi', ''],
      hi: ['स्ट्रॉबेरी', 'stro-be-ree', 'f'], ar: ['فراولة', 'faraawla', 'f']
    }
  },
  pineapple: {
    em: '🍍', cat: 'food', lvl: 2, size: 'small',
    t: {
      es: ['piña', 'PEE-nya', 'f'], fr: ['ananas', 'a-na-NA', 'm'],
      de: ['Ananas', 'A-na-nas', 'f'], it: ['ananas', 'A-na-nas', 'm'],
      pt: ['abacaxi', 'a-ba-ka-SHEE', 'm'], nl: ['ananas', 'A-na-nas', 'c'],
      ru: ['ананас', 'a-na-NAS', 'm'], ja: ['パイナップル', 'painappuru', ''],
      ko: ['파인애플', 'painaepeul', ''], zh: ['菠萝', 'bō luó', ''],
      hi: ['अनानास', 'a-naa-naas', 'm'], ar: ['أناناس', 'anaanaas', 'm']
    }
  },
  corn: {
    em: '🌽', cat: 'food', lvl: 2, size: 'small',
    t: {
      es: ['maíz', 'ma-EES', 'm'], fr: ['maïs', 'ma-EES', 'm'],
      de: ['Mais', 'mys', 'm'], it: ['mais', 'MA-ees', 'm'],
      pt: ['milho', 'MEE-lyoo', 'm'], nl: ['maïs', 'mice', 'c'],
      ru: ['кукуруза', 'koo-koo-ROO-za', 'f'], ja: ['とうもろこし', 'toumorokoshi', ''],
      ko: ['옥수수', 'oksusu', ''], zh: ['玉米', 'yù mǐ', ''],
      hi: ['मक्का', 'mak-kaa', 'm'], ar: ['ذرة', 'dhura', 'f']
    }
  },
  cucumber: {
    em: '🥒', cat: 'food', lvl: 2, size: 'small',
    t: {
      es: ['pepino', 'peh-PEE-no', 'm'], fr: ['concombre', 'kohn-KOHM-bruh', 'm'],
      de: ['Gurke', 'GOOR-ke', 'f'], it: ['cetriolo', 'cheh-tree-O-lo', 'm'],
      pt: ['pepino', 'peh-PEE-noo', 'm'], nl: ['komkommer', 'kom-KOM-mer', 'c'],
      ru: ['огурец', 'a-goo-RYETS', 'm'], ja: ['きゅうり', 'kyuuri', ''],
      ko: ['오이', 'oi', ''], zh: ['黄瓜', 'huáng guā', ''],
      hi: ['खीरा', 'khee-raa', 'm'], ar: ['خيار', 'khiyaar', 'm']
    }
  },
  cabbage: {
    em: '🥬', cat: 'food', lvl: 2, size: 'small',
    t: {
      es: ['repollo', 'reh-PO-yo', 'm'], fr: ['chou', 'shoo', 'm'],
      de: ['Kohl', 'koal', 'm'], it: ['cavolo', 'KA-vo-lo', 'm'],
      pt: ['repolho', 'heh-PO-lyoo', 'm'], nl: ['kool', 'koal', 'c'],
      ru: ['капуста', 'ka-POOS-ta', 'f'], ja: ['キャベツ', 'kyabetsu', ''],
      ko: ['양배추', 'yangbaechu', ''], zh: ['卷心菜', 'juǎn xīn cài', ''],
      hi: ['पत्ता गोभी', 'pat-taa go-bhee', 'f'], ar: ['ملفوف', 'malfoof', 'm']
    }
  },
  soup: {
    em: '🍲', cat: 'food', lvl: 2, size: 'small',
    t: {
      es: ['sopa', 'SO-pa', 'f'], fr: ['soupe', 'soop', 'f'],
      de: ['Suppe', 'ZOOP-pe', 'f'], it: ['zuppa', 'TSOOP-pa', 'f'],
      pt: ['sopa', 'SO-pa', 'f'], nl: ['soep', 'soop', 'c'],
      ru: ['суп', 'soop', 'm'], ja: ['スープ', 'suupu', ''],
      ko: ['수프', 'supeu', ''], zh: ['汤', 'tāng', ''],
      hi: ['सूप', 'soop', 'm'], ar: ['حساء', 'hisaa', 'm']
    }
  },
  icecream: {
    em: '🍦', cat: 'food', lvl: 1, size: 'tiny',
    t: {
      es: ['helado', 'eh-LA-do', 'm'], fr: ['glace', 'glas', 'f'],
      de: ['Eis', 'ice', 'n'], it: ['gelato', 'jeh-LA-to', 'm'],
      pt: ['sorvete', 'sor-VEH-chee', 'm'], nl: ['ijs', 'ice', 'n'],
      ru: ['мороженое', 'ma-RO-zhee-na-ye', 'n'], ja: ['アイスクリーム', 'aisukuriimu', ''],
      ko: ['아이스크림', 'aiseukeurim', ''], zh: ['冰淇淋', 'bīng qí lín', ''],
      hi: ['आइसक्रीम', 'aais-kreem', 'f'], ar: ['آيس كريم', 'aayis kreem', 'm']
    }
  },
  hamburger: {
    em: '🍔', cat: 'food', lvl: 1, size: 'small',
    t: {
      es: ['hamburguesa', 'am-boor-GEH-sa', 'f'], fr: ['hamburger', 'am-boor-GAIR', 'm'],
      de: ['Hamburger', 'HAM-boor-ger', 'm'], it: ['hamburger', 'AM-boor-ger', 'm'],
      pt: ['hambúrguer', 'am-BOOR-ger', 'm'], nl: ['hamburger', 'HAM-bur-ger', 'c'],
      ru: ['гамбургер', 'GAM-boor-gyer', 'm'], ja: ['ハンバーガー', 'hanbaagaa', ''],
      ko: ['햄버거', 'haembeogeo', ''], zh: ['汉堡', 'hàn bǎo', ''],
      hi: ['बर्गर', 'bar-gar', 'm'], ar: ['برغر', 'burghur', 'm']
    }
  },
  pretzel: {
    em: '🥨', cat: 'food', lvl: 3, size: 'tiny',
    t: {
      es: ['pretzel', 'PRET-sel', 'm'], fr: ['bretzel', 'bret-ZEL', 'm'],
      de: ['Brezel', 'BRAY-tsel', 'f'], it: ['pretzel', 'PRET-sel', 'm'],
      pt: ['pretzel', 'PRET-sel', 'm'], nl: ['krakeling', 'KRA-ke-ling', 'c'],
      ru: ['крендель', 'KRYEN-dyel', 'm'], ja: ['プレッツェル', 'purettseru', ''],
      ko: ['프레첼', 'peurechel', ''], zh: ['椒盐卷饼', 'jiāo yán juǎn bǐng', ''],
      hi: ['प्रेट्ज़ेल', 'pret-zel', 'm'], ar: ['بريتزل', 'briitzil', 'm']
    }
  },
  coffee: {
    em: '☕', cat: 'food', lvl: 1, size: 'tiny',
    t: {
      es: ['café', 'ka-FEH', 'm'], fr: ['café', 'ka-FAY', 'm'],
      de: ['Kaffee', 'KA-fay', 'm'], it: ['caffè', 'kaf-FEH', 'm'],
      pt: ['café', 'ka-FEH', 'm'], nl: ['koffie', 'KOF-fee', 'c'],
      ru: ['кофе', 'KO-fye', 'm'], ja: ['コーヒー', 'koohii', ''],
      ko: ['커피', 'keopi', ''], zh: ['咖啡', 'kā fēi', ''],
      hi: ['कॉफ़ी', 'ko-fee', 'f'], ar: ['قهوة', 'qahwa', 'f']
    }
  },
  wine: {
    em: '🍷', cat: 'food', lvl: 2, size: 'small',
    t: {
      es: ['vino', 'VEE-no', 'm'], fr: ['vin', 'van', 'm'],
      de: ['Wein', 'vyn', 'm'], it: ['vino', 'VEE-no', 'm'],
      pt: ['vinho', 'VEE-nyoo', 'm'], nl: ['wijn', 'vyn', 'c'],
      ru: ['вино', 'vee-NO', 'n'], ja: ['ワイン', 'wain', ''],
      ko: ['와인', 'wain', ''], zh: ['葡萄酒', 'pú tao jiǔ', ''],
      hi: ['शराब', 'sha-raab', 'f'], ar: ['نبيذ', 'nabeedh', 'm']
    }
  },

  /* ── Transport ──────────────────────────────────────────────────────── */

  ship: {
    em: '🚢', cat: 'transport', lvl: 1, size: 'huge',
    t: {
      es: ['barco', 'BAR-ko', 'm'], fr: ['bateau', 'ba-TOH', 'm'],
      de: ['Schiff', 'shif', 'n'], it: ['nave', 'NA-veh', 'f'],
      pt: ['navio', 'na-VEE-oo', 'm'], nl: ['schip', 'skhip', 'n'],
      ru: ['корабль', 'ka-RABL', 'm'], ja: ['船', 'fune', ''],
      ko: ['배', 'bae', ''], zh: ['船', 'chuán', ''],
      hi: ['जहाज़', 'ja-haaz', 'm'], ar: ['سفينة', 'safeena', 'f']
    }
  },
  tractor: {
    em: '🚜', cat: 'transport', lvl: 2, size: 'huge',
    t: {
      es: ['tractor', 'trak-TOR', 'm'], fr: ['tracteur', 'trak-TUR', 'm'],
      de: ['Traktor', 'TRAK-tor', 'm'], it: ['trattore', 'trat-TO-reh', 'm'],
      pt: ['trator', 'tra-TOR', 'm'], nl: ['tractor', 'TRAK-tor', 'c'],
      ru: ['трактор', 'TRAK-tar', 'm'], ja: ['トラクター', 'torakutaa', ''],
      ko: ['트랙터', 'teuraekteo', ''], zh: ['拖拉机', 'tuō lā jī', ''],
      hi: ['ट्रैक्टर', 'traik-tar', 'm'], ar: ['جرار', 'jarraar', 'm']
    }
  },
  ambulance: {
    em: '🚑', cat: 'transport', lvl: 2, size: 'huge',
    t: {
      es: ['ambulancia', 'am-boo-LAN-sya', 'f'], fr: ['ambulance', 'ahm-bew-LAHNS', 'f'],
      de: ['Krankenwagen', 'KRAN-ken-vaa-gen', 'm'], it: ['ambulanza', 'am-boo-LAN-tsa', 'f'],
      pt: ['ambulância', 'am-boo-LAN-sya', 'f'], nl: ['ambulance', 'am-bew-LAHN-se', 'c'],
      ru: ['скорая помощь', 'SKO-ra-ya PO-mashch', 'f'], ja: ['救急車', 'kyuukyuusha', ''],
      ko: ['구급차', 'gugeupcha', ''], zh: ['救护车', 'jiù hù chē', ''],
      hi: ['एम्बुलेंस', 'em-bu-lens', 'f'], ar: ['سيارة إسعاف', 'sayyaarat isaaf', 'f']
    }
  },
  taxi: {
    em: '🚕', cat: 'transport', lvl: 1, size: 'huge',
    t: {
      es: ['taxi', 'TAK-see', 'm'], fr: ['taxi', 'tak-SEE', 'm'],
      de: ['Taxi', 'TAK-see', 'n'], it: ['taxi', 'TAK-see', 'm'],
      pt: ['táxi', 'TAK-see', 'm'], nl: ['taxi', 'TAK-see', 'c'],
      ru: ['такси', 'tak-SEE', 'n'], ja: ['タクシー', 'takushii', ''],
      ko: ['택시', 'taeksi', ''], zh: ['出租车', 'chū zū chē', ''],
      hi: ['टैक्सी', 'taik-see', 'f'], ar: ['سيارة أجرة', 'sayyaarat ujra', 'f']
    }
  },
  van: {
    em: '🚐', cat: 'transport', lvl: 2, size: 'huge',
    t: {
      es: ['furgoneta', 'foor-go-NEH-ta', 'f'], fr: ['camionnette', 'ka-myo-NET', 'f'],
      de: ['Lieferwagen', 'LEE-fer-vaa-gen', 'm'], it: ['furgone', 'foor-GO-neh', 'm'],
      pt: ['van', 'van', 'f'], nl: ['bestelwagen', 'be-STEL-vaa-khen', 'c'],
      ru: ['фургон', 'foor-GON', 'm'], ja: ['バン', 'ban', ''],
      ko: ['승합차', 'seunghapcha', ''], zh: ['面包车', 'miàn bāo chē', ''],
      hi: ['वैन', 'vain', 'f'], ar: ['شاحنة صغيرة', 'shaahina sagheera', 'f']
    }
  },
  scooter: {
    em: '🛵', cat: 'transport', lvl: 2, size: 'large',
    t: {
      es: ['scooter', 'es-KOO-ter', 'm'], fr: ['scooter', 'skoo-TAIR', 'm'],
      de: ['Roller', 'ROL-ler', 'm'], it: ['scooter', 'SKOO-ter', 'm'],
      pt: ['scooter', 'es-KOO-ter', 'f'], nl: ['scooter', 'SKOO-ter', 'c'],
      ru: ['скутер', 'SKOO-ter', 'm'], ja: ['スクーター', 'sukuutaa', ''],
      ko: ['스쿠터', 'seukuteo', ''], zh: ['踏板车', 'tà bǎn chē', ''],
      hi: ['स्कूटर', 'skoo-tar', 'm'], ar: ['دراجة بخارية', 'darraaja bukhaariyya', 'f']
    }
  },
  balloon: {
    em: '🎈', cat: 'object', lvl: 1, size: 'small',
    t: {
      es: ['globo', 'GLO-bo', 'm'], fr: ['ballon', 'ba-LOHN', 'm'],
      de: ['Luftballon', 'LOOFT-ba-lon', 'm'], it: ['palloncino', 'pal-lon-CHEE-no', 'm'],
      pt: ['balão', 'ba-LOWN', 'm'], nl: ['ballon', 'ba-LON', 'c'],
      ru: ['воздушный шар', 'vaz-DOOSH-ny shar', 'm'], ja: ['風船', 'fuusen', ''],
      ko: ['풍선', 'pungseon', ''], zh: ['气球', 'qì qiú', ''],
      hi: ['गुब्बारा', 'gub-baa-raa', 'm'], ar: ['بالون', 'baaluun', 'm']
    }
  },
  parachute: {
    em: '🪂', cat: 'object', lvl: 3, size: 'huge',
    t: {
      es: ['paracaídas', 'pa-ra-ka-EE-das', 'm'], fr: ['parachute', 'pa-ra-SHEWT', 'm'],
      de: ['Fallschirm', 'FAL-shirm', 'm'], it: ['paracadute', 'pa-ra-ka-DOO-teh', 'm'],
      pt: ['paraquedas', 'pa-ra-KEH-das', 'm'], nl: ['parachute', 'pa-ra-SHEW-te', 'c'],
      ru: ['парашют', 'pa-ra-SHOOT', 'm'], ja: ['パラシュート', 'parashuuto', ''],
      ko: ['낙하산', 'nakhasan', ''], zh: ['降落伞', 'jiàng luò sǎn', ''],
      hi: ['पैराशूट', 'pai-raa-shoot', 'm'], ar: ['مظلة', 'mizalla', 'f']
    }
  },

  /* ── Sport ──────────────────────────────────────────────────────────── */

  basketball: {
    em: '🏀', cat: 'sport', lvl: 1, size: 'small',
    t: {
      es: ['baloncesto', 'ba-lon-SES-to', 'm'], fr: ['basket', 'bas-KET', 'm'],
      de: ['Basketball', 'BAS-ket-bal', 'm'], it: ['pallacanestro', 'pal-la-ka-NES-tro', 'f'],
      pt: ['basquete', 'bas-KEH-chee', 'm'], nl: ['basketbal', 'BAS-ket-bal', 'c'],
      ru: ['баскетбол', 'bas-keet-BOL', 'm'], ja: ['バスケットボール', 'basukettobooru', ''],
      ko: ['농구공', 'nonggugong', ''], zh: ['篮球', 'lán qiú', ''],
      hi: ['बास्केटबॉल', 'baas-ket-bol', 'm'], ar: ['كرة السلة', 'kurat as-salla', 'f']
    }
  },
  volleyball: {
    em: '🏐', cat: 'sport', lvl: 2, size: 'small',
    t: {
      es: ['voleibol', 'vo-lay-BOL', 'm'], fr: ['volley', 'vo-LAY', 'm'],
      de: ['Volleyball', 'VOL-lay-bal', 'm'], it: ['pallavolo', 'pal-la-VO-lo', 'f'],
      pt: ['vôlei', 'VO-lay', 'm'], nl: ['volleybal', 'VOL-lay-bal', 'c'],
      ru: ['волейбол', 'va-leey-BOL', 'm'], ja: ['バレーボール', 'bareebooru', ''],
      ko: ['배구공', 'baegugong', ''], zh: ['排球', 'pái qiú', ''],
      hi: ['वॉलीबॉल', 'vo-lee-bol', 'm'], ar: ['كرة الطائرة', 'kurat at-taa-ira', 'f']
    }
  },
  dumbbell: {
    em: '🏋️', cat: 'sport', lvl: 2, size: 'small',
    t: {
      es: ['mancuerna', 'man-KWEHR-na', 'f'], fr: ['haltère', 'al-TEHR', 'm'],
      de: ['Hantel', 'HAN-tel', 'f'], it: ['manubrio', 'ma-NOO-bryo', 'm'],
      pt: ['halter', 'AL-ter', 'm'], nl: ['halter', 'HAL-ter', 'c'],
      ru: ['гантель', 'gan-TEL', 'f'], ja: ['ダンベル', 'danberu', ''],
      ko: ['아령', 'aryeong', ''], zh: ['哑铃', 'yǎ líng', ''],
      hi: ['डम्बल', 'dam-bal', 'm'], ar: ['دمبل', 'dumbil', 'm']
    }
  },

  /* ── More animals ───────────────────────────────────────────────────── */

  camel: {
    em: '🐫', cat: 'animal', lvl: 2, size: 'large',
    t: {
      es: ['camello', 'ka-MEH-yo', 'm'], fr: ['chameau', 'sha-MOH', 'm'],
      de: ['Kamel', 'ka-MAYL', 'n'], it: ['cammello', 'kam-MEL-lo', 'm'],
      pt: ['camelo', 'ka-MEH-loo', 'm'], nl: ['kameel', 'ka-MAYL', 'c'],
      ru: ['верблюд', 'veerb-LYOOT', 'm'], ja: ['ラクダ', 'rakuda', ''],
      ko: ['낙타', 'nakta', ''], zh: ['骆驼', 'luò tuo', ''],
      hi: ['ऊँट', 'oont', 'm'], ar: ['جمل', 'jamal', 'm']
    }
  },
  hippopotamus: {
    em: '🦛', cat: 'animal', lvl: 3, size: 'large',
    t: {
      es: ['hipopótamo', 'ee-po-PO-ta-mo', 'm'], fr: ['hippopotame', 'ee-po-po-TAM', 'm'],
      de: ['Nilpferd', 'NEEL-pfayrt', 'n'], it: ['ippopotamo', 'eep-po-PO-ta-mo', 'm'],
      pt: ['hipopótamo', 'ee-po-PO-ta-moo', 'm'], nl: ['nijlpaard', 'NYL-paart', 'n'],
      ru: ['бегемот', 'bee-gee-MOT', 'm'], ja: ['カバ', 'kaba', ''],
      ko: ['하마', 'hama', ''], zh: ['河马', 'hé mǎ', ''],
      hi: ['दरियाई घोड़ा', 'da-ri-yaa-ee gho-raa', 'm'], ar: ['فرس النهر', 'faras an-nahr', 'm']
    }
  },
  rhinoceros: {
    em: '🦏', cat: 'animal', lvl: 3, size: 'large',
    t: {
      es: ['rinoceronte', 'ree-no-seh-RON-teh', 'm'], fr: ['rhinocéros', 'ree-no-say-ROS', 'm'],
      de: ['Nashorn', 'NAAS-horn', 'n'], it: ['rinoceronte', 'ree-no-cheh-RON-teh', 'm'],
      pt: ['rinoceronte', 'hee-no-seh-RON-chee', 'm'], nl: ['neushoorn', 'NUS-hoarn', 'c'],
      ru: ['носорог', 'na-sa-ROK', 'm'], ja: ['サイ', 'sai', ''],
      ko: ['코뿔소', 'koppulso', ''], zh: ['犀牛', 'xī niú', ''],
      hi: ['गैंडा', 'gain-daa', 'm'], ar: ['وحيد القرن', 'waheed al-qarn', 'm']
    }
  },
  fox: {
    em: '🦊', cat: 'animal', lvl: 2, size: 'medium',
    t: {
      es: ['zorro', 'SO-rro', 'm'], fr: ['renard', 'ruh-NAR', 'm'],
      de: ['Fuchs', 'fooks', 'm'], it: ['volpe', 'VOL-peh', 'f'],
      pt: ['raposa', 'ha-PO-za', 'f'], nl: ['vos', 'vos', 'c'],
      ru: ['лиса', 'lee-SA', 'f'], ja: ['キツネ', 'kitsune', ''],
      ko: ['여우', 'yeou', ''], zh: ['狐狸', 'hú li', ''],
      hi: ['लोमड़ी', 'lom-ree', 'f'], ar: ['ثعلب', 'tha-lab', 'm']
    }
  },
  wolf: {
    em: '🐺', cat: 'animal', lvl: 2, size: 'medium',
    t: {
      es: ['lobo', 'LO-bo', 'm'], fr: ['loup', 'loo', 'm'],
      de: ['Wolf', 'volf', 'm'], it: ['lupo', 'LOO-po', 'm'],
      pt: ['lobo', 'LO-boo', 'm'], nl: ['wolf', 'volf', 'c'],
      ru: ['волк', 'volk', 'm'], ja: ['オオカミ', 'ookami', ''],
      ko: ['늑대', 'neukdae', ''], zh: ['狼', 'láng', ''],
      hi: ['भेड़िया', 'bhe-ri-yaa', 'm'], ar: ['ذئب', 'dhib', 'm']
    }
  },
  pig: {
    em: '🐷', cat: 'animal', lvl: 1, size: 'medium',
    t: {
      es: ['cerdo', 'SEHR-do', 'm'], fr: ['cochon', 'ko-SHOHN', 'm'],
      de: ['Schwein', 'shvyn', 'n'], it: ['maiale', 'ma-YA-leh', 'm'],
      pt: ['porco', 'POR-koo', 'm'], nl: ['varken', 'VAR-ken', 'n'],
      ru: ['свинья', 'svee-NYA', 'f'], ja: ['豚', 'buta', ''],
      ko: ['돼지', 'dwaeji', ''], zh: ['猪', 'zhū', ''],
      hi: ['सुअर', 'su-ar', 'm'], ar: ['خنزير', 'khinzeer', 'm']
    }
  },
  lizard: {
    em: '🦎', cat: 'animal', lvl: 2, size: 'small',
    t: {
      es: ['lagarto', 'la-GAR-to', 'm'], fr: ['lézard', 'lay-ZAR', 'm'],
      de: ['Eidechse', 'AY-dek-se', 'f'], it: ['lucertola', 'loo-CHEHR-to-la', 'f'],
      pt: ['lagarto', 'la-GAR-too', 'm'], nl: ['hagedis', 'HAA-khe-dis', 'c'],
      ru: ['ящерица', 'YA-shee-reet-sa', 'f'], ja: ['トカゲ', 'tokage', ''],
      ko: ['도마뱀', 'domabaem', ''], zh: ['蜥蜴', 'xī yì', ''],
      hi: ['छिपकली', 'chhip-ka-lee', 'f'], ar: ['سحلية', 'sihliyya', 'f']
    }
  },
  crocodile: {
    em: '🐊', cat: 'animal', lvl: 2, size: 'large',
    t: {
      es: ['cocodrilo', 'ko-ko-DREE-lo', 'm'], fr: ['crocodile', 'kro-ko-DEEL', 'm'],
      de: ['Krokodil', 'kro-ko-DEEL', 'n'], it: ['coccodrillo', 'kok-ko-DREEL-lo', 'm'],
      pt: ['crocodilo', 'kro-ko-DEE-loo', 'm'], nl: ['krokodil', 'kro-ko-DIL', 'c'],
      ru: ['крокодил', 'kra-ka-DEEL', 'm'], ja: ['ワニ', 'wani', ''],
      ko: ['악어', 'ageo', ''], zh: ['鳄鱼', 'è yú', ''],
      hi: ['मगरमच्छ', 'ma-gar-machh', 'm'], ar: ['تمساح', 'timsaah', 'm']
    }
  },
  bee: {
    em: '🐝', cat: 'animal', lvl: 1, size: 'tiny',
    t: {
      es: ['abeja', 'a-BEH-kha', 'f'], fr: ['abeille', 'a-BAY', 'f'],
      de: ['Biene', 'BEE-ne', 'f'], it: ['ape', 'A-peh', 'f'],
      pt: ['abelha', 'a-BEH-lya', 'f'], nl: ['bij', 'by', 'c'],
      ru: ['пчела', 'pchee-LA', 'f'], ja: ['ハチ', 'hachi', ''],
      ko: ['벌', 'beol', ''], zh: ['蜜蜂', 'mì fēng', ''],
      hi: ['मधुमक्खी', 'ma-dhu-mak-khee', 'f'], ar: ['نحلة', 'nahla', 'f']
    }
  },
  ant: {
    em: '🐜', cat: 'animal', lvl: 1, size: 'tiny',
    t: {
      es: ['hormiga', 'or-MEE-ga', 'f'], fr: ['fourmi', 'foor-MEE', 'f'],
      de: ['Ameise', 'A-my-ze', 'f'], it: ['formica', 'for-MEE-ka', 'f'],
      pt: ['formiga', 'for-MEE-ga', 'f'], nl: ['mier', 'meer', 'c'],
      ru: ['муравей', 'moo-ra-VYAY', 'm'], ja: ['アリ', 'ari', ''],
      ko: ['개미', 'gaemi', ''], zh: ['蚂蚁', 'mǎ yǐ', ''],
      hi: ['चींटी', 'cheen-tee', 'f'], ar: ['نملة', 'namla', 'f']
    }
  },
  fly: {
    em: '🪰', cat: 'animal', lvl: 2, size: 'tiny',
    t: {
      es: ['mosca', 'MOS-ka', 'f'], fr: ['mouche', 'moosh', 'f'],
      de: ['Fliege', 'FLEE-ge', 'f'], it: ['mosca', 'MOS-ka', 'f'],
      pt: ['mosca', 'MOS-ka', 'f'], nl: ['vlieg', 'vleekh', 'c'],
      ru: ['муха', 'MOO-kha', 'f'], ja: ['ハエ', 'hae', ''],
      ko: ['파리', 'pari', ''], zh: ['苍蝇', 'cāng ying', ''],
      hi: ['मक्खी', 'mak-khee', 'f'], ar: ['ذبابة', 'dhubaaba', 'f']
    }
  },
  grasshopper: {
    em: '🦗', cat: 'animal', lvl: 3, size: 'tiny',
    t: {
      es: ['saltamontes', 'sal-ta-MON-tes', 'm'], fr: ['sauterelle', 'soh-tuh-REL', 'f'],
      de: ['Grashüpfer', 'GRAAS-huup-fer', 'm'], it: ['cavalletta', 'ka-val-LET-ta', 'f'],
      pt: ['gafanhoto', 'ga-fa-NYO-too', 'm'], nl: ['sprinkhaan', 'SPRINK-haan', 'c'],
      ru: ['кузнечик', 'kooz-NYE-cheek', 'm'], ja: ['バッタ', 'batta', ''],
      ko: ['메뚜기', 'mettugi', ''], zh: ['蚱蜢', 'zhà měng', ''],
      hi: ['टिड्डा', 'tid-daa', 'm'], ar: ['جرادة', 'jaraada', 'f']
    }
  },
  ladybug: {
    em: '🐞', cat: 'animal', lvl: 2, size: 'tiny',
    t: {
      es: ['mariquita', 'ma-ree-KEE-ta', 'f'], fr: ['coccinelle', 'kok-see-NEL', 'f'],
      de: ['Marienkäfer', 'ma-REE-en-kay-fer', 'm'], it: ['coccinella', 'koch-chee-NEL-la', 'f'],
      pt: ['joaninha', 'zho-a-NEEN-ya', 'f'], nl: ['lieveheersbeestje', 'lee-ve-HAYRS-bay-stye', 'n'],
      ru: ['божья коровка', 'BO-zhya ka-ROF-ka', 'f'], ja: ['テントウムシ', 'tentoumushi', ''],
      ko: ['무당벌레', 'mudangbeolle', ''], zh: ['瓢虫', 'piáo chóng', ''],
      hi: ['सोनपंखी', 'son-pan-khee', 'f'], ar: ['دعسوقة', 'da-suuqa', 'f']
    }
  },
  snail: {
    em: '🐌', cat: 'animal', lvl: 2, size: 'tiny',
    t: {
      es: ['caracol', 'ka-ra-KOL', 'm'], fr: ['escargot', 'es-kar-GOH', 'm'],
      de: ['Schnecke', 'SHNEK-ke', 'f'], it: ['lumaca', 'loo-MA-ka', 'f'],
      pt: ['caracol', 'ka-ra-KOL', 'm'], nl: ['slak', 'slak', 'c'],
      ru: ['улитка', 'oo-LEET-ka', 'f'], ja: ['カタツムリ', 'katatsumuri', ''],
      ko: ['달팽이', 'dalpaengi', ''], zh: ['蜗牛', 'wō niú', ''],
      hi: ['घोंघा', 'ghon-ghaa', 'm'], ar: ['حلزون', 'halazoon', 'm']
    }
  },
  crab: {
    em: '🦀', cat: 'animal', lvl: 2, size: 'small',
    t: {
      es: ['cangrejo', 'kan-GREH-kho', 'm'], fr: ['crabe', 'krab', 'm'],
      de: ['Krabbe', 'KRAB-be', 'f'], it: ['granchio', 'GRAN-kyo', 'm'],
      pt: ['caranguejo', 'ka-ran-GEH-zhoo', 'm'], nl: ['krab', 'krap', 'c'],
      ru: ['краб', 'krap', 'm'], ja: ['カニ', 'kani', ''],
      ko: ['게', 'ge', ''], zh: ['螃蟹', 'páng xiè', ''],
      hi: ['केकड़ा', 'ke-ka-raa', 'm'], ar: ['سرطان', 'saratan', 'm']
    }
  },
  starfish: {
    em: '⭐', cat: 'animal', lvl: 3, size: 'tiny',
    t: {
      es: ['estrella de mar', 'es-TREH-ya deh mar', 'f'], fr: ['étoile de mer', 'ay-TWAL duh mehr', 'f'],
      de: ['Seestern', 'ZAY-shtern', 'm'], it: ['stella marina', 'STEL-la ma-REE-na', 'f'],
      pt: ['estrela-do-mar', 'es-TREH-la doo mar', 'f'], nl: ['zeester', 'ZAY-ster', 'c'],
      ru: ['морская звезда', 'mar-SKA-ya zveez-DA', 'f'], ja: ['ヒトデ', 'hitode', ''],
      ko: ['불가사리', 'bulgasari', ''], zh: ['海星', 'hǎi xīng', ''],
      hi: ['तारा मछली', 'taa-raa mach-lee', 'f'], ar: ['نجم البحر', 'najm al-bahr', 'm']
    }
  },
  jellyfish: {
    em: '🪼', cat: 'animal', lvl: 3, size: 'small',
    t: {
      es: ['medusa', 'meh-DOO-sa', 'f'], fr: ['méduse', 'may-DEWZ', 'f'],
      de: ['Qualle', 'KVAL-le', 'f'], it: ['medusa', 'meh-DOO-za', 'f'],
      pt: ['água-viva', 'A-gwa VEE-va', 'f'], nl: ['kwal', 'kval', 'c'],
      ru: ['медуза', 'mee-DOO-za', 'f'], ja: ['クラゲ', 'kurage', ''],
      ko: ['해파리', 'haepari', ''], zh: ['水母', 'shuǐ mǔ', ''],
      hi: ['जेलीफ़िश', 'je-lee-fish', 'f'], ar: ['قنديل البحر', 'qindeel al-bahr', 'm']
    }
  },

  /* ── Appliances and more home ───────────────────────────────────────── */

  'washing machine': {
    em: '🧺', cat: 'appliance', lvl: 2, size: 'large',
    t: {
      es: ['lavadora', 'la-va-DO-ra', 'f'], fr: ['machine à laver', 'ma-SHEEN a la-VAY', 'f'],
      de: ['Waschmaschine', 'VASH-ma-shee-ne', 'f'], it: ['lavatrice', 'la-va-TREE-cheh', 'f'],
      pt: ['máquina de lavar', 'MA-kee-na jee la-VAR', 'f'], nl: ['wasmachine', 'VAS-ma-shee-ne', 'c'],
      ru: ['стиральная машина', 'stee-RAL-na-ya ma-SHEE-na', 'f'], ja: ['洗濯機', 'sentakuki', ''],
      ko: ['세탁기', 'setakgi', ''], zh: ['洗衣机', 'xǐ yī jī', ''],
      hi: ['वॉशिंग मशीन', 'vo-shing ma-sheen', 'f'], ar: ['غسالة', 'ghassaala', 'f']
    }
  },
  dishwasher: {
    em: '🍽️', cat: 'appliance', lvl: 3, size: 'large',
    t: {
      es: ['lavavajillas', 'la-va-va-KHEE-yas', 'm'], fr: ['lave-vaisselle', 'lav-vay-SEL', 'm'],
      de: ['Spülmaschine', 'SHPUUL-ma-shee-ne', 'f'], it: ['lavastoviglie', 'la-va-sto-VEE-lyeh', 'f'],
      pt: ['lava-louças', 'LA-va LOH-sas', 'f'], nl: ['vaatwasser', 'VAAT-vas-ser', 'c'],
      ru: ['посудомоечная машина', 'pa-soo-da-MO-eech-na-ya ma-SHEE-na', 'f'], ja: ['食洗機', 'shokusenki', ''],
      ko: ['식기세척기', 'sikgisecheokgi', ''], zh: ['洗碗机', 'xǐ wǎn jī', ''],
      hi: ['बर्तन धोने की मशीन', 'bar-tan dho-ne kee ma-sheen', 'f'], ar: ['غسالة صحون', 'ghassaalat suhoon', 'f']
    }
  },
  radio: {
    em: '📻', cat: 'electronics', lvl: 1, size: 'small',
    t: {
      es: ['radio', 'RA-dyo', 'f'], fr: ['radio', 'ra-DYO', 'f'],
      de: ['Radio', 'RA-dyo', 'n'], it: ['radio', 'RA-dyo', 'f'],
      pt: ['rádio', 'HA-dyoo', 'm'], nl: ['radio', 'RA-dee-yo', 'c'],
      ru: ['радио', 'RA-dee-a', 'n'], ja: ['ラジオ', 'rajio', ''],
      ko: ['라디오', 'radio', ''], zh: ['收音机', 'shōu yīn jī', ''],
      hi: ['रेडियो', 're-di-yo', 'm'], ar: ['راديو', 'raadyo', 'm']
    }
  },
  fan: {
    em: '🌀', cat: 'appliance', lvl: 2, size: 'medium',
    t: {
      es: ['ventilador', 'ven-tee-la-DOR', 'm'], fr: ['ventilateur', 'vahn-tee-la-TUR', 'm'],
      de: ['Ventilator', 'ven-tee-LAA-tor', 'm'], it: ['ventilatore', 'ven-tee-la-TO-reh', 'm'],
      pt: ['ventilador', 'ven-chee-la-DOR', 'm'], nl: ['ventilator', 'ven-tee-LAA-tor', 'c'],
      ru: ['вентилятор', 'veen-tee-LYA-tar', 'm'], ja: ['扇風機', 'senpuuki', ''],
      ko: ['선풍기', 'seonpunggi', ''], zh: ['风扇', 'fēng shàn', ''],
      hi: ['पंखा', 'pan-khaa', 'm'], ar: ['مروحة', 'mirwaha', 'f']
    }
  },
  radiator: {
    em: '🔥', cat: 'appliance', lvl: 3, size: 'medium',
    t: {
      es: ['radiador', 'ra-dya-DOR', 'm'], fr: ['radiateur', 'ra-dya-TUR', 'm'],
      de: ['Heizkörper', 'HYTS-kur-per', 'm'], it: ['radiatore', 'ra-dya-TO-reh', 'm'],
      pt: ['radiador', 'ha-dya-DOR', 'm'], nl: ['radiator', 'ra-dee-YAA-tor', 'c'],
      ru: ['радиатор', 'ra-dee-A-tar', 'm'], ja: ['ラジエーター', 'rajieetaa', ''],
      ko: ['라디에이터', 'radieiteo', ''], zh: ['暖气片', 'nuǎn qì piàn', ''],
      hi: ['रेडिएटर', 're-di-e-tar', 'm'], ar: ['مشعاع', 'mishaa', 'm']
    }
  },
  telephone: {
    em: '☎️', cat: 'electronics', lvl: 1, size: 'small',
    t: {
      es: ['teléfono', 'teh-LEH-fo-no', 'm'], fr: ['téléphone', 'tay-lay-FON', 'm'],
      de: ['Telefon', 'teh-leh-FOAN', 'n'], it: ['telefono', 'teh-LEH-fo-no', 'm'],
      pt: ['telefone', 'teh-leh-FO-nee', 'm'], nl: ['telefoon', 'tay-le-FOAN', 'c'],
      ru: ['телефон', 'tee-lee-FON', 'm'], ja: ['電話', 'denwa', ''],
      ko: ['전화', 'jeonhwa', ''], zh: ['电话', 'diàn huà', ''],
      hi: ['टेलीफ़ोन', 'te-lee-fon', 'm'], ar: ['هاتف', 'haatif', 'm']
    }
  },
  speaker: {
    em: '🔊', cat: 'electronics', lvl: 2, size: 'medium',
    t: {
      es: ['altavoz', 'al-ta-VOS', 'm'], fr: ['haut-parleur', 'oh-par-LUR', 'm'],
      de: ['Lautsprecher', 'LOWT-shprekh-er', 'm'], it: ['altoparlante', 'al-to-par-LAN-teh', 'm'],
      pt: ['alto-falante', 'AL-too fa-LAN-chee', 'm'], nl: ['luidspreker', 'LOWT-spray-ker', 'c'],
      ru: ['динамик', 'dee-NA-meek', 'm'], ja: ['スピーカー', 'supiikaa', ''],
      ko: ['스피커', 'seupikeo', ''], zh: ['音箱', 'yīn xiāng', ''],
      hi: ['स्पीकर', 'spee-kar', 'm'], ar: ['مكبر صوت', 'mukabbir sawt', 'm']
    }
  },
  microphone: {
    em: '🎤', cat: 'electronics', lvl: 2, size: 'small',
    t: {
      es: ['micrófono', 'mee-KRO-fo-no', 'm'], fr: ['microphone', 'mee-kro-FON', 'm'],
      de: ['Mikrofon', 'mee-kro-FOAN', 'n'], it: ['microfono', 'mee-KRO-fo-no', 'm'],
      pt: ['microfone', 'mee-kro-FO-nee', 'm'], nl: ['microfoon', 'mee-kro-FOAN', 'c'],
      ru: ['микрофон', 'mee-kra-FON', 'm'], ja: ['マイク', 'maiku', ''],
      ko: ['마이크', 'maikeu', ''], zh: ['麦克风', 'mài kè fēng', ''],
      hi: ['माइक्रोफ़ोन', 'maai-kro-fon', 'm'], ar: ['ميكروفون', 'meekrofoon', 'm']
    }
  },
  microscope: {
    em: '🔬', cat: 'school', lvl: 3, size: 'medium',
    t: {
      es: ['microscopio', 'mee-kro-SKO-pyo', 'm'], fr: ['microscope', 'mee-kro-SKOP', 'm'],
      de: ['Mikroskop', 'mee-kro-SKOAP', 'n'], it: ['microscopio', 'mee-kro-SKO-pyo', 'm'],
      pt: ['microscópio', 'mee-kros-KO-pyoo', 'm'], nl: ['microscoop', 'mee-kro-SKOAP', 'c'],
      ru: ['микроскоп', 'mee-kra-SKOP', 'm'], ja: ['顕微鏡', 'kenbikyou', ''],
      ko: ['현미경', 'hyeonmigyeong', ''], zh: ['显微镜', 'xiǎn wēi jìng', ''],
      hi: ['सूक्ष्मदर्शी', 'sookshm-dar-shee', 'm'], ar: ['مجهر', 'mijhar', 'm']
    }
  },
  flag: {
    em: '🚩', cat: 'object', lvl: 1, size: 'medium',
    t: {
      es: ['bandera', 'ban-DEH-ra', 'f'], fr: ['drapeau', 'dra-POH', 'm'],
      de: ['Fahne', 'FAA-ne', 'f'], it: ['bandiera', 'ban-DYEH-ra', 'f'],
      pt: ['bandeira', 'ban-DAY-ra', 'f'], nl: ['vlag', 'vlakh', 'c'],
      ru: ['флаг', 'flak', 'm'], ja: ['旗', 'hata', ''],
      ko: ['깃발', 'gitbal', ''], zh: ['旗子', 'qí zi', ''],
      hi: ['झंडा', 'jhan-daa', 'm'], ar: ['علم', 'alam', 'm']
    }
  },
  'pencil case': {
    em: '🎒', cat: 'school', lvl: 2, size: 'small',
    t: {
      es: ['estuche', 'es-TOO-cheh', 'm'], fr: ['trousse', 'troos', 'f'],
      de: ['Federmäppchen', 'FAY-der-mep-khen', 'n'], it: ['astuccio', 'as-TOOCH-cho', 'm'],
      pt: ['estojo', 'es-TO-zhoo', 'm'], nl: ['etui', 'ay-TVEE', 'n'],
      ru: ['пенал', 'pee-NAL', 'm'], ja: ['筆箱', 'fudebako', ''],
      ko: ['필통', 'piltong', ''], zh: ['铅笔盒', 'qiān bǐ hé', ''],
      hi: ['पेंसिल बॉक्स', 'pen-sil boks', 'm'], ar: ['مقلمة', 'miqlama', 'f']
    }
  },
  cart: {
    em: '🛒', cat: 'transport', lvl: 2, size: 'large',
    t: {
      es: ['carro', 'KA-rro', 'm'], fr: ['chariot', 'sha-RYOH', 'm'],
      de: ['Wagen', 'VAA-gen', 'm'], it: ['carrello', 'kar-REL-lo', 'm'],
      pt: ['carrinho', 'ka-HEEN-yoo', 'm'], nl: ['kar', 'kar', 'c'],
      ru: ['тележка', 'tee-LYESH-ka', 'f'], ja: ['カート', 'kaato', ''],
      ko: ['수레', 'sure', ''], zh: ['推车', 'tuī chē', ''],
      hi: ['गाड़ी', 'gaa-ree', 'f'], ar: ['عربة', 'araba', 'f']
    }
  },
  tent: {
    em: '⛺', cat: 'object', lvl: 2, size: 'large',
    t: {
      es: ['tienda', 'TYEN-da', 'f'], fr: ['tente', 'tahnt', 'f'],
      de: ['Zelt', 'tselt', 'n'], it: ['tenda', 'TEN-da', 'f'],
      pt: ['barraca', 'ba-HA-ka', 'f'], nl: ['tent', 'tent', 'c'],
      ru: ['палатка', 'pa-LAT-ka', 'f'], ja: ['テント', 'tento', ''],
      ko: ['텐트', 'tenteu', ''], zh: ['帐篷', 'zhàng peng', ''],
      hi: ['तंबू', 'tam-boo', 'm'], ar: ['خيمة', 'khayma', 'f']
    }
  },
  tool: {
    em: '🔧', cat: 'tool', lvl: 2, size: 'small',
    t: {
      es: ['herramienta', 'eh-rra-MYEN-ta', 'f'], fr: ['outil', 'oo-TEE', 'm'],
      de: ['Werkzeug', 'VERK-tsoyk', 'n'], it: ['attrezzo', 'at-TRET-tso', 'm'],
      pt: ['ferramenta', 'feh-ha-MEN-ta', 'f'], nl: ['gereedschap', 'khe-RAYT-skhap', 'n'],
      ru: ['инструмент', 'een-stroo-MYENT', 'm'], ja: ['道具', 'dougu', ''],
      ko: ['도구', 'dogu', ''], zh: ['工具', 'gōng jù', ''],
      hi: ['औज़ार', 'ow-zaar', 'm'], ar: ['أداة', 'adaat', 'f']
    }
  },
  food: {
    em: '🍽️', cat: 'food', lvl: 1, size: 'small',
    t: {
      es: ['comida', 'ko-MEE-da', 'f'], fr: ['nourriture', 'noo-ree-TUUR', 'f'],
      de: ['Essen', 'ES-sen', 'n'], it: ['cibo', 'CHEE-bo', 'm'],
      pt: ['comida', 'ko-MEE-da', 'f'], nl: ['eten', 'AY-ten', 'n'],
      ru: ['еда', 'ye-DA', 'f'], ja: ['食べ物', 'tabemono', ''],
      ko: ['음식', 'eumsik', ''], zh: ['食物', 'shí wù', ''],
      hi: ['खाना', 'khaa-naa', 'm'], ar: ['طعام', 'ta-aam', 'm']
    }
  }
};
