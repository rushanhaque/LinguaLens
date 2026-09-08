/**
 * imagenet.js — Route MobileNet's ImageNet labels onto Lemma's vocabulary.
 *
 * MobileNet reports the raw ImageNet-1k class name, which is a comma-separated
 * synonym list written for researchers rather than learners:
 *
 *   "ballpoint, ballpoint pen, ballpen, Biro"
 *   "Cardigan, Cardigan Welsh corgi"
 *   "notebook, notebook computer"
 *
 * Three problems follow. The labels are too specific (a hundred and twenty dog
 * breeds where a learner wants "dog"), sometimes ambiguous across senses
 * ("cardigan" is both a sweater and a corgi; "notebook" is a laptop, not a
 * pad), and never match our keys verbatim. This module resolves all three.
 *
 * Matching runs in three passes, most specific first:
 *
 *   1. EXACT      the full normalised label, for the ambiguous cases that must
 *                 be pinned down before any looser rule can claim them
 *   2. HEAD       the first synonym, which is ImageNet's canonical name
 *   3. RULES      ordered patterns that fold whole families onto one word
 *
 * A label that matches nothing is dropped rather than guessed at — showing a
 * learner the wrong word is worse than showing them none.
 */

/* ── 1. Exact whole-label overrides ───────────────────────────────────────
   Only for labels whose head word would otherwise be captured wrongly. */
const EXACT = {
  'notebook, notebook computer': 'laptop',
  'cardigan, cardigan welsh corgi': 'dog',
  'sunglasses, dark glasses, shades': 'glasses',
  'mouse, computer mouse': 'mouse',
  'screen, crt screen': 'monitor',
  'monitor': 'monitor',
  'ear, spike, capitulum': 'corn',
  'pot, flowerpot': 'potted plant',
  'crane': 'bird'
};

/* ── 2. Head-word map ─────────────────────────────────────────────────────
   Keyed on ImageNet's first synonym, lowercased. This carries the bulk of
   the mapping: every entry here is a label we can name confidently. */
const HEAD = {
  /* School, office, desk */
  ballpoint: 'pen', 'fountain pen': 'pen', quill: 'pen',
  'rubber eraser': 'eraser', 'pencil sharpener': 'sharpener',
  'pencil box': 'pencil case', binder: 'folder', envelope: 'envelope',
  rule: 'ruler', 'slide rule': 'ruler', abacus: 'calculator',
  'hand calculator': 'calculator', calculator: 'calculator',
  desk: 'desk', 'desktop computer': 'computer', 'computer keyboard': 'keyboard',
  typewriter: 'computer', 'space bar': 'keyboard', printer: 'printer',
  photocopier: 'printer', projector: 'projector', bookcase: 'bookshelf',
  library: 'bookshelf', 'book jacket': 'book', 'comic book': 'book',
  menu: 'paper', 'crossword puzzle': 'paper', 'paper towel': 'paper',
  microphone: 'microphone', 'loudspeaker': 'speaker',
  'hand-held computer': 'tablet', ipod: 'tablet', 'cellular telephone': 'cell phone',
  'dial telephone': 'telephone', 'pay-phone': 'telephone',
  microscope: 'microscope', 'flagpole': 'flag',

  /* Home */
  'table lamp': 'lamp', lampshade: 'lamp', spotlight: 'lamp', candle: 'candle',
  'sliding door': 'door', shoji: 'door', 'window shade': 'curtain',
  'window screen': 'window', 'theater curtain': 'curtain', 'shower curtain': 'curtain',
  'mosquito net': 'curtain', quilt: 'blanket', pillow: 'pillow',
  'four-poster': 'bed', crib: 'bed', cradle: 'bed', 'studio couch': 'couch',
  wardrobe: 'bookshelf', chiffonier: 'bookshelf', 'china cabinet': 'bookshelf',
  'medicine chest': 'bookshelf', 'file': 'folder',
  'rocking chair': 'chair', 'folding chair': 'chair', 'barber chair': 'chair',
  throne: 'chair', 'park bench': 'bench',
  bathtub: 'sink', washbasin: 'sink', 'toilet seat': 'toilet',
  'soap dispenser': 'soap', 'toilet tissue': 'paper', 'hand blower': 'hair drier',
  broom: 'broom', mop: 'broom', swab: 'broom', bucket: 'bucket',
  vacuum: 'broom', 'washer': 'washing machine', dishwasher: 'dishwasher',
  'espresso maker': 'coffee', coffeepot: 'teapot', teapot: 'teapot',
  'frying pan': 'pan', wok: 'pan', skillet: 'pan', caldron: 'pot',
  'dutch oven': 'pot', 'crock pot': 'pot', 'measuring cup': 'cup',
  'mixing bowl': 'bowl', 'soup bowl': 'bowl', tray: 'tray', plate: 'plate',
  'coffee mug': 'mug', 'beer glass': 'wine glass', goblet: 'wine glass',
  'water jug': 'bottle', 'whiskey jug': 'bottle', pitcher: 'bottle',
  'water bottle': 'bottle', 'pop bottle': 'bottle', 'beer bottle': 'bottle',
  'wine bottle': 'bottle', 'pill bottle': 'bottle',
  spatula: 'spoon', ladle: 'spoon', 'wooden spoon': 'spoon', strainer: 'bowl',
  corkscrew: 'tool', 'can opener': 'tool',
  mirror: 'mirror', 'hand mirror': 'mirror',
  'electric fan': 'fan', radiator: 'radiator', 'space heater': 'radiator',
  'home theater': 'tv', television: 'tv', 'entertainment center': 'tv',
  'radio': 'radio', 'cassette player': 'radio', 'tape player': 'radio',
  'cd player': 'radio', 'cassette': 'radio',
  'digital clock': 'clock', 'wall clock': 'clock', 'analog clock': 'clock',
  'digital watch': 'watch', stopwatch: 'watch', hourglass: 'clock',
  sundial: 'clock', 'parking meter': 'parking meter',
  'shopping basket': 'basket', hamper: 'basket', 'shopping cart': 'basket',
  crate: 'box', carton: 'box', chest: 'box', 'packet': 'box',
  'safe': 'box', 'piggy bank': 'box',
  'combination lock': 'lock', padlock: 'lock',
  'purse': 'wallet', wallet: 'wallet', briefcase: 'suitcase', mailbag: 'backpack',
  'plastic bag': 'box', 'vase': 'vase',

  /* Clothing */
  jersey: 'shirt', sweatshirt: 'shirt', 'lab coat': 'coat', 'trench coat': 'coat',
  'fur coat': 'coat', poncho: 'coat', kimono: 'dress', abaya: 'dress',
  gown: 'dress', 'miniskirt': 'dress', 'jean': 'trousers',
  'running shoe': 'shoe', loafer: 'shoe', sandal: 'shoe', clog: 'shoe',
  'cowboy boot': 'shoe', 'sock': 'sock', 'christmas stocking': 'sock',
  sombrero: 'hat', 'cowboy hat': 'hat', bonnet: 'hat', 'bearskin': 'hat',
  'shower cap': 'hat', mortarboard: 'hat', 'crash helmet': 'helmet',
  'football helmet': 'helmet', mitten: 'glove', 'windsor tie': 'tie',
  'bow tie': 'tie', bolo: 'tie', necklace: 'necklace', 'hair slide': 'ring',
  'sunglass': 'glasses', umbrella: 'umbrella', backpack: 'backpack',
  suit: 'coat', 'academic gown': 'coat', apron: 'coat',

  /* Music */
  'electric guitar': 'guitar', 'acoustic guitar': 'guitar', banjo: 'guitar',
  'grand piano': 'piano', 'upright': 'piano', organ: 'piano',
  violin: 'violin', cello: 'violin', harp: 'violin',
  drum: 'drum', 'bass drum': 'drum', gong: 'drum', maraca: 'drum',
  chime: 'drum', 'steel drum': 'drum', marimba: 'piano',
  flute: 'flute', panpipe: 'flute', ocarina: 'flute', oboe: 'flute',
  bassoon: 'flute', 'cornet': 'trumpet', trombone: 'trumpet',
  'french horn': 'trumpet', sax: 'trumpet', harmonica: 'flute',
  accordion: 'piano',

  /* Tools */
  hammer: 'hammer', hatchet: 'hammer', screwdriver: 'screwdriver',
  'power drill': 'screwdriver', screw: 'screwdriver', nail: 'hammer',
  'paintbrush': 'paintbrush', shovel: 'tool', plow: 'tool',
  'lawn mower': 'tool', chain: 'tool', 'safety pin': 'tool',
  hook: 'tool', buckle: 'tool', thimble: 'tool', plunger: 'tool',
  binoculars: 'binoculars', loupe: 'binoculars', ladder: 'ladder',
  'matchstick': 'candle', lighter: 'candle', torch: 'lamp',
  scale: 'tool', barometer: 'tool', 'magnetic compass': 'tool',
  syringe: 'tool', stethoscope: 'tool', 'band aid': 'tool',
  beaker: 'cup', 'petri dish': 'plate', 'tripod': 'tool',
  'reflex camera': 'camera', 'polaroid camera': 'camera', lens: 'camera',

  /* Nature */
  daisy: 'flower', 'yellow lady’s slipper': 'flower', 'rapeseed': 'flower',
  hip: 'flower', 'buckeye': 'tree', acorn: 'tree', 'coral fungus': 'mushroom',
  agaric: 'mushroom', gyromitra: 'mushroom', stinkhorn: 'mushroom',
  earthstar: 'mushroom', 'hen-of-the-woods': 'mushroom', bolete: 'mushroom',
  volcano: 'volcano', seashore: 'beach', sandbar: 'beach', lakeside: 'beach',
  'coral reef': 'beach', alp: 'mountain', cliff: 'mountain', valley: 'mountain',
  promontory: 'mountain', geyser: 'mountain',

  /* Places */
  church: 'church', mosque: 'church', monastery: 'church', 'bell cote': 'church',
  stupa: 'church', dome: 'church', castle: 'castle', palace: 'castle',
  'triumphal arch': 'castle', obelisk: 'castle', 'totem pole': 'castle',
  megalith: 'castle', 'cliff dwelling': 'house', yurt: 'house',
  barn: 'house', boathouse: 'house', greenhouse: 'house', planetarium: 'house',
  'mobile home': 'house', 'beacon': 'castle', prison: 'house',
  restaurant: 'house', 'grocery store': 'house', bookshop: 'house',
  barbershop: 'house', 'butcher shop': 'house', confectionery: 'house',
  'shoe shop': 'house', toyshop: 'house', 'tobacco shop': 'house',
  cinema: 'house', 'suspension bridge': 'bridge', viaduct: 'bridge',
  'steel arch bridge': 'bridge', pier: 'bridge', dam: 'bridge',
  fountain: 'fountain', 'picket fence': 'fence', 'chainlink fence': 'fence',
  'worm fence': 'fence', 'stone wall': 'fence', turnstile: 'fence',
  'traffic light': 'traffic light', 'street sign': 'stop sign',
  'solar dish': 'tool', birdhouse: 'house',
  mailbox: 'box', swing: 'bench', 'scoreboard': 'monitor',

  /* Food */
  'granny smith': 'apple', orange: 'orange', lemon: 'lemon', banana: 'banana',
  strawberry: 'strawberry', pineapple: 'pineapple', fig: 'food',
  pomegranate: 'food', jackfruit: 'food', 'custard apple': 'food',
  corn: 'corn', cucumber: 'cucumber', 'bell pepper': 'food',
  zucchini: 'cucumber', 'spaghetti squash': 'food', 'acorn squash': 'food',
  'butternut squash': 'food', cardoon: 'food', mushroom: 'mushroom',
  artichoke: 'food', 'head cabbage': 'cabbage', broccoli: 'broccoli',
  cauliflower: 'cabbage', 'hotdog': 'hot dog', pizza: 'pizza',
  cheeseburger: 'hamburger', bagel: 'bread', pretzel: 'pretzel',
  'french loaf': 'bread', dough: 'bread', 'meat loaf': 'food',
  'mashed potato': 'food', guacamole: 'food', consomme: 'soup',
  'hot pot': 'soup', trifle: 'cake', 'ice cream': 'icecream',
  'ice lolly': 'icecream', 'chocolate sauce': 'food', carbonara: 'food',
  burrito: 'sandwich', potpie: 'cake', 'red wine': 'wine', espresso: 'coffee',
  cup: 'cup', eggnog: 'wine', honeycomb: 'food', 'french fries': 'food',

  /* Transport */
  'sports car': 'car', convertible: 'car', cab: 'taxi', jeep: 'car',
  limousine: 'car', 'beach wagon': 'car', racer: 'car', 'model t': 'car',
  minivan: 'van', 'police van': 'van', 'moving van': 'truck',
  pickup: 'truck', 'tow truck': 'truck', 'trailer truck': 'truck',
  'garbage truck': 'truck', 'fire engine': 'truck', ambulance: 'ambulance',
  'school bus': 'bus', trolleybus: 'bus', minibus: 'bus',
  moped: 'scooter', 'motor scooter': 'scooter', 'mountain bike': 'bicycle',
  'bicycle-built-for-two': 'bicycle', unicycle: 'bicycle', tricycle: 'bicycle',
  'freight car': 'train', 'passenger car': 'train', 'electric locomotive': 'train',
  'steam locomotive': 'train', streetcar: 'train', 'bullet train': 'train',
  airliner: 'airplane', warplane: 'airplane', airship: 'balloon',
  balloon: 'balloon', parachute: 'parachute', 'space shuttle': 'airplane',
  canoe: 'boat', lifeboat: 'boat', speedboat: 'boat', fireboat: 'boat',
  gondola: 'boat', yawl: 'boat', schooner: 'boat', trimaran: 'boat',
  catamaran: 'boat', 'container ship': 'ship', liner: 'ship', pirate: 'ship',
  'aircraft carrier': 'ship', submarine: 'ship', wreck: 'ship',
  forklift: 'tractor', tractor: 'tractor', harvester: 'tractor',
  thresher: 'tractor', snowplow: 'truck', golfcart: 'car', 'go-kart': 'car',
  'horse cart': 'cart', oxcart: 'cart', jinrikisha: 'cart', barrow: 'cart',
  snowmobile: 'scooter', 'mountain tent': 'tent',

  /* Sport */
  'soccer ball': 'sports ball', basketball: 'basketball', volleyball: 'volleyball',
  'rugby ball': 'sports ball', 'tennis ball': 'sports ball',
  'golf ball': 'sports ball', 'ping-pong ball': 'sports ball',
  'croquet ball': 'sports ball', baseball: 'sports ball',
  dumbbell: 'dumbbell', barbell: 'dumbbell', 'punching bag': 'dumbbell',
  racket: 'tennis racket', ski: 'skis',

  /* Animals that deserve their own word */
  'giant panda': 'bear', 'lesser panda': 'bear', 'ice bear': 'bear',
  'brown bear': 'bear', 'american black bear': 'bear', 'sloth bear': 'bear',
  lion: 'lion', tiger: 'tiger', cheetah: 'tiger', leopard: 'tiger',
  jaguar: 'tiger', 'snow leopard': 'tiger', lynx: 'cat', cougar: 'lion',
  gorilla: 'monkey', chimpanzee: 'monkey', orangutan: 'monkey',
  gibbon: 'monkey', siamang: 'monkey', macaque: 'monkey', baboon: 'monkey',
  langur: 'monkey', 'proboscis monkey': 'monkey', marmoset: 'monkey',
  'capuchin': 'monkey', howler: 'monkey', titi: 'monkey', saki: 'monkey',
  'spider monkey': 'monkey', 'squirrel monkey': 'monkey', madagascar_cat: 'monkey',
  indri: 'monkey', koala: 'bear', wombat: 'bear', wallaby: 'bear',
  'african elephant': 'elephant', 'indian elephant': 'elephant', tusker: 'elephant',
  zebra: 'zebra', 'sorrel': 'horse', ox: 'cow', 'water buffalo': 'cow',
  bison: 'cow', bighorn: 'sheep', ram: 'sheep', ibex: 'sheep',
  'hartebeest': 'cow', impala: 'cow', gazelle: 'cow',
  'arabian camel': 'camel', llama: 'camel', hippopotamus: 'hippopotamus',
  'indian rhinoceros': 'rhinoceros', warthog: 'pig', hog: 'pig', boar: 'pig',
  'wild boar': 'pig', 'guinea pig': 'rabbit', hamster: 'rabbit',
  'wood rabbit': 'rabbit', hare: 'rabbit', angora: 'rabbit',
  porcupine: 'rabbit', hedgehog: 'rabbit', beaver: 'rabbit', otter: 'rabbit',
  skunk: 'rabbit', badger: 'rabbit', weasel: 'rabbit', mink: 'rabbit',
  polecat: 'rabbit', 'black-footed ferret': 'rabbit', mongoose: 'rabbit',
  meerkat: 'rabbit', marmot: 'rabbit', 'fox squirrel': 'rabbit',
  'red fox': 'fox', 'kit fox': 'fox', 'arctic fox': 'fox', 'grey fox': 'fox',
  'timber wolf': 'wolf', 'white wolf': 'wolf', 'red wolf': 'wolf',
  coyote: 'wolf', dingo: 'dog', dhole: 'dog', 'african hunting dog': 'dog',
  'tabby': 'cat', 'tiger cat': 'cat', 'persian cat': 'cat', 'siamese cat': 'cat',
  'egyptian cat': 'cat', 'madagascar cat': 'monkey',

  /* Birds */
  cock: 'bird', hen: 'bird', ostrich: 'bird', brambling: 'bird',
  goldfinch: 'bird', 'house finch': 'bird', junco: 'bird', 'indigo bunting': 'bird',
  robin: 'bird', bulbul: 'bird', jay: 'bird', magpie: 'bird', chickadee: 'bird',
  'water ouzel': 'bird', kite: 'bird', 'bald eagle': 'bird', vulture: 'bird',
  'great grey owl': 'owl', 'black grouse': 'bird', ptarmigan: 'bird',
  'ruffed grouse': 'bird', 'prairie chicken': 'bird', peacock: 'peacock',
  quail: 'bird', partridge: 'bird', 'african grey': 'bird', macaw: 'bird',
  'sulphur-crested cockatoo': 'bird', lorikeet: 'bird', coucal: 'bird',
  'bee eater': 'bird', hornbill: 'bird', hummingbird: 'bird', jacamar: 'bird',
  toucan: 'bird', drake: 'bird', 'red-breasted merganser': 'bird',
  goose: 'bird', 'black swan': 'bird', 'white stork': 'bird',
  'black stork': 'bird', spoonbill: 'bird', flamingo: 'bird',
  'little blue heron': 'bird', 'american egret': 'bird', bittern: 'bird',
  limpkin: 'bird', 'european gallinule': 'bird', 'american coot': 'bird',
  bustard: 'bird', 'ruddy turnstone': 'bird', 'red-backed sandpiper': 'bird',
  redshank: 'bird', dowitcher: 'bird', oystercatcher: 'bird', pelican: 'bird',
  'king penguin': 'penguin', albatross: 'bird',

  /* Water life, reptiles, insects */
  goldfish: 'fish', 'great white shark': 'fish', 'tiger shark': 'fish',
  hammerhead: 'fish', 'electric ray': 'fish', stingray: 'fish',
  'barracouta': 'fish', eel: 'fish', coho: 'fish', 'rock beauty': 'tropical fish',
  anemone_fish: 'tropical fish', 'anemone fish': 'tropical fish',
  sturgeon: 'fish', gar: 'fish', lionfish: 'tropical fish', puffer: 'tropical fish',
  tench: 'fish', 'grey whale': 'fish', 'killer whale': 'fish', dugong: 'fish',
  'sea lion': 'fish',
  loggerhead: 'turtle', 'leatherback turtle': 'turtle', 'mud turtle': 'turtle',
  terrapin: 'turtle', 'box turtle': 'turtle',
  'banded gecko': 'lizard', 'common iguana': 'lizard', 'american chameleon': 'lizard',
  whiptail: 'lizard', agama: 'lizard', 'frilled lizard': 'lizard',
  alligator_lizard: 'lizard', 'gila monster': 'lizard', 'green lizard': 'lizard',
  'african chameleon': 'lizard', 'komodo dragon': 'lizard',
  'african crocodile': 'crocodile', 'american alligator': 'crocodile',
  triceratops: 'lizard',
  'thunder snake': 'snake', 'ringneck snake': 'snake', 'hognose snake': 'snake',
  'green snake': 'snake', 'king snake': 'snake', 'garter snake': 'snake',
  'water snake': 'snake', 'vine snake': 'snake', 'night snake': 'snake',
  'boa constrictor': 'snake', 'rock python': 'snake', 'indian cobra': 'snake',
  'green mamba': 'snake', 'sea snake': 'snake', 'horned viper': 'snake',
  'diamondback': 'snake', sidewinder: 'snake',
  'tree frog': 'frog', 'tailed frog': 'frog', bullfrog: 'frog',
  'european fire salamander': 'frog', 'common newt': 'frog', eft: 'frog',
  axolotl: 'frog', 'spotted salamander': 'frog',
  'black and gold garden spider': 'spider', 'barn spider': 'spider',
  'garden spider': 'spider', 'black widow': 'spider', tarantula: 'spider',
  'wolf spider': 'spider', 'harvestman': 'spider', scorpion: 'spider',
  tick: 'spider', 'centipede': 'spider',
  monarch: 'butterfly', 'sulphur butterfly': 'butterfly',
  'cabbage butterfly': 'butterfly', admiral: 'butterfly',
  'ringlet': 'butterfly', 'lycaenid': 'butterfly',
  bee: 'bee', ant: 'ant', fly: 'fly', grasshopper: 'grasshopper',
  cricket: 'grasshopper', 'walking stick': 'grasshopper', mantis: 'grasshopper',
  cockroach: 'ant', cicada: 'grasshopper', leafhopper: 'grasshopper',
  lacewing: 'fly', dragonfly: 'butterfly', damselfly: 'butterfly',
  ladybug: 'ladybug', 'ground beetle': 'ladybug', 'long-horned beetle': 'ladybug',
  'leaf beetle': 'ladybug', 'dung beetle': 'ladybug', 'rhinoceros beetle': 'ladybug',
  weevil: 'ladybug', 'tiger beetle': 'ladybug',
  snail: 'snail', slug: 'snail', 'sea slug': 'snail', conch: 'snail',
  chiton: 'snail', 'chambered nautilus': 'snail',
  'dungeness crab': 'crab', 'rock crab': 'crab', 'fiddler crab': 'crab',
  'king crab': 'crab', 'hermit crab': 'crab', crayfish: 'crab',
  'american lobster': 'crab', 'spiny lobster': 'crab', isopod: 'crab',
  starfish: 'starfish', 'sea urchin': 'starfish', 'sea cucumber': 'starfish',
  jellyfish: 'jellyfish', 'sea anemone': 'jellyfish', 'brain coral': 'jellyfish',
  flatworm: 'jellyfish', nematode: 'jellyfish'
};

/* ── 3. Family rules ──────────────────────────────────────────────────────
   Ordered: the first pattern to match a label wins. These fold ImageNet's
   long tail — a hundred and twenty dog breeds, dozens of birds — onto the
   everyday word a learner actually wants. */
const RULES = [
  [/\bterrier\b|\bretriever\b|\bspaniel\b|\bsheepdog\b|\bcollie\b|\bschnauzer\b/, 'dog'],
  [/\bhound\b|\bsetter\b|\bpoodle\b|\bmastiff\b|\bbulldog\b|\bcorgi\b|\bhusky\b/, 'dog'],
  [/\bshepherd\b|\bpinscher\b|\bpointer\b|\bmalamute\b|\bpapillon\b|\bpekinese\b/, 'dog'],
  [/\bchihuahua\b|\bbeagle\b|\bpug\b|\bboxer\b|\bdalmatian\b|\bbasenji\b|\bkeeshond\b/, 'dog'],
  [/\bsamoyed\b|\bpomeranian\b|\bchow\b|\bnewfoundland\b|\bleonberg\b|\bdoberman\b/, 'dog'],
  [/\brottweiler\b|\bkuvasz\b|\bkomondor\b|\bbriard\b|\bkelpie\b|\bmalinois\b/, 'dog'],
  [/\bgreat dane\b|\bsaint bernard\b|\bgreat pyrenees\b|\beskimo dog\b|\bdhole\b/, 'dog'],
  [/\bcat\b$/, 'cat'],
  [/\bmonkey\b|\blemur\b/, 'monkey'],
  [/\bwhale\b|\bshark\b|\bseal\b/, 'fish'],
  [/\bsnake\b|\bviper\b|\bcobra\b|\bpython\b|\bmamba\b/, 'snake'],
  [/\bturtle\b|\btortoise\b/, 'turtle'],
  [/\blizard\b|\bgecko\b|\biguana\b|\bchameleon\b/, 'lizard'],
  [/\bfrog\b|\bsalamander\b|\bnewt\b/, 'frog'],
  [/\bspider\b/, 'spider'],
  [/\bbeetle\b/, 'ladybug'],
  [/\bbutterfly\b/, 'butterfly'],
  [/\bcrab\b|\blobster\b/, 'crab'],
  [/\bmushroom\b|\bfungus\b/, 'mushroom'],
  [/\bbridge\b/, 'bridge'],
  [/\bfence\b|\bwall\b/, 'fence'],
  [/\bclock\b/, 'clock'],
  [/\bbottle\b|\bjug\b/, 'bottle'],
  [/\bboat\b|\bship\b/, 'boat'],
  [/\btruck\b|\bvan\b/, 'truck'],
  [/\bbus\b/, 'bus'],
  [/\bcar\b(?!\w)/, 'car'],
  [/\bshop\b|\bstore\b|\bmarket\b/, 'house'],
  [/\bchair\b/, 'chair'],
  [/\btable\b/, 'dining table'],
  [/\bhat\b|\bcap\b(?!\w)/, 'hat'],
  [/\bshoe\b|\bboot\b/, 'shoe'],
  [/\bcoat\b|\bjacket\b|\bsweater\b/, 'coat'],
  [/\bbird\b|\bfinch\b|\bowl\b|\bduck\b|\bgull\b|\bheron\b/, 'bird'],
  [/\bflower\b|\borchid\b|\bdaisy\b/, 'flower'],
  [/\btree\b/, 'tree'],
  [/\bcamera\b/, 'camera'],
  [/\bcomputer\b/, 'computer'],
  [/\bphone\b|\btelephone\b/, 'cell phone'],
  [/\bpen\b(?!\w)|\bpencil\b/, 'pen'],
  [/\bbook\b/, 'book'],
  [/\bbag\b|\bpack\b/, 'backpack'],
  [/\bball\b/, 'sports ball'],
  [/\bguitar\b|\bbanjo\b|\bmandolin\b/, 'guitar'],
  [/\bpiano\b|\borgan\b/, 'piano'],
  [/\bdrum\b/, 'drum'],
  [/\bhorn\b|\btrumpet\b|\btrombone\b/, 'trumpet']
];

/** Lowercase, strip punctuation noise, collapse whitespace. */
function normalise(s) {
  return String(s || '').toLowerCase().replace(/_/g, ' ').replace(/\s+/g, ' ').trim();
}

/**
 * Map one MobileNet class name onto a Lemma vocabulary key.
 * @param {string} className the raw ImageNet label
 * @returns {string|null} a DICT key, or null when we would only be guessing
 */
export function mapImagenet(className) {
  const full = normalise(className);
  if (!full) return null;

  if (EXACT[full]) return EXACT[full];

  const head = full.split(',')[0].trim();
  if (HEAD[head]) return HEAD[head];

  // Later synonyms sometimes carry the plain word the head lacks.
  for (const part of full.split(',')) {
    const p = part.trim();
    if (HEAD[p]) return HEAD[p];
  }

  for (const [re, key] of RULES) if (re.test(full)) return key;
  return null;
}
