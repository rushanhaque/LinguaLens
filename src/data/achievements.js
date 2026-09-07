/**
 * achievements.js — Badge catalogue.
 *
 * Each badge exposes test(stats) → boolean. Stats are computed by
 * store.getStats() so badges never touch storage directly.
 */

export const ACHIEVEMENTS = [
  { id: 'first_sight', em: '👁️', title: 'First Sight', desc: 'Discover your first word', test: (s) => s.discovered >= 1 },
  { id: 'ten_words', em: '🌱', title: 'Sprouting', desc: 'Discover 10 words', test: (s) => s.discovered >= 10 },
  { id: 'quarter', em: '🌿', title: 'Growing', desc: 'Discover 25 words', test: (s) => s.discovered >= 25 },
  { id: 'half_deck', em: '🌳', title: 'Half the World', desc: 'Discover 40 words', test: (s) => s.discovered >= 40 },
  { id: 'complete', em: '🏆', title: 'Completionist', desc: 'Discover all 80 words', test: (s) => s.discovered >= 80 },

  { id: 'first_review', em: '📖', title: 'Studious', desc: 'Complete your first review', test: (s) => s.reviews >= 1 },
  { id: 'hundred_reviews', em: '💯', title: 'Century', desc: 'Complete 100 reviews', test: (s) => s.reviews >= 100 },
  { id: 'five_hundred', em: '🎓', title: 'Scholar', desc: 'Complete 500 reviews', test: (s) => s.reviews >= 500 },

  { id: 'first_mastered', em: '⭐', title: 'Locked In', desc: 'Master your first word', test: (s) => s.mastered >= 1 },
  { id: 'ten_mastered', em: '🌟', title: 'Ten Stars', desc: 'Master 10 words', test: (s) => s.mastered >= 10 },
  { id: 'fifty_mastered', em: '✨', title: 'Fluent Eye', desc: 'Master 50 words', test: (s) => s.mastered >= 50 },

  { id: 'streak_3', em: '🔥', title: 'Warming Up', desc: '3-day streak', test: (s) => s.streak >= 3 },
  { id: 'streak_7', em: '🔥', title: 'Week Strong', desc: '7-day streak', test: (s) => s.streak >= 7 },
  { id: 'streak_30', em: '🌋', title: 'Unstoppable', desc: '30-day streak', test: (s) => s.streak >= 30 },

  { id: 'polyglot_2', em: '🗺️', title: 'Bilingual', desc: 'Study 2 languages', test: (s) => s.languagesUsed >= 2 },
  { id: 'polyglot_4', em: '🌍', title: 'Polyglot', desc: 'Study 4 languages', test: (s) => s.languagesUsed >= 4 },
  { id: 'polyglot_8', em: '🌐', title: 'World Citizen', desc: 'Study 8 languages', test: (s) => s.languagesUsed >= 8 },

  { id: 'accuracy_90', em: '🎯', title: 'Sharpshooter', desc: '90% accuracy over 50+ reviews', test: (s) => s.reviews >= 50 && s.accuracy >= 0.9 },
  { id: 'perfect_10', em: '🎖️', title: 'Perfect Ten', desc: '10 correct answers in a row', test: (s) => s.bestRun >= 10 },

  { id: 'snap', em: '📸', title: 'Shutterbug', desc: 'Save your first snapshot', test: (s) => s.snapshots >= 1 },
  { id: 'category_clear', em: '🧩', title: 'Category Clear', desc: 'Discover every word in one category', test: (s) => s.clearedCategories >= 1 }
];

export const LEVELS = [
  { level: 1, title: 'Newcomer', xp: 0 },
  { level: 2, title: 'Observer', xp: 100 },
  { level: 3, title: 'Spotter', xp: 300 },
  { level: 4, title: 'Collector', xp: 600 },
  { level: 5, title: 'Linguist', xp: 1000 },
  { level: 6, title: 'Interpreter', xp: 1600 },
  { level: 7, title: 'Translator', xp: 2400 },
  { level: 8, title: 'Polyglot', xp: 3500 },
  { level: 9, title: 'Sage', xp: 5000 },
  { level: 10, title: 'Lens Master', xp: 7500 }
];

/** Resolve XP into a level, its title, and progress toward the next. */
export function levelFor(xp) {
  let idx = 0;
  for (let i = 0; i < LEVELS.length; i++) if (xp >= LEVELS[i].xp) idx = i;
  const cur = LEVELS[idx];
  const next = LEVELS[idx + 1] || null;
  const progress = next ? (xp - cur.xp) / (next.xp - cur.xp) : 1;
  return { ...cur, next, progress: Math.max(0, Math.min(1, progress)), xp };
}

export const XP = {
  discover: 10,      // first time a word is seen through the camera
  reviewCorrect: 6,
  reviewWrong: 1,    // effort still counts
  perfectSession: 25,
  streakDay: 15
};
