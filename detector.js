const DetectionPipeline = (function () {
    'use strict';

    // ── Configuration ──
    const CONFIRM_FRAMES = 8;       // frames needed to confirm a detection
    const NMS_THRESHOLD = 0.35;     // IoU threshold for Non-Maximum Suppression
    const CONFUSION_THRESHOLD = 0.3; // IoU for confusion group resolution
    const DECAY_RATE = 2;           // how fast unconfirmed candidates decay
    const CLASS_LOCK_FRAMES = 12;   // frames a confirmed class must persist before it can change

    // ── Overlapping class groups ──
    const CONFUSE_GROUPS = [
        ['cell phone', 'remote', 'mouse', 'hair drier'],
        ['cup', 'bowl', 'vase', 'bottle'],
        ['knife', 'scissors', 'fork'],
        ['couch', 'bed'],
        ['skateboard', 'surfboard'],
        ['laptop', 'tv', 'book', 'keyboard'],
        ['backpack', 'handbag', 'suitcase'],
        ['car', 'truck', 'bus'],
        ['dining table', 'desk'],
        ['potted plant', 'broccoli']
    ];
    const confuseMap = {};
    CONFUSE_GROUPS.forEach(g => g.forEach(c => confuseMap[c] = g));

    // ── Expected aspect ratios for commonly confused objects ──
    // aspect = height / width of bounding box
    const ASPECT_HINTS = {
        'cell phone': { min: 1.3, max: 2.8, weight: 0.25 },  // tall, thin rectangle
        'remote': { min: 2.2, max: 6.0, weight: 0.25 },  // very tall, very thin
        'mouse': { min: 0.5, max: 1.3, weight: 0.20 },  // squat, wider than tall
        'bottle': { min: 1.8, max: 5.0, weight: 0.25 },  // tall and thin
        'cup': { min: 0.6, max: 1.6, weight: 0.20 },  // roughly square or slightly tall
        'bowl': { min: 0.3, max: 0.9, weight: 0.20 },  // wide and short
        'vase': { min: 1.2, max: 3.5, weight: 0.15 },  // tall, narrower
        'laptop': { min: 0.5, max: 1.1, weight: 0.20 },  // wider than tall (open)
        'tv': { min: 0.4, max: 0.9, weight: 0.20 },  // wide screen
        'book': { min: 0.8, max: 1.8, weight: 0.10 },  // can vary
        'knife': { min: 2.0, max: 8.0, weight: 0.20 },  // very elongated
        'scissors': { min: 0.8, max: 2.5, weight: 0.15 },  // variable
        'keyboard': { min: 0.15, max: 0.55, weight: 0.20 },  // very wide, short
        'hair drier': { min: 0.6, max: 1.5, weight: 0.15 },  // roughly square-ish
        'fork': { min: 3.0, max: 10.0, weight: 0.20 },  // very elongated
        'spoon': { min: 2.5, max: 8.0, weight: 0.15 },  // elongated
    };

    // ── State for temporal consistency ──
    const candidates = {};

    // ── Bounding box intersection check ──
    function iou(a, b) {
        const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
        const x2 = Math.min(a[0] + a[2], b[0] + b[2]), y2 = Math.min(a[1] + a[3], b[1] + b[3]);
        if (x2 <= x1 || y2 <= y1) return 0;
        const inter = (x2 - x1) * (y2 - y1);
        return inter / (a[2] * a[3] + b[2] * b[3] - inter);
    }

    // ── Aspect ratio scoring ──
    // Returns a multiplier [0..1] — 1.0 means perfect fit, lower means bad fit
    function aspectScore(cls, bbox) {
        const hint = ASPECT_HINTS[cls];
        if (!hint) return 1.0; // no hint = no penalty
        const w = bbox[2], h = bbox[3];
        if (w === 0) return 0.5;
        const aspect = h / w;
        if (aspect >= hint.min && aspect <= hint.max) return 1.0; // perfect
        // How far outside the range?
        const dist = aspect < hint.min
            ? (hint.min - aspect) / hint.min
            : (aspect - hint.max) / hint.max;
        // Apply weighted penalty (clamp to 0.3 minimum so we don't kill detections entirely)
        return Math.max(0.3, 1.0 - dist * hint.weight * 3);
    }

    // ── Reject invalid sizes ──
    function validateSize(pred, videoArea) {
        if (!DICT || !SIZE_RANGES) return true;
        const entry = DICT[pred.class];
        if (!entry || !entry.size) return true;
        const boxArea = pred.bbox[2] * pred.bbox[3];
        const ratio = boxArea / videoArea;
        const range = SIZE_RANGES[entry.size];
        if (!range) return true;
        return ratio >= range.min * 0.5 && ratio <= range.max * 2;
    }

    // ── Primary filtering logic ──
    function filter(rawPreds, videoWidth, videoHeight) {
        const videoArea = videoWidth * videoHeight;
        let preds = [...rawPreds];

        // Step 1: Apply aspect ratio scoring — adjust confidence
        preds = preds.map(p => {
            const aScore = aspectScore(p.class, p.bbox);
            return { ...p, score: p.score * aScore, _origScore: p.score, _aspectScore: aScore };
        });

        // Step 2: Sort by adjusted confidence (highest first)
        preds.sort((a, b) => b.score - a.score);

        // Step 3: Size validation
        preds = preds.filter(p => validateSize(p, videoArea));

        // Step 4: Minimum adjusted confidence filter
        preds = preds.filter(p => p.score >= 0.45);

        // Step 5: NMS — remove overlapping boxes, keep highest adjusted confidence
        const kept = [];
        for (const p of preds) {
            let dominated = false;
            for (const k of kept) {
                if (iou(p.bbox, k.bbox) > NMS_THRESHOLD) { dominated = true; break; }
            }
            if (!dominated) kept.push(p);
        }

        // Step 6: Confusion group resolution with aspect-aware tiebreaking
        const resolved = [];
        for (const p of kept) {
            const grp = confuseMap[p.class];
            if (grp) {
                let dominated = false;
                for (const r of resolved) {
                    const rGrp = confuseMap[r.class];
                    if (rGrp && rGrp === grp && iou(p.bbox, r.bbox) > CONFUSION_THRESHOLD) {
                        // Both are in same confusion group with high IoU
                        // Prefer the one with better aspect score, then raw confidence
                        const pFit = (p._aspectScore || 1.0) * p._origScore;
                        const rFit = (r._aspectScore || 1.0) * r._origScore;
                        if (pFit > rFit) {
                            // Replace the existing resolved entry
                            const idx = resolved.indexOf(r);
                            resolved[idx] = p;
                        }
                        dominated = true;
                        break;
                    }
                }
                if (dominated) continue;
            }
            resolved.push(p);
        }

        // Step 7: Temporal consistency voting with class stability
        const nowKeys = new Set();
        const confirmed = [];

        for (const p of resolved) {
            // Find matching existing candidate by class + IoU overlap
            let matchedKey = null;
            let matchedKeyByLocation = null;
            for (const key in candidates) {
                const c = candidates[key];
                if (c.cls === p.class && iou(c.bbox, p.bbox) > 0.25) {
                    matchedKey = key; break;
                }
                // Also check spatial match regardless of class (for class stability)
                if (iou(c.bbox, p.bbox) > 0.4 && !matchedKeyByLocation) {
                    matchedKeyByLocation = key;
                }
            }

            const rKey = matchedKey || `${p.class}_${Math.round(p.bbox[0] / 40)}_${Math.round(p.bbox[1] / 40)}_${Date.now()}`;
            nowKeys.add(rKey);

            if (!candidates[rKey]) {
                // Check if there's an established candidate at this location with a different class
                if (matchedKeyByLocation && !matchedKey) {
                    const existing = candidates[matchedKeyByLocation];
                    // If the existing candidate is well-established, don't let a new class steal it easily
                    if (existing.frames >= CLASS_LOCK_FRAMES) {
                        const existingFit = aspectScore(existing.cls, p.bbox) * existing.score;
                        const newFit = aspectScore(p.class, p.bbox) * p.score;
                        // New class must be significantly better to override
                        if (newFit < existingFit * 1.3) {
                            // Keep existing class, update its bbox
                            existing.bbox = p.bbox;
                            existing.score = existing.score * 0.7 + p._origScore * 0.3;
                            nowKeys.add(matchedKeyByLocation);
                            if (existing.frames >= CONFIRM_FRAMES) {
                                confirmed.push({ ...p, class: existing.cls, score: existing.score });
                            }
                            continue;
                        }
                    }
                }
                candidates[rKey] = { cls: p.class, score: p.score, frames: 1, bbox: p.bbox };
            } else {
                const c = candidates[rKey];
                c.frames++;
                c.score = c.score * 0.6 + p.score * 0.4;
                c.bbox = p.bbox;
            }

            if (candidates[rKey].frames >= CONFIRM_FRAMES) {
                confirmed.push(p);
            }
        }

        // Decay stale candidates
        for (const key in candidates) {
            if (!nowKeys.has(key)) {
                candidates[key].frames -= DECAY_RATE;
                if (candidates[key].frames <= 0) delete candidates[key];
            }
        }

        return confirmed;
    }

    // ── Reset tracking state ──
    function reset() {
        for (const key in candidates) delete candidates[key];
    }

    return { filter, reset, iou };
})();
