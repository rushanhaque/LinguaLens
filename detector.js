// ==========================================================
//  LINGUALENS — Detection Pipeline
//  NMS, Temporal Consistency, Confusion Groups, Size Validation
// ==========================================================

const DetectionPipeline = (function () {
    'use strict';

    // ── Configuration ──
    const CONFIRM_FRAMES = 5;    // frames needed to confirm a detection
    const NMS_THRESHOLD = 0.45;  // IoU threshold for Non-Maximum Suppression
    const CONFUSION_THRESHOLD = 0.3; // IoU for confusion group resolution
    const DECAY_RATE = 2;        // how fast unconfirmed candidates decay

    // ── Confusion groups: classes COCO-SSD commonly confuses ──
    const CONFUSE_GROUPS = [
        ['cell phone', 'remote', 'mouse', 'hair drier'],
        ['cup', 'bowl', 'vase'],
        ['knife', 'scissors'],
        ['couch', 'bed'],
        ['skateboard', 'surfboard'],
        ['laptop', 'tv', 'book'],
        ['backpack', 'handbag', 'suitcase']
    ];
    const confuseMap = {};
    CONFUSE_GROUPS.forEach(g => g.forEach(c => confuseMap[c] = g));

    // ── Candidate tracker (for temporal consistency) ──
    const candidates = {};

    // ── IoU calculation ──
    function iou(a, b) {
        const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
        const x2 = Math.min(a[0] + a[2], b[0] + b[2]), y2 = Math.min(a[1] + a[3], b[1] + b[3]);
        if (x2 <= x1 || y2 <= y1) return 0;
        const inter = (x2 - x1) * (y2 - y1);
        return inter / (a[2] * a[3] + b[2] * b[3] - inter);
    }

    // ── Size validation ──
    function validateSize(pred, videoArea) {
        if (!DICT || !SIZE_RANGES) return true;
        const entry = DICT[pred.class];
        if (!entry || !entry.size) return true;
        const boxArea = pred.bbox[2] * pred.bbox[3];
        const ratio = boxArea / videoArea;
        const range = SIZE_RANGES[entry.size];
        if (!range) return true;
        // Allow some tolerance (0.5x min, 2x max)
        return ratio >= range.min * 0.5 && ratio <= range.max * 2;
    }

    // ── Main filter pipeline ──
    function filter(rawPreds, videoWidth, videoHeight) {
        const videoArea = videoWidth * videoHeight;
        let preds = [...rawPreds];

        // Step 1: Sort by confidence (highest first)
        preds.sort((a, b) => b.score - a.score);

        // Step 2: Size validation — reject physically impossible detections
        preds = preds.filter(p => validateSize(p, videoArea));

        // Step 3: NMS — remove overlapping boxes, keep highest confidence
        const kept = [];
        for (const p of preds) {
            let dominated = false;
            for (const k of kept) {
                if (iou(p.bbox, k.bbox) > NMS_THRESHOLD) { dominated = true; break; }
            }
            if (!dominated) kept.push(p);
        }

        // Step 4: Confusion group resolution
        const resolved = [];
        for (const p of kept) {
            const grp = confuseMap[p.class];
            if (grp) {
                let dominated = false;
                for (const r of resolved) {
                    const rGrp = confuseMap[r.class];
                    if (rGrp && rGrp === grp && iou(p.bbox, r.bbox) > CONFUSION_THRESHOLD) {
                        dominated = true; break;
                    }
                }
                if (dominated) continue;
            }
            resolved.push(p);
        }

        // Step 5: Temporal consistency voting
        const nowKeys = new Set();
        const confirmed = [];

        for (const p of resolved) {
            // Find matching existing candidate by class + IoU overlap
            let matchedKey = null;
            for (const key in candidates) {
                const c = candidates[key];
                if (c.cls === p.class && iou(c.bbox, p.bbox) > 0.25) {
                    matchedKey = key; break;
                }
            }

            const rKey = matchedKey || `${p.class}_${Math.round(p.bbox[0] / 40)}_${Math.round(p.bbox[1] / 40)}_${Date.now()}`;
            nowKeys.add(rKey);

            if (!candidates[rKey]) {
                candidates[rKey] = { cls: p.class, score: p.score, frames: 1, bbox: p.bbox };
            } else {
                const c = candidates[rKey];
                c.frames++;
                // Weighted score update (smooth toward current)
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

    // ── Reset (e.g. on language change) ──
    function reset() {
        for (const key in candidates) delete candidates[key];
    }

    return { filter, reset, iou };
})();
