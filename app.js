(function () {
    'use strict';

    // App state
    const state = {
        model: null, lang: 'fr', quizMode: false, detecting: true, confidence: 0.55,
        vocab: JSON.parse(localStorage.getItem('ll_vocab') || '{}'),
        quizCorrect: 0, quizTotal: 0, fps: 0, currentDetections: [],
        panelView: null, frameCount: 0, fpsUpdateTime: 0
    };

    // Positioning helpers
    const SMOOTH = 0.12, GRACE = 15, smoothed = {};
    function lerp(a, b, t) { return a + (b - a) * t; }
    function updateSmooth(cls, rx, ry, rw, rh) {
        if (!smoothed[cls]) smoothed[cls] = { x: rx, y: ry, w: rw, h: rh, age: 0, n: 0 };
        else {
            const s = smoothed[cls];
            s.x = lerp(s.x, rx, SMOOTH); s.y = lerp(s.y, ry, SMOOTH);
            s.w = lerp(s.w, rw, SMOOTH); s.h = lerp(s.h, rh, SMOOTH);
        }
        smoothed[cls].age = 0; smoothed[cls].n++;
    }
    function ageSmooth() {
        for (const c in smoothed) {
            smoothed[c].age++;
            if (smoothed[c].age > GRACE + 15) delete smoothed[c];
        }
    }

    // UI Elements
    const $ = id => document.getElementById(id);
    const video = $('webcam'), canvas = $('overlay'), ctx = canvas.getContext('2d');
    const labelContainer = $('labelContainer'), detectedBar = $('detectedBar');
    const LANG_MAP = { fr: 'fr-FR', es: 'es-ES', de: 'de-DE', ja: 'ja-JP', it: 'it-IT' };
    const LANG_NAMES = { fr: 'French', es: 'Spanish', de: 'German', ja: 'Japanese', it: 'Italian' };
    const LK = { fr: 'fr', es: 'es', de: 'de', ja: 'ja', it: 'it' };

    // Initialize app
    async function init() {
        updateLoad(10, 'Requesting camera…');
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
            });
            video.srcObject = stream;
            await video.play();
            updateLoad(30, 'Camera ready. Loading AI model…');
        } catch (e) {
            $('noCameraMsg').style.display = 'flex';
            $('loadingScreen').classList.add('hidden');
            return;
        }

        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        updateLoad(50, 'Downloading COCO-SSD model (mobilenet_v2)…');
        try {
            state.model = await cocoSsd.load({ base: 'mobilenet_v2' });
            updateLoad(90, 'Model loaded!');
        } catch (e) {
            updateLoad(70, 'Trying lighter model…');
            try {
                state.model = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
                updateLoad(90, 'Lite model loaded!');
            } catch (e2) {
                updateLoad(90, 'Model failed. Please refresh.');
                return;
            }
        }

        updateLoad(100, 'Ready!');
        setTimeout(() => $('loadingScreen').classList.add('hidden'), 400);
        setupEvents();
        updateStats();
        detectLoop();
        if (window.innerWidth > 1024) openPanel('vocab');
    }

    function updateLoad(p, m) {
        $('loadBar').style.width = p + '%';
        $('loadStatus').textContent = m;
    }

    function resizeCanvas() {
        const w = $('videoWrap');
        canvas.width = w.clientWidth;
        canvas.height = w.clientHeight;
    }

    // Processing loop
    async function detectLoop() {
        if (!state.model || !state.detecting) {
            requestAnimationFrame(detectLoop);
            return;
        }
        const now = performance.now();
        try {
            const rawPreds = await state.model.detect(video, 20, state.confidence);
            const preds = DetectionPipeline.filter(rawPreds, video.videoWidth, video.videoHeight);
            state.currentDetections = preds;
            renderSmooth(preds);
            trackVocab(preds);
            updateDetectedBar(preds);

            state.frameCount++;
            if (now - state.fpsUpdateTime > 1000) {
                state.fps = state.frameCount;
                state.frameCount = 0;
                state.fpsUpdateTime = now;
                $('statFPS').textContent = state.fps;
            }
            $('statDetecting').textContent = preds.length;
        } catch (e) { /* skip frame */ }
        requestAnimationFrame(detectLoop);
    }

    // Render labels to canvas
    function renderSmooth(preds) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const vw = video.videoWidth, vh = video.videoHeight;
        const cw = canvas.width, ch = canvas.height;
        const sc = Math.max(cw / vw, ch / vh);
        const ox = (cw - vw * sc) / 2, oy = (ch - vh * sc) / 2;
        const HDR = 48, LH = 68, LW = 130;

        // Feed raw positions into smoother
        preds.forEach(p => {
            let [bx, by, bw, bh] = p.bbox;
            bx = vw - bx - bw; // mirror
            updateSmooth(p.class, bx * sc + ox, by * sc + oy, bw * sc, bh * sc);
        });
        ageSmooth();

        // Draw all tracked objects
        for (const cls in smoothed) {
            const s = smoothed[cls];
            if (s.age > GRACE) continue;
            const entry = DICT[cls];
            if (!entry) continue;
            const fade = s.age === 0 ? 1 : Math.max(0, 1 - s.age / GRACE);
            const { x, y, w, h } = s;

            // Canvas bounding box
            ctx.globalAlpha = fade * 0.4;
            ctx.strokeStyle = 'rgba(232,232,224,0.45)';
            ctx.lineWidth = 1.5;
            roundRect(ctx, x, y, w, h, 5);
            ctx.stroke();

            // Corner accents (olive)
            ctx.globalAlpha = fade * 0.7;
            ctx.strokeStyle = '#8ba339';
            ctx.lineWidth = 2;
            const cl = Math.min(15, w * 0.13, h * 0.13);
            drawCorners(ctx, x, y, w, h, cl);
            ctx.globalAlpha = 1;

            // HTML floating label
            const word = entry[LK[state.lang]] || entry.fr;
            const phonetic = state.lang === 'fr' ? entry.ph : '';
            let el = document.querySelector(`.detection-label[data-cls="${cls}"]`);

            if (!el) {
                el = document.createElement('div');
                el.className = 'detection-label entering' + (state.quizMode ? ' quiz-label' : '');
                el.dataset.cls = cls;
                el.innerHTML = `<div class="label-card">` +
                    `<div class="label-foreign">${word}</div>` +
                    (phonetic ? `<div class="label-phonetic">/${phonetic}/</div>` : '') +
                    `<div class="label-english"><span>${entry.em} ${cls}</span>` +
                    `<button class="label-speak" onclick="event.stopPropagation();LinguaLens.speak('${word.replace(/'/g, "\\'")}')">🔊</button></div></div>`;
                el.addEventListener('click', () => {
                    if (state.quizMode) {
                        el.classList.toggle('revealed');
                        if (el.classList.contains('revealed')) state.quizCorrect++;
                        state.quizTotal++;
                        updateStats();
                    } else {
                        window.LinguaLens.speak(word);
                    }
                });
                el.addEventListener('animationend', () => el.classList.remove('entering'), { once: true });
                labelContainer.appendChild(el);
            } else {
                el.querySelector('.label-foreign').textContent = word;
                const phEl = el.querySelector('.label-phonetic');
                if (phEl) phEl.textContent = phonetic ? `/${phonetic}/` : '';
                el.querySelector('.label-english span').textContent = `${entry.em} ${cls}`;
                if (state.quizMode && !el.classList.contains('quiz-label')) el.classList.add('quiz-label');
                if (!state.quizMode && el.classList.contains('quiz-label')) el.classList.remove('quiz-label', 'revealed');
                el.style.opacity = s.age > 0 ? fade : '';
            }

            // Clamp position
            let lx = Math.max(6, Math.min(x, cw - LW - 6));
            let ly = y - 6;
            if (ly - LH < HDR + 2) ly = y + h + 6 + LH;
            if (ly > ch - 6) ly = ch - 6;
            el.style.left = lx + 'px';
            el.style.top = ly + 'px';
            el.style.transform = 'translateY(-100%)';
        }

        // Remove expired labels
        document.querySelectorAll('.detection-label').forEach(el => {
            const c = el.dataset.cls;
            if (!smoothed[c] || smoothed[c].age > GRACE) {
                if (!el.classList.contains('exiting')) {
                    el.classList.add('exiting');
                    setTimeout(() => el.remove(), 300);
                }
            }
        });
    }

    // Canvas drawing helpers
    function roundRect(ctx, x, y, w, h, r) {
        r = Math.min(r, w / 2, h / 2);
        ctx.beginPath();
        ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + r);
        ctx.lineTo(x + w, y + h - r);
        ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
        ctx.lineTo(x + r, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();
    }

    function drawCorners(ctx, x, y, w, h, l) {
        ctx.beginPath(); ctx.moveTo(x, y + l); ctx.lineTo(x, y); ctx.lineTo(x + l, y); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x + w - l, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + l); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x + w, y + h - l); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - l, y + h); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x + l, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - l); ctx.stroke();
    }

    // Detection Bar UI
    function updateDetectedBar(preds) {
        const seen = new Set();
        const current = new Set();

        preds.forEach(p => {
            if (!DICT[p.class] || seen.has(p.class)) return;
            seen.add(p.class);
            current.add(p.class);
            const entry = DICT[p.class];
            const word = entry[LK[state.lang]] || entry.fr;
            const ph = state.lang === 'fr' ? entry.ph : '';

            let chip = detectedBar.querySelector(`.detected-chip[data-cls="${p.class}"]`);
            if (!chip) {
                chip = document.createElement('div');
                chip.className = 'detected-chip';
                chip.dataset.cls = p.class;
                chip.innerHTML =
                    `<div class="chip-emoji">${entry.em}</div>` +
                    `<div class="chip-text">` +
                    `<div class="chip-foreign">${word}</div>` +
                    (ph ? `<div class="chip-phonetic">/${ph}/</div>` : '') +
                    `<div class="chip-english">${p.class}</div>` +
                    `</div>` +
                    `<button class="chip-speak" onclick="event.stopPropagation();LinguaLens.speak('${word.replace(/'/g, "\\'")}')">🔊</button>`;
                chip.addEventListener('click', () => window.LinguaLens.speak(word));
                detectedBar.appendChild(chip);
            } else {
                chip.querySelector('.chip-foreign').textContent = word;
                const phe = chip.querySelector('.chip-phonetic');
                if (phe) phe.textContent = ph ? `/${ph}/` : '';
            }
        });

        detectedBar.querySelectorAll('.detected-chip').forEach(c => {
            c.style.opacity = current.has(c.dataset.cls) ? '' : '0.25';
        });
    }

    // TTS Engine
    function speak(text) {
        if (!window.speechSynthesis) return;
        speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(text);
        u.lang = LANG_MAP[state.lang] || 'fr-FR';
        u.rate = 0.85;
        u.pitch = 1;
        const voices = speechSynthesis.getVoices();
        const match = voices.find(v => v.lang.startsWith(state.lang)) ||
            voices.find(v => v.lang.includes(state.lang));
        if (match) u.voice = match;
        speechSynthesis.speak(u);
    }
    if (window.speechSynthesis) speechSynthesis.onvoiceschanged = () => speechSynthesis.getVoices();

    // Vocabulary persistence
    function trackVocab(preds) {
        let newWord = false;
        preds.forEach(p => {
            if (DICT[p.class] && !state.vocab[p.class]) {
                state.vocab[p.class] = { discovered: Date.now(), count: 1 };
                newWord = true;
            } else if (state.vocab[p.class]) {
                state.vocab[p.class].count++;
            }
        });
        if (newWord) {
            localStorage.setItem('ll_vocab', JSON.stringify(state.vocab));
            updateStats();
            showToast('✨ New word discovered!');
        }
    }

    function updateStats() {
        $('statObjects').textContent = Object.keys(state.vocab).length;
        if (state.quizTotal > 0) {
            $('statQuizScore').textContent = Math.round(state.quizCorrect / state.quizTotal * 100) + '%';
        }
    }

    // Panel views (Vocab / Settings)
    function openPanel(view) {
        state.panelView = view;
        const body = $('panelBody');
        if (view === 'vocab') {
            $('panelTitle').textContent = '📚 Vocabulary';
            const keys = Object.keys(state.vocab);
            if (!keys.length) {
                body.innerHTML = '<div class="vocab-empty">No words yet — point your camera at objects!</div>';
            } else {
                body.innerHTML = keys.sort((a, b) => state.vocab[b].discovered - state.vocab[a].discovered).map(c => {
                    const e = DICT[c]; if (!e) return '';
                    const w = e[LK[state.lang]] || e.fr;
                    return `<div class="vocab-item" onclick="LinguaLens.speak('${w.replace(/'/g, "\\'")}')">` +
                        `<div class="vocab-icon">${e.em}</div>` +
                        `<div class="vocab-info"><div class="vocab-foreign">${w}</div>` +
                        `<div class="vocab-english">${c}</div></div></div>`;
                }).join('');
            }
        } else if (view === 'settings') {
            $('panelTitle').textContent = '⚙️ Settings';
            body.innerHTML =
                `<div class="setting-group">` +
                `<div class="setting-label">Detection Confidence</div>` +
                `<div class="setting-row"><span>Threshold</span><span class="setting-value" id="confVal">${Math.round(state.confidence * 100)}%</span></div>` +
                `<input type="range" min="20" max="90" value="${state.confidence * 100}" id="confSlider">` +
                `</div>` +
                `<div class="setting-group"><div class="setting-label">Actions</div>` +
                `<button class="btn" style="width:100%;margin-top:5px;justify-content:center" onclick="LinguaLens.clearVocab()">🗑️ Reset Vocabulary</button>` +
                `<button class="btn" style="width:100%;margin-top:5px;justify-content:center" onclick="LinguaLens.resetQuiz()">🔄 Reset Quiz Score</button>` +
                `<button class="btn" style="width:100%;margin-top:5px;justify-content:center" onclick="LinguaLens.clearBar()">🧹 Clear Detected Bar</button>` +
                `</div>` +
                `<div class="setting-group"><div class="setting-label">About</div>` +
                `<p style="font-size:11px;color:var(--text-dim);line-height:1.6">LinguaLens is a high-performance AR learning tool using TensorFlow.js. All processing happens locally for maximum privacy.</p>` +
                `<p style="font-size:10px;color:var(--olive);margin-top:8px;font-weight:600;cursor:pointer;letter-spacing:0.5px" onclick="window.open('https://www.rushanhaque.online', '_blank')">Developed by Rushan Haque</p>` +
                `</div>`;
            $('confSlider').addEventListener('input', e => {
                state.confidence = parseInt(e.target.value) / 100;
                $('confVal').textContent = e.target.value + '%';
            });
        }
        if (window.innerWidth <= 1024) {
            $('sidePanel').classList.add('open');
        } else {
            $('sidePanel').classList.add('open'); // Persistent but ensures class is there
        }
    }

    function closePanel() {
        if (window.innerWidth <= 1024) {
            $('sidePanel').classList.remove('open');
            state.panelView = null;
        }
    }

    function clearVocab() {
        state.vocab = {};
        localStorage.removeItem('ll_vocab');
        updateStats();
        showToast('Vocabulary cleared');
        if (state.panelView === 'vocab') openPanel('vocab');
    }

    function resetQuiz() {
        state.quizCorrect = 0;
        state.quizTotal = 0;
        updateStats();
        showToast('Quiz score reset');
    }

    function clearBar() {
        detectedBar.innerHTML = '';
        showToast('Detected bar cleared');
    }

    // Capture screenshot
    function takeSnapshot() {
        const snap = document.createElement('canvas');
        snap.width = canvas.width;
        snap.height = canvas.height;
        const sc = snap.getContext('2d');
        sc.save(); sc.translate(snap.width, 0); sc.scale(-1, 1);
        sc.drawImage(video, 0, 0, snap.width, snap.height);
        sc.restore();
        sc.drawImage(canvas, 0, 0);
        const link = document.createElement('a');
        link.download = `lingualens_${Date.now()}.png`;
        link.href = snap.toDataURL('image/png');
        link.click();
        showToast('📸 Snapshot saved!');
    }

    // UI Feedback
    function showToast(msg) {
        const t = $('toast');
        t.textContent = msg;
        t.classList.add('show');
        clearTimeout(t._t);
        t._t = setTimeout(() => t.classList.remove('show'), 2500);
    }

    // Interaction listeners
    function setupEvents() {
        $('langSelector').addEventListener('change', e => {
            state.lang = e.target.value;
            document.querySelectorAll('.detection-label').forEach(el => el.remove());
            for (const c in smoothed) delete smoothed[c];
            DetectionPipeline.reset();
            detectedBar.innerHTML = '';
            showToast(`Switched to ${LANG_NAMES[state.lang]}`);
            if (state.panelView === 'vocab') openPanel('vocab');
        });

        $('btnQuiz').addEventListener('click', () => {
            state.quizMode = !state.quizMode;
            $('btnQuiz').classList.toggle('active', state.quizMode);
            document.querySelectorAll('.detection-label').forEach(el => {
                if (state.quizMode) el.classList.add('quiz-label');
                else el.classList.remove('quiz-label', 'revealed');
            });
            showToast(state.quizMode ? '🎯 Quiz Mode ON — tap labels to reveal!' : 'Quiz Mode OFF');
        });

        $('btnSnapshot').addEventListener('click', takeSnapshot);
        $('btnVocab').addEventListener('click', () =>
            state.panelView === 'vocab' ? closePanel() : openPanel('vocab'));
        $('btnSettings').addEventListener('click', () =>
            state.panelView === 'settings' ? closePanel() : openPanel('settings'));
        $('panelClose').addEventListener('click', closePanel);

        document.addEventListener('keydown', e => {
            if (e.key === 'q') $('btnQuiz').click();
            if (e.key === 's') takeSnapshot();
            if (e.key === 'v') $('btnVocab').click();
            if (e.key === 'Escape') closePanel();
        });
    }

    // Public helpers for buttons
    window.LinguaLens = { speak, clearVocab, resetQuiz, clearBar };
    window.addEventListener('load', init);
})();
