/**
 * camera.js — getUserMedia lifecycle, device switching, torch and zoom.
 *
 * Capability support is wildly uneven across browsers, so every advanced
 * feature is probed against the live track and reported through `caps` rather
 * than assumed.
 */

export const CameraError = {
  DENIED: 'denied',
  NOT_FOUND: 'not-found',
  INSECURE: 'insecure',
  UNSUPPORTED: 'unsupported',
  IN_USE: 'in-use',
  UNKNOWN: 'unknown'
};

export function createCamera(videoEl) {
  let stream = null;
  let track = null;
  let facing = 'environment';
  let devices = [];
  let deviceIndex = 0;

  const caps = { torch: false, zoom: null, focus: false, multipleCameras: false };

  function isSecure() {
    return window.isSecureContext ||
      location.protocol === 'https:' ||
      ['localhost', '127.0.0.1', '::1'].includes(location.hostname);
  }

  function classify(err) {
    if (!isSecure()) return CameraError.INSECURE;
    if (!navigator.mediaDevices?.getUserMedia) return CameraError.UNSUPPORTED;
    switch (err?.name) {
      case 'NotAllowedError':
      case 'SecurityError':      return CameraError.DENIED;
      case 'NotFoundError':
      case 'OverconstrainedError': return CameraError.NOT_FOUND;
      case 'NotReadableError':
      case 'AbortError':         return CameraError.IN_USE;
      default:                   return CameraError.UNKNOWN;
    }
  }

  async function enumerate() {
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      devices = all.filter((d) => d.kind === 'videoinput');
      caps.multipleCameras = devices.length > 1;
    } catch { devices = []; }
    return devices;
  }

  function readCapabilities() {
    caps.torch = false;
    caps.zoom = null;
    caps.focus = false;
    if (!track || !track.getCapabilities) return;
    let c;
    try { c = track.getCapabilities(); } catch { return; }
    if (!c) return;
    caps.torch = !!c.torch;
    if (c.zoom && typeof c.zoom.max === 'number') {
      caps.zoom = { min: c.zoom.min ?? 1, max: c.zoom.max, step: c.zoom.step ?? 0.1 };
    }
    caps.focus = Array.isArray(c.focusMode) && c.focusMode.includes('manual');
  }

  function stop() {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
    track = null;
  }

  /**
   * Open a camera.
   * @param {object} opts { facingMode, deviceId }
   * @throws {Error & {code:string}}
   */
  async function start(opts = {}) {
    if (!isSecure()) {
      const e = new Error('Camera needs a secure (https) context.');
      e.code = CameraError.INSECURE;
      throw e;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      const e = new Error('This browser does not expose a camera API.');
      e.code = CameraError.UNSUPPORTED;
      throw e;
    }

    stop();
    const wanted = opts.facingMode || facing;

    // Try the exact facing mode first (correct rear camera on phones), then
    // relax, then accept any camera at all.
    const attempts = opts.deviceId
      ? [{ deviceId: { exact: opts.deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }]
      : [
          { facingMode: { exact: wanted }, width: { ideal: 1920 }, height: { ideal: 1080 } },
          { facingMode: wanted, width: { ideal: 1280 }, height: { ideal: 720 } },
          { width: { ideal: 1280 }, height: { ideal: 720 } },
          true
        ];

    let lastErr = null;
    for (const video of attempts) {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
        break;
      } catch (err) { lastErr = err; }
    }
    if (!stream) {
      const e = new Error(lastErr?.message || 'Could not open the camera.');
      e.code = classify(lastErr);
      throw e;
    }

    track = stream.getVideoTracks()[0];
    const settings = track.getSettings ? track.getSettings() : {};
    facing = settings.facingMode || (opts.deviceId ? facing : wanted);

    videoEl.srcObject = stream;
    videoEl.setAttribute('playsinline', '');
    videoEl.muted = true;
    await videoEl.play().catch(() => {});
    await waitForFrame();

    readCapabilities();
    await enumerate();
    if (opts.deviceId) deviceIndex = devices.findIndex((d) => d.deviceId === opts.deviceId);

    return { facing, width: videoEl.videoWidth, height: videoEl.videoHeight, caps };
  }

  /** Resolve once the element actually has pixel dimensions. */
  function waitForFrame() {
    if (videoEl.videoWidth > 0) return Promise.resolve();
    return new Promise((res) => {
      const done = () => { videoEl.removeEventListener('loadeddata', done); res(); };
      videoEl.addEventListener('loadeddata', done);
      setTimeout(done, 3000);   // never hang the boot sequence on this
    });
  }

  /** Flip front↔back, or step through devices when facingMode is unavailable. */
  async function flip() {
    if (devices.length > 1 && !/mobile|android|iphone|ipad/i.test(navigator.userAgent)) {
      deviceIndex = (deviceIndex + 1) % devices.length;
      return start({ deviceId: devices[deviceIndex].deviceId });
    }
    return start({ facingMode: facing === 'environment' ? 'user' : 'environment' });
  }

  async function setTorch(on) {
    if (!caps.torch || !track) return false;
    try { await track.applyConstraints({ advanced: [{ torch: !!on }] }); return true; }
    catch { return false; }
  }

  /** Optical zoom when the device offers it. Returns false to signal CSS fallback. */
  async function setZoom(value) {
    if (!caps.zoom || !track) return false;
    const v = Math.max(caps.zoom.min, Math.min(caps.zoom.max, value));
    try { await track.applyConstraints({ advanced: [{ zoom: v }] }); return true; }
    catch { return false; }
  }

  /** Nudge autofocus by cycling focus mode — the closest thing to tap-to-focus. */
  async function refocus() {
    if (!track?.applyConstraints) return false;
    try {
      await track.applyConstraints({ advanced: [{ focusMode: 'single-shot' }] });
      return true;
    } catch {
      try { await track.applyConstraints({ advanced: [{ focusMode: 'continuous' }] }); return true; }
      catch { return false; }
    }
  }

  /** Current frame as a canvas, un-mirrored so exports match what was seen. */
  function grabFrame(mirror = false) {
    const c = document.createElement('canvas');
    c.width = videoEl.videoWidth || 1280;
    c.height = videoEl.videoHeight || 720;
    const g = c.getContext('2d');
    if (mirror) { g.translate(c.width, 0); g.scale(-1, 1); }
    g.drawImage(videoEl, 0, 0, c.width, c.height);
    return c;
  }

  return {
    start, stop, flip, setTorch, setZoom, refocus, grabFrame, enumerate,
    get facing() { return facing; },
    get isFront() { return facing === 'user'; },
    get caps() { return caps; },
    get devices() { return devices; },
    get active() { return !!stream; },
    get videoSize() { return { w: videoEl.videoWidth, h: videoEl.videoHeight }; }
  };
}
