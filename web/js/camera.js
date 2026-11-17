import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import { config } from "../config.js";
import { recognizeSign } from "./api.js";
import { LANDMARK_INDEX } from "./hand_topology.js";
import { SkeletonOverlay } from "./overlay.js";

/**
 * Manages Tab 1 (Sign -> Text):
 * - Front-facing getUserMedia stream with explicit permission/context diagnostics
 * - MediaPipe HandLandmarker (VIDEO mode, 2 hands) running strictly in browser
 * - Unmirrored 32-frame ring buffer [left|null, right|null]
 * - Wrist steadiness detector + manual capture with debounce & single in-flight lock
 */
export class SignCaptureController {
  /**
   * @param {{
   *   onToast: (msg: string, type?: string) => void,
   *   onLoading: (busy: boolean, label?: string) => void
   * }} callbacks
   */
  constructor({ onToast, onLoading }) {
    this.onToast = onToast;
    this.onLoading = onLoading;

    this.video = document.getElementById("webcam-video");
    this.canvas = document.getElementById("overlay-canvas");
    this.placeholder = document.getElementById("camera-placeholder");
    this.statusMsg = document.getElementById("camera-status-msg");

    this.btnStartOverlay = document.getElementById("btn-start-camera");
    this.btnToggle = document.getElementById("btn-toggle-camera");
    this.btnCapture = document.getElementById("btn-capture-sign");

    this.steadinessLabel = document.getElementById("steadiness-label");
    this.steadinessFill = document.getElementById("steadiness-fill");
    this.bufferCounter = document.getElementById("buffer-counter");
    this.metricFps = document.getElementById("metric-fps");
    this.metricLatency = document.getElementById("metric-latency");
    this.metricBuffer = document.getElementById("metric-buffer");

    this.statusBadge = document.getElementById("recog-status-badge");
    this.currentSignEl = document.getElementById("current-sign-label");
    this.confidenceValEl = document.getElementById("confidence-value");
    this.confidenceFillEl = document.getElementById("confidence-fill");
    this.confidenceBarWrap = document.getElementById("confidence-bar-wrap");
    this.alternativesList = document.getElementById("alternatives-list");
    this.sentenceBox = document.getElementById("sentence-output");

    this.overlay = new SkeletonOverlay(this.canvas);
    this.landmarker = null;
    this.stream = null;
    this.rafId = null;
    this.isRunning = false;

    // Ring buffer of [leftHand|null, rightHand|null]
    this.frameBuffer = [];
    this.lastVideoTime = -1;
    this.lastFrameTimestamp = 0;
    this.smoothedFps = 24;

    // Steadiness & recognition gating
    this.prevWrists = { left: null, right: null };
    this.steadyCount = 0;
    this.inFlight = false;
    this.lastRecognizeAt = 0;
    this.autoTriggeredThisHold = false;

    this.#bindEvents();
  }

  #bindEvents() {
    this.btnStartOverlay.addEventListener("click", () => this.start());
    this.btnToggle.addEventListener("click", () => {
      if (this.isRunning) {
        this.stop();
      } else {
        this.start();
      }
    });

    this.btnCapture.addEventListener("click", () => this.#triggerRecognize(true));

    document.getElementById("btn-clear-sentence").addEventListener("click", () => {
      this.sentenceBox.value = "";
    });

    document.getElementById("btn-copy-sentence").addEventListener("click", async () => {
      const text = this.sentenceBox.value.trim();
      if (!text) return;
      try {
        await navigator.clipboard.writeText(text);
        this.onToast("Sentence copied to clipboard.");
      } catch {
        this.sentenceBox.select();
        document.execCommand("copy");
        this.onToast("Sentence copied.");
      }
    });
  }

  async #ensureLandmarker() {
    if (this.landmarker) return;
    this.onLoading(true, "Loading MediaPipe HandLandmarker…");
    try {
      const vision = await FilesetResolver.forVisionTasks(config.MEDIAPIPE_WASM_BASE);
      this.landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: config.HAND_LANDMARKER_MODEL,
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
      });
    } finally {
      this.onLoading(false);
    }
  }

  async start() {
    if (this.isRunning) return;

    // 1. Check secure context & mediaDevices availability
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      const msg =
        "Camera requires a secure context (HTTPS or http://localhost). Open this app over localhost or HTTPS.";
      this.statusMsg.textContent = msg;
      this.onToast(msg, "error");
      return;
    }

    try {
      await this.#ensureLandmarker();

      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: { ideal: 960 },
          height: { ideal: 720 }
        }
      });

      this.video.srcObject = this.stream;
      await this.video.play();

      this.overlay.syncSize(this.video.videoWidth || 640, this.video.videoHeight || 480);

      this.isRunning = true;
      this.placeholder.classList.add("hidden");
      this.btnToggle.textContent = "Stop Camera";
      this.btnCapture.disabled = false;

      this.frameBuffer = [];
      this.steadyCount = 0;
      this.lastVideoTime = -1;
      this.lastFrameTimestamp = performance.now();

      this.#loop();
    } catch (err) {
      this.#handleCameraError(err);
    }
  }

  stop() {
    this.isRunning = false;
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    if (this.stream) {
      for (const track of this.stream.getTracks()) {
        track.stop();
      }
      this.stream = null;
    }
    this.video.srcObject = null;
    this.overlay.clear();

    this.placeholder.classList.remove("hidden");
    this.statusMsg.textContent = "Camera is paused.";
    this.btnToggle.textContent = "Start Camera";
    this.btnCapture.disabled = true;
    this.#updateSteadinessUI(0, "Idle");
  }

  #handleCameraError(err) {
    let userMsg = "Unable to access camera.";
    if (err?.name === "NotAllowedError" || err?.name === "PermissionDeniedError") {
      userMsg =
        "Camera permission denied. Allow webcam access in your browser site settings and retry.";
    } else if (err?.name === "NotFoundError" || err?.name === "DevicesNotFoundError") {
      userMsg = "No camera device found. Connect a webcam and try again.";
    } else if (err?.name === "NotReadableError") {
      userMsg = "Camera is in use by another application.";
    } else if (err?.message) {
      userMsg = `Camera error: ${err.message}`;
    }

    this.statusMsg.textContent = userMsg;
    this.placeholder.classList.remove("hidden");
    this.onToast(userMsg, "error");
  }

  #loop = () => {
    if (!this.isRunning) return;

    const now = performance.now();
    if (this.video.readyState >= 2 && this.video.currentTime !== this.lastVideoTime) {
      const dt = now - this.lastFrameTimestamp;
      if (dt > 0) {
        const instantFps = 1000 / dt;
        this.smoothedFps = this.smoothedFps * 0.85 + instantFps * 0.15;
        this.metricFps.textContent = `${Math.round(this.smoothedFps)} FPS`;
      }
      this.lastFrameTimestamp = now;
      this.lastVideoTime = this.video.currentTime;

      this.overlay.syncSize(this.video.videoWidth, this.video.videoHeight);

      const results = this.landmarker.detectForVideo(this.video, now);
      this.#processDetections(results);
    }

    this.rafId = requestAnimationFrame(this.#loop);
  };

  /**
   * Extract unmirrored 21x[x,y,z] landmarks and resolve true Left/Right handedness.
   * MediaPipe reports handedness relative to the unmirrored camera sensor image
   * (where user's anatomical left hand appears on the right side of the raw frame
   * and is labeled "Left" by MediaPipe's selfie-mode convention — wait:
   * MediaPipe HandLandmarker trained on selfie images labels the user's anatomical
   * right hand as "Left" when given an unmirrored front-camera frame, so we invert
   * the raw MediaPipe categoryName to match true anatomical Left/Right).
   */
  #processDetections(results) {
    let leftHand = null;
    let rightHand = null;

    const landmarksList = results?.landmarks || [];
    const handednessList = results?.handedness || [];

    for (let i = 0; i < landmarksList.length; i++) {
      const rawPts = landmarksList[i];
      if (!rawPts || rawPts.length !== 21) continue;

      // Keep raw unmirrored [x, y, z] coordinates for API contract
      const formatted = rawPts.map((pt) => [pt.x, pt.y, pt.z]);

      const rawLabel = handednessList[i]?.[0]?.categoryName || "";
      // Account for front-camera mirror flip on MediaPipe's selfie handedness classifier
      const trueHand = rawLabel === "Left" ? "right" : "left";

      if (trueHand === "left" && !leftHand) {
        leftHand = formatted;
      } else if (trueHand === "right" && !rightHand) {
        rightHand = formatted;
      } else if (!leftHand) {
        leftHand = formatted;
      } else if (!rightHand) {
        rightHand = formatted;
      }
    }

    // Draw mirrored overlay for visual alignment with CSS scaleX(-1) video
    this.overlay.draw(leftHand, rightHand);

    // Push [left|null, right|null] into 32-frame ring buffer
    if (leftHand || rightHand) {
      this.frameBuffer.push([leftHand, rightHand]);
      if (this.frameBuffer.length > config.RING_BUFFER_SIZE) {
        this.frameBuffer.shift();
      }
      this.#evaluateSteadiness(leftHand, rightHand);
    } else {
      this.steadyCount = 0;
      this.prevWrists = { left: null, right: null };
      this.autoTriggeredThisHold = false;
      this.#updateSteadinessUI(0, "No hand");
    }

    const count = this.frameBuffer.length;
    this.bufferCounter.textContent = `${count} / ${config.RING_BUFFER_SIZE}f`;
    this.metricBuffer.textContent = `${count} frames`;
  }

  #evaluateSteadiness(leftHand, rightHand) {
    let maxVelocity = 0;
    let hasPrev = false;

    for (const [key, hand] of [["left", leftHand], ["right", rightHand]]) {
      if (!hand) {
        this.prevWrists[key] = null;
        continue;
      }
      const wrist = hand[LANDMARK_INDEX.WRIST];
      const prev = this.prevWrists[key];
      if (prev) {
        hasPrev = true;
        const dx = wrist[0] - prev[0];
        const dy = wrist[1] - prev[1];
        const dz = wrist[2] - prev[2];
        const dist = Math.hypot(dx, dy, dz);
        if (dist > maxVelocity) maxVelocity = dist;
      }
      this.prevWrists[key] = wrist;
    }

    if (!hasPrev) return;

    if (maxVelocity < config.WRIST_VELOCITY_THRESHOLD) {
      this.steadyCount = Math.min(config.STEADY_FRAMES_REQUIRED, this.steadyCount + 1);
      const pct = Math.round((this.steadyCount / config.STEADY_FRAMES_REQUIRED) * 100);
      this.#updateSteadinessUI(
        pct,
        this.steadyCount >= config.STEADY_FRAMES_REQUIRED ? "Steady" : "Holding…"
      );

      if (
        this.steadyCount >= config.STEADY_FRAMES_REQUIRED &&
        !this.autoTriggeredThisHold &&
        this.frameBuffer.length >= 8
      ) {
        this.autoTriggeredThisHold = true;
        this.#triggerRecognize(false);
      }
    } else {
      this.steadyCount = 0;
      this.autoTriggeredThisHold = false;
      this.#updateSteadinessUI(0, "Moving");
    }
  }

  #updateSteadinessUI(percent, text) {
    this.steadinessFill.style.width = `${percent}%`;
    this.steadinessLabel.textContent = text;
  }

  async #triggerRecognize(isManual) {
    const now = performance.now();

    if (this.inFlight) return;
    if (!isManual && now - this.lastRecognizeAt < config.RECOGNIZE_DEBOUNCE_MS) return;

    if (this.frameBuffer.length === 0) {
      if (isManual) {
        this.onToast("No hand landmarks captured yet. Hold your hand in view first.", "error");
      }
      return;
    }

    // Snapshot up to 64 frames (ring buffer holds max 32)
    const framesSnapshot = this.frameBuffer.slice(-64);
    const fps = Math.max(1, Math.round(this.smoothedFps));

    this.inFlight = true;
    this.lastRecognizeAt = now;
    this.btnCapture.disabled = true;

    const t0 = performance.now();
    try {
      const result = await recognizeSign(framesSnapshot, fps);
      const latencyMs = Math.round(performance.now() - t0);
      this.metricLatency.textContent = `${latencyMs} ms`;
      this.#renderRecognition(result);
    } catch (err) {
      this.onToast(`Recognition failed: ${err.message}`, "error");
    } finally {
      this.inFlight = false;
      if (this.isRunning) {
        this.btnCapture.disabled = false;
      }
    }
  }

  #renderRecognition(res) {
    const status = res?.status || "reject";
    const label = res?.label || "—";
    const confidence = typeof res?.confidence === "number" ? res.confidence : 0;
    const pct = Math.max(0, Math.min(100, Math.round(confidence * 100)));

    // Status pill
    this.statusBadge.textContent = status;
    this.statusBadge.className = `badge badge-${status}`;

    // Current sign & confidence bar
    this.currentSignEl.textContent = status === "reject" ? "Unrecognized" : label;
    this.confidenceValEl.textContent = `${pct}%`;
    this.confidenceFillEl.style.width = `${pct}%`;
    this.confidenceBarWrap.setAttribute("aria-valuenow", String(pct));

    if (status === "confident") {
      this.confidenceFillEl.style.backgroundColor = "var(--status-ok)";
    } else if (status === "uncertain") {
      this.confidenceFillEl.style.backgroundColor = "var(--status-warn)";
    } else {
      this.confidenceFillEl.style.backgroundColor = "var(--ink-muted)";
    }

    // Append ONLY confident results automatically
    if (status === "confident" && res?.label) {
      this.#appendWordToSentence(res.label);
    }

    // Show uncertain predictions & alternatives as clickable suggestion chips
    this.#renderAlternatives(res);
  }

  #renderAlternatives(res) {
    const items = [];
    const seen = new Set();

    if (res?.status === "uncertain" && res?.label) {
      items.push({ label: res.label, score: res.confidence });
      seen.add(res.label);
    }

    if (Array.isArray(res?.alternatives)) {
      for (const alt of res.alternatives) {
        if (alt?.label && !seen.has(alt.label)) {
          items.push(alt);
          seen.add(alt.label);
        }
      }
    }

    this.alternativesList.innerHTML = "";
    if (items.length === 0) {
      this.alternativesList.innerHTML = `<span class="empty-hint">No alternatives available</span>`;
      return;
    }

    for (const item of items) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "chip";
      const scorePct = typeof item.score === "number" ? ` ${Math.round(item.score * 100)}%` : "";
      btn.textContent = `${item.label}${scorePct}`;
      btn.title = `Insert "${item.label}" into sentence`;
      btn.addEventListener("click", () => this.#appendWordToSentence(item.label));
      this.alternativesList.appendChild(btn);
    }
  }

  #appendWordToSentence(word) {
    const clean = String(word).trim();
    if (!clean) return;
    const current = this.sentenceBox.value.trim();
    this.sentenceBox.value = current ? `${current} ${clean}` : clean;
  }
}