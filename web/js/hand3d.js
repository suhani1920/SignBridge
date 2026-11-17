import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { config } from "../config.js";
import { synthesizeText } from "./api.js";
import { HAND_BONES, HAND_COLORS, LANDMARK_INDEX, NUM_LANDMARKS } from "./hand_topology.js";

const UP_AXIS = new THREE.Vector3(0, 1, 0);

/**
 * Creates a reusable Three.js hand rig with 21 joint spheres and 23 bone cylinders.
 */
class HandRig {
  constructor(scene, palette) {
    this.group = new THREE.Group();
    this.group.visible = false;
    scene.add(this.group);

    const jointGeo = new THREE.SphereGeometry(0.022, 16, 16);
    const jointMat = new THREE.MeshStandardMaterial({
      color: palette.hex,
      roughness: 0.35,
      metalness: 0.1
    });

    this.joints = [];
    for (let i = 0; i < NUM_LANDMARKS; i++) {
      const mesh = new THREE.Mesh(jointGeo, jointMat);
      if (i === 0) mesh.scale.setScalar(1.3);
      this.group.add(mesh);
      this.joints.push(mesh);
    }

    // Unit-height cylinder (height 1) oriented via quaternion & scaled per frame
    const boneGeo = new THREE.CylinderGeometry(0.011, 0.011, 1, 12);
    const boneMat = new THREE.MeshStandardMaterial({
      color: palette.boneHex,
      roughness: 0.45,
      metalness: 0.05
    });

    this.bones = HAND_BONES.map(([a, b]) => {
      const mesh = new THREE.Mesh(boneGeo, boneMat);
      this.group.add(mesh);
      return { mesh, a, b };
    });

    this._dir = new THREE.Vector3();
    this._mid = new THREE.Vector3();
  }

  /**
   * Update 3D joint positions and orient every bone cylinder with quaternions.
   * @param {THREE.Vector3[]|null} positions - 21 normalized 3D vectors
   */
  update(positions) {
    if (!positions || positions.length !== NUM_LANDMARKS) {
      this.group.visible = false;
      return;
    }

    this.group.visible = true;

    for (let i = 0; i < NUM_LANDMARKS; i++) {
      this.joints[i].position.copy(positions[i]);
    }

    for (const { mesh, a, b } of this.bones) {
      const start = positions[a];
      const end = positions[b];

      this._mid.addVectors(start, end).multiplyScalar(0.5);
      mesh.position.copy(this._mid);

      this._dir.subVectors(end, start);
      const length = this._dir.length();
      if (length > 1e-5) {
        this._dir.divideScalar(length);
        mesh.quaternion.setFromUnitVectors(UP_AXIS, this._dir);
        mesh.scale.set(1, length, 1);
        mesh.visible = true;
      } else {
        mesh.visible = false;
      }
    }
  }
}

/**
 * Tab 2 (Text -> Sign) Controller:
 * - Calls POST /synthesize and displays matched/unmatched gloss chips
 * - Renders dual 3D hands in Three.js with OrbitControls, lighting, and ground grid
 * - Plays multi-word sequences with linear frame interpolation and ~10-frame cross-word blending
 */
export class TextToSignController {
  /**
   * @param {{
   *   onToast: (msg: string, type?: string) => void,
   *   onLoading: (busy: boolean, label?: string) => void
   * }} callbacks
   */
  constructor({ onToast, onLoading }) {
    this.onToast = onToast;
    this.onLoading = onLoading;

    this.form = document.getElementById("synthesize-form");
    this.textInput = document.getElementById("text-input");
    this.btnTranslate = document.getElementById("btn-translate");
    this.wordChipsEl = document.getElementById("word-chips");
    this.sequenceSummaryEl = document.getElementById("sequence-summary");
    this.activeWordEl = document.getElementById("active-sign-word");

    this.refVideoCard = document.getElementById("ref-video-card");
    this.refVideoLabel = document.getElementById("ref-video-label");
    this.refVideo = document.getElementById("ref-video");

    this.container = document.getElementById("three-container");
    this.emptyState = document.getElementById("three-empty-state");

    this.btnPlayPause = document.getElementById("btn-play-pause");
    this.btnRestart = document.getElementById("btn-restart");
    this.speedSlider = document.getElementById("speed-slider");
    this.speedValue = document.getElementById("speed-value");
    this.loopToggle = document.getElementById("loop-toggle");

    // Sequence playback state
    this.playlist = []; // [{ label, video_path, fps, frames: [[leftVecs|null, rightVecs|null], ...] }]
    this.chipElements = [];
    this.currentWordIdx = 0;
    this.frameCursor = 0;
    this.isPlaying = false;
    this.speed = 1.0;
    this.loop = true;
    this.lastRenderTime = 0;

    this.#initScene();
    this.#bindEvents();
    this.#startRenderLoop();
  }

  #initScene() {
    this.scene = new THREE.Scene();

    const width = this.container.clientWidth || 640;
    const height = this.container.clientHeight || 480;

    this.camera = new THREE.PerspectiveCamera(42, width / height, 0.05, 50);
    this.camera.position.set(0, 0.35, 1.85);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(width, height);
    this.container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.target.set(0, 0.15, 0);

    // Studio Lighting
    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 1.35);
    hemiLight.position.set(0, 2, 0);
    this.scene.add(hemiLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.25);
    dirLight.position.set(1.5, 2.5, 2.0);
    this.scene.add(dirLight);

    // Ground Reference Grid
    const grid = new THREE.GridHelper(2.4, 12, 0x888888, 0xcccccc);
    grid.position.y = -0.35;
    grid.material.opacity = 0.25;
    grid.material.transparent = true;
    this.scene.add(grid);

    // Left & Right 3D Hand Rigs
    this.leftRig = new HandRig(this.scene, HAND_COLORS.left);
    this.rightRig = new HandRig(this.scene, HAND_COLORS.right);

    // Responsive ResizeObserver
    this.resizeObserver = new ResizeObserver(() => this.handleResize());
    this.resizeObserver.observe(this.container);
  }

  handleResize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (!width || !height) return;

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  #bindEvents() {
    this.form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const text = this.textInput.value.trim();
      if (!text) return;
      await this.#translate(text);
    });

    this.btnPlayPause.addEventListener("click", () => {
      if (this.playlist.length === 0) return;
      this.isPlaying = !this.isPlaying;
      this.btnPlayPause.textContent = this.isPlaying ? "Pause" : "Play";
      if (this.isPlaying) {
        this.lastRenderTime = performance.now();
      }
    });

    this.btnRestart.addEventListener("click", () => {
      if (this.playlist.length === 0) return;
      this.currentWordIdx = 0;
      this.frameCursor = 0;
      this.isPlaying = true;
      this.btnPlayPause.textContent = "Pause";
      this.lastRenderTime = performance.now();
      this.#syncActiveWordUI();
    });

    this.speedSlider.addEventListener("input", () => {
      this.speed = parseFloat(this.speedSlider.value) || 1.0;
      this.speedValue.textContent = `${this.speed.toFixed(2)}x`;
    });

    this.loopToggle.addEventListener("change", () => {
      this.loop = this.loopToggle.checked;
    });
  }

  async #translate(text) {
    this.btnTranslate.disabled = true;
    this.onLoading(true, "Synthesizing ISL sequence…");

    try {
      const response = await synthesizeText(text, 3, true);
      const signs = Array.isArray(response?.signs) ? response.signs : [];
      this.#loadSigns(signs);
    } catch (err) {
      this.onToast(`Synthesis failed: ${err.message}`, "error");
    } finally {
      this.btnTranslate.disabled = false;
      this.onLoading(false);
    }
  }

  /**
   * Convert raw MediaPipe landmarks [x,y,z] to 3D scene coordinates:
   * - Flip Y (MediaPipe Y points downward)
   * - Scale Z for realistic hand depth
   * - Center on wrist landmark (index 0) and apply lateral offset for two-hand separation
   */
  #convertHandLandmarks(rawHand, lateralOffsetX) {
    if (!Array.isArray(rawHand) || rawHand.length !== NUM_LANDMARKS) return null;

    const wrist = rawHand[LANDMARK_INDEX.WRIST];
    const wx = wrist[0];
    const wy = wrist[1];
    const wz = wrist[2];

    const scaleXY = 2.1;
    const scaleZ = 1.6;

    return rawHand.map(([x, y, z]) => {
      const nx = (x - wx) * scaleXY + lateralOffsetX;
      const ny = -(y - wy) * scaleXY; // Flip Y axis
      const nz = -(z - wz) * scaleZ;
      return new THREE.Vector3(nx, ny, nz);
    });
  }

  #loadSigns(signs) {
    this.wordChipsEl.innerHTML = "";
    this.playlist = [];
    this.chipElements = [];

    if (signs.length === 0) {
      this.wordChipsEl.innerHTML = `<span class="empty-hint">No tokens returned</span>`;
      this.sequenceSummaryEl.textContent = "0 signs";
      return;
    }

    for (const sign of signs) {
      const chip = document.createElement("span");
      chip.className = `chip ${sign.matched ? "chip-matched" : "chip-unmatched"}`;
      chip.textContent = sign.label;
      chip.title = sign.matched
        ? `Matched (${Math.round((sign.similarity || 0) * 100)}%)`
        : "No sign pose available in dictionary";
      this.wordChipsEl.appendChild(chip);

      if (sign.matched && Array.isArray(sign.pose?.frames) && sign.pose.frames.length > 0) {
        const convertedFrames = sign.pose.frames.map(([leftRaw, rightRaw]) => [
          this.#convertHandLandmarks(leftRaw, -0.28),
          this.#convertHandLandmarks(rightRaw, 0.28)
        ]);

        const playlistIndex = this.playlist.length;
        this.playlist.push({
          label: sign.label,
          videoPath: sign.video_path || null,
          fps: sign.pose.fps || config.DEFAULT_FPS,
          frames: convertedFrames
        });

        chip.style.cursor = "pointer";
        chip.addEventListener("click", () => {
          this.currentWordIdx = playlistIndex;
          this.frameCursor = 0;
          this.isPlaying = true;
          this.btnPlayPause.textContent = "Pause";
          this.#syncActiveWordUI();
        });

        this.chipElements[playlistIndex] = chip;
      }
    }

    this.sequenceSummaryEl.textContent = `${this.playlist.length} / ${signs.length} matched`;

    if (this.playlist.length === 0) {
      this.isPlaying = false;
      this.btnPlayPause.disabled = true;
      this.btnRestart.disabled = true;
      this.emptyState.classList.remove("hidden");
      this.activeWordEl.textContent = "—";
      this.leftRig.update(null);
      this.rightRig.update(null);
      this.refVideoCard.classList.add("hidden");
      this.onToast("No matching signs with 3D pose data were found.", "error");
      return;
    }

    this.emptyState.classList.add("hidden");
    this.btnPlayPause.disabled = false;
    this.btnRestart.disabled = false;

    this.currentWordIdx = 0;
    this.frameCursor = 0;
    this.isPlaying = true;
    this.btnPlayPause.textContent = "Pause";
    this.lastRenderTime = performance.now();
    this.#syncActiveWordUI();
  }

  #syncActiveWordUI() {
    const active = this.playlist[this.currentWordIdx];
    if (!active) return;

    this.activeWordEl.textContent = active.label;

    this.chipElements.forEach((el, idx) => {
      if (!el) return;
      el.classList.toggle("chip-active", idx === this.currentWordIdx);
    });

    // Optional side-by-side reference video when video_path is available
    if (active.videoPath) {
      const resolvedSrc = active.videoPath.startsWith("http")
        ? active.videoPath
        : `${config.API_BASE.replace(/\/$/, "")}/${active.videoPath.replace(/^\//, "")}`;
      if (this.refVideo.src !== resolvedSrc) {
        this.refVideo.src = resolvedSrc;
      }
      this.refVideoLabel.textContent = active.label;
      this.refVideoCard.classList.remove("hidden");
      this.refVideo.play().catch(() => {});
    } else {
      this.refVideoCard.classList.add("hidden");
      this.refVideo.pause();
    }
  }

  /**
   * Linearly interpolate two 21-point Vector3 arrays.
   */
  #lerpHand(handA, handB, alpha) {
    if (!handA && !handB) return null;
    if (!handA) return handB;
    if (!handB) return handA;

    const out = new Array(NUM_LANDMARKS);
    for (let i = 0; i < NUM_LANDMARKS; i++) {
      out[i] = new THREE.Vector3().lerpVectors(handA[i], handB[i], alpha);
    }
    return out;
  }

  #evaluatePoseAtCursor() {
    const currentItem = this.playlist[this.currentWordIdx];
    if (!currentItem) return [null, null];

    const frames = currentItem.frames;
    const total = frames.length;
    const idxA = Math.min(total - 1, Math.floor(this.frameCursor));
    const idxB = Math.min(total - 1, idxA + 1);
    const frac = this.frameCursor - idxA;

    const frameA = frames[idxA];
    const frameB = frames[idxB];

    let left = this.#lerpHand(frameA[0], frameB[0], frac);
    let right = this.#lerpHand(frameA[1], frameB[1], frac);

    // Smooth ~10-frame transition blend into the next word at the end of a sign
    const blendWindow = Math.min(config.TRANSITION_BLEND_FRAMES, Math.floor(total / 2));
    const remaining = total - 1 - this.frameCursor;
    const hasNextWord = this.currentWordIdx + 1 < this.playlist.length || this.loop;

    if (blendWindow > 0 && remaining < blendWindow && hasNextWord && this.playlist.length > 1) {
      const nextIdx = (this.currentWordIdx + 1) % this.playlist.length;
      const nextFirstFrame = this.playlist[nextIdx].frames[0];
      const blendAlpha = 1 - Math.max(0, remaining) / blendWindow;

      left = this.#lerpHand(left, nextFirstFrame[0], blendAlpha);
      right = this.#lerpHand(right, nextFirstFrame[1], blendAlpha);
    }

    return [left, right];
  }

  #startRenderLoop() {
    const tick = (now) => {
      requestAnimationFrame(tick);

      if (this.playlist.length > 0) {
        const currentItem = this.playlist[this.currentWordIdx];
        if (this.isPlaying && currentItem) {
          const dtSec = Math.min(0.1, (now - this.lastRenderTime) / 1000);
          this.frameCursor += dtSec * currentItem.fps * this.speed;

          if (this.frameCursor >= currentItem.frames.length - 1) {
            if (this.currentWordIdx + 1 < this.playlist.length) {
              this.currentWordIdx += 1;
              this.frameCursor = 0;
              this.#syncActiveWordUI();
            } else if (this.loop) {
              this.currentWordIdx = 0;
              this.frameCursor = 0;
              this.#syncActiveWordUI();
            } else {
              this.frameCursor = currentItem.frames.length - 1;
              this.isPlaying = false;
              this.btnPlayPause.textContent = "Play";
            }
          }
        }

        const [leftPose, rightPose] = this.#evaluatePoseAtCursor();
        this.leftRig.update(leftPose);
        this.rightRig.update(rightPose);
      }

      this.lastRenderTime = now;
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };

    this.lastRenderTime = performance.now();
    requestAnimationFrame(tick);
  }
}