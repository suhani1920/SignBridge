import { HAND_BONES, HAND_COLORS } from "./hand_topology.js";

/**
 * 2D HTML5 Canvas skeleton renderer aligned to the CSS-mirrored video viewport.
 * Input landmarks are raw unmirrored MediaPipe normalized coordinates [x, y, z].
 * Because <video> uses CSS `transform: scaleX(-1)`, we draw X at `(1 - x) * width`
 * so the skeleton visually tracks the user's mirrored reflection without mutating
 * the underlying unmirrored coordinates sent to the API.
 */
export class SkeletonOverlay {
  /**
   * @param {HTMLCanvasElement} canvas
   */
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
  }

  clear() {
    if (!this.ctx) return;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  /**
   * Sync internal canvas pixel buffer with displayed element size.
   */
  syncSize(width, height) {
    if (width > 0 && height > 0) {
      if (this.canvas.width !== width || this.canvas.height !== height) {
        this.canvas.width = width;
        this.canvas.height = height;
      }
    }
  }

  /**
   * Render left and right hand skeletons.
   * @param {Array<number[]>|null} leftHand - 21x[x,y,z] unmirrored
   * @param {Array<number[]>|null} rightHand - 21x[x,y,z] unmirrored
   */
  draw(leftHand, rightHand) {
    if (!this.ctx) return;
    this.clear();

    if (leftHand) {
      this.#drawHand(leftHand, HAND_COLORS.left);
    }
    if (rightHand) {
      this.#drawHand(rightHand, HAND_COLORS.right);
    }
  }

  #drawHand(landmarks, palette) {
    const w = this.canvas.width;
    const h = this.canvas.height;
    const ctx = this.ctx;

    // Draw bones
    ctx.lineWidth = 2.5;
    ctx.lineCap = "round";
    ctx.strokeStyle = palette.bone;

    ctx.beginPath();
    for (const [a, b] of HAND_BONES) {
      const ptA = landmarks[a];
      const ptB = landmarks[b];
      if (!ptA || !ptB) continue;

      const ax = (1 - ptA[0]) * w;
      const ay = ptA[1] * h;
      const bx = (1 - ptB[0]) * w;
      const by = ptB[1] * h;

      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
    }
    ctx.stroke();

    // Draw 21 joints
    for (let i = 0; i < landmarks.length; i++) {
      const pt = landmarks[i];
      if (!pt) continue;

      const x = (1 - pt[0]) * w;
      const y = pt[1] * h;
      const radius = i === 0 ? 5 : 3.5;

      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = palette.joint;
      ctx.fill();

      ctx.lineWidth = 1.2;
      ctx.strokeStyle = "#ffffff";
      ctx.stroke();
    }
  }
}