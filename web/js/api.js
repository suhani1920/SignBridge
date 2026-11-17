import { config } from "../config.js";

/**
 * Fetch wrapper with AbortController timeout and uniform error normalization.
 */
async function requestJson(path, options = {}) {
  const url = `${config.API_BASE.replace(/\/$/, "")}${path}`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), config.REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {})
      }
    });

    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const errBody = await response.json();
        if (errBody?.detail) detail = String(errBody.detail);
      } catch {
        // Ignore JSON parse failure on non-JSON error body
      }
      throw new Error(detail);
    }

    return await response.json();
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error(`Request timed out after ${config.REQUEST_TIMEOUT_MS}ms`);
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}

/* --------------------------------------------------------------------------
   Mock Data Generators (used only when config.USE_MOCK === true)
   -------------------------------------------------------------------------- */

function generateSyntheticHand(handedness = "right", phase = 0) {
  const sideSign = handedness === "left" ? -1 : 1;
  const wristX = 0.5 + sideSign * 0.14;
  const wristY = 0.72;
  const pts = [];

  // Wrist (0)
  pts.push([wristX, wristY, 0]);

  // 5 fingers x 4 joints = 20 landmarks
  for (let f = 0; f < 5; f++) {
    const spread = (f - 2) * 0.032 * sideSign;
    const curl = Math.sin(phase + f * 0.45) * 0.035;
    for (let j = 1; j <= 4; j++) {
      const len = j * 0.048;
      const x = wristX + spread * j + Math.cos(phase) * 0.01;
      const y = wristY - len + Math.abs(curl) * j;
      const z = -0.02 * j + curl;
      pts.push([x, y, z]);
    }
  }
  return pts;
}

function buildMockPoseSequence(numFrames = 28) {
  const frames = [];
  for (let i = 0; i < numFrames; i++) {
    const phase = (i / numFrames) * Math.PI * 2;
    frames.push([
      generateSyntheticHand("left", phase),
      generateSyntheticHand("right", phase + 0.6)
    ]);
  }
  return { fps: 24, frames };
}

const MOCK_VOCAB = ["HELLO", "NAMASTE", "THANK YOU", "WELCOME", "YES", "NO", "PLEASE", "HELP"];

function mockRecognize() {
  const idx = Math.floor(Math.random() * MOCK_VOCAB.length);
  const label = MOCK_VOCAB[idx];
  const alt1 = MOCK_VOCAB[(idx + 1) % MOCK_VOCAB.length];
  const alt2 = MOCK_VOCAB[(idx + 2) % MOCK_VOCAB.length];
  const isConfident = Math.random() > 0.3;

  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        label,
        confidence: isConfident ? 0.87 : 0.54,
        status: isConfident ? "confident" : "uncertain",
        method: "mock-stgcn",
        alternatives: [
          { label, score: isConfident ? 0.87 : 0.54 },
          { label: alt1, score: 0.21 },
          { label: alt2, score: 0.09 }
        ]
      });
    }, 180);
  });
}

function mockSynthesize(text) {
  const tokens = text
    .trim()
    .split(/\s+/)
    .filter(Boolean);

  const signs = tokens.map((raw) => {
    const clean = raw.replace(/[^\w]/g, "").toUpperCase();
    const matched = clean.length > 0 && clean !== "XYZ";
    return {
      label: clean || raw,
      similarity: matched ? 0.94 : 0.0,
      video_path: null,
      pose: matched ? buildMockPoseSequence(30) : null,
      matched
    };
  });

  return new Promise((resolve) => {
    setTimeout(() => resolve({ signs }), 220);
  });
}

/* --------------------------------------------------------------------------
   Public Backend Contract API
   -------------------------------------------------------------------------- */

export async function checkHealth() {
  if (config.USE_MOCK) {
    return { status: "ok", mock: true };
  }
  return requestJson("/health", { method: "GET" });
}

export async function listSigns({ search = "", limit = 50, offset = 0 } = {}) {
  if (config.USE_MOCK) {
    return MOCK_VOCAB.map((label) => ({
      label,
      video_path: null,
      pose_path: null,
      num_samples: 12
    }));
  }
  const params = new URLSearchParams({
    search,
    limit: String(limit),
    offset: String(offset)
  });
  return requestJson(`/signs?${params.toString()}`, { method: "GET" });
}

/**
 * POST /recognize
 * @param {Array<[Array<number[]>|null, Array<number[]>|null]>} frames - 1..64 unmirrored frames
 * @param {number} fps
 */
export async function recognizeSign(frames, fps) {
  if (config.USE_MOCK) {
    return mockRecognize();
  }
  return requestJson("/recognize", {
    method: "POST",
    body: JSON.stringify({ frames, fps })
  });
}

/**
 * POST /synthesize
 * @param {string} text
 * @param {number} topK
 * @param {boolean} includePose
 */
export async function synthesizeText(text, topK = 3, includePose = true) {
  if (config.USE_MOCK) {
    return mockSynthesize(text);
  }
  return requestJson("/synthesize", {
    method: "POST",
    body: JSON.stringify({
      text,
      top_k: topK,
      include_pose: includePose
    })
  });
}