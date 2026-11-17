/**
 * Runtime configuration for ISL Studio frontend.
 * Update API_BASE to point to your FastAPI server.
 */
export const config = Object.freeze({
  API_BASE: "http://localhost:8000",
  USE_MOCK: true, // Set true to demo UI offline with synthetic pose & recognition data

  REQUEST_TIMEOUT_MS: 8000,

  // Sign -> Text capture & steadiness detection
  RING_BUFFER_SIZE: 32,
  STEADY_FRAMES_REQUIRED: 15,
  WRIST_VELOCITY_THRESHOLD: 0.014, // Normalized landmark units per frame
  RECOGNIZE_DEBOUNCE_MS: 900,

  // Text -> Sign 3D playback
  DEFAULT_FPS: 24,
  TRANSITION_BLEND_FRAMES: 10,

  // Pinned MediaPipe HandLandmarker model & WASM assets
  MEDIAPIPE_WASM_BASE:
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm",
  HAND_LANDMARKER_MODEL:
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
});