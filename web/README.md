# ISL Two-Way Translator Frontend

A dependency-light, high-performance web frontend for two-way Indian Sign Language translation supporting browser-side MediaPipe Hand landmark tracking, Three.js 3D avatar rendering, and seamless FastAPI backend integration.

## Features
- **Tab 1: Sign → Text**: Real-time camera capture, MediaPipe HandLandmarker tracking (up to 2 hands), 21-point bone skeleton overlay, automatic steady-state detection & manual capture, confidence scoring, alternatives, and running sentence history with copy/clear functions.
- **Tab 2: Text → Sign**: Text-to-sign synthesis displaying matched/unmatched word chips, interactive 3D Three.js skeleton avatar playback engine with smooth interpolation, speed control, looping, and reference video preview.
- **Robust Quality & Developer Experience**: Configurable API base (`config.js`), built-in Mock Mode (`USE_MOCK = true`) for immediate offline demonstration without a running backend, responsive dark-mode UI, error toasts, and FPS/latency debugging metrics.

## How to Run

### Option 1: Using Python Built-in HTTP Server
```bash
cd web
python3 -m http.server 3000