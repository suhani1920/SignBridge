# ISL Studio — Two-Way Indian Sign Language Translator (Frontend)

A responsive, real-time two-way Indian Sign Language (ISL) translator frontend supporting:
1. **Sign → Text**: On-device webcam hand-landmark capture (MediaPipe) + sign recognition stream & sentence builder.
2. **Text → Sign**: English text translation into interactive 3D articulated hand animations (Three.js) and reference clips.

---

## 1. Project Structure

```text
web/
├── login.html           # Entry authentication portal (Sign In / Create Account / Demo Bypass)
├── index.html           # Main two-way ISL translator workspace
├── styles.css           # Responsive design system (light/dark mode + tactile hover states)
├── config.js            # Backend API URL & Offline Mock Mode toggle
├── download.jpg         # Showcase banner image used on login.html
├── package.json         # Vite dev server & dependencies (three, @mediapipe/tasks-vision)
├── vite.config.js       # Configured to run on http://localhost:5173
├── .gitignore           # Excludes node_modules/ from Git
└── js/
    ├── app.js           # Tab routing, theme toggle, backend health polling
    ├── api.js           # Backend HTTP client (/health, /recognize, /synthesize) + Mock fallback
    ├── camera.js        # Webcam & MediaPipe hand tracking controller
    ├── overlay.js       # 2D skeletal hand landmark overlay renderer
    ├── hand3d.js        # Three.js 3D dual-hand articulation player
    └── hand_topology.js # 21-point hand joint & bone connection mappings
```

---

## 2. How to Setup & Run the Frontend Locally

> **Important (Camera Permission Rule):** Modern browsers block webcam access on local network IPs (`10.x.x.x` or `192.168.x.x`). Always open the app using **`http://localhost:5173`**.

1. **Open the frontend folder in terminal:**
   ```bash
   cd web
   ```

2. **Install all required packages (`vite`, `three`, `@mediapipe/tasks-vision`):**
   ```bash
   npm install
   ```
   *(If `package.json` is ever missing dependencies, run: `npm install vite three @mediapipe/tasks-vision`)*

3. **Start the development server:**
   ```bash
   npm run dev
   ```

4. **Open in Browser:**
   - Vite automatically launches **`http://localhost:5173/login.html`**.
   - Click **Demo Access (Bypass)** (or sign in with `demo` / `demo`, or register a new account) to enter the translator workspace (`index.html`).

---

## 3. How to Connect the FastAPI Backend (`config.js`)

By default, the frontend runs in **Offline Mock Mode** (`USE_MOCK: true`) so you can test the UI without a server. When the backend is ready, follow these steps:

### Step 1: Paste Backend URL & Disable Mock Mode in `config.js`
Open **`config.js`** and update these two lines at the top of the `config` object:

```javascript
export const config = {
  // 1. Paste your FastAPI backend URL here (default local port is 8000)
  API_BASE: "http://localhost:8000",

  // 2. Change true -> false to connect to the live backend instead of offline mock data
  USE_MOCK: false,

  // Keep existing thresholds & frame settings below unchanged
  ...
};
```

### Step 2: Enable CORS in FastAPI Backend
In your FastAPI `main.py` / `app.py`, allow requests from `http://localhost:5173`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "[http://127.0.0.1:5173](http://127.0.0.1:5173)"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Step 3: Verify Connection
1. Start the FastAPI server (e.g., `uvicorn main:app --reload --port 8000`).
2. Start the frontend (`npm run dev`).
3. Check the top-right status badge in the UI—it should turn green and display **`Online`** (instead of `Mock Mode`).

---

## 4. Required Backend API Endpoints (`js/api.js`)

The frontend sends requests to these three endpoints:

1. **`GET /health`**
   - **Response:** `{ "status": "ok" }`
   - Used by the top status indicator (`Online` / `Offline` / `Mock Mode`).

2. **`POST /recognize`** (Sign → Text)
   - **Request Body:** Captured sequence of hand landmark frames (`landmarks`, `handedness`, `timestamp`).
   - **Response:**
     ```json
     {
       "gloss": "HELLO",
       "confidence": 0.92,
       "status": "confident",
       "alternatives": [
         { "label": "HELLO", "score": 0.92 },
         { "label": "NAMASTE", "score": 0.05 }
       ]
     }
     ```

3. **`POST /synthesize`** (Text → Sign)
   - **Request Body:** `{ "text": "Hello how are you" }`
   - **Response:**
     ```json
     {
       "tokens": [
         { "word": "HELLO", "matched": true },
         { "word": "HOW", "matched": true }
       ],
       "sequence": [
         {
           "word": "HELLO",
           "duration_ms": 1200,
           "frames": [ /* 21-point 3D hand landmark frames */ ]
         }
       ],
       "video_url": null
     }
     ```

---

## 5. Git Commands for Team Collaboration

### Pulling Latest Changes & Running
```bash
git pull origin main
npm install
npm run dev
```

### Pushing Updates (Ensure `.gitignore` excludes `node_modules/`)
```bash
git add .
git commit -m "Describe your changes"
git push origin main
```

---

## 6. Notes on Authentication (`login.html`)

- **Session Guard:** `index.html` checks `sessionStorage.getItem("isl_authenticated")` and redirects unauthenticated tabs to `login.html`. Closing the browser tab resets the session.
- **Connecting Real Backend Auth:** Currently, `login.html` validates accounts via `localStorage` (`isl_studio_users`). To connect real user authentication, update the `signin-form` and `signup-form` submit handlers at the bottom of `login.html` to `fetch()` your `/auth/login` and `/auth/register` endpoints before calling `authorizeAndEnter()`.