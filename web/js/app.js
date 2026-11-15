import { config } from '../config.js';
import { api } from './api.js';
import { CameraManager } from './camera.js';
import { OverlayRenderer } from './overlay.js';
import { Hand3DRenderer } from './hand3d.js';

class App {
    constructor() {
        this.ringBuffer = [];
        this.sentenceHistory = [];
        this.isProcessingRequest = false;
        this.fpsCounter = { frames: 0, lastTime: performance.now(), current: 0 };
        this.lastWristPositions = { left: null, right: null };
        this.steadyFrameCount = 0;

        this.initDOM();
        this.initTabs();
        this.initBackendHealth();
        this.initTab1();
        this.initTab2();
    }

    initDOM() {
        this.elements = {
            backendStatus: document.getElementById('backend-status'),
            tabBtns: document.querySelectorAll('.tab-btn'),
            tabPanels: document.querySelectorAll('.tab-panel'),
            toastContainer: document.getElementById('toast-container'),
            // Tab 1
            webcam: document.getElementById('webcam'),
            outputCanvas: document.getElementById('output-canvas'),
            toggleCameraBtn: document.getElementById('toggle-camera-btn'),
            captureSignBtn: document.getElementById('capture-sign-btn'),
            mockModeToggle: document.getElementById('mock-mode-toggle'),
            cameraOverlayMsg: document.getElementById('camera-overlay-msg'),
            currentLabel: document.getElementById('current-label'),
            currentStatus: document.getElementById('current-status'),
            confidenceBar: document.getElementById('confidence-bar'),
            alternativesBox: document.getElementById('alternatives-box'),
            sentenceOutput: document.getElementById('sentence-output'),
            copySentenceBtn: document.getElementById('copy-sentence-btn'),
            clearSentenceBtn: document.getElementById('clear-sentence-btn'),
            debugFps: document.getElementById('debug-fps'),
            debugLatency: document.getElementById('debug-latency'),
            debugHands: document.getElementById('debug-hands'),
            // Tab 2
            signTextInput: document.getElementById('sign-text-input'),
            translateTextBtn: document.getElementById('translate-text-btn'),
            wordChips: document.getElementById('word-chips'),
            playPauseBtn: document.getElementById('play-pause-btn'),
            restartBtn: document.getElementById('restart-btn'),
            speedSlider: document.getElementById('speed-slider'),
            loopToggle: document.getElementById('loop-toggle'),
            currentWordSign: document.getElementById('current-word-sign'),
            referenceVideo: document.getElementById('reference-video'),
            noVideoMsg: document.getElementById('no-video-msg')
        };

        this.elements.mockModeToggle.checked = config.USE_MOCK;
        this.elements.mockModeToggle.addEventListener('change', (e) => {
            config.USE_MOCK = e.target.checked;
            this.showToast(`Mock Mode ${config.USE_MOCK ? 'Enabled' : 'Disabled'}`, 'success');
        });
    }

    initTabs() {
        this.elements.tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const targetId = btn.getAttribute('aria-controls');
                this.elements.tabBtns.forEach(b => {
                    b.classList.remove('active');
                    b.setAttribute('aria-selected', 'false');
                });
                this.elements.tabPanels.forEach(p => p.classList.add('hidden'));

                btn.classList.add('active');
                btn.setAttribute('aria-selected', 'true');
                document.getElementById(targetId).classList.remove('hidden');
            });
        });
    }

    async initBackendHealth() {
        try {
            await api.checkHealth();
            this.elements.backendStatus.textContent = config.USE_MOCK ? 'Mock Mode' : 'Connected';
            this.elements.backendStatus.className = 'badge success';
        } catch (err) {
            this.elements.backendStatus.textContent = 'Backend Offline';
            this.elements.backendStatus.className = 'badge danger';
            this.showToast('Could not connect to backend. Enable Mock Mode to test UI.', 'error');
        }
    }

    showToast(message, type = 'success') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        this.elements.toastContainer.appendChild(toast);
        setTimeout(() => toast.remove(), 4000);
    }

    // --- TAB 1 LOGIC ---
    initTab1() {
        this.overlayRenderer = new OverlayRenderer(this.elements.outputCanvas);
        this.cameraManager = new CameraManager(this.elements.webcam, (results, w, h) => this.handleCameraResults(results, w, h));

        this.elements.toggleCameraBtn.addEventListener('click', async () => {
            if (this.cameraManager.webcamRunning) {
                this.cameraManager.stopCamera();
                this.elements.toggleCameraBtn.textContent = 'Start Camera';
                this.elements.toggleCameraBtn.className = 'btn primary';
                this.elements.captureSignBtn.disabled = true;
                this.elements.cameraOverlayMsg.classList.remove('hidden');
            } else {
                try {
                    this.elements.toggleCameraBtn.disabled = true;
                    await this.cameraManager.initialize();
                    await this.cameraManager.startCamera();
                    this.elements.toggleCameraBtn.textContent = 'Stop Camera';
                    this.elements.toggleCameraBtn.className = 'btn danger';
                    this.elements.captureSignBtn.disabled = false;
                    this.elements.cameraOverlayMsg.classList.add('hidden');
                } catch (err) {
                    this.showToast(err.message, 'error');
                } finally {
                    this.elements.toggleCameraBtn.disabled = false;
                }
            }
        });

        this.elements.captureSignBtn.addEventListener('click', () => this.triggerRecognition(true));

        this.elements.clearSentenceBtn.addEventListener('click', () => {
            this.sentenceHistory = [];
            this.elements.sentenceOutput.textContent = '';
        });

        this.elements.copySentenceBtn.addEventListener('click', () => {
            const text = this.elements.sentenceOutput.textContent;
            navigator.clipboard.writeText(text);
            this.showToast('Sentence copied to clipboard!', 'success');
        });
    }

    handleCameraResults(results, width, height) {
        // FPS calculation
        this.fpsCounter.frames++;
        const now = performance.now();
        if (now - this.fpsCounter.lastTime >= 1000) {
            this.fpsCounter.current = Math.round((this.fpsCounter.frames * 1000) / (now - this.fpsCounter.lastTime));
            this.elements.debugFps.textContent = this.fpsCounter.current;
            this.fpsCounter.frames = 0;
            this.fpsCounter.lastTime = now;
        }

        this.overlayRenderer.drawResults(results, width, height);
        this.elements.debugHands.textContent = results.landmarks ? results.landmarks.length : 0;

        // Process landmarks into unmirrored format for ring buffer
        const frameData = [null, null]; // [left_hand, right_hand]
        if (results.landmarks && results.handedness) {
            for (let i = 0; i < results.landmarks.length; i++) {
                const rawHand = results.landmarks[i].map(pt => [pt.x, pt.y, pt.z]);
                const handednessCat = results.handedness[i][0].categoryName;
                
                // Account for mirror flip: MediaPipe Left is user's Right in mirrored preview
                const isLeftHand = handednessCat === 'Right'; 
                const slotIndex = isLeftHand ? 0 : 1;
                frameData[slotIndex] = rawHand;
            }
        }

        // Ring buffer push
        this.ringBuffer.push(frameData);
        if (this.ringBuffer.length > config.RING_BUFFER_SIZE) {
            this.ringBuffer.shift();
        }

        // Check motion steadiness for auto-trigger
        this.checkSteadinessAndAutoTrigger(frameData);
    }

    checkSteadinessAndAutoTrigger(currentFrame) {
        const leftWrist = currentFrame[0] ? currentFrame[0][0] : null;
        const rightWrist = currentFrame[1] ? currentFrame[1][0] : null;

        let maxVelocity = 0;
        if (leftWrist && this.lastWristPositions.left) {
            const dx = leftWrist[0] - this.lastWristPositions.left[0];
            const dy = leftWrist[1] - this.lastWristPositions.left[1];
            maxVelocity = Math.max(maxVelocity, Math.sqrt(dx*dx + dy*dy));
        }
        if (rightWrist && this.lastWristPositions.right) {
            const dx = rightWrist[0] - this.lastWristPositions.right[0];
            const dy = rightWrist[1] - this.lastWristPositions.right[1];
            maxVelocity = Math.max(maxVelocity, Math.sqrt(dx*dx + dy*dy));
        }

        this.lastWristPositions = { left: leftWrist, right: rightWrist };

        if (maxVelocity < config.STEADY_VELOCITY_THRESHOLD && (leftWrist || rightWrist)) {
            this.steadyFrameCount++;
            if (this.steadyFrameCount >= config.STEADY_FRAME_COUNT && !this.isProcessingRequest && this.ringBuffer.length >= 15) {
                this.steadyFrameCount = 0;
                this.triggerRecognition(false);
            }
        } else {
            this.steadyFrameCount = 0;
        }
    }

    async triggerRecognition(forced) {
        if (this.isProcessingRequest || this.ringBuffer.length === 0) return;
        this.isProcessingRequest = true;

        try {
            const framesSnapshot = [...this.ringBuffer];
            const { result, latency } = await api.recognize(framesSnapshot, config.FPS);
            this.elements.debugLatency.textContent = `${latency}ms`;
            this.updateRecognitionUI(result);
        } catch (err) {
            console.error('Recognition error:', err);
        } finally {
            this.isProcessingRequest = false;
        }
    }

    updateRecognitionUI(res) {
        this.elements.currentStatus.textContent = res.status;
        this.elements.currentStatus.className = `status-badge ${res.status}`;
        const confidencePct = Math.round((res.confidence || 0) * 100);
        this.elements.confidenceBar.style.width = `${confidencePct}%`;

        if (res.label && res.status === 'confident') {
            this.elements.currentLabel.textContent = res.label;
            this.elements.currentLabel.classList.remove('placeholder');
            this.sentenceHistory.push(res.label);
            this.elements.sentenceOutput.textContent = this.sentenceHistory.join(' ');
        } else if (res.label && res.status === 'uncertain') {
            this.elements.currentLabel.textContent = `${res.label} (Uncertain)`;
            this.elements.currentLabel.classList.remove('placeholder');
        } else {
            this.elements.currentLabel.textContent = 'Sign rejected or unrecognized';
            this.elements.currentLabel.classList.add('placeholder');
        }

        // Render alternatives
        this.elements.alternativesBox.innerHTML = '';
        if (res.alternatives && res.alternatives.length > 0) {
            res.alternatives.forEach(alt => {
                const chip = document.createElement('span');
                chip.className = 'alt-chip';
                chip.textContent = `${alt.label} (${Math.round(alt.score * 100)}%)`;
                this.elements.alternativesBox.appendChild(chip);
            });
        }
    }

    // --- TAB 2 LOGIC ---
    initTab2() {
        const container = document.getElementById('canvas-3d-container');
        this.hand3D = new Hand3DRenderer(container);
        this.hand3D.onWordChange = (word) => {
            this.elements.currentWordSign.textContent = `Signing: ${word}`;
        };

        this.elements.translateTextBtn.addEventListener('click', async () => {
            const text = this.elements.signTextInput.value.trim();
            if (!text) return;

            try {
                this.elements.translateTextBtn.disabled = true;
                const response = await api.synthesize(text, 3, true);
                this.renderWordChips(response.signs);
                this.hand3D.loadSequence(response.signs);

                this.elements.playPauseBtn.disabled = false;
                this.elements.restartBtn.disabled = false;
                this.elements.playPauseBtn.textContent = 'Pause';
            } catch (err) {
                this.showToast('Synthesis failed: ' + err.message, 'error');
            } finally {
                this.elements.translateTextBtn.disabled = false;
            }
        });

        this.elements.playPauseBtn.addEventListener('click', () => {
            if (this.hand3D.isPlaying) {
                this.hand3D.pause();
                this.elements.playPauseBtn.textContent = 'Play';
            } else {
                this.hand3D.play();
                this.elements.playPauseBtn.textContent = 'Pause';
            }
        });

        this.elements.restartBtn.addEventListener('click', () => {
            this.hand3D.restart();
            this.elements.playPauseBtn.textContent = 'Pause';
        });

        this.elements.speedSlider.addEventListener('input', (e) => {
            this.hand3D.setSpeed(parseFloat(e.target.value));
        });

        this.elements.loopToggle.addEventListener('change', (e) => {
            this.hand3D.setLoop(e.target.checked);
        });
    }

    renderWordChips(signs) {
        this.elements.wordChips.innerHTML = '';
        signs.forEach(sign => {
            const chip = document.createElement('div');
            chip.className = `word-chip ${sign.matched ? 'matched' : 'unmatched'}`;
            chip.textContent = `${sign.label} ${sign.matched ? '✓' : '✗'}`;
            
            if (sign.video_path) {
                chip.addEventListener('click', () => {
                    document.querySelectorAll('.word-chip').forEach(c => c.classList.remove('active-chip'));
                    chip.classList.add('active-chip');
                    this.elements.referenceVideo.src = sign.video_path;
                    this.elements.referenceVideo.classList.remove('hidden');
                    this.elements.noVideoMsg.classList.add('hidden');
                    this.elements.referenceVideo.play();
                });
            }
            this.elements.wordChips.appendChild(chip);
        });
    }
}

window.addEventListener('DOMContentLoaded', () => new App());