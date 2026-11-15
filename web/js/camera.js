import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

export class CameraManager {
    constructor(videoElement, onResultsCallback) {
        this.videoElement = videoElement;
        this.onResultsCallback = onResultsCallback;
        this.handLandmarker = null;
        this.webcamRunning = false;
        this.lastVideoTime = -1;
        this.animationFrameId = null;
    }

    async initialize() {
        try {
            const vision = await FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm"
            );
            this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                    delegate: "GPU"
                },
                runningMode: "VIDEO",
                numHands: 2
            });
            return true;
        } catch (error) {
            console.error("Failed to initialize MediaPipe HandLandmarker:", error);
            throw error;
        }
    }

    async startCamera() {
        if (!navigator.mediaDevices?.getUserMedia) {
            throw new Error("getUserMedia is not supported by your browser or context is insecure (HTTPS required).");
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }
            });
            this.videoElement.srcObject = stream;
            await new Promise((resolve) => {
                this.videoElement.onloadedmetadata = () => {
                    this.videoElement.play();
                    resolve();
                };
            });
            this.webcamRunning = true;
            this.predictLoop();
        } catch (error) {
            console.error("Camera permission or initialization error:", error);
            throw error;
        }
    }

    stopCamera() {
        this.webcamRunning = false;
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
        }
        const stream = this.videoElement.srcObject;
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            this.videoElement.srcObject = null;
        }
    }

    predictLoop = () => {
        if (!this.webcamRunning) return;

        let startTimeMs = performance.now();
        if (this.videoElement.currentTime !== this.lastVideoTime && this.handLandmarker) {
            this.lastVideoTime = this.videoElement.currentTime;
            const results = this.handLandmarker.detectForVideo(this.videoElement, startTimeMs);
            if (this.onResultsCallback) {
                this.onResultsCallback(results, this.videoElement.videoWidth, this.videoElement.videoHeight);
            }
        }
        this.animationFrameId = requestAnimationFrame(this.predictLoop);
    }
}