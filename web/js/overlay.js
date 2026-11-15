import { HAND_CONNECTIONS } from './hand_topology.js';

export class OverlayRenderer {
    constructor(canvasElement) {
        this.canvas = canvasElement;
        this.ctx = canvasElement.getContext('2d');
    }

    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    drawResults(results, videoWidth, videoHeight) {
        if (this.canvas.width !== videoWidth || this.canvas.height !== videoHeight) {
            this.canvas.width = videoWidth;
            this.canvas.height = videoHeight;
        }

        this.clear();
        if (!results.landmarks || results.landmarks.length === 0) return;

        for (let i = 0; i < results.landmarks.length; i++) {
            const landmarks = results.landmarks[i];
            const handedness = results.handedness[i][0].categoryName; // "Left" or "Right"
            
            // Note: MediaPipe reports handedness from camera perspective. 
            // Since video is mirrored via CSS, flip color assignment or label appropriately.
            const color = handedness === 'Left' ? '#3b82f6' : '#22c55e'; // Blue for left, green for right

            // Draw bones
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 3;
            for (const [start, end] of HAND_CONNECTIONS) {
                const p1 = landmarks[start];
                const p2 = landmarks[end];
                this.ctx.beginPath();
                this.ctx.moveTo(p1.x * this.canvas.width, p1.y * this.canvas.height);
                this.ctx.lineTo(p2.x * this.canvas.width, p2.y * this.canvas.height);
                this.ctx.stroke();
            }

            // Draw joints
            this.ctx.fillStyle = '#ffffff';
            for (const pt of landmarks) {
                this.ctx.beginPath();
                this.ctx.arc(pt.x * this.canvas.width, pt.y * this.canvas.height, 4, 0, 2 * Math.PI);
                this.ctx.fill();
                this.ctx.strokeStyle = color;
                this.ctx.lineWidth = 1.5;
                this.ctx.stroke();
            }
        }
    }
}