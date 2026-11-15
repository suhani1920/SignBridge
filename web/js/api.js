import { config } from '../config.js';

class ApiService {
    constructor() {
        this.timeoutMs = 8000;
    }

    async _fetch(endpoint, options = {}) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeoutMs);

        try {
            const response = await fetch(`${config.API_BASE}${endpoint}`, {
                ...options,
                signal: controller.signal,
                headers: {
                    'Content-Type': 'application/json',
                    ...(options.headers || {})
                }
            });
            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            clearTimeout(timeoutId);
            if (error.name === 'AbortError') {
                throw new Error('Request timed out connecting to backend.');
            }
            throw error;
        }
    }

    async checkHealth() {
        if (config.USE_MOCK) return { status: 'ok', mode: 'mock' };
        try {
            return await this._fetch('/health');
        } catch (err) {
            console.warn('Backend health check failed:', err);
            throw err;
        }
    }

    async getSigns(search = '', limit = 50, offset = 0) {
        if (config.USE_MOCK) {
            return [
                { label: 'HELLO', video_path: null, pose_path: null, num_samples: 12 },
                { label: 'THANK_YOU', video_path: null, pose_path: null, num_samples: 10 },
                { label: 'YES', video_path: null, pose_path: null, num_samples: 15 },
                { label: 'NO', video_path: null, pose_path: null, num_samples: 8 },
                { label: 'PLEASE', video_path: null, pose_path: null, num_samples: 9 }
            ].filter(s => s.label.toLowerCase().includes(search.toLowerCase()));
        }
        return await this._fetch(`/signs?search=${encodeURIComponent(search)}&limit=${limit}&offset=${offset}`);
    }

    async recognize(frames, fps) {
        const startTime = performance.now();
        if (config.USE_MOCK) {
            await new Promise(r => setTimeout(r, 150)); // simulate latency
            const labels = ['HELLO', 'THANK YOU', 'PLEASE', 'HELP', 'GOOD MORNING'];
            const randomLabel = labels[Math.floor(Math.random() * labels.length)];
            const confidence = +(Math.random() * 0.4 + 0.6).toFixed(2);
            return {
                result: {
                    label: confidence > 0.75 ? randomLabel : null,
                    confidence: confidence,
                    status: confidence > 0.85 ? 'confident' : (confidence > 0.7 ? 'uncertain' : 'reject'),
                    method: 'mock_classifier',
                    alternatives: [
                        { label: randomLabel, score: confidence },
                        { label: 'NAMASTE', score: +(confidence - 0.15).toFixed(2) }
                    ]
                },
                latency: Math.round(performance.now() - startTime)
            };
        }

        const data = await this._fetch('/recognize', {
            method: 'POST',
            body: JSON.stringify({ frames, fps })
        });
        const latency = Math.round(performance.now() - startTime);
        return { result: data, latency };
    }

    async synthesize(text, topK = 3, includePose = true) {
        if (config.USE_MOCK) {
            await new Promise(r => setTimeout(r, 200));
            const words = text.toUpperCase().trim().split(/\s+/).filter(Boolean);
            return {
                signs: words.map(word => ({
                    label: word,
                    similarity: 0.95,
                    video_path: null,
                    matched: true,
                    pose: {
                        fps: 30,
                        frames: Array.from({ length: 25 }, () => [
                            // Mock 21 points for left hand
                            Array.from({ length: 21 }, (_, i) => [i * 0.01, Math.sin(i) * 0.1, 0]),
                            // Mock 21 points for right hand
                            Array.from({ length: 21 }, (_, i) => [0.5 + i * 0.01, Math.cos(i) * 0.1, 0])
                        ])
                    }
                }))
            };
        }

        return await this._fetch('/synthesize', {
            method: 'POST',
            body: JSON.stringify({ text, top_k: topK, include_pose: includePose })
        });
    }
}

export const api = new ApiService();