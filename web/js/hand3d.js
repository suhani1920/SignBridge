import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { HAND_CONNECTIONS } from './hand_topology.js';

export class Hand3DRenderer {
    constructor(containerElement) {
        this.container = containerElement;
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(60, containerElement.clientWidth / containerElement.clientHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.controls = null;
        
        this.handsGroup = new THREE.Group();
        this.sequences = [];
        this.currentSequenceIndex = 0;
        this.currentFrameIndex = 0;
        this.isPlaying = false;
        this.isLooping = true;
        this.playbackSpeed = 1.0;
        this.interpolationProgress = 0;
        this.onWordChange = null;

        this.init();
    }

    init() {
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);

        this.camera.position.set(0, 0, 1.2);
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
        this.scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(0, 10, 10);
        this.scene.add(directionalLight);

        // Grid helper
        const grid = new THREE.GridHelper(10, 10, 0x334155, 0x1e293b);
        grid.position.y = -0.5;
        this.scene.add(grid);

        this.scene.add(this.handsGroup);

        // Create initial placeholder hand meshes
        this.buildHandMeshes();

        window.addEventListener('resize', () => this.onWindowResize());
        this.animate();
    }

    buildHandMeshes() {
        // Left & Right hand containers (21 spheres, 21 cylinders each approx)
        this.handMeshes = {
            left: this.createHandStructure(0x3b82f6),
            right: this.createHandStructure(0x22c55e)
        };
    }

    createHandStructure(color) {
        const group = new THREE.Group();
        const jointGeometry = new THREE.SphereGeometry(0.012, 16, 16);
        const material = new THREE.MeshStandardMaterial({ color, roughness: 0.3 });

        const joints = [];
        for (let i = 0; i < 21; i++) {
            const mesh = new THREE.Mesh(jointGeometry, material);
            group.add(mesh);
            joints.push(mesh);
        }

        const boneGeometry = new THREE.CylinderGeometry(0.005, 0.005, 1, 8);
        const boneMaterial = new THREE.MeshStandardMaterial({ color: 0x94a3b8, roughness: 0.5 });
        const bones = [];
        for (let i = 0; i < HAND_CONNECTIONS.length; i++) {
            const mesh = new THREE.Mesh(boneGeometry, boneMaterial);
            group.add(mesh);
            bones.push(mesh);
        }

        this.scene.add(group);
        return { group, joints, bones };
    }

    loadSequence(signDataList) {
        this.sequences = signDataList.filter(s => s.matched && s.pose && s.pose.frames);
        this.currentSequenceIndex = 0;
        this.currentFrameIndex = 0;
        this.interpolationProgress = 0;
        this.isPlaying = this.sequences.length > 0;
    }

    play() { this.isPlaying = true; }
    pause() { this.isPlaying = false; }
    restart() {
        this.currentSequenceIndex = 0;
        this.currentFrameIndex = 0;
        this.interpolationProgress = 0;
        this.isPlaying = true;
    }
    setSpeed(speed) { this.playbackSpeed = speed; }
    setLoop(loop) { this.isLooping = loop; }

    update(deltaTime) {
        if (!this.isPlaying || this.sequences.length === 0) return;

        const currentSeq = this.sequences[this.currentSequenceIndex];
        if (this.onWordChange) this.onWordChange(currentSeq.label);

        const frames = currentSeq.pose.frames;
        const fps = currentSeq.pose.fps || 30;
        const frameDuration = 1 / fps;

        this.interpolationProgress += deltaTime * this.playbackSpeed;
        if (this.interpolationProgress >= frameDuration) {
            this.interpolationProgress = 0;
            this.currentFrameIndex++;

            if (this.currentFrameIndex >= frames.length - 1) {
                this.currentSequenceIndex++;
                this.currentFrameIndex = 0;
                if (this.currentSequenceIndex >= this.sequences.length) {
                    if (this.isLooping) {
                        this.currentSequenceIndex = 0;
                    } else {
                        this.isPlaying = false;
                        this.currentSequenceIndex = this.sequences.length - 1;
                    }
                }
            }
        }

        const alpha = Math.min(1, this.interpolationProgress / frameDuration);
        const f1 = frames[this.currentFrameIndex];
        const f2 = frames[Math.min(this.currentFrameIndex + 1, frames.length - 1)];

        if (f1 && f2) {
            this.renderHandData(f1[0], f2[0], alpha, this.handMeshes.left);
            this.renderHandData(f1[1], f2[1], alpha, this.handMeshes.right);
        }
    }

    renderHandData(l1, l2, alpha, handObj) {
        if (!l1 || !l2) {
            handObj.group.visible = false;
            return;
        }
        handObj.group.visible = true;

        const interpolated = l1.map((p1, idx) => {
            const p2 = l2[idx];
            if (!p1 || !p2) return null;
            // Coordinate transformation: flip y (MediaPipe y points down), scale z, center on wrist (index 0)
            const wx = l1[0][0];
            const wy = l1[0][1];
            const wz = l1[0][2];

            return [
                (p1[0] * (1 - alpha) + p2[0] * alpha) - wx,
                -((p1[1] * (1 - alpha) + p2[1] * alpha) - wy), // Flip Y
                ((p1[2] * (1 - alpha) + p2[2] * alpha) - wz) * 0.5
            ];
        });

        // Update joints
        interpolated.forEach((coords, idx) => {
            if (coords) {
                handObj.joints[idx].position.set(coords[0], coords[1], coords[2]);
                handObj.joints[idx].visible = true;
            } else {
                handObj.joints[idx].visible = false;
            }
        });

        // Update bones
        HAND_CONNECTIONS.forEach(([start, end], idx) => {
            const pStart = interpolated[start];
            const pEnd = interpolated[end];
            const bone = handObj.bones[idx];

            if (pStart && pEnd) {
                const vStart = new THREE.Vector3(...pStart);
                const vEnd = new THREE.Vector3(...pEnd);
                const distance = vStart.distanceTo(vEnd);

                bone.position.copy(vStart).add(vEnd).multiplyScalar(0.5);
                bone.scale.set(1, distance, 1);
                bone.lookAt(vEnd);
                bone.rotateX(Math.PI / 2);
                bone.visible = true;
            } else {
                bone.visible = false;
            }
        });
    }

    animate = () => {
        requestAnimationFrame(this.animate);
        const clockDelta = 0.016; // approximate frame delta
        this.update(clockDelta);
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }
}