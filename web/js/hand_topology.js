/**
 * Shared 21-point MediaPipe hand landmark topology and styling tokens.
 * Used identically by both the 2D canvas overlay and 3D Three.js hand renderer.
 */

export const NUM_LANDMARKS = 21;

export const LANDMARK_INDEX = Object.freeze({
  WRIST: 0,
  THUMB_CMC: 1,
  THUMB_MCP: 2,
  THUMB_IP: 3,
  THUMB_TIP: 4,
  INDEX_MCP: 5,
  INDEX_PIP: 6,
  INDEX_DIP: 7,
  INDEX_TIP: 8,
  MIDDLE_MCP: 9,
  MIDDLE_PIP: 10,
  MIDDLE_DIP: 11,
  MIDDLE_TIP: 12,
  RING_MCP: 13,
  RING_PIP: 14,
  RING_DIP: 15,
  RING_TIP: 16,
  PINKY_MCP: 17,
  PINKY_PIP: 18,
  PINKY_DIP: 19,
  PINKY_TIP: 20
});

/**
 * 21-point skeletal bone connections [startIdx, endIdx].
 */
export const HAND_BONES = Object.freeze([
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index finger
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle finger
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring finger
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky finger
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Metacarpal palm arch
  [5, 9], [9, 13], [13, 17]
]);

export const HAND_COLORS = Object.freeze({
  left: {
    joint: "#3b82f6",
    bone: "#93c5fd",
    hex: 0x3b82f6,
    boneHex: 0x60a5fa
  },
  right: {
    joint: "#ea580c",
    bone: "#fdba74",
    hex: 0xea580c,
    boneHex: 0xfb923c
  }
});