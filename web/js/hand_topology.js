// Shared 21-point hand skeleton bone connections based on MediaPipe Hands topology
export const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],         // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],         // Index finger
    [5, 9], [9, 10], [10, 11], [11, 12],    // Middle finger
    [9, 13], [13, 14], [14, 15], [15, 16],  // Ring finger
    [13, 17], [17, 18], [18, 19], [19, 20], // Little finger
    [0, 17]                                 // Palm base connection
];

export const JOINT_NAMES = [
    "Wrist",
    "Thumb CMC", "Thumb MCP", "Thumb IP", "Thumb Tip",
    "Index MCP", "Index PIP", "Index DIP", "Index Tip",
    "Middle MCP", "Middle PIP", "Middle DIP", "Middle Tip",
    "Ring MCP", "Ring PIP", "Ring DIP", "Ring Tip",
    "Pinky MCP", "Pinky PIP", "Pinky DIP", "Pinky Tip"
];