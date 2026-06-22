#version 450

// Gradient skydome: a camera-centred dome whose per-vertex colours form the sky
// gradient (Preetham model, computed by the simulation). No texture, no light.

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

layout(push_constant) uniform PushConstants {
  mat4 uMVP;  // proj * rotation-only view * scale
} pc;

layout(location = 0) out vec3 vColor;

void main() {
  gl_Position = pc.uMVP * vec4(aPos, 1.0);
  vColor = aColor;
}
