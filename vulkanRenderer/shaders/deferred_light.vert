#version 450

// Fullscreen triangle (no vertex buffer): covers the screen for the deferred
// lighting pass. vUV runs 0..1 across the framebuffer.
layout(location = 0) out vec2 vUV;

void main() {
  vUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = vec4(vUV * 2.0 - 1.0, 0.0, 1.0);
}
