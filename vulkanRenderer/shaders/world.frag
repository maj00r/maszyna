#version 450

// Temporary flat directional shading by normal so untextured scenery reads
// well. Replaced once materials/textures land.

layout(location = 0) in vec3 vNormal;
layout(location = 0) out vec4 outColor;

void main() {
  vec3 n = normalize(vNormal);
  float l = clamp(dot(n, normalize(vec3(0.4, 1.0, 0.3))), 0.0, 1.0) * 0.8 + 0.2;
  outColor = vec4(vec3(l), 1.0);
}
