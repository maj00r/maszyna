#version 450

// Samples the bound material texture (set 0) by UV and applies simple flat
// directional shading by normal. With the default white texture bound this is
// identical to plain normal shading; real per-material textures plug in later.

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec2 vUV;

layout(set = 0, binding = 0) uniform sampler2D uTexture;

layout(location = 0) out vec4 outColor;

void main() {
  vec4 tex = texture(uTexture, vUV);
  // Alpha-cutout: drop nearly-transparent texels (masks, grilles, button
  // labels, glass) instead of drawing them as black. Full alpha blending for
  // glass/soft shadows is a later pass.
  if (tex.a < 0.5) discard;
  vec3 n = normalize(vNormal);
  float l = clamp(dot(n, normalize(vec3(0.4, 1.0, 0.3))), 0.0, 1.0) * 0.7 + 0.3;
  outColor = vec4(tex.rgb * l, 1.0);
}
