#version 450

layout(location = 0) in vec3 vLight;
layout(location = 1) in vec2 vUV;
layout(location = 2) in float vOpacity;

layout(set = 0, binding = 0) uniform sampler2D uTexture;

layout(location = 0) out vec4 outColor;

void main() {
  vec4 tex = texture(uTexture, vUV);
  float a = tex.a * vOpacity;  // material opacity (1 for opaque materials)
  // Drop only fully-transparent texels; the rest is alpha-blended by the
  // pipeline so glass is see-through and dark shadow decals darken the ground
  // instead of showing as black.
  if (a < 0.04) discard;
  outColor = vec4(tex.rgb * vLight, a);
}
