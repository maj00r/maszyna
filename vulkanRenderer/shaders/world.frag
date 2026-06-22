#version 450

layout(location = 0) in vec3 vLight;
layout(location = 1) in vec2 vUV;
layout(location = 2) in float vOpacity;

layout(set = 0, binding = 0) uniform sampler2D uTexture;

layout(location = 0) out vec4 outColor;

void main() {
  vec4 tex = texture(uTexture, vUV);
  // vOpacity is the material's alpha-test threshold (0 for blended atlases,
  // ~0.5 for cutout). Drop texels below it (with a tiny floor so fully
  // transparent texels never show), then keep the texture's own alpha so the
  // pipeline alpha-blends glass while opaque surfaces (alpha 1) stay solid.
  if (tex.a < max(vOpacity, 0.04)) discard;
  outColor = vec4(tex.rgb * vLight, tex.a);
}
