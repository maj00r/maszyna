#version 450

// Deferred geometry pass: write the G-buffer (no lighting). Shares world.vert,
// so it gets the same parallax + normal-mapping inputs as forward shading.
// Textures come from the bindless array, indexed by the pushed slot indices.

layout(location = 0) in vec3 vCamRel;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vUV;
layout(location = 3) in float vOpacity;
layout(location = 4) in float vEmission;
layout(location = 5) in float vInterior;
layout(location = 6) in vec3 vTangent;
layout(location = 7) in vec3 vBitangent;
layout(location = 8) in float vHasNormal;
layout(location = 9) flat in int vTexDiffuse;
layout(location = 10) flat in int vTexNormal;
layout(location = 11) in float vGloss;

layout(set = 0, binding = 0) uniform sampler2D uTextures[8192];  // bindless

layout(location = 0) out vec4 oAlbedo;    // rgb albedo, a emission strength
layout(location = 1) out vec4 oNormal;    // rgb world normal, a interior flag
layout(location = 2) out vec4 oPosition;  // rgb camera-rel position, a glossiness

const float HEIGHT_SCALE = 0.05;
vec2 parallax_uv(vec2 uv, vec3 viewDir) {
  const float minLayers = 8.0;
  const float maxLayers = 32.0;
  float numLayers = mix(maxLayers, minLayers, clamp(abs(viewDir.z), 0.0, 1.0));
  float layerDepth = 1.0 / numLayers;
  float curDepth = 0.0;
  vec2 dUV = (viewDir.xy * HEIGHT_SCALE) / numLayers;
  vec2 cur = uv;
  float curMap = texture(uTextures[vTexNormal], cur).b;
  for (int i = 0; i < 32 && curDepth < curMap; ++i) {
    cur -= dUV;
    curMap = texture(uTextures[vTexNormal], cur).b;
    curDepth += layerDepth;
  }
  vec2 prev = cur + dUV;
  float after = curMap - curDepth;
  float before = texture(uTextures[vTexNormal], prev).b - curDepth + layerDepth;
  float w = after / (after - before);
  return mix(cur, prev, clamp(w, 0.0, 1.0));
}

void main() {
  mat3 tbn = mat3(normalize(vTangent), normalize(vBitangent), normalize(vNormal));

  vec2 uv = vUV;
  if (vHasNormal >= 1.5) {
    vec3 vdir = normalize(transpose(tbn) * normalize(-vCamRel));
    uv = parallax_uv(vUV, vdir);
  }

  vec4 tex = texture(uTextures[vTexDiffuse], uv);
  if (tex.a < max(vOpacity, 0.04)) discard;

  vec3 n = normalize(vNormal);
  if (vHasNormal > 0.5) {
    vec4 nm = texture(uTextures[vTexNormal], uv);
    vec3 nt;
    nt.xy = nm.rg * 2.0 - 1.0;
    nt.z = sqrt(max(1.0 - dot(nt.xy, nt.xy), 0.0));
    n = normalize(tbn * nt);
  }

  oAlbedo = vec4(tex.rgb, vEmission);
  oNormal = vec4(n, vInterior);
  // a >= 1 doubles as the "has geometry" marker the lighting pass tests.
  oPosition = vec4(vCamRel, max(vGloss, 1.0));
}
