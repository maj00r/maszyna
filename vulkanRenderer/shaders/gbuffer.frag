#version 450

// Deferred geometry pass: write the G-buffer (no lighting). Shares world.vert,
// so it gets the same parallax + normal-mapping inputs as forward shading.

layout(location = 0) in vec3 vCamRel;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vUV;
layout(location = 3) in float vOpacity;
layout(location = 4) in float vEmission;
layout(location = 5) in float vInterior;
layout(location = 6) in vec3 vTangent;
layout(location = 7) in vec3 vBitangent;
layout(location = 8) in float vHasNormal;

layout(set = 0, binding = 0) uniform sampler2D uTexture;
layout(set = 2, binding = 0) uniform sampler2D uNormal;  // rg: normal, b: height

layout(location = 0) out vec4 oAlbedo;    // rgb albedo, a emission strength
layout(location = 1) out vec4 oNormal;    // rgb world normal, a interior flag
layout(location = 2) out vec4 oPosition;  // rgb camera-relative position, a = 1

const float HEIGHT_SCALE = 0.05;
vec2 parallax_uv(vec2 uv, vec3 viewDir) {
  const float minLayers = 8.0;
  const float maxLayers = 32.0;
  float numLayers = mix(maxLayers, minLayers, clamp(abs(viewDir.z), 0.0, 1.0));
  float layerDepth = 1.0 / numLayers;
  float curDepth = 0.0;
  vec2 dUV = (viewDir.xy * HEIGHT_SCALE) / numLayers;
  vec2 cur = uv;
  float curMap = texture(uNormal, cur).b;
  for (int i = 0; i < 32 && curDepth < curMap; ++i) {
    cur -= dUV;
    curMap = texture(uNormal, cur).b;
    curDepth += layerDepth;
  }
  vec2 prev = cur + dUV;
  float after = curMap - curDepth;
  float before = texture(uNormal, prev).b - curDepth + layerDepth;
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

  vec4 tex = texture(uTexture, uv);
  if (tex.a < max(vOpacity, 0.04)) discard;

  vec3 n = normalize(vNormal);
  if (vHasNormal > 0.5) {
    vec4 nm = texture(uNormal, uv);
    vec3 nt;
    nt.xy = nm.rg * 2.0 - 1.0;
    nt.z = sqrt(max(1.0 - dot(nt.xy, nt.xy), 0.0));
    n = normalize(tbn * nt);
  }

  oAlbedo = vec4(tex.rgb, vEmission);
  oNormal = vec4(n, vInterior);
  oPosition = vec4(vCamRel, 1.0);
}
