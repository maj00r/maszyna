#version 450

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

struct GpuLight {
  vec4 pos;    // xyz: camera-relative position, w: 1 = spot
  vec4 dir;    // xyz: spot direction, w: range
  vec4 color;  // rgb: colour * intensity, a: cos(outer cone)
  vec4 extra;  // x: cos(inner cone)
};

layout(set = 1, binding = 0) uniform LightData {
  mat4 viewproj;
  vec4 sun_dir;
  vec4 sun_color;
  vec4 ambient;
  vec4 interior_light;
  mat4 lightspace[4];
  vec4 cascade_splits;
  vec4 cab_light;        // xyz: camera-relative pos, w: enable
  vec4 fog;              // rgb: fog colour, a: visibility range
  ivec4 count;
  GpuLight lights[8];
} u;

layout(set = 1, binding = 1) uniform sampler2DArrayShadow uShadow;

layout(location = 0) out vec4 outColor;

// PCF cascaded sun-shadow factor (1 = lit, 0 = shadowed). Picks the cascade by
// distance from the camera, then samples that cascade's layer.
float sun_shadow(vec3 camrel) {
  float dist = length(camrel);
  int c = dist < u.cascade_splits.x ? 0 : (dist < u.cascade_splits.y ? 1 : 2);
  vec4 ls = u.lightspace[c] * vec4(camrel, 1.0);
  if (ls.w <= 0.0) return 1.0;
  vec3 p = ls.xyz / ls.w;
  p.xy = p.xy * 0.5 + 0.5;
  if (p.xy != clamp(p.xy, 0.0, 1.0) || p.z > 1.0) return 1.0;  // outside map
  float bias = 0.0015;
  vec2 texel = 1.0 / vec2(textureSize(uShadow, 0).xy);
  float s = 0.0;
  for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x)
      s += texture(uShadow,
                   vec4(p.xy + vec2(x, y) * texel, float(c), p.z - bias));
  return s / 9.0;
}

// Cab-light shadow (layer 3): occlusion of the cab interior lamp so cab objects
// cast shadows from it. 1 = lit, 0 = occluded.
float cab_shadow(vec3 camrel) {
  vec4 ls = u.lightspace[3] * vec4(camrel, 1.0);
  if (ls.w <= 0.0) return 1.0;
  vec3 p = ls.xyz / ls.w;
  p.xy = p.xy * 0.5 + 0.5;
  if (p.xy != clamp(p.xy, 0.0, 1.0) || p.z > 1.0 || p.z < 0.0) return 1.0;
  float bias = 0.002;
  vec2 texel = 1.0 / vec2(textureSize(uShadow, 0).xy);
  float s = 0.0;
  for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x)
      s += texture(uShadow, vec4(p.xy + vec2(x, y) * texel, 3.0, p.z - bias));
  return s / 9.0;
}

// Parallax occlusion mapping: ray-march the height map (normal map .b) along the
// tangent-space view direction and return the displaced UV. Only used for
// parallax materials (their .b really is a height). Mirrors the engine's POM.
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
  // Build the TBN once (used by parallax and normal mapping).
  mat3 tbn = mat3(normalize(vTangent), normalize(vBitangent), normalize(vNormal));

  // Parallax/displacement: offset the UVs using the height map before sampling
  // (parallax materials only; vHasNormal == 2).
  vec2 uv = vUV;
  if (vHasNormal >= 1.5) {
    vec3 vdir = normalize(transpose(tbn) * normalize(-vCamRel));
    uv = parallax_uv(vUV, vdir);
  }

  vec4 tex = texture(uTextures[vTexDiffuse], uv);
  // Alpha-test threshold (0 for blended atlases, ~0.5 for cutout); a small
  // floor drops fully transparent texels. Glass keeps its own alpha for blend.
  if (tex.a < max(vOpacity, 0.04)) discard;

  // Per-fragment normal: from the normal map (tangent space -> world via TBN)
  // when a real one is bound, otherwise the interpolated geometric normal.
  vec3 n = normalize(vNormal);
  if (vHasNormal > 0.5) {
    vec4 nm = texture(uTextures[vTexNormal], uv);
    vec3 nt;
    nt.xy = nm.rg * 2.0 - 1.0;
    nt.z = sqrt(max(1.0 - dot(nt.xy, nt.xy), 0.0));
    n = normalize(tbn * nt);
  }
  // Directional sun (shadowed) + ambient floor.
  float ndl = max(dot(n, normalize(-u.sun_dir.xyz)), 0.0);
  float shadow = sun_shadow(vCamRel);
  vec3 light = max(u.ambient.rgb, vec3(0.2)) +
               u.sun_color.rgb * (ndl * shadow);

  // Cab interior lamp. At night it's a real point light: Lambert (surfaces
  // facing the lamp light up), distance falloff, and a real shadow (occluded ->
  // dark, with a small bounce floor so it isn't pitch black). In daylight the
  // lamp is off, so the dimmed colour is just a flat residual fill.
  if (vInterior > 0.5) {
    if (u.cab_light.w > 0.5) {
      vec3 Lc = u.cab_light.xyz - vCamRel;
      float dc = length(Lc);
      float ndlc = max(dot(n, Lc / max(dc, 0.001)), 0.0);
      float atten = 1.0 / (1.0 + 0.3 * dc * dc);
      float csh = cab_shadow(vCamRel);
      light += u.interior_light.rgb * (0.12 + ndlc * atten * csh);
    } else {
      light += u.interior_light.rgb;
    }
  }

  // Dynamic point/spot lights (headlights, lamps, signals).
  int c = u.count.x;
  for (int i = 0; i < c; ++i) {
    vec3 L = u.lights[i].pos.xyz - vCamRel;
    float d = length(L);
    vec3 Ld = L / max(d, 0.001);
    float range = max(u.lights[i].dir.w, 0.001);
    float atten = clamp(1.0 - d / range, 0.0, 1.0);
    atten *= atten;
    float ndl = max(dot(n, Ld), 0.0);
    float spot = 1.0;
    if (u.lights[i].pos.w > 0.5) {  // spot cone
      float cd = dot(normalize(u.lights[i].dir.xyz), -Ld);
      float co = u.lights[i].color.a;   // cos(outer)
      float ci = u.lights[i].extra.x;   // cos(inner)
      spot = clamp((cd - co) / max(ci - co, 0.001), 0.0, 1.0);
    }
    light += u.lights[i].color.rgb * (ndl * atten * spot);
  }

  // Self-illumination brightens the texture so gauges/lamps glow in the dark.
  outColor = vec4(tex.rgb * (light + vEmission), tex.a);
}
