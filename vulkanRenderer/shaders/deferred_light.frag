#version 450

// Deferred lighting: read the G-buffer + the same light UBO / shadow maps as the
// forward path and shade each pixel. Background pixels (no geometry) are
// discarded so the sky drawn underneath shows through.

layout(location = 0) in vec2 vUV;

layout(set = 0, binding = 0) uniform sampler2D gAlbedo;    // rgb albedo, a emission
layout(set = 0, binding = 1) uniform sampler2D gNormal;    // rgb normal, a interior
layout(set = 0, binding = 2) uniform sampler2D gPosition;  // rgb camera-rel pos

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
  vec4 cab_light;
  ivec4 count;
  GpuLight lights[8];
} u;

layout(set = 1, binding = 1) uniform sampler2DArrayShadow uShadow;     // PCF
layout(set = 1, binding = 2) uniform sampler2DArray uShadowDepth;     // raw depth

layout(location = 0) out vec4 outColor;

// PCSS: a blocker search estimates how far the occluder is, then the PCF kernel
// grows with that distance -> sharp contact shadows, soft far penumbrae.
float sun_shadow(vec3 camrel) {
  float dist = length(camrel);
  int c = dist < u.cascade_splits.x ? 0 : (dist < u.cascade_splits.y ? 1 : 2);
  vec4 ls = u.lightspace[c] * vec4(camrel, 1.0);
  if (ls.w <= 0.0) return 1.0;
  vec3 p = ls.xyz / ls.w;
  p.xy = p.xy * 0.5 + 0.5;
  if (p.xy != clamp(p.xy, 0.0, 1.0) || p.z > 1.0) return 1.0;
  float recv = p.z;
  float bias = 0.0015;
  vec2 texel = 1.0 / vec2(textureSize(uShadow, 0).xy);

  // 1. Blocker search: average depth of texels nearer the light than us.
  float blockerSum = 0.0;
  int blockerCount = 0;
  for (int y = -2; y <= 2; ++y)
    for (int x = -2; x <= 2; ++x) {
      float d = texture(uShadowDepth,
                        vec3(p.xy + vec2(x, y) * texel * 1.5, float(c))).r;
      if (d < recv - bias) {
        blockerSum += d;
        blockerCount++;
      }
    }
  if (blockerCount == 0) return 1.0;  // unoccluded
  float avgBlocker = blockerSum / float(blockerCount);

  // 2. Penumbra (similar triangles); grows with receiver-to-blocker distance.
  float penumbra = (recv - avgBlocker) / max(avgBlocker, 1e-4);
  float radius = clamp(penumbra * 40.0, 1.0, 6.0);  // PCF kernel in texels

  // 3. PCF with the penumbra-sized kernel.
  float s = 0.0;
  for (int y = -2; y <= 2; ++y)
    for (int x = -2; x <= 2; ++x)
      s += texture(uShadow, vec4(p.xy + vec2(x, y) * texel * radius * 0.5,
                                 float(c), recv - bias));
  return s / 25.0;
}

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

void main() {
  vec4 pos = texture(gPosition, vUV);
  if (pos.a < 0.5) discard;  // background: keep the sky drawn underneath

  vec3 camrel = pos.xyz;
  vec4 alb = texture(gAlbedo, vUV);
  vec4 nrm = texture(gNormal, vUV);
  vec3 n = normalize(nrm.xyz);
  float emission = alb.a;
  float interior = nrm.a;

  float ndl = max(dot(n, normalize(-u.sun_dir.xyz)), 0.0);
  float shadow = sun_shadow(camrel);
  vec3 light = max(u.ambient.rgb, vec3(0.2)) + u.sun_color.rgb * (ndl * shadow);

  if (interior > 0.5) {
    if (u.cab_light.w > 0.5) {
      vec3 Lc = u.cab_light.xyz - camrel;
      float dc = length(Lc);
      float ndlc = max(dot(n, Lc / max(dc, 0.001)), 0.0);
      float atten = 1.0 / (1.0 + 0.3 * dc * dc);
      light += u.interior_light.rgb * (0.12 + ndlc * atten * cab_shadow(camrel));
    } else {
      light += u.interior_light.rgb;
    }
  }

  int c = u.count.x;
  for (int i = 0; i < c; ++i) {
    vec3 L = u.lights[i].pos.xyz - camrel;
    float d = length(L);
    vec3 Ld = L / max(d, 0.001);
    float range = max(u.lights[i].dir.w, 0.001);
    float atten = clamp(1.0 - d / range, 0.0, 1.0);
    atten *= atten;
    float ndlp = max(dot(n, Ld), 0.0);
    float spot = 1.0;
    if (u.lights[i].pos.w > 0.5) {
      float cd = dot(normalize(u.lights[i].dir.xyz), -Ld);
      float co = u.lights[i].color.a;
      float ci = u.lights[i].extra.x;
      spot = clamp((cd - co) / max(ci - co, 0.001), 0.0, 1.0);
    }
    light += u.lights[i].color.rgb * (ndlp * atten * spot);
  }

  outColor = vec4(alb.rgb * (light + emission), 1.0);
}
