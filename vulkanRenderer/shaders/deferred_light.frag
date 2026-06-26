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
  vec4 fog;
  ivec4 count;
  GpuLight lights[8];
} u;

layout(set = 1, binding = 1) uniform sampler2DArrayShadow uShadow;     // PCF
layout(set = 1, binding = 2) uniform sampler2DArray uShadowDepth;     // raw depth

layout(location = 0) out vec4 outColor;

// 16-tap Poisson disk, reused for both the blocker search and the PCF kernel.
// Rotating it per pixel turns the regular-grid banding into cheap dithered noise.
const vec2 POISSON[16] = vec2[](
    vec2(-0.94201624, -0.39906216), vec2(0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870), vec2(0.34495938, 0.29387760),
    vec2(-0.91588581, 0.45771432), vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543, 0.27676845), vec2(0.97484398, 0.75648379),
    vec2(0.44323325, -0.97511554), vec2(0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023), vec2(0.79197514, 0.19090188),
    vec2(-0.24188840, 0.99706507), vec2(-0.81409955, 0.91437590),
    vec2(0.19984126, 0.78641367), vec2(0.14383161, -0.14100790));

// Sun softness: penumbra width as a fraction of the receiver->blocker distance
// (a directional light has no real size, so this is an artistic stand-in).
const float SUN_SIZE = 0.04;
const float SEARCH_RADIUS = 3.0;     // blocker-search radius, in texels
const float MAX_PCF = 8.0;           // max penumbra kernel radius, in texels
const float NORMAL_OFFSET = 2.0;     // receiver push-off, in texels of world
const float DEPTH_BIAS = 0.12;       // residual depth bias, in world units

// Interleaved gradient noise -> a per-pixel rotation angle for the kernel.
float ign(vec2 p) {
  return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

float cascade_half_extent(int c) {
  return c == 0 ? u.cascade_splits.x
                : (c == 1 ? u.cascade_splits.y : u.cascade_splits.z);
}

// PCSS for one cascade: a blocker search estimates the occluder distance, then
// the PCF kernel grows with it -> sharp contact shadows, soft far penumbrae.
float eval_cascade(int c, vec3 camrel, vec3 n, vec2 rotcs) {
  // Cascade geometry mirrors the CPU side: ortho half-extent R, depth 2R+200.
  float R = cascade_half_extent(c);
  float depth = 2.0 * R + 200.0;
  vec2 size = vec2(textureSize(uShadow, 0).xy);
  float texelWorld = 2.0 * R / size.x;  // world units per shadow texel (XY)

  // Slope-scaled normal-offset: push the receiver off the surface along its
  // normal so self-shadowing acne disappears without much peter-panning.
  float ndl = clamp(dot(n, normalize(-u.sun_dir.xyz)), 0.0, 1.0);
  float slope = sqrt(max(0.0, 1.0 - ndl * ndl)) / max(ndl, 0.15);  // tan(angle)
  vec3 wp = camrel + n * (texelWorld * NORMAL_OFFSET * (1.0 + slope));

  vec4 ls = u.lightspace[c] * vec4(wp, 1.0);
  if (ls.w <= 0.0) return 1.0;
  vec3 p = ls.xyz / ls.w;
  p.xy = p.xy * 0.5 + 0.5;
  if (p.xy != clamp(p.xy, 0.0, 1.0) || p.z > 1.0) return 1.0;
  float recv = p.z;
  float bias = DEPTH_BIAS / depth;  // world bias -> normalized [0,1] depth
  vec2 texel = 1.0 / size;

  // 1. Blocker search: average depth of taps nearer the light than the receiver.
  float blockerSum = 0.0;
  int blockerCount = 0;
  for (int i = 0; i < 16; ++i) {
    vec2 o = vec2(POISSON[i].x * rotcs.x - POISSON[i].y * rotcs.y,
                  POISSON[i].x * rotcs.y + POISSON[i].y * rotcs.x);
    float d = texture(uShadowDepth,
                      vec3(p.xy + o * SEARCH_RADIUS * texel, float(c))).r;
    if (d < recv - bias) {
      blockerSum += d;
      blockerCount++;
    }
  }
  if (blockerCount == 0) return 1.0;  // unoccluded
  float avgBlocker = blockerSum / float(blockerCount);

  // 2. Penumbra: receiver->blocker distance (in world units) scaled by the sun
  //    size, converted back to texels. Same world softness across cascades.
  float recvBlockerWorld = (recv - avgBlocker) * depth;
  float radius = clamp(recvBlockerWorld * SUN_SIZE / texelWorld, 0.5, MAX_PCF);

  // 3. PCF with the penumbra-sized, per-pixel-rotated kernel (hardware compare).
  float s = 0.0;
  for (int i = 0; i < 16; ++i) {
    vec2 o = vec2(POISSON[i].x * rotcs.x - POISSON[i].y * rotcs.y,
                  POISSON[i].x * rotcs.y + POISSON[i].y * rotcs.x);
    s += texture(uShadow,
                 vec4(p.xy + o * radius * texel, float(c), recv - bias));
  }
  return s / 16.0;
}

float sun_shadow(vec3 camrel, vec3 n, vec2 fragcoord) {
  float dist = length(camrel);
  float a = ign(fragcoord) * 6.2831853;
  vec2 rotcs = vec2(cos(a), sin(a));
  int c = dist < u.cascade_splits.x ? 0 : (dist < u.cascade_splits.y ? 1 : 2);
  float v = eval_cascade(c, camrel, n, rotcs);

  // Cross-fade into the next cascade over the last 10% of the split so the
  // resolution change at the boundary doesn't show as a hard seam.
  if (c < 2) {
    float split = c == 0 ? u.cascade_splits.x : u.cascade_splits.y;
    float t = smoothstep(split * 0.9, split, dist);
    if (t > 0.0) v = mix(v, eval_cascade(c + 1, camrel, n, rotcs), t);
  }
  return v;
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
  float shadow = sun_shadow(camrel, n, gl_FragCoord.xy);
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

  // Specular highlight from the sun (Blinn-Phong; glossiness from G-buffer .a).
  vec3 V = normalize(-camrel);
  vec3 Ls = normalize(-u.sun_dir.xyz);
  vec3 H = normalize(Ls + V);
  float gloss = max(pos.a, 1.0);
  float spec = pow(max(dot(n, H), 0.0), gloss) * shadow * step(0.0, dot(n, Ls));
  vec3 color = alb.rgb * (light + emission) + u.sun_color.rgb * (spec * 0.35);

  // Distance fog: blend to the horizon colour over the visibility range.
  float fdist = length(camrel);
  float fogf = 1.0 - exp(-pow(fdist * 1.5 / max(u.fog.a, 1.0), 2.0));
  color = mix(color, u.fog.rgb, clamp(fogf, 0.0, 1.0));

  outColor = vec4(color, 1.0);
}
