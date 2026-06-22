#version 450

layout(location = 0) in vec3 vCamRel;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vUV;
layout(location = 3) in float vOpacity;
layout(location = 4) in float vEmission;
layout(location = 5) in float vInterior;

layout(set = 0, binding = 0) uniform sampler2D uTexture;

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
  ivec4 count;
  GpuLight lights[8];
} u;

layout(location = 0) out vec4 outColor;

void main() {
  vec4 tex = texture(uTexture, vUV);
  // Alpha-test threshold (0 for blended atlases, ~0.5 for cutout); a small
  // floor drops fully transparent texels. Glass keeps its own alpha for blend.
  if (tex.a < max(vOpacity, 0.04)) discard;

  vec3 n = normalize(vNormal);
  // Directional sun + ambient floor + cab interior glow (cab geometry only).
  vec3 light = max(u.ambient.rgb, vec3(0.2)) +
               u.sun_color.rgb * max(dot(n, normalize(-u.sun_dir.xyz)), 0.0) +
               vInterior * u.interior_light.rgb;

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
