#version 450

// Scene geometry vertex (gfx::basic_vertex). The push constant holds the
// camera-relative model matrix; the per-frame light UBO (set 1) holds the
// view-projection plus the sun and dynamic lights. Lighting is per-fragment,
// so we pass the camera-relative position and world normal through.

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;
layout(location = 3) in vec4 aTangent;

layout(push_constant) uniform PushConstants {
  mat4 uLocal;   // offset 0  : camera-relative model matrix
  vec4 uMisc;    // offset 64 : .x alpha-test threshold, .y emission, .w mode
  ivec2 uTex;    // offset 80 : .x diffuse bindless slot, .y normal slot
  float uGloss;  // offset 88 : material glossiness (specular exponent)
} pc;

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
  vec4 interior_light;   // cab glow (colour * level)
  mat4 lightspace[4];    // [0..2] sun cascades, [3] cab light
  vec4 cascade_splits;   // .xyz: far distance of each cascade
  vec4 cab_light;        // xyz: camera-relative pos, w: enable
  vec4 fog;              // rgb: fog colour, a: visibility range
  ivec4 count;           // x: active light count
  GpuLight lights[8];
} u;

layout(location = 0) out vec3 vCamRel;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec2 vUV;
layout(location = 3) out float vOpacity;
layout(location = 4) out float vEmission;
layout(location = 5) out float vInterior;  // 1 for cab geometry, else 0
layout(location = 6) out vec3 vTangent;    // world-space TBN basis
layout(location = 7) out vec3 vBitangent;
layout(location = 8) out float vHasNormal; // 0 none, 1 normalmap, 2 +parallax
layout(location = 9) flat out int vTexDiffuse;   // bindless slot indices
layout(location = 10) flat out int vTexNormal;
layout(location = 11) out float vGloss;          // material glossiness

void main() {
  vec3 camrel = (pc.uLocal * vec4(aPos, 1.0)).xyz;
  vCamRel = camrel;
  mat3 m = mat3(pc.uLocal);
  vNormal = m * aNormal;
  vTangent = m * aTangent.xyz;
  // Bitangent from N x T with the stored handedness (aTangent.w).
  vBitangent = cross(vNormal, vTangent) * aTangent.w;
  vUV = aUV;
  vOpacity = pc.uMisc.x;
  vEmission = pc.uMisc.y;
  vInterior = pc.uMisc.z;
  vHasNormal = pc.uMisc.w;
  vTexDiffuse = pc.uTex.x;
  vTexNormal = pc.uTex.y;
  vGloss = pc.uGloss;
  gl_Position = u.viewproj * vec4(camrel, 1.0);
}
