#version 450

// Scene geometry vertex (gfx::basic_vertex). MVP in the push constant plus a
// per-frame directional sun (direction/colour/ambient). Lighting is computed
// per-vertex here; the fragment stage just modulates the texture by it.

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;
layout(location = 3) in vec4 aTangent;

layout(push_constant) uniform PushConstants {
  mat4 uMVP;        // offset 0
  vec4 uSunDir;     // offset 64 (xyz: travel direction)
  vec4 uSunColor;   // offset 80
  vec4 uAmbient;    // offset 96
  vec4 uMisc;       // offset 112 (.x: material opacity)
} pc;

layout(location = 0) out vec3 vLight;
layout(location = 1) out vec2 vUV;
layout(location = 2) out float vOpacity;

void main() {
  gl_Position = pc.uMVP * vec4(aPos, 1.0);
  vUV = aUV;
  vec3 n = normalize(aNormal);
  float ndl = max(dot(n, normalize(-pc.uSunDir.xyz)), 0.0);
  vec3 ambient = max(pc.uAmbient.rgb, vec3(0.2));  // floor to avoid black
  vLight = ambient + pc.uSunColor.rgb * ndl;
  vOpacity = pc.uMisc.x;
}
