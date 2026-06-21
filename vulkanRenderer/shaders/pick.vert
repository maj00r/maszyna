#version 450

// Control picking: draw cab geometry with a flat per-submodel ID colour into
// an offscreen target. Same vertex format as the world shader (basic_vertex).

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;
layout(location = 3) in vec4 aTangent;

layout(push_constant) uniform PushConstants {
  mat4 uMVP;     // offset 0
  vec4 uColor;   // offset 64 (encoded submodel index)
} pc;

layout(location = 0) out vec4 vColor;

void main() {
  gl_Position = pc.uMVP * vec4(aPos, 1.0);
  vColor = pc.uColor;
}
