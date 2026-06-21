#version 450

// Scene geometry vertex format = gfx::basic_vertex
// (position, normal, texture, tangent). Texture/tangent are declared so the
// vertex input matches the engine's buffers; they're unused until texturing
// lands. A single MVP push constant transforms to clip space.

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;
layout(location = 3) in vec4 aTangent;

layout(push_constant) uniform PushConstants {
  mat4 uMVP;
} pc;

layout(location = 0) out vec3 vNormal;

void main() {
  gl_Position = pc.uMVP * vec4(aPos, 1.0);
  vNormal = aNormal;
}
