#version 450

// Scene geometry vertex format = gfx::basic_vertex
// (position, normal, texture, tangent). A single MVP push constant transforms
// to clip space; normal + UV are passed to the fragment stage.

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;
layout(location = 3) in vec4 aTangent;

layout(push_constant) uniform PushConstants {
  mat4 uMVP;
} pc;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec2 vUV;

void main() {
  gl_Position = pc.uMVP * vec4(aPos, 1.0);
  vNormal = aNormal;
  vUV = aUV;
}
