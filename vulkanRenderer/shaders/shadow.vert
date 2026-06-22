#version 450

// Sun depth pre-pass: render scene geometry from the sun's point of view into
// the shadow map. Shares the world pipeline layout (push: camera-relative model
// matrix; set 1: light UBO holding the light-space matrix). Depth-only.

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;
layout(location = 3) in vec4 aTangent;

layout(push_constant) uniform PushConstants {
  mat4 uLocal;
  vec4 uMisc;
} pc;

struct GpuLight {
  vec4 pos;
  vec4 dir;
  vec4 color;
  vec4 extra;
};

layout(set = 1, binding = 0) uniform LightData {
  mat4 viewproj;
  vec4 sun_dir;
  vec4 sun_color;
  vec4 ambient;
  vec4 interior_light;
  mat4 lightspace;
  ivec4 count;
  GpuLight lights[8];
} u;

void main() {
  gl_Position = u.lightspace * (pc.uLocal * vec4(aPos, 1.0));
}
