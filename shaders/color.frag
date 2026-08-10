in vec3 f_color;
in vec4 f_pos;

#include <common>
#include <apply_fog.glsl>
#include <tonemapping.glsl>

layout(location = 0) out vec4 out_color;
#if MOTIONBLUR_ENABLED
layout(location = 1) out vec4 out_motion;
#endif

void main()
{
	// Skydome only. Gamma decode is meaningful below 1; above it sits Perez's
	// circumsolar term, which washes half the dome without a knee.
	const float SKY_GLOW_ROLLOFF = 1.5;
	vec3 sky = max(f_color.rgb, 0.0);
	vec3 excess = max(sky - 1.0, 0.0);
	vec3 col = pow(min(sky, 1.0), vec3(2.2)) + excess / (1.0 + excess * SKY_GLOW_ROLLOFF);
#if POSTFX_ENABLED
	out_color = vec4(apply_fog(col), 1.0f);
#else
    out_color = tonemap(vec4(apply_fog(col), 1.0f));
#endif
#if MOTIONBLUR_ENABLED
	out_motion = vec4(0.0f);
#endif
}
