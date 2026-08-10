vec3 get_fog_color()
{
	vec3 fog_color_v = fog_color;
	if (lights_count >= 1U && lights[0].type == LIGHT_DIR) {
		float sun_amount = max(dot(normalize(f_pos.xyz), normalize(-lights[0].dir)), 0.0);
		fog_color_v += lights[0].color * pow(sun_amount, 30.0);
	}
	return fog_color_v;
}

float get_fog_amount()
{
	float z = length(f_pos.xyz);
//	float fog_amount_v = 1.0 - z * fog_density; // linear
//	float fog_amount_v = exp( -fog_density * z ); // exp
	float fog_amount_v = exp2( -fog_density * fog_density * z * z * 1.442695 ); // exp2

	fog_amount_v = 1.0 - clamp(fog_amount_v, 0.0, 1.0);
	return fog_amount_v;
}

// Haze that does not wait for the scenery's fog range, which is routinely set
// to several km - without it the horizon just stops.
const float AERIAL_RANGE = 3000.0;
const float AERIAL_MAX = 0.35;

float get_aerial_amount()
{
	return AERIAL_MAX * (1.0 - exp(-length(f_pos.xyz) / AERIAL_RANGE));
}

vec3 apply_fog(vec3 color)
{
	// Fills in only where the scenery's own fog is weak, never stacking on it.
	return mix(color, get_fog_color(), max(get_fog_amount(), get_aerial_amount()));
}

vec3 add_fog(vec3 color, float amount)
{
	return color + get_fog_color() * get_fog_amount() * amount;
}
