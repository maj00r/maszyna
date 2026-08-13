#if SHADOWMAP_ENABLED
in vec4 f_light_pos[MAX_CASCADES];
uniform sampler2DArrayShadow shadowmap;
#endif
uniform sampler2D headlightmap;

#include <envmapping.glsl>
#include <conversion.glsl>

float glossiness = 1.0;
float metalic = 0.0;

float length2(vec3 v)
{
        return dot(v, v);
}

float calc_shadow()
{
#if SHADOWMAP_ENABLED
	float distance = dot(f_pos.xyz, f_pos.xyz);
	uint cascade;
	for (cascade = 0U; cascade < MAX_CASCADES; cascade++)
		if (distance <= cascade_end[cascade])
			break;
	float dist_casc = distance / cascade_end[cascade];
	vec3 coords = f_light_pos[cascade].xyz / f_light_pos[cascade].w;
	if (coords.z < 0.0)
		return 0.0f;
		
		

	float bias = 0.00005f * float(cascade + 1U);
	vec2 texel = vec2(1.0) / vec2(textureSize(shadowmap, 0));
	//float radius = 1.0; f_light_pos[cascade].w; //0.5 + 2.0 * max(abs(2.0 * coords.x - 1.0), abs(2.0 * coords.y - 1.0));
	float radius = 1.0;
	float minradius = 0.0;
	if (cascade == 0U)
		minradius = 1.0;
	if (cascade < MAX_CASCADES - 1U)
		radius = mix(minradius, f_light_pos[cascade+1U].w/f_light_pos[cascade].w, dist_casc);
	else
		radius = 0.5;

#if defined(GL_ARB_gpu_shader5) || defined(GL_EXT_gpu_shader5) || __VERSION__ >= 400
	// Fast path -- replace the original 4x4 grid of individual hardware-PCF
	// lookups with 4 textureGather() calls. Each gather returns the 4 raw
	// shadow comparisons of a 2x2 texel footprint, so 4 gathers laid out at
	// (+-1, +-1) * radius * texel from the sample center cover the same 4x4
	// sample area as the original kernel; summing all 16 comparisons and
	// dividing by 16 reproduces the original loop's averaging. The cost on
	// the TMUs drops from 16 hardware-PCF samples to 4 gathers (the gather
	// path returns 4 values per fetch where the original needed 4 fetches),
	// roughly a 4x reduction in shadow-sample work. The only thing dropped
	// vs. the hardware-PCF path is the implicit bilinear blending inside
	// each 2x2 footprint -- effectively turning a tent-weighted kernel into
	// a box-weighted one of the same extent, which is imperceptible in
	// motion. calc_shadow() is by far the heaviest piece of the lighting
	// shader, so this is a measurable GPU saving on every shaded fragment.
	float refz = coords.z + bias;
	float layer = float(cascade);
	vec2 off = radius * texel;
	vec4 g0 = textureGather(shadowmap, vec3(coords.xy + vec2(-off.x, -off.y), layer), refz);
	vec4 g1 = textureGather(shadowmap, vec3(coords.xy + vec2( off.x, -off.y), layer), refz);
	vec4 g2 = textureGather(shadowmap, vec3(coords.xy + vec2(-off.x,  off.y), layer), refz);
	vec4 g3 = textureGather(shadowmap, vec3(coords.xy + vec2( off.x,  off.y), layer), refz);
	float shadow = dot(g0 + g1 + g2 + g3, vec4(1.0 / 16.0));
	return shadow;
#else
	// Fallback for drivers without textureGather on shadow samplers
	// (notably GLES 3.0 and any 3.3 desktop driver that doesn't expose
	// GL_ARB_texture_gather). Identical to the previous implementation.
	float shadow = 0.0;
	for (float y = -1.5; y <= 1.5; y += 1.0)
		for (float x = -1.5; x <= 1.5; x += 1.0)
			shadow += texture(shadowmap, vec4(coords.xy + vec2(x, y) * radius * texel, cascade, coords.z + bias) );
	shadow /= 16.0;

	return shadow;
#endif
#else
	return 0.0;
#endif
}

vec2 calc_light(vec3 light_dir, vec3 fragnormal)
{
	vec3 view_dir = normalize(vec3(0.0f, 0.0f, 0.0f) - f_pos.xyz);
	vec3 halfway_dir = normalize(light_dir + view_dir);

	float diffuse_v = max(dot(fragnormal, light_dir), 0.0);

	// Energy-conserving Blinn-Phong normalization:
	// (n+8)/(8*pi) ensures the specular lobe integrates to the same
	// total energy regardless of glossiness — low glossiness stays dim
	// and spreads wide (blurry), high glossiness is bright and tight (sharp).
	// Capped at 4.0 so very high glossiness (n>~92) does not produce pinhole
	// highlights that blow past the tonemap shoulder and read as burnt white.
	float n = max(glossiness, 0.01);
	float normalization = min((n + 8.0) / (8.0 * 3.14159265), 4.0);
	float NdotH = max(dot(fragnormal, halfway_dir), 0.0);
	float specular_v = normalization * pow(NdotH, n);

	return vec2(diffuse_v, specular_v);
}

vec2 calc_point_light(light_s light, vec3 fragnormal)
{
	vec3 light_dir = normalize(light.pos - f_pos.xyz);
	vec2 val = calc_light(light_dir, fragnormal);
	val.x += light.ambient;
	val *= light.intensity;
	
	float distance = length(light.pos - f_pos.xyz);
	float atten = 1.0f / (distance * distance);
	//float atten = 1.0f / (1.0f + light.linear * distance + light.quadratic * (distance * distance));
	
	return val * atten;
}

vec2 calc_spot_light(light_s light, vec3 fragnormal)
{
	vec3 light_dir = normalize(light.pos - f_pos.xyz);
	
	float theta = dot(light_dir, normalize(-light.dir));
	float epsilon = light.in_cutoff - light.out_cutoff;
	float intensity = clamp((theta - light.out_cutoff) / epsilon, 0.0, 1.0);

	vec2 point = calc_point_light(light, fragnormal);	
	return point * intensity;
}

vec2 calc_dir_light(light_s light, vec3 fragnormal)
{
	vec3 light_dir = normalize(-light.dir);
	return calc_light(light_dir, fragnormal);
}

vec2 calc_headlights(light_s light, vec3 fragnormal)
{
	vec4 headlightpos = light.headlight_projection * f_pos;
	vec3 coords = headlightpos.xyz / headlightpos.w;

	if (coords.z > 1.0)
		return vec2(0.0);
	if (coords.z < 0.0)
		return vec2(0.0);

	vec3 light_dir = normalize(light.pos - f_pos.xyz);
	// Tighter wrap (was +0.25): faces angled away from the headlight cone
	// fall off to dark much faster, so cab/exterior surfaces read with a
	// clear directional shape instead of a flat half-lit wash.
	vec2 part = vec2(1.0) * clamp(dot(fragnormal, light_dir) + 0.10, 0.0, 1.0);
	float distance = length(light.pos - f_pos.xyz);
	float atten = 1.0f / (1.0f + light.linear * distance + light.quadratic * (distance * distance));
	atten *= mix(1.0, 0.0, clamp((coords.z - 0.998) * 500.0, 0.0, 1.0));
	vec3 lights = textureProj(headlightmap, headlightpos).rgb * light.headlight_weights.rgb;
	float lightintensity = max(max(lights.r, lights.g), lights.b);
	return part * atten * lightintensity;
}

// [0] - diffuse, [1] - specular
// do magic here
vec3 apply_lights(vec3 fragcolor, vec3 fragnormal, vec3 texturecolor, float reflectivity, float specularity, float shadowtone)
{
    vec3 basecolor = param[0].rgb;
    fragcolor *= basecolor;

    vec3 emissioncolor = basecolor * emission;

    vec3 view_dir = normalize(-f_pos.xyz);
    float NdotV = max(dot(fragnormal, view_dir), 0.0);
    vec3 F0 = mix(vec3(0.04), texturecolor, metalic);
    vec3 fresnel = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);

    const float MAX_REFLECTION_LOD = 8.0;
    float env_roughness = 1.0 - clamp(glossiness / max(abs(param[1].w), 1.0), 0.0, 1.0);
    vec3 envcolor = envmap_color_lod(fragnormal, env_roughness * MAX_REFLECTION_LOD);

    // Tint texture toward fully-saturated under strong env, weighted by fresnel
    vec3 texturecoloryuv = rgb2yuv(texturecolor);
    vec3 texturecolorfullv = yuv2rgb(vec3(0.2176, texturecoloryuv.gb));
    vec3 envyuv = rgb2yuv(envcolor);
    texturecolor = mix(texturecolor, texturecolorfullv, envyuv.r * reflectivity * fresnel.r);

    if (lights_count == 0U)
        return (fragcolor + emissioncolor) * texturecolor
             + envcolor * fresnel * reflectivity;

    vec2 sunlight = calc_dir_light(lights[0], fragnormal);
    // Sharpen sun N.L falloff so the lit-to-shaded terminator on cab
    // panels, vehicle bodies and terrain reads as a clear edge rather
    // than a soft Lambertian ramp. Stays close to physical Lambert.
    float sun_NdotL = pow(sunlight.x, 1.25);
    float diffuseamount = sun_NdotL * param[1].x * lights[0].intensity;

    float shadow1 = 0.0;
    if (shadowtone < 1.0)
        shadow1 = (1.0 - shadowtone) * clamp(calc_shadow(), 0.0, 1.0);

    // Sun HDR scale lowered from 5.0 -> 3.5: with the previous broken
    // identity-Reinhard tonemap the 5x boost just clipped at 1.0; with
    // ACES tonemapping in place 3.5 keeps lit areas strong but leaves
    // real headroom for spec highlights instead of crushing to white.
    fragcolor += lights[0].color * 3.5 * (1.0 - shadow1) * diffuseamount;

    for (uint i = 1U; i < lights_count; i++)
    {
        light_s light = lights[i];
        vec2 part = calc_headlights(light, fragnormal);
        fragcolor += light.color * (part.x * param[1].x + part.y * param[1].y) * light.intensity;
    }

    float specularamount = sunlight.y * param[1].y * specularity * lights[0].intensity
                         * clamp(1.0 - shadowtone, 0.0, 1.0);
    if (shadowtone < 1.0)
        specularamount *= clamp(1.0 - shadow1, 0.0, 1.0);

    fragcolor += emissioncolor;
    vec3 specularcolor = specularamount * lights[0].color;

    // Env reflection tracked separately — must NOT go through the albedo multiply below
    vec3 env_reflection = envcolor * fresnel * reflectivity * (1.0 - shadow1 * 0.5);

    // Diffuse + specular: albedo tints diffuse, metals also tint specular
    vec3 result = mix(
        (fragcolor + specularcolor) * texturecolor,
         fragcolor * texturecolor + specularcolor,
        metalic);

    // Env added after albedo multiply: raw for dielectrics, albedo-tinted for metals
    result += mix(env_reflection, env_reflection * texturecolor, metalic);

    return result;
}