#if ENVMAP_ENABLED
uniform samplerCube envmap;
#endif

vec3 envmap_color( vec3 normal )
{
#if ENVMAP_ENABLED
	vec3 refvec = reflect(f_pos.xyz, normal); // view space
	refvec = vec3(inv_view * vec4(refvec, 0.0)); // world space
	vec3 envcolor = texture(envmap, refvec).rgb;
	// Sanitize. The env cubemap can briefly contain NaN/Inf after a
	// reflection-pass regeneration (m_empty_cubemap was bound during the
	// face render and on NVIDIA can sample as undefined). Without this
	// guard, a single bad texel propagates through env_reflection in
	// apply_lights() and produces one-frame black flashes on glossy
	// (esp. specgloss) surfaces.
	if (any(isnan(envcolor)) || any(isinf(envcolor))) envcolor = vec3(0.0);
	envcolor = max(envcolor, vec3(0.0));
#else
	vec3 envcolor = vec3(0.5);
#endif
	return envcolor;
}
