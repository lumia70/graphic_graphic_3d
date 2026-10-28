#version 460 core
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

// Copy pass: UNORM offscreen -> sRGB backbuffer.
// The offscreen holds RAW (gamma-space) blended splat colors, exactly like
// the web viewer's canvas. The sRGB backbuffer hardware-encodes whatever we
// write, so we decode here to cancel it: stored code == offscreen value.
// No blending — the pass replaces the backbuffer content (the offscreen
// already contains the composited background).

layout(set = 0, binding = 0) uniform texture2D uScene;
layout(set = 0, binding = 1) uniform sampler uSampler;

layout(location = 0) in vec2 vUV;

layout(location = 0) out vec4 outColor;

vec3 decodeSrgb(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.0031308), c));
}

void main() {
    vec4 s = textureLod(sampler2D(uScene, uSampler), vUV, 0.0);
    outColor = vec4(decodeSrgb(s.rgb), 1.0);
}
