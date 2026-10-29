#version 460 core
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

// Copy pass: UNORM offscreen -> sRGB backbuffer.
// The offscreen holds RAW (gamma-space) blended splat colours, exactly like
// the web viewer's canvas. The sRGB backbuffer hardware-encodes whatever we
// write, so we decode here to cancel it: stored code == offscreen value.
// A mild unsharp mask restores some micro-contrast lost versus the reference
// system renderer. No blending — the pass replaces the backbuffer content.

layout(set = 0, binding = 0) uniform texture2D uScene;
layout(set = 0, binding = 1) uniform sampler uSampler;

layout(location = 0) in vec2 vUV;

layout(location = 0) out vec4 outColor;

// Unsharp-mask strength (0 = off, ~0.3-0.5 mild, 1.0 strong)
const float SHARPEN = 0.35;

vec3 decodeSrgb(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.0031308), c));
}

vec3 sampleDecoded(vec2 uv) {
    return decodeSrgb(textureLod(sampler2D(uScene, uSampler), uv, 0.0).rgb);
}

void main() {
    vec2 texel = 1.0 / vec2(textureSize(sampler2D(uScene, uSampler), 0));

    vec3 c = sampleDecoded(vUV);
    vec3 blur = 0.25 * (
        sampleDecoded(vUV + vec2(texel.x, 0.0)) +
        sampleDecoded(vUV - vec2(texel.x, 0.0)) +
        sampleDecoded(vUV + vec2(0.0, texel.y)) +
        sampleDecoded(vUV - vec2(0.0, texel.y)));

    vec3 sharp = clamp(c + (c - blur) * SHARPEN, 0.0, 1.0);
    outColor = vec4(sharp, 1.0);
}
