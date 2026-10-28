#version 460 core

/*
 * 3D Gaussian Splatting — Fragment Shader (raw output)
 *
 * Matches the PlayCanvas/web convention exactly: the splat color (DC + SH)
 * is a GAMMA-SPACE (display-ready) value. It is output premultiplied and
 * blended RAW in code space on the UNORM offscreen target — no linear/sRGB
 * conversions anywhere in the splat path.
 *
 * Falloff: normalized exponential (PlayCanvas normExp), support |uv| <= 1.
 */

layout(location = 0) in vec4 vColor;
layout(location = 1) in vec2 vOffset;
layout(location = 2) in float vOpacity;

layout(location = 0) out vec4 outColor;

const float EXP4 = exp(-4.0);
const float INV_EXP4 = 1.0 / (1.0 - EXP4);

float normExp(float x) {
    return (exp(x * -4.0) - EXP4) * INV_EXP4;
}

void main() {
    float A = dot(vOffset, vOffset);
    if (A > 1.0) {
        discard;
    }

    float alpha = normExp(A) * vOpacity;
    if (alpha < 1.0 / 255.0) {
        discard;
    }

    // Raw gamma-space color, premultiplied output
    outColor = vec4(vColor.rgb * alpha, alpha);
}
