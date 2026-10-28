#version 460 core

/*
 * 3D Gaussian Splatting — Vertex Shader
 *
 * Anisotropic covariance projection + full SH degree-3 view-dependent color:
 *   Sigma_world = R * S * S^T * R^T
 *   Sigma_view  = W * Sigma_world * W^T          (W = view rotation)
 *   cov2d       = J * Sigma_view * J^T           (J = perspective Jacobian, y-down)
 *   quad        = eigen-axes of cov2d, 3-sigma extent
 *   color       = DC + SH3(dir), dir = normalize(pos - camPos)
 *
 * Gaussians are consumed in back-to-front sorted order (index buffer, binding 2).
 */

layout(set = 0, binding = 0, std140) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec4 viewport;
    vec4 cameraPos;
};

struct Gaussian {
    vec4 posAndPad;     // xyz = position
    vec4 scaleOpacity;  // xyz = scale, w = opacity
    vec4 rotation;      // (w, x, y, z) quaternion
    vec4 color;         // rgba (DC term already decoded)
};

layout(set = 0, binding = 1, std430) readonly buffer GaussianSSBO {
    Gaussian gaussians[];
};

layout(set = 0, binding = 2, std430) readonly buffer SortSSBO {
    uint sortedIndices[];
};

// SH coefficients: 48 floats per gaussian, linear PLY layout sh[base + j*3 + c]
layout(set = 0, binding = 3, std430) readonly buffer ShSSBO {
    float shArr[];
};

layout(location = 0) out vec4 vColor;
layout(location = 1) out vec2 vOffset;
layout(location = 2) out float vOpacity;

const vec2 QUAD_OFFSETS[6] = vec2[6](
    vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
    vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0)
);

const float SH_C1 = 0.4886025119029199;
const float SH_C2[5] = float[5](
    1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
    -1.0925484305920792, 0.5462742152960396);
const float SH_C3[7] = float[7](
    -0.5900435899266435, 2.890611442640554, -0.4570457994644658,
    0.3731763325901154, -0.4570457994644658, 1.445305721320277,
    -0.5900435899266435);

// Evaluate SH degree 1..3 for all channels. dir = normalize(pos - camPos).
vec3 evalSH(uint idx, vec3 dir) {
    int base = int(idx) * 48;
    float x = dir.x, y = dir.y, z = dir.z;
    vec3 result = vec3(0.0);

    // degree 1: -y, z, -x
    result += SH_C1 * (-y) * vec3(shArr[base + 0], shArr[base + 1], shArr[base + 2]);
    result += SH_C1 * ( z) * vec3(shArr[base + 3], shArr[base + 4], shArr[base + 5]);
    result += SH_C1 * (-x) * vec3(shArr[base + 6], shArr[base + 7], shArr[base + 8]);

    // degree 2: xy, yz, (2zz-xx-yy), xz, (xx-yy)
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    result += SH_C2[0] * xy * vec3(shArr[base +  9], shArr[base + 10], shArr[base + 11]);
    result += SH_C2[1] * yz * vec3(shArr[base + 12], shArr[base + 13], shArr[base + 14]);
    result += SH_C2[2] * (2.0 * zz - xx - yy) * vec3(shArr[base + 15], shArr[base + 16], shArr[base + 17]);
    result += SH_C2[3] * xz * vec3(shArr[base + 18], shArr[base + 19], shArr[base + 20]);
    result += SH_C2[4] * (xx - yy) * vec3(shArr[base + 21], shArr[base + 22], shArr[base + 23]);

    // degree 3
    result += SH_C3[0] * y * (3.0 * xx - yy) * vec3(shArr[base + 24], shArr[base + 25], shArr[base + 26]);
    result += SH_C3[1] * xy * z * vec3(shArr[base + 27], shArr[base + 28], shArr[base + 29]);
    result += SH_C3[2] * y * (4.0 * zz - xx - yy) * vec3(shArr[base + 30], shArr[base + 31], shArr[base + 32]);
    result += SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * vec3(shArr[base + 33], shArr[base + 34], shArr[base + 35]);
    result += SH_C3[4] * x * (4.0 * zz - xx - yy) * vec3(shArr[base + 36], shArr[base + 37], shArr[base + 38]);
    result += SH_C3[5] * z * (xx - yy) * vec3(shArr[base + 39], shArr[base + 40], shArr[base + 41]);
    result += SH_C3[6] * x * (xx - 3.0 * yy) * vec3(shArr[base + 42], shArr[base + 43], shArr[base + 44]);

    return result;
}

mat3 quatToMat3(vec4 q) {
    float w = q.x, x = q.y, y = q.z, z = q.w;
    float x2 = x + x, y2 = y + y, z2 = z + z;
    float xx = x * x2, xy = x * y2, xz = x * z2;
    float yy = y * y2, yz = y * z2, zz = z * z2;
    float wx = w * x2, wy = w * y2, wz = w * z2;
    return mat3(
        1.0 - (yy + zz), xy + wz, xz - wy,
        xy - wz, 1.0 - (xx + zz), yz + wx,
        xz + wy, yz - wx, 1.0 - (xx + yy)
    );
}

void main() {
    uint gaussianIdx = sortedIndices[uint(gl_VertexIndex) / 6u];

    // Sentinel padding entry — skip (stale-frame safety)
    if (gaussianIdx == 0xFFFFFFFFu) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        vColor = vec4(0.0); vOffset = vec2(0.0); vOpacity = 0.0;
        return;
    }

    uint cornerIdx = uint(gl_VertexIndex) % 6u;

    Gaussian g = gaussians[gaussianIdx];

    vec3 worldPos = g.posAndPad.xyz;
    vec3 scale = g.scaleOpacity.xyz;
    float opacity = g.scaleOpacity.w;

    // Skip transparent
    if (opacity < 1.0 / 255.0) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        vColor = vec4(0.0); vOffset = vec2(0.0); vOpacity = 0.0;
        return;
    }

    vec4 viewPos = view * vec4(worldPos, 1.0);
    float z = -viewPos.z;

    // Near/far cull
    if (z < 0.05 || z > 100.0) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        vColor = vec4(0.0); vOffset = vec2(0.0); vOpacity = 0.0;
        return;
    }

    // Smooth fade above the near plane — prevents pop flicker when gaussians
    // cross behind the camera (wider band for close-up zoom).
    float nearFade = clamp((z - 0.05) / 0.15, 0.0, 1.0);

    // ---- View-dependent color: DC (precomputed) + SH degree 1..3 ----
    // SH direction = camera -> gaussian (incoming). Verified by two-viewpoint
    // trend analysis against the web reference: this convention matches the
    // web's per-channel angular color trends; the flipped direction produces
    // an opposite blue trend (phone blue +9% vs web -7% front->side).
    vec3 viewDir = normalize(worldPos - cameraPos.xyz);
    vec3 color = g.color.rgb + evalSH(gaussianIdx, viewDir);
    color = max(color, vec3(0.0));

    // ---- 3D covariance in world space ----
    mat3 R = quatToMat3(g.rotation);
    mat3 M = R * mat3(
        scale.x, 0.0, 0.0,
        0.0, scale.y, 0.0,
        0.0, 0.0, scale.z
    );
    mat3 Sigma = M * transpose(M);

    // ---- Transform to view space ----
    mat3 W = mat3(view);
    mat3 Sv = W * Sigma * transpose(W);

    // ---- Perspective Jacobian (y-down screen space, pixels) ----
    float fx = proj[0][0] * viewport.z * 0.5;         // positive
    float fy = -abs(proj[1][1]) * viewport.w * 0.5;   // negative: y-down
    float invz = 1.0 / z;
    vec3 Jx = vec3(fx * invz, 0.0, fx * viewPos.x * invz * invz);
    vec3 Jy = vec3(0.0, fy * invz, fy * viewPos.y * invz * invz);

    // ---- 2D covariance (2x2 symmetric) ----
    float c00 = dot(Jx, Sv * Jx);
    float c01 = dot(Jx, Sv * Jy);
    float c11 = dot(Jy, Sv * Jy);

    // Low-pass filter: minimum 0.3 pixel variance (matches fragment falloff)
    float a = c00 + 0.3;
    float b = c01;
    float c = c11 + 0.3;

    // Eigenvalues
    float mid = 0.5 * (a + c);
    float disc = sqrt(max(mid * mid - (a * c - b * b), 0.0));
    float l1 = mid + disc;
    float l2 = mid - disc;

    float r1 = 3.0 * sqrt(max(l1, 0.0));  // major radius (pixels, 3-sigma)
    float r2 = 3.0 * sqrt(max(l2, 0.0));  // minor radius

    // Cull degenerate only; clamp oversized to bound fill rate
    if (r1 < 0.5) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        vColor = vec4(0.0); vOffset = vec2(0.0); vOpacity = 0.0;
        return;
    }
    r1 = min(r1, 1536.0);
    r2 = min(r2, 1536.0);

    // Eigenvector of major axis: (b, l1 - a), fallback for b ~ 0
    vec2 e1;
    if (abs(b) > 1e-9) {
        e1 = normalize(vec2(b, l1 - a));
    } else {
        e1 = (a >= c) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    }
    vec2 e2 = vec2(-e1.y, e1.x);

    // Quad corner offset in pixels
    vec2 offset = QUAD_OFFSETS[cornerIdx];
    vec2 pixelOffset = e1 * (offset.x * r1) + e2 * (offset.y * r2);

    vec4 clipPos = proj * viewPos;
    clipPos.xy += pixelOffset * clipPos.w / (viewport.zw * 0.5);

    gl_Position = clipPos;
    vColor = vec4(color, 1.0);
    vOffset = offset;
    vOpacity = opacity * nearFade;
}
