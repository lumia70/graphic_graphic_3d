/*
 * splat_loader.h — .splat (PLY packed format) parser for 3D Gaussian Splatting
 *
 * File format (binary little-endian PLY):
 *   element chunk (N):  48 bytes each — AABB min/max xyz + scale min/max xyz (6×float×2)
 *   element vertex (M): 16 bytes each — packed_position, packed_rotation, packed_scale, packed_color (4×uint32)
 *
 * Each chunk covers 256 consecutive vertices. Vertex positions and scales are
 * quantized relative to their chunk's AABB and scale range.
 *
 * Decoded per-gaussian struct for GPU upload:
 *   vec3 position   (world space)
 *   vec3 scale      (exp space — already exponentiated after dequantization)
 *   vec4 rotation   (unit quaternion wxyz)
 *   vec4 color      (RGBA [0,1])
 */
#pragma once

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

struct SplatChunk {
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
    float minScaleX, minScaleY, minScaleZ;
    float maxScaleX, maxScaleY, maxScaleZ;
};
static_assert(sizeof(SplatChunk) == 48, "SplatChunk must be 48 bytes");

struct PackedVertex {
    uint32_t packedPosition;
    uint32_t packedRotation;
    uint32_t packedScale;
    uint32_t packedColor;
};
static_assert(sizeof(PackedVertex) == 16, "PackedVertex must be 16 bytes");

// GPU-ready gaussian struct: 64 bytes, 16-byte aligned
struct alignas(16) GaussianSplatGPU {
    float px, py, pz, pad0;     // 16 bytes
    float sx, sy, sz, opacity;  // 16 bytes
    float qw, qx, qy, qz;      // 16 bytes
    float cr, cg, cb, ca;       // 16 bytes
};
static_assert(sizeof(GaussianSplatGPU) == 64, "GaussianSplatGPU must be 64 bytes");

struct SplatData {
    std::vector<SplatChunk> chunks;
    std::vector<GaussianSplatGPU> gaussians;
    // SH coefficients, 48 floats per gaussian (f_rest_0..44 + 3 pad).
    // Linear PLY layout: f_rest_m where m = j*3+c (j=basis index, c=channel).
    std::vector<float> sh;
    uint32_t gaussianCount = 0;
    float boundsMin[3] = {1e30f, 1e30f, 1e30f};
    float boundsMax[3] = {-1e30f, -1e30f, -1e30f};
};

// Decode packed 10-10-10-2 position within chunk AABB
inline void DecodePosition(uint32_t packed, const SplatChunk& chunk, float& x, float& y, float& z) {
    uint32_t ix = (packed >> 0) & 0x3FF;   // 10 bits
    uint32_t iy = (packed >> 10) & 0x3FF;  // 10 bits
    uint32_t iz = (packed >> 20) & 0x3FF;  // 10 bits
    // Normalize to [0, 1]
    float fx = static_cast<float>(ix) / 1023.0f;
    float fy = static_cast<float>(iy) / 1023.0f;
    float fz = static_cast<float>(iz) / 1023.0f;
    // Lerp within chunk AABB
    x = chunk.minX + fx * (chunk.maxX - chunk.minX);
    y = chunk.minY + fy * (chunk.maxY - chunk.minY);
    z = chunk.minZ + fz * (chunk.maxZ - chunk.minZ);
}

// Decode packed 10-10-10-2 scale within chunk scale range
inline void DecodeScale(uint32_t packed, const SplatChunk& chunk, float& sx, float& sy, float& sz) {
    uint32_t ix = (packed >> 0) & 0x3FF;
    uint32_t iy = (packed >> 10) & 0x3FF;
    uint32_t iz = (packed >> 20) & 0x3FF;
    float fx = static_cast<float>(ix) / 1023.0f;
    float fy = static_cast<float>(iy) / 1023.0f;
    float fz = static_cast<float>(iz) / 1023.0f;
    // Scales are in log space in the chunk; dequantize then exp
    float logSx = chunk.minScaleX + fx * (chunk.maxScaleX - chunk.minScaleX);
    float logSy = chunk.minScaleY + fy * (chunk.maxScaleY - chunk.minScaleY);
    float logSz = chunk.minScaleZ + fz * (chunk.maxScaleZ - chunk.minScaleZ);
    sx = std::exp(logSx);
    sy = std::exp(logSy);
    sz = std::exp(logSz);
}

// Decode packed rotation: 4 × 8-bit normalized quaternion components
// Stored as unsigned bytes [0,255] mapping to [-1, 1]
inline void DecodeRotation(uint32_t packed, float& qw, float& qx, float& qy, float& qz) {
    uint8_t bw = (packed >> 0) & 0xFF;
    uint8_t bx = (packed >> 8) & 0xFF;
    uint8_t by = (packed >> 16) & 0xFF;
    uint8_t bz = (packed >> 24) & 0xFF;
    qw = (static_cast<float>(bw) / 127.5f) - 1.0f;
    qx = (static_cast<float>(bx) / 127.5f) - 1.0f;
    qy = (static_cast<float>(by) / 127.5f) - 1.0f;
    qz = (static_cast<float>(bz) / 127.5f) - 1.0f;
    // Normalize
    float len = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
    if (len > 1e-8f) {
        float inv = 1.0f / len;
        qw *= inv; qx *= inv; qy *= inv; qz *= inv;
    } else {
        qw = 1.0f; qx = qy = qz = 0.0f;
    }
}

// Decode packed RGBA color
inline void DecodeColor(uint32_t packed, float& r, float& g, float& b, float& a) {
    r = static_cast<float>((packed >> 0) & 0xFF) / 255.0f;
    g = static_cast<float>((packed >> 8) & 0xFF) / 255.0f;
    b = static_cast<float>((packed >> 16) & 0xFF) / 255.0f;
    a = static_cast<float>((packed >> 24) & 0xFF) / 255.0f;
}

// Sigmoid for opacity
inline float Sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Parse PLY header, return offset to binary data
// Returns false on error
inline bool ParsePlyHeader(FILE* f, uint32_t& chunkCount, uint32_t& vertexCount) {
    chunkCount = 0;
    vertexCount = 0;
    char line[256];
    bool inHeader = true;
    
    while (inHeader && fgets(line, sizeof(line), f)) {
        // Remove trailing newline
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
            line[--len] = '\0';
        }
        
        if (strncmp(line, "element chunk ", 14) == 0) {
            chunkCount = static_cast<uint32_t>(atoi(line + 14));
        } else if (strncmp(line, "element vertex ", 15) == 0) {
            vertexCount = static_cast<uint32_t>(atoi(line + 15));
        } else if (strcmp(line, "end_header") == 0) {
            inHeader = false;
        }
    }
    return (chunkCount > 0 && vertexCount > 0);
}

// Load a .splat PLY file and decode all gaussians
inline bool LoadSplatFile(const char* path, SplatData& out) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    
    uint32_t chunkCount = 0, vertexCount = 0;
    if (!ParsePlyHeader(f, chunkCount, vertexCount)) {
        fclose(f);
        return false;
    }
    
    // Read chunk table
    out.chunks.resize(chunkCount);
    size_t readChunks = fread(out.chunks.data(), sizeof(SplatChunk), chunkCount, f);
    if (readChunks != chunkCount) {
        fclose(f);
        return false;
    }
    
    // Read packed vertices
    std::vector<PackedVertex> packedVerts(vertexCount);
    size_t readVerts = fread(packedVerts.data(), sizeof(PackedVertex), vertexCount, f);
    fclose(f);
    if (readVerts != vertexCount) {
        return false;
    }
    
    // Decode all gaussians
    out.gaussians.resize(vertexCount);
    out.gaussianCount = vertexCount;
    
    for (uint32_t i = 0; i < vertexCount; i++) {
        uint32_t chunkIdx = i / 256;
        if (chunkIdx >= chunkCount) chunkIdx = chunkCount - 1;
        const auto& chunk = out.chunks[chunkIdx];
        const auto& pv = packedVerts[i];
        auto& g = out.gaussians[i];
        
        DecodePosition(pv.packedPosition, chunk, g.px, g.py, g.pz);
        DecodeScale(pv.packedScale, chunk, g.sx, g.sy, g.sz);
        DecodeRotation(pv.packedRotation, g.qw, g.qx, g.qy, g.qz);
        
        float r, gb, b, a;
        DecodeColor(pv.packedColor, r, gb, b, a);
        g.cr = r;
        g.cg = gb;
        g.cb = b;
        // Alpha channel in .splat is the opacity (already in [0,1] after /255)
        g.opacity = a;
        g.ca = a;
        
        g.pad0 = 0.0f;
        
        // Track bounds
        if (g.px < out.boundsMin[0]) out.boundsMin[0] = g.px;
        if (g.py < out.boundsMin[1]) out.boundsMin[1] = g.py;
        if (g.pz < out.boundsMin[2]) out.boundsMin[2] = g.pz;
        if (g.px > out.boundsMax[0]) out.boundsMax[0] = g.px;
        if (g.py > out.boundsMax[1]) out.boundsMax[1] = g.py;
        if (g.pz > out.boundsMax[2]) out.boundsMax[2] = g.pz;
    }
    
    return true;
}

// ---------------------------------------------------------------------------
// Standard INRIA 3DGS PLY format loader (fruit_bowl.ply etc.)
//
// Binary PLY with float properties: x y z, rot_0..3, scale_0..2, opacity,
// f_dc_0..2 (SH DC color), optional f_rest_* (higher SH, ignored).
//
// Decoding (per 3DGS spec):
//   position = (x, y, z)
//   color    = 0.5 + SH_C0 * f_dc_i        (SH_C0 = 0.28209479177387814)
//   opacity  = sigmoid(opacity)
//   scale    = exp(scale_i)
//   rotation = normalize(rot_0..3)          (w, x, y, z)
//
// worldScale multiplies positions AND scales (models are often tiny, e.g. ±0.15).
// ---------------------------------------------------------------------------

// Load a standard 3DGS PLY and decode all gaussians
inline bool LoadStandard3DgsPly(const char* path, SplatData& out, float worldScale = 100.0f)
{
    FILE* f = fopen(path, "rb");
    if (!f) return false;

    // Parse header, collecting property names in order
    char line[512];
    uint32_t vertexCount = 0;
    std::vector<std::string> propNames;
    bool inHeader = true;
    while (inHeader && fgets(line, sizeof(line), f)) {
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
            line[--len] = '\0';
        }
        if (strncmp(line, "element vertex ", 15) == 0) {
            vertexCount = static_cast<uint32_t>(atoi(line + 15));
        } else if (strncmp(line, "property float ", 15) == 0) {
            propNames.push_back(std::string(line + 15));
        } else if (strncmp(line, "end_header", 10) == 0) {
            inHeader = false;
        }
    }
    if (vertexCount == 0 || propNames.size() < 14) {
        fclose(f);
        return false;
    }

    // Locate property offsets by name
    const size_t propCount = propNames.size();
    auto findProp = [&](const char* name) -> int {
        for (size_t i = 0; i < propCount; i++) {
            if (propNames[i] == name) return static_cast<int>(i);
        }
        return -1;
    };
    const int ix = findProp("x"), iy = findProp("y"), iz = findProp("z");
    const int ir0 = findProp("rot_0"), ir1 = findProp("rot_1"), ir2 = findProp("rot_2"), ir3 = findProp("rot_3");
    const int is0 = findProp("scale_0"), is1 = findProp("scale_1"), is2 = findProp("scale_2");
    const int iop = findProp("opacity");
    const int idc0 = findProp("f_dc_0"), idc1 = findProp("f_dc_1"), idc2 = findProp("f_dc_2");
    if (ix < 0 || iy < 0 || iz < 0 || ir0 < 0 || ir1 < 0 || ir2 < 0 || ir3 < 0 ||
        is0 < 0 || is1 < 0 || is2 < 0 || iop < 0 || idc0 < 0 || idc1 < 0 || idc2 < 0) {
        fclose(f);
        return false;
    }

    const size_t stride = propCount * sizeof(float);  // bytes per vertex
    constexpr float SH_C0 = 0.28209479177387814f;

    out.gaussians.resize(vertexCount);
    out.gaussianCount = vertexCount;

    // Locate f_rest property indices (may be absent in low-degree files)
    int restIdx[45];
    bool hasRest = false;
    for (int m = 0; m < 45; m++) {
        char name[16];
        snprintf(name, sizeof(name), "f_rest_%d", m);
        restIdx[m] = findProp(name);
        if (restIdx[m] >= 0) hasRest = true;
    }
    if (hasRest) {
        out.sh.assign((size_t)vertexCount * 48, 0.0f);
    }

    // Chunked read to bound peak memory
    constexpr uint32_t CHUNK_VERTS = 8192;
    std::vector<float> buf(CHUNK_VERTS * propCount);
    uint32_t decoded = 0;
    while (decoded < vertexCount) {
        const uint32_t n = (vertexCount - decoded < CHUNK_VERTS) ? (vertexCount - decoded) : CHUNK_VERTS;
        if (fread(buf.data(), stride, n, f) != n) {
            fclose(f);
            return false;
        }
        for (uint32_t j = 0; j < n; j++) {
            const float* v = &buf[(size_t)j * propCount];
            GaussianSplatGPU& g = out.gaussians[decoded + j];

            g.px = v[ix] * worldScale;
            g.py = v[iy] * worldScale;
            g.pz = v[iz] * worldScale;
            g.pad0 = 0.0f;

            g.sx = std::exp(v[is0]) * worldScale;
            g.sy = std::exp(v[is1]) * worldScale;
            g.sz = std::exp(v[is2]) * worldScale;
            g.opacity = 1.0f / (1.0f + std::exp(-v[iop]));
            g.ca = g.opacity;

            // Quaternion (w, x, y, z), normalized
            float qw = v[ir0], qx = v[ir1], qy = v[ir2], qz = v[ir3];
            float ql = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
            if (ql > 1e-8f) {
                qw /= ql; qx /= ql; qy /= ql; qz /= ql;
            } else {
                qw = 1.0f; qx = qy = qz = 0.0f;
            }
            g.qw = qw; g.qx = qx; g.qy = qy; g.qz = qz;

            // SH DC term → base color
            g.cr = 0.5f + SH_C0 * v[idc0];
            g.cg = 0.5f + SH_C0 * v[idc1];
            g.cb = 0.5f + SH_C0 * v[idc2];

            // SH degree 1..3 coefficients (f_rest linear layout)
            if (hasRest) {
                float* dst = &out.sh[(size_t)(decoded + j) * 48];
                for (int m = 0; m < 45; m++) {
                    dst[m] = (restIdx[m] >= 0) ? v[restIdx[m]] : 0.0f;
                }
            }

            // Track bounds
            if (g.px < out.boundsMin[0]) out.boundsMin[0] = g.px;
            if (g.py < out.boundsMin[1]) out.boundsMin[1] = g.py;
            if (g.pz < out.boundsMin[2]) out.boundsMin[2] = g.pz;
            if (g.px > out.boundsMax[0]) out.boundsMax[0] = g.px;
            if (g.py > out.boundsMax[1]) out.boundsMax[1] = g.py;
            if (g.pz > out.boundsMax[2]) out.boundsMax[2] = g.pz;
        }
        decoded += n;
    }
    fclose(f);
    return true;
}
