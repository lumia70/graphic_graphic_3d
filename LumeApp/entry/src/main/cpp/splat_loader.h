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
#include <cstddef>
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

// Decode packed position within chunk AABB.
// super-splat compressed-chunk packing (verified against splat-transform
// src/lib/writers/compressed-chunk.ts): pack111011 = (x:11 << 21) | (y:10 << 11) | z:11
// i.e. LOW bits hold z, then y, then x in the top 11 bits.
inline void DecodePosition(uint32_t packed, const SplatChunk& chunk, float& x, float& y, float& z) {
    uint32_t iz = (packed >> 0) & 0x7FF;   // 11 bits
    uint32_t iy = (packed >> 11) & 0x3FF;  // 10 bits
    uint32_t ix = (packed >> 21) & 0x7FF;  // 11 bits
    float fx = static_cast<float>(ix) / 2047.0f;
    float fy = static_cast<float>(iy) / 1023.0f;
    float fz = static_cast<float>(iz) / 2047.0f;
    x = chunk.minX + fx * (chunk.maxX - chunk.minX);
    y = chunk.minY + fy * (chunk.maxY - chunk.minY);
    z = chunk.minZ + fz * (chunk.maxZ - chunk.minZ);
}

// Decode packed scale (same pack111011 layout as position) and exponentiate.
inline void DecodeScale(uint32_t packed, const SplatChunk& chunk, float& sx, float& sy, float& sz) {
    uint32_t iz = (packed >> 0) & 0x7FF;
    uint32_t iy = (packed >> 11) & 0x3FF;
    uint32_t ix = (packed >> 21) & 0x7FF;
    float fx = static_cast<float>(ix) / 2047.0f;
    float fy = static_cast<float>(iy) / 1023.0f;
    float fz = static_cast<float>(iz) / 2047.0f;
    // Scales are stored in log space in the chunk; dequantize then exp
    float logSx = chunk.minScaleX + fx * (chunk.maxScaleX - chunk.minScaleX);
    float logSy = chunk.minScaleY + fy * (chunk.maxScaleY - chunk.minScaleY);
    float logSz = chunk.minScaleZ + fz * (chunk.maxScaleZ - chunk.minScaleZ);
    sx = std::exp(logSx);
    sy = std::exp(logSy);
    sz = std::exp(logSz);
}

// Decode packed rotation (super-splat 2-10-10-10 packing):
//   bits 30-31: index of the largest-magnitude component (0..3 = x,y,z,w)
//   three 10-bit fields follow from bit 20 down to 0, holding the remaining
//   components in ascending index order, quantised as round((a*√2/2 + 0.5)*1023)
inline void DecodeRotation(uint32_t packed, float& qw, float& qx, float& qy, float& qz) {
    const uint32_t largest = (packed >> 30) & 0x3;
    float comps[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t shift = 20;
    for (uint32_t i = 0; i < 4; i++) {
        if (i == largest) continue;
        const uint32_t raw = (packed >> shift) & 0x3FF;
        shift -= 10;
        const float v = static_cast<float>(raw) / 1023.0f;
        comps[i] = (v - 0.5f) * 1.41421356f;  // undo (a*√2/2 + 0.5)
    }
    // Reconstruct the largest component (sign chosen positive at pack time)
    float sumSq = comps[0]*comps[0] + comps[1]*comps[1] + comps[2]*comps[2] + comps[3]*comps[3];
    comps[largest] = std::sqrt(sumSq < 1.0f ? (1.0f - sumSq) : 0.0f);
    // The writer calls packRot(rot_0..rot_3) which are PLY-ordered (w, x, y, z),
    // so the indexed components are (w, x, y, z) — not PlayCanvas' internal
    // (x, y, z, w). Mapping them the other way bends every ellipsoid incorrectly.
    qw = comps[0];
    qx = comps[1];
    qy = comps[2];
    qz = comps[3];
    // Normalize (quantisation noise)
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

// Decode a super-splat compressed PLY payload held in memory.
//
// The header declares a sequence of elements; this parser walks them in order
// and decodes every element it understands, keeping the SH data available for
// the render pipeline (which may or may not consume it):
//   chunk            12 floats  — AABB + log-scale range per 256 gaussians
//   vertex            4 uints   — packed position / rotation / scale / colour
//   feature_rest_vq   1 uint    — per-gaussian index into the SH codebook
//   feature_rest     45 chars   — vector-quantised SH codebook (f_rest_0..44)
// Inline per-vertex f_rest_0..44 float properties (uncompressed variant) are
// also supported. Unknown trailing elements are skipped by their declared size.
inline size_t PlyTypeSize(const char* type)
{
    if (!strcmp(type, "char") || !strcmp(type, "uchar") || !strcmp(type, "int8") || !strcmp(type, "uint8")) return 1;
    if (!strcmp(type, "short") || !strcmp(type, "ushort") || !strcmp(type, "int16") || !strcmp(type, "uint16")) return 2;
    if (!strcmp(type, "int") || !strcmp(type, "uint") || !strcmp(type, "int32") || !strcmp(type, "uint32") ||
        !strcmp(type, "float") || !strcmp(type, "float32")) return 4;
    if (!strcmp(type, "double") || !strcmp(type, "float64")) return 8;
    return 0;
}

struct PlyElementDesc {
    std::string name;
    uint32_t count = 0;
    size_t stride = 0;
    std::vector<std::pair<std::string, size_t>> props;  // (name, byte offset in record)
    size_t dataOffset = 0;                              // relative to header end
};

// Colour byte layouts seen in compressed splat files:
//  0 = RGBA bytes (b0=R, b1=G, b2=B, b3=A)      — legacy .splat variant
//  1 = official pack8888 (b0=A, b1=B, b2=G, b3=R)
//  2 = alpha-first, BGR (b0=A, b1=R, b2=G, b3=B)
// PLY / chunk SH rest is stored channel-major (f_rest_0..14 = red, 15..29 = green,
// 30..44 = blue), while the shader reads coefficient-major (sh[j*3 + c]).
inline void StoreShRestChannelMajorToCoefficientMajor(float* dst, const float* src)
{
    for (int c = 0; c < 3; c++) {
        for (int j = 0; j < 15; j++) {
            dst[j * 3 + c] = src[c * 15 + j];
        }
    }
}

inline bool LoadSplatPlyFromMemory(const uint8_t* data, size_t size, SplatData& out,
    const int colorOrder = 0)
{
    if (!data || size < 16) return false;
    const char* text = reinterpret_cast<const char*>(data);
    size_t hdrEnd = SIZE_MAX;
    for (size_t i = 0; i + 10 <= size; i++) {
        if (memcmp(text + i, "end_header", 10) == 0) {
            size_t j = i + 10;
            if (j < size && text[j] == '\r') j++;
            if (j < size && text[j] == '\n') j++;
            hdrEnd = j;
            break;
        }
    }
    if (hdrEnd == SIZE_MAX) return false;

    // ---- Parse the header into element/property descriptors and declares ----
    std::vector<PlyElementDesc> elements;
    float restScale = 1.0f;
    float restZeroPoint = 0.0f;
    float dcRangeMin = 0.0f;
    float dcRangeMax = 0.0f;
    bool hasDcMin = false;
    bool hasDcMax = false;
    size_t lineStart = 0;
    for (size_t i = 0; i <= hdrEnd; i++) {
        if (i != hdrEnd && text[i] != '\n') continue;
        size_t len = i - lineStart;
        while (len > 0 && (text[lineStart + len - 1] == '\r' || text[lineStart + len - 1] == ' ')) len--;
        if (len > 0) {
            std::string line(text + lineStart, len);
            if (line.rfind("element ", 0) == 0) {
                PlyElementDesc el;
                char name[64] = {0};
                unsigned count = 0;
                if (sscanf(line.c_str(), "element %63s %u", name, &count) == 2 && count > 0) {
                    el.name = name;
                    el.count = count;
                    elements.push_back(std::move(el));
                }
            } else if (line.rfind("property ", 0) == 0 && !elements.empty()) {
                char type[32] = {0};
                char name[64] = {0};
                if (sscanf(line.c_str(), "property %31s %63s", type, name) == 2) {
                    size_t propSize = PlyTypeSize(type);
                    if (propSize > 0) {
                        auto& el = elements.back();
                        el.props.emplace_back(name, el.stride);
                        el.stride += propSize;
                    }
                }
            } else if (line.rfind("declare feature_rest_scale ", 0) == 0) {
                restScale = strtof(line.c_str() + 27, nullptr);
            } else if (line.rfind("declare feature_rest_zero_point ", 0) == 0) {
                restZeroPoint = strtof(line.c_str() + 32, nullptr);
            } else if (line.rfind("declare feature_dc_min ", 0) == 0) {
                dcRangeMin = strtof(line.c_str() + 22, nullptr);
                hasDcMin = true;
            } else if (line.rfind("declare feature_dc_max ", 0) == 0) {
                dcRangeMax = strtof(line.c_str() + 22, nullptr);
                hasDcMax = true;
            }
        }
        lineStart = i + 1;
    }
    if (elements.empty()) return false;

    // ---- Element data offsets (elements are concatenated without padding) ----
    size_t offset = hdrEnd;
    for (auto& el : elements) {
        el.dataOffset = offset;
        offset += (size_t)el.count * el.stride;
    }
    if (offset > size) return false;

    const PlyElementDesc* elChunk = nullptr;
    const PlyElementDesc* elVertex = nullptr;
    const PlyElementDesc* elVq = nullptr;
    const PlyElementDesc* elRest = nullptr;
    for (const auto& el : elements) {
        if (el.name == "chunk") elChunk = &el;
        else if (el.name == "vertex") elVertex = &el;
        else if (el.name == "feature_rest_vq") elVq = &el;
        else if (el.name == "feature_rest") elRest = &el;
    }
    if (!elChunk || !elVertex) return false;

    auto propOffset = [](const PlyElementDesc& el, const char* name) -> size_t {
        for (const auto& p : el.props) {
            if (p.first == name) return p.second;
        }
        return SIZE_MAX;
    };

    const size_t offChunk = elChunk->dataOffset;
    const size_t offVertex = elVertex->dataOffset;
    if (elVertex->stride < sizeof(PackedVertex)) return false;

    // Chunk records: 12-float (old) or 18-float (with per-chunk colour range)
    // layouts; read the AABB / scale fields by name.
    const char* kChunkNames[12] = {
        "min_x", "min_y", "min_z", "max_x", "max_y", "max_z",
        "min_scale_x", "min_scale_y", "min_scale_z", "max_scale_x", "max_scale_y", "max_scale_z",
    };
    size_t chunkOff[12];
    for (int k = 0; k < 12; k++) {
        chunkOff[k] = propOffset(*elChunk, kChunkNames[k]);
        if (chunkOff[k] == SIZE_MAX) return false;
    }
    // Optional per-chunk colour ranges (newer writer variant)
    const size_t offRMin = propOffset(*elChunk, "min_r");
    const size_t offRMax = propOffset(*elChunk, "max_r");
    const size_t offGMin = propOffset(*elChunk, "min_g");
    const size_t offGMax = propOffset(*elChunk, "max_g");
    const size_t offBMin = propOffset(*elChunk, "min_b");
    const size_t offBMax = propOffset(*elChunk, "max_b");
    const bool hasColorRange = offRMin != SIZE_MAX && offRMax != SIZE_MAX &&
                               offGMin != SIZE_MAX && offGMax != SIZE_MAX &&
                               offBMin != SIZE_MAX && offBMax != SIZE_MAX;

    out.chunks.resize(elChunk->count);
    for (uint32_t k = 0; k < elChunk->count; k++) {
        const uint8_t* rec = data + offChunk + (size_t)k * elChunk->stride;
        float* dst = reinterpret_cast<float*>(&out.chunks[k]);  // 12 floats
        for (int f = 0; f < 12; f++) {
            memcpy(&dst[f], rec + chunkOff[f], sizeof(float));
        }
    }
    float crMin = 0.f, crMax = 1.f, cgMin = 0.f, cgMax = 1.f, cbMin = 0.f, cbMax = 1.f;
    if (hasColorRange) {
        // Range values are uniform across chunks in practice; read chunk 0.
        const uint8_t* rec = data + offChunk;
        memcpy(&crMin, rec + offRMin, sizeof(float));
        memcpy(&crMax, rec + offRMax, sizeof(float));
        memcpy(&cgMin, rec + offGMin, sizeof(float));
        memcpy(&cgMax, rec + offGMax, sizeof(float));
        memcpy(&cbMin, rec + offBMin, sizeof(float));
        memcpy(&cbMax, rec + offBMax, sizeof(float));
    }
    const uint32_t chunkCount = elChunk->count;
    const uint32_t vertexCount = elVertex->count;
    const size_t vtxStride = elVertex->stride;

    // ---- Gaussians ----
    out.gaussians.resize(vertexCount);
    out.gaussianCount = vertexCount;
    for (uint32_t i = 0; i < vertexCount; i++) {
        uint32_t chunkIdx = i / 256;
        if (chunkIdx >= chunkCount) chunkIdx = chunkCount - 1;
        const auto& chunk = out.chunks[chunkIdx];
        PackedVertex pv;
        memcpy(&pv, data + offVertex + (size_t)i * vtxStride, sizeof(PackedVertex));
        auto& g = out.gaussians[i];
        DecodePosition(pv.packedPosition, chunk, g.px, g.py, g.pz);
        DecodeScale(pv.packedScale, chunk, g.sx, g.sy, g.sz);
        DecodeRotation(pv.packedRotation, g.qw, g.qx, g.qy, g.qz);
        float r, gb, b, a;
        if (colorOrder != 0) {
            a = static_cast<float>(pv.packedColor & 0xFF) / 255.0f;
            if (colorOrder == 1) {
                // official pack8888: b1=blue, b2=green, b3=red
                b = static_cast<float>((pv.packedColor >> 8) & 0xFF) / 255.0f;
                gb = static_cast<float>((pv.packedColor >> 16) & 0xFF) / 255.0f;
                r = static_cast<float>((pv.packedColor >> 24) & 0xFF) / 255.0f;
            } else {
                // BGR variant: b1=red, b2=green, b3=blue
                r = static_cast<float>((pv.packedColor >> 8) & 0xFF) / 255.0f;
                gb = static_cast<float>((pv.packedColor >> 16) & 0xFF) / 255.0f;
                b = static_cast<float>((pv.packedColor >> 24) & 0xFF) / 255.0f;
            }
            if (hasColorRange) {
                // Newer writer variant: colours are normalised to the chunk range
                r = crMin + r * (crMax - crMin);
                gb = cgMin + gb * (cgMax - cgMin);
                b = cbMin + b * (cbMax - cbMin);
            }
            if (hasDcMin && hasDcMax) {
                // Exports that declare feature_dc_min/max store NORMALISED SH DC
                // coefficients instead of final colours: reconstruct
                // colour = 0.5 + SH_C0 * (dcMin + n * (dcMax - dcMin)).
                // This restores roughly 2x the channel spread (saturation) the
                // raw byte values alone would give.
                constexpr float SH_C0 = 0.28209479177387814f;
                const float span = dcRangeMax - dcRangeMin;
                auto dcToColor = [&](float n) {
                    float c = 0.5f + SH_C0 * (dcRangeMin + n * span);
                    return c < 0.0f ? 0.0f : (c > 1.0f ? 1.0f : c);
                };
                r = dcToColor(r);
                gb = dcToColor(gb);
                b = dcToColor(b);
            }
        } else {
            DecodeColor(pv.packedColor, r, gb, b, a);
        }
        g.cr = r; g.cg = gb; g.cb = b;
        g.opacity = a;
        g.ca = a;
        g.pad0 = 0.0f;
        if (g.px < out.boundsMin[0]) out.boundsMin[0] = g.px;
        if (g.py < out.boundsMin[1]) out.boundsMin[1] = g.py;
        if (g.pz < out.boundsMin[2]) out.boundsMin[2] = g.pz;
        if (g.px > out.boundsMax[0]) out.boundsMax[0] = g.px;
        if (g.py > out.boundsMax[1]) out.boundsMax[1] = g.py;
        if (g.pz > out.boundsMax[2]) out.boundsMax[2] = g.pz;
    }

    // ---- SH degrees 1..3 (parsed for all supported layouts; the render
    // pipeline decides whether to evaluate it) ----
    // Layout A: inline per-vertex float f_rest_0..44 in the vertex element.
    // Layout B: vector-quantised codebook (feature_rest_vq indices + int8 codebook).
    {
        // Layout A detection
        size_t inlineOff[45];
        bool inlineSh = true;
        for (int m = 0; m < 45; m++) {
            char name[32];
            snprintf(name, sizeof(name), "f_rest_%d", m);
            inlineOff[m] = propOffset(*elVertex, name);
            if (inlineOff[m] == SIZE_MAX) {
                inlineSh = false;
                break;
            }
        }
        if (inlineSh) {
            out.sh.assign((size_t)vertexCount * 48, 0.0f);
            for (uint32_t i = 0; i < vertexCount; i++) {
                float channelMajor[45];
                const uint8_t* rec = data + offVertex + (size_t)i * vtxStride;
                for (int m = 0; m < 45; m++) {
                    memcpy(&channelMajor[m], rec + inlineOff[m], sizeof(float));
                }
                StoreShRestChannelMajorToCoefficientMajor(&out.sh[(size_t)i * 48], channelMajor);
            }
        } else if (elVq && elRest) {
            // Layout B: VQ codebook rest_scale / rest_zero_point reconstruction
            size_t vqOff = elVq->props.empty() ? SIZE_MAX : elVq->props[0].second;
            if (vqOff != SIZE_MAX && elVq->stride >= 4 && elRest->stride >= 45) {
                out.sh.assign((size_t)vertexCount * 48, 0.0f);
                const uint8_t* codebook = data + elRest->dataOffset;
                for (uint32_t i = 0; i < vertexCount; i++) {
                    uint32_t idx = 0;
                    memcpy(&idx, data + elVq->dataOffset + (size_t)i * elVq->stride + vqOff, sizeof(uint32_t));
                    if (idx >= elRest->count) continue;  // defensive: malformed input
                    const int8_t* entry = reinterpret_cast<const int8_t*>(codebook + (size_t)idx * elRest->stride);
                    float* dst = &out.sh[(size_t)i * 48];
                    for (int c = 0; c < 3; c++) {
                        for (int j = 0; j < 15; j++) {
                            dst[j * 3 + c] = (float)entry[c * 15 + j] * restScale + restZeroPoint;
                        }
                    }
                }
            }
        }
    }
    return true;
}

// Load a .splat PLY file and decode all gaussians
inline bool LoadSplatFile(const char* path, SplatData& out) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) {
        fclose(f);
        return false;
    }
    std::vector<uint8_t> buf((size_t)sz);
    size_t got = fread(buf.data(), 1, (size_t)sz, f);
    fclose(f);
    if (got != (size_t)sz) return false;
    return LoadSplatPlyFromMemory(buf.data(), buf.size(), out, /*colorOrder=*/1);
}

// Load a glTF 2.0 GLB carrying a UWA_gaussian_splatting extension.
// The extension stores a super-splat compressed PLY (chunk table + packed
// vertices + optional vector-quantised SH) verbatim inside the BIN chunk;
// this extracts that payload and decodes it with the shared decoder.
inline bool LoadUwaSplatGlb(const char* path, SplatData& out)
{
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 20) {
        fclose(f);
        return false;
    }
    std::vector<uint8_t> file((size_t)sz);
    size_t got = fread(file.data(), 1, (size_t)sz, f);
    fclose(f);
    if (got != (size_t)sz) return false;

    uint32_t magic = 0, version = 0, length = 0;
    memcpy(&magic, file.data(), 4);
    memcpy(&version, file.data() + 4, 4);
    memcpy(&length, file.data() + 8, 4);
    constexpr uint32_t GLB_MAGIC = 0x46546C67u;  // 'glTF'
    constexpr uint32_t CHUNK_JSON = 0x4E4F534Au; // 'JSON'
    constexpr uint32_t CHUNK_BIN = 0x004E4942u;  // 'BIN\0'
    if (magic != GLB_MAGIC || version != 2) return false;

    size_t off = 12;
    const size_t end = (length <= (uint32_t)sz) ? length : (size_t)sz;
    const uint8_t* binData = nullptr;
    size_t binSize = 0;
    while (off + 8 <= end) {
        uint32_t clen = 0, ctype = 0;
        memcpy(&clen, file.data() + off, 4);
        memcpy(&ctype, file.data() + off + 4, 4);
        if (off + 8 + clen > end) break;
        if (ctype == CHUNK_BIN) {
            binData = file.data() + off + 8;
            binSize = clen;
            break;
        }
        off += 8 + clen;
    }
    if (!binData || binSize == 0) return false;

    return LoadSplatPlyFromMemory(binData, binSize, out, /*colorOrder=*/1);
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

            // SH degree 1..3 coefficients: PLY is channel-major, shader is
            // coefficient-major, so transpose while copying.
            if (hasRest) {
                float channelMajor[45];
                for (int m = 0; m < 45; m++) {
                    channelMajor[m] = (restIdx[m] >= 0) ? v[restIdx[m]] : 0.0f;
                }
                StoreShRestChannelMajorToCoefficientMajor(&out.sh[(size_t)(decoded + j) * 48], channelMajor);
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
