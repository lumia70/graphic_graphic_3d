/*
 * render_node_gaussian_splatting.cpp
 *
 * 3D Gaussian Splatting rendering as a Lume RenderNode.
 * Reads gaussian data from liblumeapp.so via dlsym, uploads to GPU SSBO,
 * and renders billboard quads with 2D gaussian alpha blending.
 */
#include "render_node_gaussian_splatting.h"

#include <cstring>
#include <cmath>
#include <dlfcn.h>

#include <base/containers/array_view.h>
#include <render/device/intf_gpu_resource_manager.h>
#include <render/device/intf_shader_manager.h>
#include <render/device/pipeline_layout_desc.h>
#include <render/device/pipeline_state_desc.h>
#include <render/nodecontext/intf_render_command_list.h>
#include <render/nodecontext/intf_render_node_context_manager.h>
#include <render/nodecontext/intf_node_context_pso_manager.h>
#include <render/nodecontext/intf_node_context_descriptor_set_manager.h>
#include <render/nodecontext/intf_render_node_util.h>
#include <render/render_data_structures.h>
#include <render/namespace.h>

#include "gaussian_splat_data.h"
#include "default_engine_constants.h"
#include "util/log.h"

using BASE_NS::array_view;
using BASE_NS::countof;

RENDER_BEGIN_NAMESPACE()

// Factory
IRenderNode* RenderNodeGaussianSplatting::Create() { return new RenderNodeGaussianSplatting(); }
void RenderNodeGaussianSplatting::Destroy(IRenderNode* instance) {
    delete static_cast<RenderNodeGaussianSplatting*>(instance);
}

// ---- dlsym helper to read shared globals from liblumeapp.so ----
namespace {
struct SharedPtrs {
    const GpuGaussianData** dataPtr = nullptr;
    uint32_t* countPtr = nullptr;
    bool* readyPtr = nullptr;
    GaussianCameraData* cameraPtr = nullptr;  // direct pointer to struct (not pointer-to-pointer)
    bool* cameraReadyPtr = nullptr;
    const float** shPtr = nullptr;
    bool resolved = false;

    void Resolve() {
        if (resolved) return;
        void* h = dlopen("liblumeapp.so", RTLD_NOLOAD);
        if (!h) h = dlopen("liblumeapp.so", RTLD_NOW | RTLD_GLOBAL);
        if (h) {
            dataPtr = (const GpuGaussianData**)dlsym(h, "g_gaussianSplatData");
            countPtr = (uint32_t*)dlsym(h, "g_gaussianSplatCount");
            readyPtr = (bool*)dlsym(h, "g_gaussianDataReady");
            cameraPtr = (GaussianCameraData*)dlsym(h, "g_gaussianCameraData");
            cameraReadyPtr = (bool*)dlsym(h, "g_gaussianCameraReady");
            shPtr = (const float**)dlsym(h, "g_gaussianShData");
        }
        resolved = true;
    }
    bool Ready() const { return readyPtr && *readyPtr && dataPtr && *dataPtr && countPtr && *countPtr > 0; }
    const GpuGaussianData* Data() const { return dataPtr ? *dataPtr : nullptr; }
    uint32_t Count() const { return countPtr ? *countPtr : 0; }
    bool CameraReady() const { return cameraReadyPtr && *cameraReadyPtr && cameraPtr; }
    const GaussianCameraData& Camera() const { return *cameraPtr; }
    const float* Sh() const { return shPtr ? *shPtr : nullptr; }
};
SharedPtrs g_sp;

// Build a simple look-at view matrix (column-major for GLSL)
void BuildLookAt(float* out, float ex, float ey, float ez, float tx, float ty, float tz) {
    float ux = 0.f, uy = 1.f, uz = 0.f;
    // forward = normalize(eye - target)
    float fx = ex - tx, fy = ey - ty, fz = ez - tz;
    float fl = sqrtf(fx*fx + fy*fy + fz*fz);
    if (fl > 1e-8f) { fx /= fl; fy /= fl; fz /= fl; }
    // right = normalize(up x forward)
    float rx = uy*fz - uz*fy, ry = uz*fx - ux*fz, rz = ux*fy - uy*fx;
    float rl = sqrtf(rx*rx + ry*ry + rz*rz);
    if (rl > 1e-8f) { rx /= rl; ry /= rl; rz /= rl; }
    // true up = forward x right
    float tux = fy*rz - fz*ry, tuy = fz*rx - fx*rz, tuz = fx*ry - fy*rx;
    // Column-major layout
    out[ 0]=rx;  out[ 1]=tux;  out[ 2]=fx;  out[ 3]=0.f;
    out[ 4]=ry;  out[ 5]=tuy;  out[ 6]=fy;  out[ 7]=0.f;
    out[ 8]=rz;  out[ 9]=tuz;  out[10]=fz;  out[11]=0.f;
    out[12]=-(rx*ex+ry*ey+rz*ez);
    out[13]=-(tux*ex+tuy*ey+tuz*ez);
    out[14]=-(fx*ex+fy*ey+fz*ez);
    out[15]=1.f;
}

// Build a perspective projection matrix (column-major, Vulkan clip [0,1] depth)
void BuildPerspective(float* out, float fovYDeg, float aspect, float zn, float zf) {
    float fovRad = fovYDeg * 3.14159265f / 180.f;
    float f = 1.f / tanf(fovRad * 0.5f);
    std::memset(out, 0, 16 * sizeof(float));
    out[0]  = f / aspect;
    out[5]  = -f;  // Vulkan Y-flip
    out[10] = zf / (zn - zf);
    out[11] = -1.f;
    out[14] = (zn * zf) / (zn - zf);
}

// Number of depth buckets for the back-to-front counting sort.
// 4096 buckets make slices thin enough that within-bucket order errors
// (source of banding stripes on dense surfaces) become invisible.
constexpr uint32_t SORT_BUCKETS = 4096;

// ---------------------------------------------------------------------------
// DESIGN NOTE: why the depth sort runs on the CPU (recorded for later review)
// ---------------------------------------------------------------------------
// 1. Data locality: gaussians are decoded and uploaded from CPU memory; the
//    sort reuses the same arrays with zero copies.
// 2. Cost profile: O(n) counting sort (2 passes over 750K) ~5-15ms, and it
//    only runs on frames where the camera rotated/panned past a threshold —
//    static scenes cost nothing (render-on-demand + axial zoom skips resort).
// 3. GPU alternatives (for future discussion):
//    - Bitonic sort: O(n log^2 n), ~200+ compute dispatches for 750K — too slow.
//    - GPU radix sort: ~8 scatter rounds + prefix sums + ping-pong buffers —
//      significant complexity, plus a fixed per-frame cost: dispatch +
//      compute->vertex barrier must run even when the camera is static
//      (change detection on GPU would need readback, which is worse).
// 4. CPU-side headroom first: multi-thread the bucket sort (Lume task queue)
//    and/or SIMD the key pass before considering a GPU move.
// ---------------------------------------------------------------------------

}  // namespace

// ===================================================================
// InitNode — called once when the render node graph is created
// ===================================================================
void RenderNodeGaussianSplatting::InitNode(IRenderNodeContextManager& mgr)
{
    renderNodeContextMgr_ = &mgr;
    g_sp.Resolve();
    PLUGIN_LOG_I("GS::InitNode ready=%d count=%u", g_sp.Ready()?1:0, g_sp.Count());

    auto& gpuResMgr = mgr.GetGpuResourceManager();
    auto& shaderMgr = mgr.GetShaderManager();
    const auto& renderNodeUtil = mgr.GetRenderNodeUtil();

    // ---- 1. Create GPU buffers ----

    // Camera UBO (host-visible, updated every frame) — triple buffered
    for (uint32_t s = 0; s < FRAME_SLOTS; s++) {
        GpuBufferDesc uboDesc {};
        uboDesc.usageFlags = CORE_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        uboDesc.memoryPropertyFlags = CORE_MEMORY_PROPERTY_HOST_VISIBLE_BIT | CORE_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        uboDesc.byteSize = sizeof(CameraUboData);
        uboDesc.engineCreationFlags = CORE_ENGINE_BUFFER_CREATION_DYNAMIC_BARRIERS;
        char name[32];
        snprintf(name, sizeof(name), "GS_CamUBO_%u", s);
        cameraUbo_[s] = gpuResMgr.Create(name, uboDesc);
    }

    // Gaussian SSBO (host-visible, uploaded once when data is ready)
    if (g_sp.Ready()) {
        uint32_t cnt = g_sp.Count();
        GpuBufferDesc desc {};
        desc.usageFlags = CORE_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        desc.memoryPropertyFlags = CORE_MEMORY_PROPERTY_HOST_VISIBLE_BIT | CORE_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        desc.byteSize = cnt * sizeof(GpuGaussian);
        desc.engineCreationFlags = CORE_ENGINE_BUFFER_CREATION_DYNAMIC_BARRIERS;
        gaussianBuffer_ = gpuResMgr.Create("GS_SSBO", desc);
        gaussianCount_ = cnt;
        needsUpload_ = true;

        // Sort index buffer (one uint32 per gaussian) — triple buffered
        for (uint32_t s = 0; s < FRAME_SLOTS; s++) {
            GpuBufferDesc sortDesc {};
            sortDesc.usageFlags = CORE_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            sortDesc.memoryPropertyFlags = CORE_MEMORY_PROPERTY_HOST_VISIBLE_BIT | CORE_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            sortDesc.byteSize = cnt * sizeof(uint32_t);
            sortDesc.engineCreationFlags = CORE_ENGINE_BUFFER_CREATION_DYNAMIC_BARRIERS;
            char name[32];
            snprintf(name, sizeof(name), "GS_SortIdx_%u", s);
            sortIndexBuffer_[s] = gpuResMgr.Create(name, sortDesc);
        }

        // SH coefficient buffer (48 floats per gaussian, uploaded once)
        GpuBufferDesc shDesc {};
        shDesc.usageFlags = CORE_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        shDesc.memoryPropertyFlags = CORE_MEMORY_PROPERTY_HOST_VISIBLE_BIT | CORE_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        shDesc.byteSize = (uint32_t)(cnt * 48ULL * sizeof(float));
        shDesc.engineCreationFlags = CORE_ENGINE_BUFFER_CREATION_DYNAMIC_BARRIERS;
        shBuffer_ = gpuResMgr.Create("GS_Sh", shDesc);

        sortIndices_.resize(cnt);
        sortScatter_.resize(cnt);
        sortKeys_.resize(cnt);
        bucketCounts_.resize(SORT_BUCKETS + 1);
        bucketStarts_.resize(SORT_BUCKETS + 1);
        for (uint32_t i = 0; i < cnt; i++) sortIndices_[i] = i;

        PLUGIN_LOG_I("GS: SSBO %u gaussians (%u MB)", cnt, (uint32_t)(cnt * 64ULL / (1024 * 1024)));
    }

    // ---- 2. Get shader handle ----
    // The .shader JSON + .spv + .spv.lsb are loaded from "splatshaders://gaussian_render.shader"
    shaderHandle_ = shaderMgr.GetShaderHandle("splatshaders://gaussian_render.shader");
    PLUGIN_LOG_I("GS: shader valid=%d", RenderHandleUtil::IsValid(shaderHandle_) ? 1 : 0);

    if (!RenderHandleUtil::IsValid(shaderHandle_)) {
        PLUGIN_LOG_E("GS: FATAL — shader 'splatshaders://gaussian_render.shader' not found or invalid");
        return;
    }

    // ---- 3. Create pipeline layouts from shader reflection ----
    pipelineLayout_ = renderNodeUtil.CreatePipelineLayout(shaderHandle_);
    PLUGIN_LOG_I("GS: pipelineLayout set count: ds0=%u bindings, ds1=%u bindings",
        (uint32_t)pipelineLayout_.descriptorSetLayouts[0].bindings.size(),
        (uint32_t)pipelineLayout_.descriptorSetLayouts[1].bindings.size());

    // Copy pass layout (offscreen -> sRGB backbuffer)
    copyShaderHandle_ = shaderMgr.GetShaderHandle("splatshaders://gaussian_copy.shader");
    PLUGIN_LOG_I("GS: copy shader valid=%d", RenderHandleUtil::IsValid(copyShaderHandle_) ? 1 : 0);
    if (RenderHandleUtil::IsValid(copyShaderHandle_)) {
        copyPipelineLayout_ = renderNodeUtil.CreatePipelineLayout(copyShaderHandle_);
    }

    // ---- 4. Reserve descriptor sets (ONE combined reserve for both binders) ----
    // ResetAndReserve resets the pool; calling it twice would corrupt the
    // first binder's sets — always reserve all counts in a single call.
    if (RenderHandleUtil::IsValid(copyShaderHandle_)) {
        DescriptorCounts allCounts[] = {
            renderNodeUtil.GetDescriptorCounts(pipelineLayout_),
            renderNodeUtil.GetDescriptorCounts(copyPipelineLayout_),
        };
        mgr.GetDescriptorSetManager().ResetAndReserve({ allCounts, countof(allCounts) });
    } else {
        mgr.GetDescriptorSetManager().ResetAndReserve(renderNodeUtil.GetDescriptorCounts(pipelineLayout_));
    }

    pipelineDescriptorSetBinder_ = renderNodeUtil.CreatePipelineDescriptorSetBinder(pipelineLayout_);

    // Bind our UBO and SSBOs (slot 0 initially; re-bound per frame)
    if (pipelineDescriptorSetBinder_) {
        pipelineDescriptorSetBinder_->BindBuffer(0u, 0u, BindableBuffer { cameraUbo_[0].GetHandle() });
        if (gaussianBuffer_) {
            pipelineDescriptorSetBinder_->BindBuffer(0u, 1u, BindableBuffer { gaussianBuffer_.GetHandle() });
        }
        if (sortIndexBuffer_[0]) {
            pipelineDescriptorSetBinder_->BindBuffer(0u, 2u, BindableBuffer { sortIndexBuffer_[0].GetHandle() });
        }
        if (shBuffer_) {
            pipelineDescriptorSetBinder_->BindBuffer(0u, 3u, BindableBuffer { shBuffer_.GetHandle() });
        }
    }

    // ---- 5. Set up custom graphics state (alpha blending, no depth test) ----
    {
        graphicsState_ = {};
        graphicsState_.inputAssembly.primitiveTopology = PrimitiveTopology::CORE_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        graphicsState_.inputAssembly.enablePrimitiveRestart = false;
        graphicsState_.rasterizationState.polygonMode = PolygonMode::CORE_POLYGON_MODE_FILL;
        graphicsState_.rasterizationState.cullModeFlags = CullModeFlagBits::CORE_CULL_MODE_NONE;
        graphicsState_.rasterizationState.frontFace = FrontFace::CORE_FRONT_FACE_COUNTER_CLOCKWISE;
        // No depth test: gaussians are alpha-blended in back-to-front sorted order,
        // correct compositing accumulates color and opacity per pixel.
        graphicsState_.depthStencilState.enableDepthTest = false;
        graphicsState_.depthStencilState.enableDepthWrite = false;

        // Pre-multiplied alpha blending: outColor = src + (1-srcAlpha)*dst
        graphicsState_.colorBlendState.colorAttachmentCount = 1;
        auto& att = graphicsState_.colorBlendState.colorAttachments[0];
        att.enableBlend = true;
        att.srcColorBlendFactor = BlendFactor::CORE_BLEND_FACTOR_ONE;
        att.dstColorBlendFactor = BlendFactor::CORE_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        att.colorBlendOp = BlendOp::CORE_BLEND_OP_ADD;
        att.srcAlphaBlendFactor = BlendFactor::CORE_BLEND_FACTOR_ONE;
        att.dstAlphaBlendFactor = BlendFactor::CORE_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        att.alphaBlendOp = BlendOp::CORE_BLEND_OP_ADD;
        att.colorWriteMask = ColorComponentFlagBits::CORE_COLOR_COMPONENT_R_BIT |
                             ColorComponentFlagBits::CORE_COLOR_COMPONENT_G_BIT |
                             ColorComponentFlagBits::CORE_COLOR_COMPONENT_B_BIT |
                             ColorComponentFlagBits::CORE_COLOR_COMPONENT_A_BIT;
    }

    // ---- 6. Create PSO (using GraphicsState& overload, no need for a handle) ----
    {
        constexpr DynamicStateEnum DYNAMIC_STATES[] = {
            CORE_DYNAMIC_STATE_ENUM_VIEWPORT,
            CORE_DYNAMIC_STATE_ENUM_SCISSOR,
        };
        auto& psoMgr = mgr.GetPsoManager();
        psoHandle_ = psoMgr.GetGraphicsPsoHandle(
            shaderHandle_,
            graphicsState_,
            pipelineLayout_,
            {},  // no vertex input (procedural from gl_VertexIndex)
            {},  // no specialization constants
            { DYNAMIC_STATES, countof(DYNAMIC_STATES) }
        );
        PLUGIN_LOG_I("GS: PSO valid=%d", RenderHandleUtil::IsValid(psoHandle_) ? 1 : 0);
    }

    // ---- 7. Copy pass: offscreen -> sRGB backbuffer (decode cancels hw encode) ----
    {
        if (RenderHandleUtil::IsValid(copyShaderHandle_)) {
            copyBinder_ = renderNodeUtil.CreatePipelineDescriptorSetBinder(copyPipelineLayout_);
            if (copyBinder_) {
                // Bind a default image placeholder until the offscreen exists
                // (re-bound with the real offscreen every frame in ExecuteFrame)
                const RenderHandle defImg =
                    gpuResMgr.GetImageHandle(DefaultEngineGpuResourceConstants::CORE_DEFAULT_GPU_IMAGE);
                const RenderHandle defSamp =
                    gpuResMgr.GetSamplerHandle(DefaultEngineGpuResourceConstants::CORE_DEFAULT_SAMPLER_LINEAR_CLAMP);
                BindableImage bi {};
                bi.handle = defImg;
                bi.imageLayout = ImageLayout::CORE_IMAGE_LAYOUT_UNDEFINED;
                copyBinder_->BindImage(0u, 0u, bi);
                copyBinder_->BindSampler(0u, 1u, BindableSampler { defSamp });
            }
            GraphicsState gs2 {};
            gs2.inputAssembly.primitiveTopology = PrimitiveTopology::CORE_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            gs2.rasterizationState.cullModeFlags = CullModeFlagBits::CORE_CULL_MODE_NONE;
            gs2.depthStencilState.enableDepthTest = false;
            gs2.depthStencilState.enableDepthWrite = false;
            gs2.colorBlendState.colorAttachmentCount = 1;
            auto& att2 = gs2.colorBlendState.colorAttachments[0];
            att2.enableBlend = false;  // replace: the offscreen already contains the composited bg
            constexpr DynamicStateEnum DYNAMIC_STATES2[] = {
                CORE_DYNAMIC_STATE_ENUM_VIEWPORT,
                CORE_DYNAMIC_STATE_ENUM_SCISSOR,
            };
            copyPsoHandle_ = mgr.GetPsoManager().GetGraphicsPsoHandle(
                copyShaderHandle_, gs2, copyPipelineLayout_, {}, {},
                { DYNAMIC_STATES2, countof(DYNAMIC_STATES2) });
            PLUGIN_LOG_I("GS: copy PSO valid=%d", RenderHandleUtil::IsValid(copyPsoHandle_) ? 1 : 0);
        }
    }

    initialized_ = true;
    PLUGIN_LOG_I("GS::InitNode complete — gaussians=%u, shader=%d, pso=%d",
        gaussianCount_,
        RenderHandleUtil::IsValid(shaderHandle_) ? 1 : 0,
        RenderHandleUtil::IsValid(psoHandle_) ? 1 : 0);
}

// ===================================================================
// PreExecuteFrame — called each frame before ExecuteFrame
// ===================================================================
void RenderNodeGaussianSplatting::PreExecuteFrame()
{
    if (!initialized_) return;

    // Lazy creation of SSBO if data became ready after InitNode
    if (gaussianCount_ == 0 && g_sp.Ready()) {
        uint32_t cnt = g_sp.Count();
        auto& gpuResMgr = renderNodeContextMgr_->GetGpuResourceManager();
        GpuBufferDesc desc {};
        desc.usageFlags = CORE_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        desc.memoryPropertyFlags = CORE_MEMORY_PROPERTY_HOST_VISIBLE_BIT | CORE_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        desc.byteSize = cnt * sizeof(GpuGaussian);
        desc.engineCreationFlags = CORE_ENGINE_BUFFER_CREATION_DYNAMIC_BARRIERS;
        gaussianBuffer_ = gpuResMgr.Create("GS_SSBO", desc);
        gaussianCount_ = cnt;
        needsUpload_ = true;

        // Sort index buffer
        // Sort index buffer — triple buffered
        for (uint32_t s = 0; s < FRAME_SLOTS; s++) {
            GpuBufferDesc sortDesc {};
            sortDesc.usageFlags = CORE_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            sortDesc.memoryPropertyFlags = CORE_MEMORY_PROPERTY_HOST_VISIBLE_BIT | CORE_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            sortDesc.byteSize = cnt * sizeof(uint32_t);
            sortDesc.engineCreationFlags = CORE_ENGINE_BUFFER_CREATION_DYNAMIC_BARRIERS;
            sortIndexBuffer_[s] = gpuResMgr.Create("GS_SortIdx_Late", sortDesc);
        }

        sortIndices_.resize(cnt);
        sortScatter_.resize(cnt);
        sortKeys_.resize(cnt);
        bucketCounts_.resize(SORT_BUCKETS + 1);
        bucketStarts_.resize(SORT_BUCKETS + 1);
        for (uint32_t i = 0; i < cnt; i++) sortIndices_[i] = i;

        PLUGIN_LOG_I("GS: late SSBO creation %u gaussians", cnt);

        // Rebind SSBOs
        if (pipelineDescriptorSetBinder_) {
            pipelineDescriptorSetBinder_->BindBuffer(0u, 1u, BindableBuffer { gaussianBuffer_.GetHandle() });
            pipelineDescriptorSetBinder_->BindBuffer(0u, 2u, BindableBuffer { sortIndexBuffer_[0].GetHandle() });
        }
    }
}

// ===================================================================
// ExecuteFrame — called each frame, issues GPU commands
// ===================================================================
void RenderNodeGaussianSplatting::ExecuteFrame(IRenderCommandList& cmdList)
{
    if (!initialized_ || gaussianCount_ == 0) return;
    if (!RenderHandleUtil::IsValid(psoHandle_)) {
        if (frameCount_ < 3) {
            PLUGIN_LOG_E("GS::ExecuteFrame: PSO invalid, skipping");
        }
        frameCount_++;
        return;
    }

    auto& gpuResMgr = renderNodeContextMgr_->GetGpuResourceManager();

    // ---- 1. Upload gaussian data (once) ----
    if (needsUpload_ && g_sp.Ready() && gaussianBuffer_) {
        void* mapped = gpuResMgr.MapBuffer(gaussianBuffer_.GetHandle());
        if (mapped) {
            std::memcpy(mapped, g_sp.Data(), gaussianCount_ * sizeof(GpuGaussian));
            gpuResMgr.UnmapBuffer(gaussianBuffer_.GetHandle());
            PLUGIN_LOG_I("GS: uploaded %u gaussians to GPU", gaussianCount_);
        }
        // Upload SH coefficients (zeros when absent — SH eval contributes nothing)
        if (shBuffer_) {
            const size_t shBytes = (size_t)gaussianCount_ * 48ULL * sizeof(float);
            void* shMapped = gpuResMgr.MapBuffer(shBuffer_.GetHandle());
            if (shMapped) {
                if (g_sp.Sh()) {
                    std::memcpy(shMapped, g_sp.Sh(), shBytes);
                } else {
                    std::memset(shMapped, 0, shBytes);
                }
                gpuResMgr.UnmapBuffer(shBuffer_.GetHandle());
                PLUGIN_LOG_I("GS: uploaded SH coefficients (%s)",
                    g_sp.Sh() ? "present" : "absent — zero filled");
            }
        }
        needsUpload_ = false;
    }

    // ---- 2. Get backbuffer and its dimensions ----
    const RenderHandle backbufferHandle = gpuResMgr.GetImageHandle("CORE_DEFAULT_BACKBUFFER");
    if (!RenderHandleUtil::IsValid(backbufferHandle)) {
        if (frameCount_ < 3) {
            PLUGIN_LOG_E("GS: CORE_DEFAULT_BACKBUFFER not found");
        }
        frameCount_++;
        return;
    }
    const GpuImageDesc bbDesc = gpuResMgr.GetImageDescriptor(backbufferHandle);
    const uint32_t width = bbDesc.width;
    const uint32_t height = bbDesc.height;

    // ---- 2b. Create (or recreate) UNORM offscreen target matching backbuffer size ----
    // Raw code-space compositing target: splats blend here exactly like the
    // web canvas (no sRGB decode/encode), then a copy pass transfers it to
    // the sRGB backbuffer with a decode that cancels the hardware encode.
    if (!offscreenImage_ || offscreenWidth_ != width || offscreenHeight_ != height) {
        GpuImageDesc offDesc {};
        offDesc.format = BASE_NS::Format::BASE_FORMAT_R8G8B8A8_UNORM;
        offDesc.width = width;
        offDesc.height = height;
        offDesc.depth = 1;
        offDesc.mipCount = 1;
        offDesc.layerCount = 1;
        offDesc.sampleCountFlags = CORE_SAMPLE_COUNT_1_BIT;
        offDesc.imageTiling = CORE_IMAGE_TILING_OPTIMAL;
        offDesc.usageFlags = CORE_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | CORE_IMAGE_USAGE_SAMPLED_BIT;
        offDesc.memoryPropertyFlags = CORE_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        offDesc.engineCreationFlags = CORE_ENGINE_IMAGE_CREATION_DYNAMIC_BARRIERS;
        offscreenImage_ = gpuResMgr.Create("GS_Offscreen", offDesc);
        offscreenWidth_ = width;
        offscreenHeight_ = height;
        PLUGIN_LOG_I("GS: UNORM offscreen created %ux%u", width, height);
        // Re-bind the offscreen to the copy binder (handle changed)
        if (copyBinder_) {
            BindableImage bi {};
            bi.handle = offscreenImage_.GetHandle();
            bi.imageLayout = ImageLayout::CORE_IMAGE_LAYOUT_UNDEFINED;
            copyBinder_->BindImage(0u, 0u, bi);
        }
    }

    // ---- 3. Update camera UBO ----
    // Pick this frame's buffer slot (triple buffered against in-flight frames)
    const uint32_t slot = currentSlot_;
    currentSlot_ = (currentSlot_ + 1) % FRAME_SLOTS;
    {
        CameraUboData camData {};
        if (g_sp.CameraReady()) {
            // Use shared camera data from app's orbit camera
            const auto& cam = g_sp.Camera();
            std::memcpy(camData.view, cam.view, sizeof(camData.view));
            std::memcpy(camData.proj, cam.proj, sizeof(camData.proj));
            std::memcpy(camData.viewport, cam.viewport, sizeof(camData.viewport));
            std::memcpy(camData.cameraPos, cam.cameraPos, sizeof(camData.cameraPos));
        } else {
            // Fallback: default camera
            const float camX = 0.f, camY = 10.f, camZ = 150.f;
            BuildLookAt(camData.view, camX, camY, camZ,  0.f, 0.f, 0.f);
            BuildPerspective(camData.proj, 60.f, (float)width / (float)height, 1.f, 500.f);
            camData.viewport[0] = 0.f;
            camData.viewport[1] = 0.f;
            camData.viewport[2] = (float)width;
            camData.viewport[3] = (float)height;
            camData.cameraPos[0] = camX;
            camData.cameraPos[1] = camY;
            camData.cameraPos[2] = camZ;
            camData.cameraPos[3] = 1.f;
        }

        void* mapped = gpuResMgr.MapBuffer(cameraUbo_[slot].GetHandle());
        if (mapped) {
            std::memcpy(mapped, &camData, sizeof(camData));
            gpuResMgr.UnmapBuffer(cameraUbo_[slot].GetHandle());
        }

        // ---- 3.5 Frustum + size cull, back-to-front bucket sort ----
        // Re-sorted only when the view rotated/zoomed past a small threshold;
        // the cull pass reduces both GPU vertex load and sort work.
        if (sortIndexBuffer_[slot] && gaussianCount_ > 0 && g_sp.Data()) {
            const float* vm = camData.view;
            const float* pm = camData.proj;

            // Coarse change detection: re-sort only on rotation or LATERAL eye
            // movement. Pure axial zoom (along the view axis) preserves depth
            // order, so it reuses the previous sort — no per-frame CPU cost.
            float dirX = -vm[2], dirY = -vm[6], dirZ = -vm[10];
            const float dlen = sqrtf(dirX*dirX + dirY*dirY + dirZ*dirZ);
            if (dlen > 1e-8f) { dirX /= dlen; dirY /= dlen; dirZ /= dlen; }
            const float dirDot = dirX*lastDirX_ + dirY*lastDirY_ + dirZ*lastDirZ_;
            const float ex = camData.cameraPos[0], ey = camData.cameraPos[1], ez = camData.cameraPos[2];
            const float dex = ex - lastEyeX_, dey = ey - lastEyeY_, dez = ez - lastEyeZ_;
            const float axial = dex*dirX + dey*dirY + dez*dirZ;
            const float latx = dex - axial*dirX, laty = dey - axial*dirY, latz = dez - axial*dirZ;
            const float latT = sqrtf(latx*latx + laty*laty + latz*latz);
            const float eyeLen = sqrtf(ex*ex + ey*ey + ez*ez);
            const float latThreshold = 0.001f * (eyeLen > 1.f ? eyeLen : 1.f);
            const bool viewChanged = !sortBufferUploaded_ ||
                (dirDot < 0.99999f) ||      // rotated ~0.25 degrees
                (latT > latThreshold);       // lateral (pan/zoom-orbit) movement

            if (viewChanged) {
                lastDirX_ = dirX; lastDirY_ = dirY; lastDirZ_ = dirZ;
                lastEyeX_ = ex; lastEyeY_ = ey; lastEyeZ_ = ez;

                const GpuGaussianData* data = g_sp.Data();
                // View rows (column-major): right, up, forward
                const float r0 = vm[0], r1 = vm[4], r2 = vm[8], t0 = vm[12];
                const float u0 = vm[1], u1 = vm[5], u2 = vm[9], t1 = vm[13];
                const float f0 = vm[2], f1 = vm[6], f2 = vm[10], t2 = vm[14];
                // Frustum half-tangents with generous 2x margin (no false culls)
                const float tanX = 2.0f / fabsf(pm[0]);
                const float tanY = 2.0f / fabsf(pm[5]);
                // Size cull: focal_y in pixels for screen radius estimate
                const float focalY = fabsf(pm[5]) * (float)height * 0.5f;

                // Pass 1: cull + collect visible + depth range
                float zmin = 1e30f, zmax = -1e30f;
                visibleCount_ = 0;
                for (uint32_t i = 0; i < gaussianCount_; i++) {
                    const float px = data[i].px, py = data[i].py, pz = data[i].pz;
                    const float vz = f0*px + f1*py + f2*pz + t2;
                    const float z = -vz;  // positive distance in front
                    if (z < 0.05f || z > 100.f) continue;
                    // Conservative world radius (3 sigma) for frustum margin
                    const float msx = data[i].sx, msy = data[i].sy, msz = data[i].sz;
                    const float ms = msx > msy ? (msx > msz ? msx : msz) : (msy > msz ? msy : msz);
                    const float wr = ms * 3.f;
                    // Frustum cull with world-radius margin (center + extent)
                    const float vx = r0*px + r1*py + r2*pz + t0;
                    if (fabsf(vx) > z * tanX + wr) continue;
                    const float vy = u0*px + u1*py + u2*pz + t1;
                    if (fabsf(vy) > z * tanY + wr) continue;
                    // Size cull: projected 2*sqrt(2)*sigma extent below 2px (engine minPixelSize=2.0)
                    if (focalY * ms * 2.83f / z < 2.0f) continue;
                    if (visibleCount_ < sortIndices_.size()) {
                        sortIndices_[visibleCount_] = i;
                        sortKeys_[visibleCount_] = z;
                    }
                    visibleCount_++;
                    if (z < zmin) zmin = z;
                    if (z > zmax) zmax = z;
                }

                if (visibleCount_ > 0 && visibleCount_ <= sortIndices_.size()) {
                    // Bucket back-to-front: farthest (large z) first → small bucket index
                    const float range = zmax - zmin;
                    const float invRange = (range > 1e-6f) ? (float)SORT_BUCKETS / range : 0.f;
                    std::memset(bucketCounts_.data(), 0, bucketCounts_.size() * sizeof(uint32_t));
                    for (uint32_t j = 0; j < visibleCount_; j++) {
                        uint32_t b = (uint32_t)((zmax - sortKeys_[j]) * invRange);
                        if (b >= SORT_BUCKETS) b = SORT_BUCKETS - 1;
                        bucketCounts_[b]++;
                    }
                    uint32_t total = 0;
                    for (uint32_t b = 0; b < SORT_BUCKETS; b++) {
                        bucketStarts_[b] = total;
                        total += bucketCounts_[b];
                    }
                    // Scatter visible list into final back-to-front order
                    uint32_t cursor[SORT_BUCKETS];
                    std::memcpy(cursor, bucketStarts_.data(), SORT_BUCKETS * sizeof(uint32_t));
                    for (uint32_t j = 0; j < visibleCount_; j++) {
                        uint32_t b = (uint32_t)((zmax - sortKeys_[j]) * invRange);
                        if (b >= SORT_BUCKETS) b = SORT_BUCKETS - 1;
                        sortScatter_[cursor[b]++] = sortIndices_[j];
                    }
                    sortBufferUploaded_ = true;
                } else if (visibleCount_ > sortIndices_.size()) {
                    visibleCount_ = 0;  // safety, should not happen
                }
            }

            // Upload the master sort order to THIS frame's slot every frame
            // (slots rotate; even a skipped re-sort must refresh the slot).
            // Tail padded with 0xFFFFFFFF sentinel — shader skips those entries.
            void* sortMapped = gpuResMgr.MapBuffer(sortIndexBuffer_[slot].GetHandle());
            if (sortMapped) {
                if (visibleCount_ > 0) {
                    std::memcpy(sortMapped, sortScatter_.data(), visibleCount_ * sizeof(uint32_t));
                }
                std::memset((uint8_t*)sortMapped + visibleCount_ * sizeof(uint32_t), 0xFF,
                    (size_t)(gaussianCount_ - visibleCount_) * sizeof(uint32_t));
                gpuResMgr.UnmapBuffer(sortIndexBuffer_[slot].GetHandle());
            }
        }
    }

    // ---- 3.6 Early out when nothing is visible ----
    if (visibleCount_ == 0) {
        if (frameCount_ < 3) {
            PLUGIN_LOG_I("GS::ExecuteFrame: no visible gaussians, skipping render pass");
        }
        frameCount_++;
        return;
    }

    // ---- 4. Update descriptor sets ----
    if (pipelineDescriptorSetBinder_) {
        // Bind this frame's slot buffers
        pipelineDescriptorSetBinder_->BindBuffer(0u, 0u, BindableBuffer { cameraUbo_[slot].GetHandle() });
        if (gaussianBuffer_) {
            pipelineDescriptorSetBinder_->BindBuffer(0u, 1u, BindableBuffer { gaussianBuffer_.GetHandle() });
        }
        if (sortIndexBuffer_[slot]) {
            pipelineDescriptorSetBinder_->BindBuffer(0u, 2u, BindableBuffer { sortIndexBuffer_[slot].GetHandle() });
        }
        if (shBuffer_) {
            pipelineDescriptorSetBinder_->BindBuffer(0u, 3u, BindableBuffer { shBuffer_.GetHandle() });
        }

        for (auto refIndex : pipelineDescriptorSetBinder_->GetSetIndices()) {
            const auto descHandle = pipelineDescriptorSetBinder_->GetDescriptorSetHandle(refIndex);
            const auto bindings = pipelineDescriptorSetBinder_->GetDescriptorSetLayoutBindingResources(refIndex);
            cmdList.UpdateDescriptorSet(descHandle, bindings);
        }
    }

    // ---- 5. PASS 1: splats -> UNORM offscreen (raw code-space compositing) ----
    // Cleared to the web background color (raw display values) — the offscreen
    // holds the fully composited splat layer, exactly like the web canvas.
    ViewportDesc vp {};
    vp.x = 0.f;
    vp.y = 0.f;
    vp.width = (float)width;
    vp.height = (float)height;
    vp.minDepth = 0.f;
    vp.maxDepth = 1.f;

    ScissorDesc sc {};
    sc.offsetX = 0;
    sc.offsetY = 0;
    sc.extentWidth = width;
    sc.extentHeight = height;

    if (visibleCount_ > 0) {
        RenderPassDesc rpDesc {};
        rpDesc.attachmentCount = 1;
        rpDesc.attachmentHandles[0] = offscreenImage_.GetHandle();
        rpDesc.attachments[0].loadOp = AttachmentLoadOp::CORE_ATTACHMENT_LOAD_OP_CLEAR;
        rpDesc.attachments[0].storeOp = AttachmentStoreOp::CORE_ATTACHMENT_STORE_OP_STORE;
        // Raw (gamma-space) background, matching the web capture (18,12,6)
        rpDesc.attachments[0].clearValue = ClearValue(0.071f, 0.047f, 0.024f, 1.0f);
        rpDesc.renderArea.offsetX = 0;
        rpDesc.renderArea.offsetY = 0;
        rpDesc.renderArea.extentWidth = width;
        rpDesc.renderArea.extentHeight = height;
        rpDesc.subpassCount = 1;
        rpDesc.subpassContents = SubpassContents::CORE_SUBPASS_CONTENTS_INLINE;

        RenderPassSubpassDesc subpassDesc {};
        subpassDesc.colorAttachmentCount = 1;
        subpassDesc.colorAttachmentIndices[0] = 0;

        cmdList.BeginRenderPass(rpDesc, 0u, subpassDesc);
        cmdList.BindPipeline(psoHandle_);
        if (pipelineDescriptorSetBinder_) {
            const uint32_t firstSetIndex = pipelineDescriptorSetBinder_->GetFirstSet();
            cmdList.BindDescriptorSets(firstSetIndex, pipelineDescriptorSetBinder_->GetDescriptorSetHandles());
        }
        cmdList.SetDynamicStateViewport(vp);
        cmdList.SetDynamicStateScissor(sc);

        // Visible gaussians only (CPU frustum + size culled, sorted back-to-front)
        const uint32_t vertexCount = visibleCount_ * 6u;
        cmdList.Draw(vertexCount, 1u, 0u, 0u);
        cmdList.EndRenderPass();
    } else {
        // Nothing visible — clear the offscreen to bg directly (no splat draw)
        RenderPassDesc rpDesc {};
        rpDesc.attachmentCount = 1;
        rpDesc.attachmentHandles[0] = offscreenImage_.GetHandle();
        rpDesc.attachments[0].loadOp = AttachmentLoadOp::CORE_ATTACHMENT_LOAD_OP_CLEAR;
        rpDesc.attachments[0].storeOp = AttachmentStoreOp::CORE_ATTACHMENT_STORE_OP_STORE;
        rpDesc.attachments[0].clearValue = ClearValue(0.071f, 0.047f, 0.024f, 1.0f);
        rpDesc.renderArea.offsetX = 0;
        rpDesc.renderArea.offsetY = 0;
        rpDesc.renderArea.extentWidth = width;
        rpDesc.renderArea.extentHeight = height;
        rpDesc.subpassCount = 1;
        rpDesc.subpassContents = SubpassContents::CORE_SUBPASS_CONTENTS_INLINE;
        RenderPassSubpassDesc subpassDesc {};
        subpassDesc.colorAttachmentCount = 1;
        subpassDesc.colorAttachmentIndices[0] = 0;
        cmdList.BeginRenderPass(rpDesc, 0u, subpassDesc);
        cmdList.SetDynamicStateViewport(vp);
        cmdList.SetDynamicStateScissor(sc);
        cmdList.EndRenderPass();
    }

    // ---- 6. PASS 2: fullscreen copy offscreen -> sRGB backbuffer ----
    // The copy shader decodes the raw values; the sRGB hardware encode on
    // write cancels it — stored code == offscreen raw value == web output.
    if (RenderHandleUtil::IsValid(copyPsoHandle_) && copyBinder_) {
        // Re-bind the offscreen (may have been recreated on resize)
        BindableImage bi {};
        bi.handle = offscreenImage_.GetHandle();
        bi.imageLayout = ImageLayout::CORE_IMAGE_LAYOUT_UNDEFINED;
        copyBinder_->BindImage(0u, 0u, bi);
        for (auto refIndex : copyBinder_->GetSetIndices()) {
            const auto descHandle = copyBinder_->GetDescriptorSetHandle(refIndex);
            const auto bindings = copyBinder_->GetDescriptorSetLayoutBindingResources(refIndex);
            cmdList.UpdateDescriptorSet(descHandle, bindings);
        }

        RenderPassDesc rpDesc2 {};
        rpDesc2.attachmentCount = 1;
        rpDesc2.attachmentHandles[0] = backbufferHandle;
        rpDesc2.attachments[0].loadOp = AttachmentLoadOp::CORE_ATTACHMENT_LOAD_OP_DONT_CARE;
        rpDesc2.attachments[0].storeOp = AttachmentStoreOp::CORE_ATTACHMENT_STORE_OP_STORE;
        rpDesc2.renderArea.offsetX = 0;
        rpDesc2.renderArea.offsetY = 0;
        rpDesc2.renderArea.extentWidth = width;
        rpDesc2.renderArea.extentHeight = height;
        rpDesc2.subpassCount = 1;
        rpDesc2.subpassContents = SubpassContents::CORE_SUBPASS_CONTENTS_INLINE;

        RenderPassSubpassDesc subpassDesc2 {};
        subpassDesc2.colorAttachmentCount = 1;
        subpassDesc2.colorAttachmentIndices[0] = 0;

        cmdList.BeginRenderPass(rpDesc2, 0u, subpassDesc2);
        cmdList.BindPipeline(copyPsoHandle_);
        const uint32_t firstSetIndex2 = copyBinder_->GetFirstSet();
        cmdList.BindDescriptorSets(firstSetIndex2, copyBinder_->GetDescriptorSetHandles());
        cmdList.SetDynamicStateViewport(vp);
        cmdList.SetDynamicStateScissor(sc);
        cmdList.Draw(3u, 1u, 0u, 0u);  // fullscreen triangle
        cmdList.EndRenderPass();
    }

    if (frameCount_ < 3 || (frameCount_ % 300 == 0)) {
        PLUGIN_LOG_I("GS::ExecuteFrame #%u: %u/%u visible, %ux%u",
            frameCount_, visibleCount_, gaussianCount_, width, height);
    }
    frameCount_++;
}

RENDER_END_NAMESPACE()
