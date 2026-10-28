/*
 * render_node_gaussian_splatting.h
 *
 * Custom RenderNode for 3D Gaussian Splatting rendering.
 */
#pragma once

#include <render/nodecontext/intf_render_node.h>
#include <render/nodecontext/intf_pipeline_descriptor_set_binder.h>
#include <render/device/pipeline_layout_desc.h>
#include <render/namespace.h>
#include <render/resource_handle.h>
#include <base/containers/string.h>
#include <base/containers/unique_ptr.h>
#include <base/containers/vector.h>
#include <base/util/uid.h>

RENDER_BEGIN_NAMESPACE()

class IRenderNodeContextManager;

class RenderNodeGaussianSplatting final : public IRenderNode {
public:
    RenderNodeGaussianSplatting() = default;
    ~RenderNodeGaussianSplatting() override = default;

    // IRenderNode interface
    void InitNode(IRenderNodeContextManager& renderNodeContextMgr) override;
    void PreExecuteFrame() override;
    void ExecuteFrame(IRenderCommandList& cmdList) override;
    ExecuteFlags GetExecuteFlags() const override { return 0U; }

    // for plugin / factory interface
    static constexpr BASE_NS::Uid UID { "a1b2c3d4-e5f6-7890-abcd-ef0123456789" };
    static constexpr const char* TYPE_NAME = "RenderNodeGaussianSplatting";
    static constexpr IRenderNode::BackendFlags BACKEND_FLAGS = IRenderNode::BackendFlagBits::BACKEND_FLAG_BITS_DEFAULT;
    static constexpr IRenderNode::ClassType CLASS_TYPE = IRenderNode::ClassType::CLASS_TYPE_NODE;
    static IRenderNode* Create();
    static void Destroy(IRenderNode* instance);

    struct GpuGaussian {
        float px, py, pz, pad0;
        float sx, sy, sz, opacity;
        float qw, qx, qy, qz;
        float cr, cg, cb, ca;
    };

    // Camera UBO layout (must match shader)
    struct CameraUboData {
        float view[16];
        float proj[16];
        float viewport[4];   // x, y, w, h
        float cameraPos[4];  // xyz + pad
    };

private:
    IRenderNodeContextManager* renderNodeContextMgr_ = nullptr;

    // GPU resources
    RenderHandleReference gaussianBuffer_;
    // Triple-buffered per-frame written buffers: while the GPU executes up to
    // 2 in-flight frames, the CPU writes the next slot — no torn reads.
    static constexpr uint32_t FRAME_SLOTS = 3;
    RenderHandleReference cameraUbo_[FRAME_SLOTS];
    RenderHandleReference sortIndexBuffer_[FRAME_SLOTS];
    uint32_t currentSlot_ = 0;
    RenderHandleReference shBuffer_;
    // UNORM offscreen target: raw (code-space) splat compositing, exactly
    // like the web canvas; copied to the sRGB backbuffer by a decode pass.
    RenderHandleReference offscreenImage_;
    uint32_t offscreenWidth_ = 0;
    uint32_t offscreenHeight_ = 0;
    // Copy pass resources
    RenderHandle copyShaderHandle_;
    PipelineLayout copyPipelineLayout_;
    RenderHandle copyPsoHandle_;
    IPipelineDescriptorSetBinder::Ptr copyBinder_;

    // CPU-side back-to-front sort state
    BASE_NS::vector<uint32_t> sortIndices_;   // visible gaussian ids (pre-scatter)
    BASE_NS::vector<uint32_t> sortScatter_;  // final sorted order
    BASE_NS::vector<float> sortKeys_;        // per-visible depth key
    BASE_NS::vector<uint32_t> bucketCounts_;
    BASE_NS::vector<uint32_t> bucketStarts_;
    uint32_t visibleCount_ = 0;
    // Coarse view-change tracking (re-sort threshold)
    float lastDirX_ = 0.f, lastDirY_ = 0.f, lastDirZ_ = -1.f;
    float lastEyeX_ = 0.f, lastEyeY_ = 0.f, lastEyeZ_ = 0.f;
    bool sortBufferUploaded_ = false;

    // Shader + PSO
    RenderHandle shaderHandle_;
    GraphicsState graphicsState_;
    PipelineLayout pipelineLayout_;
    RenderHandle psoHandle_;

    // Descriptor set management
    IPipelineDescriptorSetBinder::Ptr pipelineDescriptorSetBinder_;

    uint32_t gaussianCount_ = 0;
    bool needsUpload_ = true;
    bool initialized_ = false;

    uint32_t frameCount_ = 0;
};

RENDER_END_NAMESPACE()
