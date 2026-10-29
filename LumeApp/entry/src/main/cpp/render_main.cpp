/*
 * Lume 3D engine rendering demo — full engine init via XComponent.
 *
 * Phase 3: Initialize the Lume engine with Vulkan backend, create ECS,
 * load a glTF model, set up camera + light, and render via TickFrame.
 */
#include <ace/xcomponent/native_interface_xcomponent.h>
#include <dlfcn.h>
#include <time.h>
#include <hilog/log.h>
#include <napi/native_api.h>
#include <native_window/external_window.h>
#include <rawfile/raw_file_manager.h>

#include <cstdint>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

#include <rawfile/raw_dir.h>
#include <rawfile/raw_file.h>

#include "splat_loader.h"
#include "gaussian_splat_data.h"

// --- LumeBase ---
#include <base/containers/array_view.h>
#include <base/containers/refcnt_ptr.h>
#include <base/containers/string.h>
#include <base/containers/unique_ptr.h>
#include <base/math/mathf.h>
#include <base/math/matrix.h>
#include <base/math/quaternion.h>
#include <base/math/vector.h>
#include <base/util/uid.h>

// --- LumeEngine ---
#include <core/ecs/intf_ecs.h>
#include <core/ecs/intf_system_graph_loader.h>
#include <core/engine_info.h>
#include <core/implementation_uids.h>
#include <core/intf_engine.h>
#include <core/io/intf_file_manager.h>
#include <core/os/platform_create_info.h>
#include <core/plugin/intf_class_factory.h>
#include <core/plugin/intf_class_register.h>
#include <core/plugin/intf_plugin_register.h>

// --- LumeRender ---
#include <render/device/intf_device.h>
#include <render/device/intf_gpu_resource_manager.h>
#include <render/device/intf_shader_manager.h>
#include <render/datastore/intf_render_data_store_manager.h>
#include <render/nodecontext/intf_render_node_graph_manager.h>
#include <render/implementation_uids.h>
#include <3d/ecs/components/render_configuration_component.h>
#include <3d/ecs/components/environment_component.h>

#include <render/intf_render_context.h>
#include <render/intf_renderer.h>

// --- Lume_3D ---
#include <3d/ecs/components/camera_component.h>
#include <3d/ecs/components/light_component.h>
#include <3d/ecs/components/name_component.h>
#include <3d/ecs/components/node_component.h>
#include <3d/ecs/components/render_configuration_component.h>
#include <3d/ecs/components/transform_component.h>
#include <3d/ecs/components/mesh_component.h>
#include <3d/ecs/components/render_mesh_component.h>
#include <3d/ecs/components/layer_component.h>
#include <3d/ecs/components/world_matrix_component.h>
#include <3d/ecs/components/material_component.h>
#include <3d/render/intf_render_data_store_default_material.h>
#include <3d/gltf/gltf.h>
#include <3d/implementation_uids.h>
#include <3d/intf_graphics_context.h>
#include <3d/util/intf_scene_util.h>

// ---------------------------------------------------------------------------
// Log helpers
// ---------------------------------------------------------------------------
#undef LOG_DOMAIN
#define LOG_DOMAIN 0x4C4D /* "LM" */
#undef LOG_TAG
#define LOG_TAG "LumeApp"
#define LOGI(...) OH_LOG_Print(LOG_APP, LOG_INFO, LOG_DOMAIN, LOG_TAG, __VA_ARGS__)
#define LOGW(...) OH_LOG_Print(LOG_APP, LOG_WARN, LOG_DOMAIN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_DOMAIN, LOG_TAG, __VA_ARGS__)

// ---------------------------------------------------------------------------
// Engine state (all access on XComponent callback thread)
// ---------------------------------------------------------------------------
namespace {

// Rawfile base path inside the OHOS HAP sandbox.
constexpr const char* RAWFILE_BASE = "/data/storage/el1/bundle/entry/resources/rawfile/";

struct LumeState {
    OHNativeWindow* nativeWindow = nullptr;
    uint64_t width = 0;
    uint64_t height = 0;
    bool ready = false;

    // Engine objects — ordered by destruction dependency (reverse init order).
    CORE_NS::IEcs::Ptr ecs;
    CORE3D_NS::IGraphicsContext::Ptr graphicsContext;
    RENDER_NS::IRenderContext::Ptr renderContext;
    CORE_NS::IEngine::Ptr engine;

    // Non-owning pointers (owned by contexts above).
    RENDER_NS::IDevice* device = nullptr;
    RENDER_NS::IRenderer* renderer = nullptr;
    RENDER_NS::RenderHandleReference swapchainHandle;

    // Keep importer alive to prevent EntityReference cleanup from destroying imported entities.
    CORE3D_NS::IGLTF2Importer::Ptr gltfImporter;

    // 3DGS splat data
    SplatData splatData;
    bool splatLoaded = false;
    RENDER_NS::RenderHandleReference splatRngHandle;  // RNG for gaussian splatting

    // Frame counter for logging.
    uint64_t frameCount = 0;
};

LumeState g_lume;
bool g_pluginRegistryCreated = false;
napi_env g_napiEnv = nullptr;
NativeResourceManager* g_resMan = nullptr;
std::string g_filesDir;  // app sandbox writable directory

// ---------------------------------------------------------------------------
// Orbit Camera for 3DGS scene exploration
// Free-tumble arcball: orientation accumulated as world-space basis vectors
// (right/up/back) with Rodrigues rotations — no gimbal lock, no pitch clamp.
// ---------------------------------------------------------------------------
struct OrbitCamera {
    // Camera basis in world space (orthonormal)
    float rightX = 1.f, rightY = 0.f, rightZ = 0.f;
    float upX = 0.f, upY = 1.f, upZ = 0.f;
    float backX = 0.f, backY = 0.f, backZ = 1.f;

    float fovYDeg = 60.0f;  // vertical field of view (per-model)
    float shDeg1 = 0.0f;    // SH degree-1 weight (per-model)
    float shDeg23 = 0.0f;   // SH degree 2-3 weight (per-model)
    float distance = 1.2f;  // distance from target
    float targetX = 0.0f, targetY = 0.0f, targetZ = 0.0f;

    // Touch state
    float lastX = 0.0f, lastY = 0.0f;
    float lastDist = 0.0f;  // pinch distance
    int activePointers = 0;
    bool dragging = false;
    bool pinching = false;

    // Double-tap shortcut: toggle between far framing and the model centre
    double lastTapTimeMs = 0.0;
    float lastTapX = 0.0f, lastTapY = 0.0f;
    float savedDistance = 0.0f;
    bool insideView = false;

    // Rotate vector v around unit axis k by angle (Rodrigues)
    static void RotateVec(float& vx, float& vy, float& vz,
        float kx, float ky, float kz, float angle) {
        const float c = cosf(angle), s = sinf(angle);
        const float kxv = ky * vz - kz * vy;
        const float kyv = kz * vx - kx * vz;
        const float kzv = kx * vy - ky * vx;
        const float kdotv = kx * vx + ky * vy + kz * vz;
        const float nx = vx * c + kxv * s + kx * kdotv * (1.f - c);
        const float ny = vy * c + kyv * s + ky * kdotv * (1.f - c);
        const float nz = vz * c + kzv * s + kz * kdotv * (1.f - c);
        vx = nx; vy = ny; vz = nz;
    }

    // Tumble: rotate around camera up (yaw feel) and camera right (pitch feel)
    void Tumble(float yawDelta, float pitchDelta) {
        if (yawDelta != 0.f) {
            // rotate right & back around up
            RotateVec(rightX, rightY, rightZ, upX, upY, upZ, yawDelta);
            RotateVec(backX, backY, backZ, upX, upY, upZ, yawDelta);
        }
        if (pitchDelta != 0.f) {
            // rotate up & back around right
            RotateVec(upX, upY, upZ, rightX, rightY, rightZ, pitchDelta);
            RotateVec(backX, backY, backZ, rightX, rightY, rightZ, pitchDelta);
        }
    }

    void ComputeViewMatrix(float* out) const {
        // Camera looks along -back; eye = target + back * distance
        const float ex = targetX + backX * distance;
        const float ey = targetY + backY * distance;
        const float ez = targetZ + backZ * distance;
        // Column-major view matrix (world→view): rows = right, up, back
        // (view space: camera at origin looking down -Z, so in-front points have z < 0)
        out[ 0]=rightX;  out[ 1]=upX;   out[ 2]=backX;  out[ 3]=0.f;
        out[ 4]=rightY;  out[ 5]=upY;   out[ 6]=backY;  out[ 7]=0.f;
        out[ 8]=rightZ;  out[ 9]=upZ;   out[10]=backZ;  out[11]=0.f;
        out[12]=-(rightX*ex + rightY*ey + rightZ*ez);
        out[13]=-(upX*ex + upY*ey + upZ*ez);
        out[14]=-(backX*ex + backY*ey + backZ*ez);
        out[15]=1.f;
    }

    void ComputeProjMatrix(float* out, float aspect) const {
        float fovRad = fovYDeg * 3.14159265f / 180.f;
        float f = 1.f / tanf(fovRad * 0.5f);
        float zn = 0.05f, zf = 2000.f;
        memset(out, 0, 16 * sizeof(float));
        out[0]  = f / aspect;
        out[5]  = -f;  // Vulkan Y-flip
        out[10] = zf / (zn - zf);
        out[11] = -1.f;
        out[14] = (zn * zf) / (zn - zf);
    }

    void GetPosition(float& x, float& y, float& z) const {
        x = targetX + backX * distance;
        y = targetY + backY * distance;
        z = targetZ + backZ * distance;
    }

    // Pan target along camera right/up so content follows the finger
    void Pan(float dxScreen, float dyScreen, float scale) {
        targetX -= (rightX * dxScreen - upX * dyScreen) * scale;
        targetY -= (rightY * dxScreen - upY * dyScreen) * scale;
        targetZ -= (rightZ * dxScreen - upZ * dyScreen) * scale;
    }

    void LogState(const char* when) const {
        LOGI("ORBITCAM[%{public}s]: right=(%{public}.6f,%{public}.6f,%{public}.6f) up=(%{public}.6f,%{public}.6f,%{public}.6f) back=(%{public}.6f,%{public}.6f,%{public}.6f) dist=%{public}.6f target=(%{public}.6f,%{public}.6f,%{public}.6f)",
            when,
            rightX, rightY, rightZ,
            upX, upY, upZ,
            backX, backY, backZ,
            distance,
            targetX, targetY, targetZ);
    }

    void UpdateSharedCameraData(uint32_t width, uint32_t height) {
        float aspect = (float)width / (float)height;
        ComputeViewMatrix(g_gaussianCameraData.view);
        ComputeProjMatrix(g_gaussianCameraData.proj, aspect);
        g_gaussianCameraData.viewport[0] = 0.f;
        g_gaussianCameraData.viewport[1] = 0.f;
        g_gaussianCameraData.viewport[2] = (float)width;
        g_gaussianCameraData.viewport[3] = (float)height;
        float cx, cy, cz;
        GetPosition(cx, cy, cz);
        g_gaussianCameraData.cameraPos[0] = cx;
        g_gaussianCameraData.cameraPos[1] = cy;
        g_gaussianCameraData.cameraPos[2] = cz;
        g_gaussianCameraData.cameraPos[3] = 1.f;
        g_gaussianCameraData.shScale[0] = shDeg1;
        g_gaussianCameraData.shScale[1] = shDeg23;
        g_gaussianCameraData.shScale[2] = 0.f;
        g_gaussianCameraData.shScale[3] = 0.f;
        g_gaussianCameraReady = true;
    }
};
OrbitCamera g_orbitCam;

// Render-on-demand state: only render when the view changed or shortly after interaction.
static uint32_t g_forceRenderFrames = 30;  // render this many frames after startup/touch
static float g_lastRenderedView[16] = {};

// ---------------------------------------------------------------------------
// Rawfile extraction — copy HAP rawfile assets to app sandbox for file:// access
// ---------------------------------------------------------------------------

static void MkdirRecursive(const std::string& path)
{
    size_t pos = 0;
    while ((pos = path.find('/', pos + 1)) != std::string::npos) {
        mkdir(path.substr(0, pos).c_str(), 0755);
    }
    mkdir(path.c_str(), 0755);
}

static bool ExtractRawDir(NativeResourceManager* resMgr, const std::string& rawDir, const std::string& destDir)
{
    if (!resMgr) return false;
    MkdirRecursive(destDir);

    RawDir* dir = OH_ResourceManager_OpenRawDir(resMgr, rawDir.c_str());
    if (!dir) {
        LOGW("Cannot open rawdir: %{public}s", rawDir.c_str());
        return false;
    }

    int count = OH_ResourceManager_GetRawFileCount(dir);
    int extracted = 0;
    for (int i = 0; i < count; i++) {
        const char* name = OH_ResourceManager_GetRawFileName(dir, i);
        if (!name) continue;

        std::string rawPath = rawDir.empty() ? std::string(name) : rawDir + "/" + name;
        std::string destPath = destDir + "/" + name;

        // Try as file first
        RawFile* file = OH_ResourceManager_OpenRawFile(resMgr, rawPath.c_str());
        if (file) {
            long size = OH_ResourceManager_GetRawFileSize(file);
            if (size > 0) {
                std::vector<uint8_t> buf(size);
                OH_ResourceManager_ReadRawFile(file, buf.data(), size);
                OH_ResourceManager_CloseRawFile(file);

                FILE* out = fopen(destPath.c_str(), "wb");
                if (out) {
                    fwrite(buf.data(), 1, size, out);
                    fclose(out);
                    extracted++;
                }
            } else {
                OH_ResourceManager_CloseRawFile(file);
                // Size 0 might be a directory — try recursion
                ExtractRawDir(resMgr, rawPath, destPath);
            }
        } else {
            // Not a file — try as subdirectory
            ExtractRawDir(resMgr, rawPath, destPath);
        }
    }
    OH_ResourceManager_CloseRawDir(dir);
    LOGI("Extracted %{public}d files from rawfile/%{public}s", extracted, rawDir.c_str());
    return true;
}

static std::string ExtractShadersToFilesDir()
{
    if (!g_resMan || g_filesDir.empty()) {
        LOGE("Cannot extract shaders: resMan=%{public}p filesDir=%{public}s",
            g_resMan, g_filesDir.c_str());
        return "";
    }

    std::string shaderBase = g_filesDir + "/shaders";

    // Check if already extracted (skip re-extraction for speed)
    struct stat st;
    // Always extract model assets (they might be updated independently of shaders)
    std::string assetsBase = g_filesDir + "/assets";
    ExtractRawDir(g_resMan, "assets", assetsBase);

    // Always extract splat shaders (may have been added/updated after initial extraction)
    ExtractRawDir(g_resMan, "shaders/splat", shaderBase + "/splat");

    if (stat((shaderBase + "/render/shaders").c_str(), &st) == 0) {
        LOGI("Shaders already extracted at %{public}s (splat shaders refreshed)", shaderBase.c_str());
        return shaderBase;
    }

    LOGI("Extracting shaders from rawfile to %{public}s ...", shaderBase.c_str());
    ExtractRawDir(g_resMan, "shaders/render", shaderBase + "/render");
    ExtractRawDir(g_resMan, "shaders/3d", shaderBase + "/3d");
    ExtractRawDir(g_resMan, "shaders/splat", shaderBase + "/splat");

    LOGI("Shader extraction complete");
    return shaderBase;
}

// ---------------------------------------------------------------------------
// Engine initialization helpers
// ---------------------------------------------------------------------------

// Build the base filesystem path as a file:// URI for the rawfile directory.
BASE_NS::string GetRawfileUri()
{
    return BASE_NS::string("file://") + RAWFILE_BASE;
}

bool InitPluginRegistry()
{
    if (g_pluginRegistryCreated) {
        LOGI("Plugin registry already created");
        return true;
    }

    // Load plugin .so files via dlopen BEFORE creating the registry.
    // They must NOT be in liblumeapp.so's NEEDED deps, otherwise the linker
    // loads them too early (before LumeEngine globals are initialized) and
    // the constructor-based RegisterStaticPlugin writes to uninitialized memory.
    void* hMeta = dlopen("libLumeMeta.so", RTLD_NOW | RTLD_GLOBAL);
    LOGI("dlopen libLumeMeta.so: %{public}s", hMeta ? "OK" : dlerror());
    void* hRender = dlopen("libLumeRender.so", RTLD_NOW | RTLD_GLOBAL);
    LOGI("dlopen libLumeRender.so: %{public}s", hRender ? "OK" : dlerror());
    void* h3D = dlopen("libLume_3D.so", RTLD_NOW | RTLD_GLOBAL);
    LOGI("dlopen libLume_3D.so: %{public}s", h3D ? "OK" : dlerror());

    LOGI("Creating plugin registry...");
    CORE_NS::PlatformCreateInfo platformCI {};
    platformCI.jsEnv = g_napiEnv;
    platformCI.jsObject = nullptr;
    platformCI.resMan = g_resMan;
    platformCI.appPluginPath = RAWFILE_BASE;

    CORE_NS::CreatePluginRegistry(platformCI);
    g_pluginRegistryCreated = true;
    LOGI("Plugin registry created OK");

    return true;
}

bool InitEngine()
{
    LOGI("Creating engine...");

    // Diagnostic: step-by-step verify each call
    LOGI("Step 1: GetPluginRegister...");
    CORE_NS::IPluginRegister& pluginReg = CORE_NS::GetPluginRegister();
    LOGI("Step 1 OK");

    LOGI("Step 2: GetClassRegister...");
    CORE_NS::IClassRegister& classReg = pluginReg.GetClassRegister();
    LOGI("Step 2 OK");

    LOGI("Step 3: GetInstance(UID_ENGINE_FACTORY)...");
    auto* factoryIntf = classReg.GetInstance(CORE_NS::UID_ENGINE_FACTORY);
    LOGI("Step 3: factoryIntf = 0x%{public}lx", (unsigned long)(uintptr_t)factoryIntf);

    auto* factory = static_cast<CORE_NS::IEngineFactory*>(factoryIntf);

    if (!factory) {
        LOGE("Failed to get IEngineFactory");
        return false;
    }
    LOGI("IEngineFactory obtained");

    CORE_NS::EngineCreateInfo engineCI {};
    engineCI.platformCreateInfo.jsEnv = g_napiEnv;
    engineCI.platformCreateInfo.jsObject = nullptr;
    engineCI.platformCreateInfo.resMan = g_resMan;
    engineCI.platformCreateInfo.appPluginPath = RAWFILE_BASE;
    engineCI.applicationVersion.name = "LumeApp";
    engineCI.applicationVersion.versionMajor = 0;
    engineCI.applicationVersion.versionMinor = 1;
    engineCI.applicationVersion.versionPatch = 0;
    if (!factory) {
        LOGE("Failed to get IEngineFactory");
        return false;
    }
    LOGI("IEngineFactory obtained");

    g_lume.engine = factory->Create(engineCI);
    if (!g_lume.engine) {
        LOGE("IEngineFactory::Create returned null");
        return false;
    }
    LOGI("Engine instance created");

    // Extract shaders from HAP rawfile to app sandbox (file:// accessible)
    std::string shaderBase = ExtractShadersToFilesDir();
    if (shaderBase.empty()) {
        LOGE("Shader extraction failed");
        return false;
    }

    // Register filesystem paths using file:// protocol pointing at extracted files
    auto& fm = g_lume.engine->GetFileManager();
    const BASE_NS::string renderUri = BASE_NS::string("file://") + shaderBase.c_str() + "/render/";
    const BASE_NS::string tdUri = BASE_NS::string("file://") + shaderBase.c_str() + "/3d/";
    const BASE_NS::string appUri = BASE_NS::string("file://") + g_filesDir.c_str() + "/";

    fm.RegisterPath("rofsRndr", renderUri, false);
    fm.RegisterPath("rofs3D", tdUri, false);

    // Root protocol paths — 3d:// and render:// point to the root of 3d/render asset dirs
    fm.RegisterPath("3d", tdUri, false);
    fm.RegisterPath("render", renderUri, false);

    // Render paths
    fm.RegisterPath("rendershaders", renderUri + "shaders/", false);
    fm.RegisterPath("rendershaderstates", renderUri + "shaderstates/", false);
    fm.RegisterPath("rendervertexinputdeclarations", renderUri + "vertexinputdeclarations/", false);
    fm.RegisterPath("renderpipelinelayouts", renderUri + "pipelinelayouts/", false);
    fm.RegisterPath("renderrenderdataconfigurations", renderUri + "renderdataconfigurations/", false);
    fm.RegisterPath("renderrendernodegraphs", renderUri + "rendernodegraphs/", false);

    // 3D paths
    fm.RegisterPath("3dshaders", tdUri + "shaders/", false);
    fm.RegisterPath("3dshaderstates", tdUri + "shaderstates/", false);
    fm.RegisterPath("3dvertexinputdeclarations", tdUri + "vertexinputdeclarations/", false);
    fm.RegisterPath("3dpipelinelayouts", tdUri + "pipelinelayouts/", false);
    fm.RegisterPath("3drenderdataconfigurations", tdUri + "renderdataconfigurations/", false);
    fm.RegisterPath("3drendernodegraphs", tdUri + "rendernodegraphs/", false);

    // Splat shader paths
    const BASE_NS::string splatUri = BASE_NS::string("file://") + shaderBase.c_str() + "/splat/";
    fm.RegisterPath("splatshaders", splatUri, false);

    // App assets
    fm.RegisterPath("app", appUri, true);

    LOGI("All shader paths registered (file:// backed)");

    // CRITICAL: Must call LoadPlugins on the PluginRegistry to trigger
    // GatherStaticPlugins → RegisterPlugin → registerInterfaces.
    LOGI("Loading plugins via PluginRegistry...");
    CORE_NS::GetPluginRegister().LoadPlugins({});
    auto loadedPlugins = CORE_NS::GetPluginRegister().GetPlugins();
    LOGI("PluginRegistry::LoadPlugins done — %{public}zu plugins loaded", loadedPlugins.size());
    for (size_t i = 0; i < loadedPlugins.size(); ++i) {
        if (loadedPlugins[i]) {
            LOGI("  Plugin[%{public}zu]: %{public}s", i, loadedPlugins[i]->name);
        }
    }

    // Check registered IEnginePlugin type infos before engine init
    auto enginePluginInfos = CORE_NS::GetPluginRegister().GetTypeInfos(CORE_NS::IEnginePlugin::UID);
    LOGI("IEnginePlugin type infos before Init: %{public}zu", enginePluginInfos.size());

    g_lume.engine->Init();

    // After Init, check engine plugins
    enginePluginInfos = CORE_NS::GetPluginRegister().GetTypeInfos(CORE_NS::IEnginePlugin::UID);
    LOGI("IEnginePlugin type infos after Init: %{public}zu", enginePluginInfos.size());

    // Check what interfaces engine now exposes
    auto* renderCtx = g_lume.engine->GetInterface<RENDER_NS::IRenderContext>();
    LOGI("Engine->GetInterface<IRenderContext> after Init = 0x%{public}lx", (unsigned long)(uintptr_t)renderCtx);

    LOGI("Engine initialized (version: %{public}s)",
        BASE_NS::string(g_lume.engine->GetVersion()).c_str());

    return true;
}

bool InitRenderContext()
{
    LOGI("Creating render context...");

    auto* engineClassFactory = g_lume.engine->GetInterface<CORE_NS::IClassFactory>();
    if (!engineClassFactory) {
        LOGE("Engine does not expose IClassFactory");
        return false;
    }

    auto renderCtxPtr = CORE_NS::CreateInstance<RENDER_NS::IRenderContext>(*engineClassFactory, RENDER_NS::UID_RENDER_CONTEXT);
    if (!renderCtxPtr) {
        LOGE("Failed to create IRenderContext");
        return false;
    }
    g_lume.renderContext = BASE_NS::move(renderCtxPtr);
    LOGI("IRenderContext instance created");

    RENDER_NS::RenderCreateInfo renderCI {};
    renderCI.applicationInfo.name = "LumeApp";
    renderCI.deviceCreateInfo.backendType = RENDER_NS::DeviceBackendType::VULKAN;

    // CRITICAL: RenderContext::Init() registers rendershaders → rofsRndr://shaders/, etc.
    // Since RENDER_EMBEDDED_ASSETS_ENABLED=0, rofsRndr filesystem doesn't exist.
    // Register rofsRndr as a path alias to our file:// render shader directory BEFORE Init().
    {
        auto& fm = g_lume.engine->GetFileManager();
        const std::string shaderBase = g_filesDir + "/shaders";
        const BASE_NS::string renderUri = BASE_NS::string("file://") + shaderBase.c_str() + "/render/";
        fm.RegisterPath("rofsRndr", renderUri, false);
        LOGI("Registered rofsRndr → %{public}s (before RenderContext::Init)", renderUri.c_str());
    }

    auto result = g_lume.renderContext->Init(renderCI);
    if (result != RENDER_NS::RenderResultCode::RENDER_SUCCESS) {
        LOGE("IRenderContext::Init failed (result=%{public}u)", static_cast<uint32_t>(result));
        return false;
    }
    LOGI("Render context initialized with Vulkan backend");

    g_lume.device = &g_lume.renderContext->GetDevice();
    g_lume.renderer = &g_lume.renderContext->GetRenderer();

    // Create swapchain handle — only use CreateSwapchainHandle, not CreateSwapchain
    RENDER_NS::SwapchainCreateInfo swapCI {};
    swapCI.window.window = reinterpret_cast<uintptr_t>(g_lume.nativeWindow);
    swapCI.swapchainFlags =
        RENDER_NS::CORE_SWAPCHAIN_COLOR_BUFFER_BIT |
        RENDER_NS::CORE_SWAPCHAIN_DEPTH_BUFFER_BIT |
        RENDER_NS::CORE_SWAPCHAIN_SRGB_BIT;
    swapCI.imageUsageFlags = RENDER_NS::CORE_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    g_lume.swapchainHandle = g_lume.device->CreateSwapchainHandle(swapCI);
    LOGI("Swapchain handle created: valid=%{public}d", g_lume.swapchainHandle ? 1 : 0);

    return true;
}

bool InitGraphicsContext()
{
    LOGI("Creating 3D graphics context...");

    auto* classFactory = g_lume.renderContext->GetInterface<CORE_NS::IClassFactory>();
    if (!classFactory) {
        LOGE("RenderContext does not expose IClassFactory");
        return false;
    }

    g_lume.graphicsContext = CORE_NS::CreateInstance<CORE3D_NS::IGraphicsContext>(
        *classFactory, CORE3D_NS::UID_GRAPHICS_CONTEXT);
    if (!g_lume.graphicsContext) {
        LOGE("Failed to create IGraphicsContext");
        return false;
    }

    CORE3D_NS::IGraphicsContext::CreateInfo gfxCI {};

    // CRITICAL: GraphicsContext::Init() registers 3dshaders → rofs3D://shaders/, etc.
    // In normal OpenHarmony builds, rofs3D is an embedded ROFS filesystem in the .so.
    // Since CORE3D_EMBEDDED_ASSETS_ENABLED=0, we need to register rofs3D as a path
    // pointing to our file:// shader directory BEFORE Init() runs.
    {
        auto& fm = g_lume.engine->GetFileManager();
        const std::string shaderBase = g_filesDir + "/shaders";
        const BASE_NS::string tdUri = BASE_NS::string("file://") + shaderBase.c_str() + "/3d/";
        fm.RegisterPath("rofs3D", tdUri, false);
        LOGI("Registered rofs3D → %{public}s (before GraphicsContext::Init)", tdUri.c_str());
    }

    g_lume.graphicsContext->Init(gfxCI);
    LOGI("Graphics context initialized");

    // Load splat shaders
    {
        RENDER_NS::IShaderManager::ShaderFilePathDesc desc;
        desc.shaderPath = "splatshaders://";
        desc.pipelineLayoutPath = "splatshaders://";
        desc.shaderStatePath = "splatshaders://";
        desc.vertexInputDeclarationPath = "splatshaders://";
        g_lume.renderContext->GetDevice().GetShaderManager().LoadShaderFiles(desc);
        LOGI("Splat shaders loaded");
    }

    return true;
}

bool InitEcs()
{
    LOGI("Creating ECS...");

    g_lume.ecs = g_lume.engine->CreateEcs();
    if (!g_lume.ecs) {
        LOGE("Failed to create ECS");
        return false;
    }

    // Load the system graph (defines which systems run and in what order).
    // Try to load from the 3D plugin's embedded assets first.
    auto* sgFactory = CORE_NS::GetInstance<CORE_NS::ISystemGraphLoaderFactory>(
        CORE_NS::UID_SYSTEM_GRAPH_LOADER);
    if (sgFactory) {
        auto loader = sgFactory->Create(g_lume.engine->GetFileManager());
        // Try the 3D plugin's embedded asset path.
        auto loadResult = loader->Load("3d://systemGraph.json", *g_lume.ecs);
        if (!loadResult.success) {
            LOGW("Failed to load 3d://systemGraph.json: %{public}s, trying app://",
                loadResult.error.c_str());
            // Fallback: try app:// which maps to our rawfile dir.
            loadResult = loader->Load("app://systemGraph.json", *g_lume.ecs);
            if (!loadResult.success) {
                LOGW("Failed to load app://systemGraph.json: %{public}s, "
                     "ECS will use default systems", loadResult.error.c_str());
            } else {
                LOGI("Loaded system graph from app://systemGraph.json");
            }
        } else {
            LOGI("Loaded system graph from 3d://systemGraph.json");
        }
    } else {
        LOGW("ISystemGraphLoaderFactory not available, using default ECS init");
    }

    g_lume.ecs->Initialize();
    LOGI("ECS initialized");
    return true;
}

void SetupCameraAndLight()
{
    if (!g_lume.graphicsContext || !g_lume.ecs) {
        LOGW("Cannot set up camera/light: missing graphicsContext or ECS");
        return;
    }

    auto& sceneUtil = g_lume.graphicsContext->GetSceneUtil();
    auto& ecs = *g_lume.ecs;

    // Create a camera looking at the origin from a reasonable distance.
    const BASE_NS::Math::Vec3 cameraPos { 0.0f, 1.0f, 3.0f };
    const BASE_NS::Math::Quat cameraRot {}; // identity — we use LookAt below
    const float zNear = 0.1f;
    const float zFar = 100.0f;
    const float fovDeg = 60.0f;

    CORE_NS::Entity cameraEntity = sceneUtil.CreateCamera(
        ecs, cameraPos, cameraRot, zNear, zFar, fovDeg);

    // Point the camera at the origin.
    const BASE_NS::Math::Vec3 target { 0.0f, 0.0f, 0.0f };
    const BASE_NS::Math::Vec3 up { 0.0f, 1.0f, 0.0f };
    sceneUtil.CameraLookAt(ecs, cameraEntity, cameraPos, target, up);

    // Update the camera viewport to match surface resolution.
    const BASE_NS::Math::UVec2 resolution {
        static_cast<uint32_t>(g_lume.width),
        static_cast<uint32_t>(g_lume.height)
    };
    sceneUtil.UpdateCameraViewport(ecs, cameraEntity, resolution);

    // Set a bright blue clear color so we can tell if the camera is rendering to screen
    auto* camMgr = CORE_NS::GetManager<CORE3D_NS::ICameraComponentManager>(ecs);
    if (camMgr) {
        auto camHandle = camMgr->Write(cameraEntity);
        if (camHandle) {
            camHandle->clearColorValue = { 0.006f, 0.004f, 0.002f, 1.0f }; // warm black (matches web viewer bg)
            // CRITICAL: Without MAIN_CAMERA_BIT, RenderNodeDefaultCameraController renders
            // to an internal buffer instead of CORE_DEFAULT_BACKBUFFER (swapchain).
            // The scene renders correctly but nothing gets presented to screen.
            camHandle->sceneFlags |= CORE3D_NS::CameraComponent::SceneFlagBits::MAIN_CAMERA_BIT;
            // CRITICAL: CLEAR_COLOR_BIT forces loadOp=clear on color attachment.
            // Without it, the default LWRP pipeline uses loadOp=dont_care and never clears.
            camHandle->pipelineFlags |= CORE3D_NS::CameraComponent::PipelineFlagBits::CLEAR_COLOR_BIT;
            LOGI("Camera: clear=dark-neutral, MAIN_CAMERA_BIT + CLEAR_COLOR_BIT set");
        }
    }

    // CRITICAL: Add RenderConfigurationComponent to enable render pipeline creation.
    // Without this, RenderSystem won't create any render node graphs and the screen stays black.
    auto* rcMgr = CORE_NS::GetManager<CORE3D_NS::IRenderConfigurationComponentManager>(ecs);
    if (rcMgr) {
        rcMgr->Create(cameraEntity);

        // Create an environment entity with a visible background color
        auto* envMgr = CORE_NS::GetManager<CORE3D_NS::IEnvironmentComponentManager>(ecs);
        if (envMgr) {
            CORE_NS::Entity envEntity = ecs.GetEntityManager().Create();
            envMgr->Create(envEntity);
            // Link environment to the render configuration
            auto rcHandle = rcMgr->Write(cameraEntity);
            if (rcHandle) {
                rcHandle->environment = envEntity;
            }
            LOGI("EnvironmentComponent created and linked to camera");
        }

        LOGI("RenderConfigurationComponent added to camera entity (CREATE_RNGS_BIT enabled by default)");
    } else {
        LOGE("IRenderConfigurationComponentManager not found — rendering will be black");
    }

    LOGI("Camera created and positioned at (0, 1, 3) looking at origin");

    // Create a directional light.
    CORE3D_NS::LightComponent lightComp {};
    lightComp.type = CORE3D_NS::LightComponent::Type::DIRECTIONAL;
    lightComp.intensity = 1.0f;
    lightComp.color = { 1.0f, 1.0f, 1.0f };

    const BASE_NS::Math::Vec3 lightPos { 2.0f, 4.0f, 2.0f };
    const BASE_NS::Math::Quat lightRot {}; // identity
    sceneUtil.CreateLight(ecs, lightComp, lightPos, lightRot);

    LOGI("Directional light created at (2, 4, 2)");
}

void LoadGltfModel()
{
    if (!g_lume.graphicsContext || !g_lume.ecs) {
        LOGW("Cannot load glTF: missing graphicsContext or ECS");
        return;
    }

    auto& gltf = g_lume.graphicsContext->GetGltf();
    auto& ecs = *g_lume.ecs;

    // Try loading the model.
    const BASE_NS::string uri = "app://assets/cube.glb";
    LOGI("Loading glTF: %{public}s", uri.c_str());

    auto loadResult = gltf.LoadGLTF(uri);
    if (!loadResult.success) {
        LOGW("glTF load failed: %{public}s — engine will render clear color only",
            loadResult.error.c_str());
        return;
    }
    LOGI("glTF parsed successfully");

    // Create an importer and import synchronously.
    // CRITICAL: Store the importer in g_lume to keep it alive. The importer holds EntityReferences
    // to all imported entities (meshes, materials, buffers). If the importer is destroyed,
    // the reference counts drop to zero and all imported entities are deleted.
    g_lume.gltfImporter = gltf.CreateGLTF2Importer(ecs);
    auto& importer = g_lume.gltfImporter;
    if (!importer) {
        LOGE("Failed to create GLTF2 importer");
        return;
    }

    CORE3D_NS::GltfImportOptions importOptions {};
    importer->ImportGLTF(*loadResult.data, importOptions);

    // Drive the import to completion.
    constexpr uint32_t IMPORT_TIME_BUDGET_MS = 1000;
    while (!importer->IsCompleted()) {
        importer->Execute(IMPORT_TIME_BUDGET_MS);
    }

    auto& importResult = importer->GetResult();
    if (!importResult.success) {
        LOGW("glTF import failed — engine will render clear color only");
        return;
    }
    LOGI("glTF imported: %{public}zu meshes, %{public}zu materials, %{public}zu images, %{public}zu textures, %{public}zu skins",
        importResult.data.meshes.size(),
        importResult.data.materials.size(),
        importResult.data.images.size(),
        importResult.data.textures.size(),
        importResult.data.skins.size());

    // Import scene 0 into ECS.
    gltf.ImportGltfScene(0, *loadResult.data, importResult.data, ecs);
    LOGI("glTF scene imported into ECS");

    // CRITICAL: Force-touch all MeshComponent entities to bump the generation counter.
    // This ensures RenderSystem detects the mesh as new on the next tick, even if the mesh
    // was created before RenderSystem's Initialize() registered its event listeners.
    {
        auto* meshMgr = CORE_NS::GetManager<CORE3D_NS::IMeshComponentManager>(ecs);
        if (meshMgr) {
            for (uint32_t i = 0; i < meshMgr->GetComponentCount(); i++) {
                auto entity = meshMgr->GetEntity(i);
                auto handle = meshMgr->Write(entity);
                // Just writing (even without changes) bumps the generation counter
                LOGI("Force-touched MeshComponent on entity 0x%{public}llx",
                    (unsigned long long)entity.id);
            }
        }
        // Also touch materials
        auto* matMgr = CORE_NS::GetManager<CORE3D_NS::IMaterialComponentManager>(ecs);
        if (matMgr) {
            for (uint32_t i = 0; i < matMgr->GetComponentCount(); i++) {
                auto entity = matMgr->GetEntity(i);
                auto handle = matMgr->Write(entity);
                LOGI("Force-touched MaterialComponent on entity 0x%{public}llx",
                    (unsigned long long)entity.id);
            }
        }
    }

    // ===== DIAGNOSTIC: Dump all entities with MeshComponent =====
    {
        auto* meshMgr = CORE_NS::GetManager<CORE3D_NS::IMeshComponentManager>(ecs);
        auto* rmMgr = CORE_NS::GetManager<CORE3D_NS::IRenderMeshComponentManager>(ecs);
        auto* transMgr = CORE_NS::GetManager<CORE3D_NS::ITransformComponentManager>(ecs);
        auto* nameMgr = CORE_NS::GetManager<CORE3D_NS::INameComponentManager>(ecs);
        auto* nodeMgr = CORE_NS::GetManager<CORE3D_NS::INodeComponentManager>(ecs);
        auto* layerMgr = CORE_NS::GetManager<CORE3D_NS::ILayerComponentManager>(ecs);
        auto* worldMgr = CORE_NS::GetManager<CORE3D_NS::IWorldMatrixComponentManager>(ecs);
        auto* matMgr = CORE_NS::GetManager<CORE3D_NS::IMaterialComponentManager>(ecs);

        LOGI("DIAG ECS: meshMgr=%{public}d rmMgr=%{public}d transMgr=%{public}d nameMgr=%{public}d nodeMgr=%{public}d layerMgr=%{public}d worldMgr=%{public}d matMgr=%{public}d",
            meshMgr?1:0, rmMgr?1:0, transMgr?1:0, nameMgr?1:0, nodeMgr?1:0, layerMgr?1:0, worldMgr?1:0, matMgr?1:0);

        if (meshMgr) {
            uint32_t meshCount = meshMgr->GetComponentCount();
            LOGI("DIAG ECS: %{public}u entities with MeshComponent", meshCount);
            for (uint32_t i = 0; i < meshCount; i++) {
                auto entity = meshMgr->GetEntity(i);
                const char* name = "?";
                if (nameMgr) {
                    if (auto nh = nameMgr->Read(entity); nh) {
                        name = nh->name.c_str();
                    }
                }
                bool hasRenderMesh = rmMgr && rmMgr->HasComponent(entity);
                bool hasTransform = transMgr && transMgr->HasComponent(entity);
                bool hasNode = nodeMgr && nodeMgr->HasComponent(entity);
                bool hasLayer = layerMgr && layerMgr->HasComponent(entity);
                bool hasWorld = worldMgr && worldMgr->HasComponent(entity);

                LOGI("DIAG Mesh[%{public}u] name='%{public}s' RM=%{public}d T=%{public}d N=%{public}d L=%{public}d W=%{public}d",
                    i, name,
                    hasRenderMesh?1:0, hasTransform?1:0, hasNode?1:0, hasLayer?1:0, hasWorld?1:0);

                if (hasTransform) {
                    auto th = transMgr->Read(entity);
                    if (th) {
                        LOGI("  pos=(%{public}.2f,%{public}.2f,%{public}.2f) scale=(%{public}.2f,%{public}.2f,%{public}.2f)",
                            th->position.x, th->position.y, th->position.z,
                            th->scale.x, th->scale.y, th->scale.z);
                    }
                }
                if (hasNode) {
                    auto nh = nodeMgr->Read(entity);
                    if (nh) {
                        LOGI("  node: enabled=%{public}d effectivelyEnabled=%{public}d",
                            nh->enabled?1:0, nh->effectivelyEnabled?1:0);
                    }
                }
                if (hasLayer) {
                    auto lh = layerMgr->Read(entity);
                    if (lh) {
                        LOGI("  layer: mask=0x%{public}llx", (unsigned long long)lh->layerMask);
                    }
                }
            }
        }
        // Also dump camera info
        auto* camMgr = CORE_NS::GetManager<CORE3D_NS::ICameraComponentManager>(ecs);
        if (camMgr) {
            uint32_t camCount = camMgr->GetComponentCount();
            LOGI("DIAG ECS: %{public}u cameras", camCount);
            for (uint32_t i = 0; i < camCount; i++) {
                auto entity = camMgr->GetEntity(i);
                auto ch = camMgr->Read(i);
                if (ch) {
                    LOGI("  cam[%{public}u] sceneFlags=0x%{public}x pipeFlags=0x%{public}x layerMask=0x%{public}llx res=%{public}ux%{public}u",
                        i, ch->sceneFlags, ch->pipelineFlags,
                        (unsigned long long)ch->layerMask,
                        ch->renderResolution[0], ch->renderResolution[1]);
                }
            }
        }
        // Material count
        if (matMgr) {
            LOGI("DIAG ECS: %{public}u materials", matMgr->GetComponentCount());
            for (uint32_t i = 0; i < matMgr->GetComponentCount(); i++) {
                auto entity = matMgr->GetEntity(i);
                auto mh = matMgr->Read(entity);
                if (mh) {
                    LOGI("  mat[%{public}u] type=%{public}u shaderValid=%{public}d gsValid=%{public}d depthValid=%{public}d baseColor=(%{public}.2f,%{public}.2f,%{public}.2f,%{public}.2f)",
                        i, (unsigned)mh->type,
                        CORE_NS::EntityUtil::IsValid(mh->materialShader.shader)?1:0,
                        CORE_NS::EntityUtil::IsValid(mh->materialShader.graphicsState)?1:0,
                        CORE_NS::EntityUtil::IsValid(mh->depthShader.shader)?1:0,
                        mh->textures[CORE3D_NS::MaterialComponent::TextureIndex::BASE_COLOR].factor.x,
                        mh->textures[CORE3D_NS::MaterialComponent::TextureIndex::BASE_COLOR].factor.y,
                        mh->textures[CORE3D_NS::MaterialComponent::TextureIndex::BASE_COLOR].factor.z,
                        mh->textures[CORE3D_NS::MaterialComponent::TextureIndex::BASE_COLOR].factor.w);
                }
            }
        }
        // RenderMesh count
        if (rmMgr) {
            uint32_t rmCount = rmMgr->GetComponentCount();
            LOGI("DIAG ECS: %{public}u entities with RenderMeshComponent", rmCount);
            for (uint32_t i = 0; i < rmCount; i++) {
                auto entity = rmMgr->GetEntity(i);
                const char* name = "?";
                if (nameMgr) {
                    if (auto nh = nameMgr->Read(entity); nh) {
                        name = nh->name.c_str();
                    }
                }
                bool hasTransform = transMgr && transMgr->HasComponent(entity);
                bool hasNode = nodeMgr && nodeMgr->HasComponent(entity);
                bool hasLayer = layerMgr && layerMgr->HasComponent(entity);
                bool hasWorld = worldMgr && worldMgr->HasComponent(entity);
                LOGI("DIAG RM[%{public}u] name='%{public}s' T=%{public}d N=%{public}d L=%{public}d W=%{public}d",
                    i, name, hasTransform?1:0, hasNode?1:0, hasLayer?1:0, hasWorld?1:0);
                // Check what mesh entity the RenderMeshComponent references
                {
                    auto rmh = rmMgr->Read(entity);
                    if (rmh) {
                        bool meshValid = CORE_NS::EntityUtil::IsValid(rmh->mesh);
                        bool meshHasMeshComp = meshValid && meshMgr && meshMgr->HasComponent(rmh->mesh);
                        LOGI("  RM->mesh: valid=%{public}d hasMeshComp=%{public}d meshEntity=0x%{public}llx",
                            meshValid?1:0, meshHasMeshComp?1:0, (unsigned long long)rmh->mesh.id);
                        // Dump submesh details from the MeshComponent
                        if (meshHasMeshComp) {
                            auto msh = meshMgr->Read(rmh->mesh);
                            if (msh) {
                                LOGI("  MeshComp: %{public}zu submeshes", msh->submeshes.size());
                                for (size_t s = 0; s < msh->submeshes.size(); s++) {
                                    auto& sm = msh->submeshes[s];
                                    LOGI("    sm[%{public}zu]: vtx=%{public}u idx=%{public}u mat=0x%{public}llx buf0Valid=%{public}d idxBufValid=%{public}d",
                                        s, sm.vertexCount, sm.indexCount,
                                        (unsigned long long)sm.material.id,
                                        CORE_NS::EntityUtil::IsValid(sm.bufferAccess[0].buffer)?1:0,
                                        CORE_NS::EntityUtil::IsValid(sm.indexBuffer.buffer)?1:0);
                                }
                            }
                        }
                    }
                }
                if (hasTransform) {
                    auto th = transMgr->Read(entity);
                    if (th) {
                        LOGI("  pos=(%{public}.2f,%{public}.2f,%{public}.2f) scale=(%{public}.2f,%{public}.2f,%{public}.2f)",
                            th->position.x, th->position.y, th->position.z,
                            th->scale.x, th->scale.y, th->scale.z);
                    }
                }
                if (hasNode) {
                    auto nh = nodeMgr->Read(entity);
                    if (nh) {
                        LOGI("  node: enabled=%{public}d effectivelyEnabled=%{public}d sceneId=%{public}u",
                            nh->enabled?1:0, nh->effectivelyEnabled?1:0, nh->sceneId);
                    }
                }
            }
        }
        // Total entity count
        LOGI("DIAG ECS: entity mgr active");
    }
    // ===== END DIAGNOSTIC =====
}

// Generate synthetic test gaussians: colored cube corners + edges + axes
// This bypasses splat file loading to verify the rendering pipeline independently.
static void GenerateTestGaussians(SplatData& out)
{
    out.gaussians.clear();
    out.gaussianCount = 0;

    auto addGaussian = [&](float x, float y, float z, float s, float r, float g, float b, float a) {
        GaussianSplatGPU gp {};
        gp.px = x; gp.py = y; gp.pz = z; gp.pad0 = 0.f;
        gp.sx = s; gp.sy = s; gp.sz = s; gp.opacity = a;
        gp.qw = 1.f; gp.qx = 0.f; gp.qy = 0.f; gp.qz = 0.f;  // identity rotation
        gp.cr = r; gp.cg = g; gp.cb = b; gp.ca = a;
        out.gaussians.push_back(gp);
        out.gaussianCount++;
    };

    const float S = 5.0f;  // half-size of cube
    const float gs = 0.3f; // gaussian scale (size)

    // 8 cube corners — each a different color
    addGaussian(-S, -S, -S, gs, 1, 0, 0, 1);  // red
    addGaussian( S, -S, -S, gs, 0, 1, 0, 1);  // green
    addGaussian( S,  S, -S, gs, 0, 0, 1, 1);  // blue
    addGaussian(-S,  S, -S, gs, 1, 1, 0, 1);  // yellow
    addGaussian(-S, -S,  S, gs, 1, 0, 1, 1);  // magenta
    addGaussian( S, -S,  S, gs, 0, 1, 1, 1);  // cyan
    addGaussian( S,  S,  S, gs, 1, 1, 1, 1);  // white
    addGaussian(-S,  S,  S, gs, 1, 0.5, 0, 1); // orange

    // Edges — 10 gaussians per edge (12 edges), in the edge's color blend
    auto addEdge = [&](float x0, float y0, float z0, float x1, float y1, float z1,
                       float r, float g, float b) {
        const int N = 10;
        for (int i = 0; i <= N; i++) {
            float t = (float)i / (float)N;
            float x = x0 + t * (x1 - x0);
            float y = y0 + t * (y1 - y0);
            float z = z0 + t * (z1 - z0);
            addGaussian(x, y, z, gs * 0.5f, r, g, b, 0.9f);
        }
    };

    // Bottom face edges (y = -S)
    addEdge(-S,-S,-S,  S,-S,-S, 1,0,0);
    addEdge( S,-S,-S,  S,-S, S, 0,1,0);
    addEdge( S,-S, S, -S,-S, S, 0,0,1);
    addEdge(-S,-S, S, -S,-S,-S, 1,1,0);
    // Top face edges (y = S)
    addEdge(-S, S,-S,  S, S,-S, 1,0,1);
    addEdge( S, S,-S,  S, S, S, 0,1,1);
    addEdge( S, S, S, -S, S, S, 1,1,1);
    addEdge(-S, S, S, -S, S,-S, 1,0.5,0);
    // Vertical edges
    addEdge(-S,-S,-S, -S, S,-S, 0.5,0,0);
    addEdge( S,-S,-S,  S, S,-S, 0,0.5,0);
    addEdge( S,-S, S,  S, S, S, 0,0,0.5);
    addEdge(-S,-S, S, -S, S, S, 0.5,0.5,0);

    // 3 axes from origin — longer lines
    for (int i = 0; i <= 20; i++) {
        float t = (float)i / 20.f * S * 2.f;
        addGaussian(t, 0, 0, gs*0.4f, 1, 0, 0, 1);  // X = red
        addGaussian(0, t, 0, gs*0.4f, 0, 1, 0, 1);  // Y = green
        addGaussian(0, 0, t, gs*0.4f, 0, 0, 1, 1);  // Z = blue
    }

    out.boundsMin[0] = out.boundsMin[1] = out.boundsMin[2] = -S;
    out.boundsMax[0] = out.boundsMax[1] = out.boundsMax[2] = S;

    LOGI("Test gaussians generated: %{public}u total", out.gaussianCount);
}

void LoadSplatModel()
{
    // Demo model selection:
    //   0 = fruit_bowl.ply  (standard 3DGS PLY, colour-calibration reference)
    //   1 = godzilla.splat  (compressed .splat chunk format, indoor scene)
    //   2 = doll.glb        (glTF 2.0 + UWA_gaussian_splatting extension)
    constexpr int kDemoModel = 2;

    bool loaded = false;
    if (kDemoModel == 1) {
        std::string splatPath = g_filesDir + "/assets/godzilla.splat";
        LOGI("Loading splat: %{public}s", splatPath.c_str());
        loaded = LoadSplatFile(splatPath.c_str(), g_lume.splatData);
    } else if (kDemoModel == 2) {
        std::string glbPath = g_filesDir + "/assets/doll.glb";
        LOGI("Loading splat glb: %{public}s", glbPath.c_str());
        loaded = LoadUwaSplatGlb(glbPath.c_str(), g_lume.splatData);
    } else {
        std::string plyPath = g_filesDir + "/assets/fruit_bowl.ply";
        LOGI("Loading splat: %{public}s", plyPath.c_str());
        loaded = LoadStandard3DgsPly(plyPath.c_str(), g_lume.splatData, 100.0f);
    }

    if (loaded) {
        g_lume.splatLoaded = true;
        LOGI("Splat loaded: %{public}u gaussians, bounds=(%{public}.2f,%{public}.2f,%{public}.2f)-(%{public}.2f,%{public}.2f,%{public}.2f)",
            g_lume.splatData.gaussianCount,
            g_lume.splatData.boundsMin[0], g_lume.splatData.boundsMin[1], g_lume.splatData.boundsMin[2],
            g_lume.splatData.boundsMax[0], g_lume.splatData.boundsMax[1], g_lume.splatData.boundsMax[2]);
        // Print first few gaussians for verification
        for (uint32_t i = 0; i < 3 && i < g_lume.splatData.gaussianCount; i++) {
            auto& g = g_lume.splatData.gaussians[i];
            LOGI("  g[%{public}u] pos=(%{public}.3f,%{public}.3f,%{public}.3f) scale=(%{public}.4f,%{public}.4f,%{public}.4f) color=(%{public}.2f,%{public}.2f,%{public}.2f,%{public}.2f)",
                i, g.px, g.py, g.pz, g.sx, g.sy, g.sz, g.cr, g.cg, g.cb, g.opacity);
        }

        // Per-model FOV: godzilla's reference app uses 80°, everything else 60°.
        g_orbitCam.fovYDeg = (kDemoModel == 1) ? 80.0f : 60.0f;
        // Per-model SH: the doll's VQ-SH matches its reference renderer (measured
        // saturation/contrast inside the reference range); the fruit-bowl PLY is
        // matched against a DC-dominant web viewer, and godzilla carries no SH.
        g_orbitCam.shDeg1 = (kDemoModel == 2) ? 1.0f : 0.0f;
        g_orbitCam.shDeg23 = (kDemoModel == 2) ? 1.0f : 0.0f;

        if (kDemoModel == 0) {
            // Reference viewpoint — recorded from the user's manual match to the
            // web capture (ORBITCAM[after-touch] log, same framing as web.png).
            // The app now always starts at this angle so color comparisons stay valid.
            g_orbitCam.rightX = -0.717920f; g_orbitCam.rightY = 0.024674f; g_orbitCam.rightZ = -0.695681f;
            g_orbitCam.upX    =  0.326028f; g_orbitCam.upY    = -0.871066f; g_orbitCam.upZ    = -0.367345f;
            g_orbitCam.backX  = -0.615049f; g_orbitCam.backY  = -0.490535f; g_orbitCam.backZ  =  0.617311f;
            g_orbitCam.targetX = 3.886603f; g_orbitCam.targetY = -1.901034f; g_orbitCam.targetZ = 0.267847f;
            g_orbitCam.distance = 57.762306f;
        } else {
            // Auto-frame from the model bounds: axis-aligned identity basis,
            // target at the bounding-box centre, distance fitted to the extent.
            float cx = (g_lume.splatData.boundsMin[0] + g_lume.splatData.boundsMax[0]) * 0.5f;
            float cy = (g_lume.splatData.boundsMin[1] + g_lume.splatData.boundsMax[1]) * 0.5f;
            float cz = (g_lume.splatData.boundsMin[2] + g_lume.splatData.boundsMax[2]) * 0.5f;
            float ex = g_lume.splatData.boundsMax[0] - g_lume.splatData.boundsMin[0];
            float ey = g_lume.splatData.boundsMax[1] - g_lume.splatData.boundsMin[1];
            float ez = g_lume.splatData.boundsMax[2] - g_lume.splatData.boundsMin[2];
            float maxExtent = ex > ey ? (ex > ez ? ex : ez) : (ey > ez ? ey : ez);
            g_orbitCam.rightX = 1.f; g_orbitCam.rightY = 0.f; g_orbitCam.rightZ = 0.f;
            g_orbitCam.upX = 0.f; g_orbitCam.upY = 1.f; g_orbitCam.upZ = 0.f;
            g_orbitCam.backX = 0.f; g_orbitCam.backY = 0.f; g_orbitCam.backZ = 1.f;
            g_orbitCam.targetX = cx;
            g_orbitCam.targetY = cy;
            g_orbitCam.targetZ = cz;
            g_orbitCam.distance = (maxExtent > 0.01f ? maxExtent * 1.6f : 50.f);
            if (kDemoModel == 2) {
                // Deterministic comparison view for the doll: inside the
                // environment sphere, close to the subject (matches the
                // reference capture framing).
                g_orbitCam.targetX = 0.f; g_orbitCam.targetY = 0.f; g_orbitCam.targetZ = 0.f;
                g_orbitCam.distance = 42.f;
            }
        }
        g_orbitCam.LogState("initial");

        // Feed data to the RenderNode via shared globals
        static_assert(sizeof(GaussianSplatGPU) == sizeof(GpuGaussianData),
            "GaussianSplatGPU must match GpuGaussianData layout");
        g_gaussianSplatData = reinterpret_cast<const GpuGaussianData*>(g_lume.splatData.gaussians.data());
        g_gaussianSplatCount = g_lume.splatData.gaussianCount;
        g_gaussianShData = g_lume.splatData.sh.empty() ? nullptr : g_lume.splatData.sh.data();
        g_gaussianDataReady = true;
        LOGI("Splat data passed to shared globals for RenderNode");

        // Create a RNG (Render Node Graph) for the gaussian splatting node
        auto& rngMgr = g_lume.renderContext->GetRenderNodeGraphManager();
        RENDER_NS::RenderNodeGraphDesc rngDesc {};
        rngDesc.renderNodeGraphName = "GaussianSplatRNG";

        RENDER_NS::RenderNodeDesc nodeDesc {};
        nodeDesc.typeName = "RenderNodeGaussianSplatting";
        nodeDesc.nodeName = "GaussianSplatNode";
        // No renderPass config here — the RenderNode handles its own render pass in ExecuteFrame
        rngDesc.nodes.push_back(BASE_NS::move(nodeDesc));

        g_lume.splatRngHandle = rngMgr.Create(
            RENDER_NS::IRenderNodeGraphManager::RenderNodeGraphUsageType::RENDER_NODE_GRAPH_STATIC,
            rngDesc);
        LOGI("Gaussian splat RNG created: valid=%{public}d",
            g_lume.splatRngHandle ? 1 : 0);
    } else {
        LOGE("Failed to load splat model %{public}d", kDemoModel);
    }
}

// ---------------------------------------------------------------------------
// Frame rendering
// ---------------------------------------------------------------------------

void DrawFrame()
{
    if (!g_lume.ready || !g_lume.engine) {
        return;
    }

    // Update shared camera data for the gaussian splatting RenderNode
    if (g_lume.width > 0 && g_lume.height > 0) {
        g_orbitCam.UpdateSharedCameraData((uint32_t)g_lume.width, (uint32_t)g_lume.height);
    }

    // TickFrame updates internal time and drives all ECS systems.
    CORE_NS::IEcs* ecsArr[] = { g_lume.ecs.get() };
    bool needsRender = g_lume.engine->TickFrame(
        BASE_NS::array_view<CORE_NS::IEcs*>(ecsArr, 1u));

    // Render-on-demand: skip the expensive RenderFrame when the view is static.
    // This keeps the GPU idle (and the device cool) between interactions.
    bool viewChanged = (std::memcmp(g_lastRenderedView, g_gaussianCameraData.view,
                            sizeof(g_lastRenderedView)) != 0);
    if (g_forceRenderFrames > 0) {
        g_forceRenderFrames--;
    }
    const bool doRender = viewChanged || (g_forceRenderFrames > 0) || (g_lume.frameCount < 5);

    if (doRender && needsRender && g_lume.graphicsContext && g_lume.renderer) {
        std::memcpy(g_lastRenderedView, g_gaussianCameraData.view, sizeof(g_lastRenderedView));
        // Check render data store for submitted submeshes
        if (g_lume.frameCount == 5 || g_lume.frameCount == 10) {
            // Also check mesh component manager generation counter
            auto* meshMgr = CORE_NS::GetManager<CORE3D_NS::IMeshComponentManager>(*g_lume.ecs);
            auto* matMgr = CORE_NS::GetManager<CORE3D_NS::IMaterialComponentManager>(*g_lume.ecs);
            LOGI("DIAG frame %{public}llu: meshGen=%{public}u meshCount=%{public}u matGen=%{public}u matCount=%{public}u",
                (unsigned long long)g_lume.frameCount,
                meshMgr ? (unsigned)meshMgr->GetGenerationCounter() : 0,
                meshMgr ? meshMgr->GetComponentCount() : 0,
                matMgr ? (unsigned)matMgr->GetGenerationCounter() : 0,
                matMgr ? matMgr->GetComponentCount() : 0);

            auto& rdsMgr = g_lume.renderContext->GetRenderDataStoreManager();
            // Try multiple names (the data store may have a scene prefix)
            const char* names[] = {
                "RenderDataStoreDefaultMaterial",
                "0_RenderDataStoreDefaultMaterial",
                "RenderDataStoreDefaultMaterial0",
            };
            for (auto n : names) {
                auto ds = rdsMgr.GetRenderDataStore(n);
                if (ds) {
                    auto* mat = static_cast<CORE3D_NS::IRenderDataStoreDefaultMaterial*>(ds.get());
                    auto submeshes = mat->GetSubmeshes();
                    auto meshData = mat->GetMeshData();
                    LOGI("DIAG frame %{public}llu ds='%{public}s': submeshes=%{public}zu meshData=%{public}zu",
                        (unsigned long long)g_lume.frameCount, n, submeshes.size(), meshData.size());
                } else {
                    LOGI("DIAG frame %{public}llu ds='%{public}s': NOT FOUND",
                        (unsigned long long)g_lume.frameCount, n);
                }
            }
        }
        auto rngs = g_lume.graphicsContext->GetRenderNodeGraphs(*g_lume.ecs);
        if (g_lume.frameCount <= 2) {
            LOGI("DrawFrame thread: %{public}ld, RNG: %{public}zu",
                (long)gettid(), rngs.size());
            // Check CORE_DEFAULT_BACKBUFFER descriptor size
            auto& gpuResMgr = g_lume.renderContext->GetDevice().GetGpuResourceManager();
            auto bbHandle = gpuResMgr.GetImageHandle("CORE_DEFAULT_BACKBUFFER");
            if (bbHandle) {
                auto desc = gpuResMgr.GetImageDescriptor(bbHandle);
                LOGI("CORE_DEFAULT_BACKBUFFER: %{public}ux%{public}u format=%{public}u",
                    desc.width, desc.height, (unsigned)desc.format);
            } else {
                LOGE("CORE_DEFAULT_BACKBUFFER handle NOT FOUND");
            }
        }

        // Append gaussian splatting RNG if available
        if (g_lume.splatRngHandle) {
            // Build a combined list: scene RNGs + splat RNG
            BASE_NS::vector<RENDER_NS::RenderHandleReference> allRngs;
            allRngs.reserve(rngs.size() + 1);
            for (size_t i = 0; i < rngs.size(); i++) {
                allRngs.push_back(rngs[i]);
            }
            allRngs.push_back(g_lume.splatRngHandle);
            g_lume.renderer->RenderFrame(
                BASE_NS::array_view<const RENDER_NS::RenderHandleReference>(allRngs.data(), allRngs.size()));
        } else {
            g_lume.renderer->RenderFrame(rngs);
        }
    }

    g_lume.frameCount++;
    if (g_lume.frameCount <= 5 || (g_lume.frameCount % 300 == 0)) {
        LOGI("Tick %{public}llu (rendered=%{public}d)",
            (unsigned long long)g_lume.frameCount, doRender ? 1 : 0);
    }
}

// ---------------------------------------------------------------------------
// Teardown
// ---------------------------------------------------------------------------

void DestroyEngine()
{
    LOGI("Destroying engine resources...");
    g_lume.ready = false;

    // Destroy in reverse init order.
    if (g_lume.ecs) {
        g_lume.ecs->Uninitialize();
        g_lume.ecs->ProcessEvents();
    }
    g_lume.ecs.reset();

    g_lume.graphicsContext.reset();

    if (g_lume.device) {
        g_lume.device->DestroySwapchain();
    }

    g_lume.renderer = nullptr;
    g_lume.device = nullptr;
    g_lume.renderContext.reset();
    g_lume.engine.reset();
    g_lume.nativeWindow = nullptr;
    g_lume.frameCount = 0;

    LOGI("Engine resources destroyed");
}

// ---------------------------------------------------------------------------
// XComponent callbacks
// ---------------------------------------------------------------------------

void OnSurfaceCreated(OH_NativeXComponent* component, void* window)
{
    LOGI("OnSurfaceCreated (tid=%{public}d)", gettid());
    uint64_t w = 0, h = 0;
    OH_NativeXComponent_GetXComponentSize(component, window, &w, &h);
    LOGI("Surface size: %{public}lux%{public}lu", (unsigned long)w, (unsigned long)h);

    g_lume.nativeWindow = static_cast<OHNativeWindow*>(window);
    g_lume.width = w;
    g_lume.height = h;

    // Set buffer geometry so the native window allocates the right size.
    OH_NativeWindow_NativeWindowHandleOpt(
        g_lume.nativeWindow, SET_BUFFER_GEOMETRY,
        static_cast<int32_t>(w), static_cast<int32_t>(h));

    // Full initialization sequence.
    bool ok = InitPluginRegistry();
    if (ok) ok = InitEngine();
    if (ok) ok = InitRenderContext();  // creates device, swapchain, then Init
    if (ok) ok = InitGraphicsContext();
    if (ok) ok = InitEcs();

    if (ok) {
        g_lume.ready = true;
        LOGI("=== Lume engine fully initialized ===");

        // Set up scene: camera, light.
        SetupCameraAndLight();

        // CRITICAL: Tick one frame BEFORE loading the model. This lets RenderSystem::Initialize()
        // run and register its event listeners. If we load the model before the first tick,
        // RenderSystem's meshGeneration_ counter absorbs the mesh creation and the generation
        // check never detects it as new, leaving meshModifiedEvents_ permanently empty.
        DrawFrame();
        LOGI("First tick completed — RenderSystem initialized, now loading model");

        // glTF demo cube disabled — 3DGS-only debugging (the Lume scene pass
        // still runs and provides the backbuffer + clear color).
        // LoadGltfModel();

        // Load 3DGS splat data (if available)
        LoadSplatModel();

        // Render second frame to process the newly loaded model.
        DrawFrame();
    } else {
        LOGE("=== Lume engine initialization FAILED ===");
    }
}

void OnSurfaceChanged(OH_NativeXComponent* component, void* window)
{
    uint64_t w = 0, h = 0;
    OH_NativeXComponent_GetXComponentSize(component, window, &w, &h);
    LOGI("OnSurfaceChanged %{public}lux%{public}lu", (unsigned long)w, (unsigned long)h);
    g_lume.width = w;
    g_lume.height = h;
    OH_NativeWindow_NativeWindowHandleOpt(
        g_lume.nativeWindow, SET_BUFFER_GEOMETRY,
        static_cast<int32_t>(w), static_cast<int32_t>(h));
}

void OnSurfaceDestroyed(OH_NativeXComponent* /*component*/, void* /*window*/)
{
    LOGI("OnSurfaceDestroyed");
    DestroyEngine();
}

void DispatchTouchEvent(OH_NativeXComponent* component, void* window)
{
    OH_NativeXComponent_TouchEvent touchEvent;
    if (OH_NativeXComponent_GetTouchEvent(component, window, &touchEvent) != 0) {
        DrawFrame();
        return;
    }

    const float sensitivity = 0.005f;  // radians per pixel
    const float zoomSensitivity = 0.01f;

    float x = touchEvent.x;
    float y = touchEvent.y;

    switch (touchEvent.type) {
    case OH_NATIVEXCOMPONENT_DOWN: {
        if (touchEvent.numPoints == 1) {
            g_orbitCam.lastX = x;
            g_orbitCam.lastY = y;
            g_orbitCam.dragging = true;
            g_orbitCam.pinching = false;
        } else if (touchEvent.numPoints >= 2) {
            // Start pinch
            g_orbitCam.dragging = false;
            g_orbitCam.pinching = true;
            float x1 = touchEvent.touchPoints[0].x;
            float y1 = touchEvent.touchPoints[0].y;
            float x2 = touchEvent.touchPoints[1].x;
            float y2 = touchEvent.touchPoints[1].y;
            float dx = x2 - x1, dy = y2 - y1;
            g_orbitCam.lastDist = sqrtf(dx*dx + dy*dy);
            // Also store for pan
            g_orbitCam.lastX = (x1 + x2) * 0.5f;
            g_orbitCam.lastY = (y1 + y2) * 0.5f;
        }
        break;
    }
    case OH_NATIVEXCOMPONENT_MOVE: {
        // Keep rendering for a couple frames after the last touch event
        g_forceRenderFrames = 3;
        if (touchEvent.numPoints >= 2 && g_orbitCam.pinching) {
            // Pinch zoom
            float x1 = touchEvent.touchPoints[0].x;
            float y1 = touchEvent.touchPoints[0].y;
            float x2 = touchEvent.touchPoints[1].x;
            float y2 = touchEvent.touchPoints[1].y;
            float dx = x2 - x1, dy = y2 - y1;
            float dist = sqrtf(dx*dx + dy*dy);
            if (g_orbitCam.lastDist > 1.f) {
                float ratio = g_orbitCam.lastDist / dist;
                g_orbitCam.distance *= ratio;
                if (g_orbitCam.distance < 0.1f) g_orbitCam.distance = 0.1f;
                if (g_orbitCam.distance > 2000.f) g_orbitCam.distance = 2000.f;
            }
            g_orbitCam.lastDist = dist;

            // Two-finger pan
            float mx = (x1 + x2) * 0.5f;
            float my = (y1 + y2) * 0.5f;
            float panDx = mx - g_orbitCam.lastX;
            float panDy = my - g_orbitCam.lastY;
            float panScale = g_orbitCam.distance * 0.002f;
            g_orbitCam.Pan(panDx, panDy, panScale);
            g_orbitCam.lastX = mx;
            g_orbitCam.lastY = my;
        } else if (touchEvent.numPoints == 1 && g_orbitCam.dragging) {
            // Free tumble — no angle limits
            float dx = x - g_orbitCam.lastX;
            float dy = y - g_orbitCam.lastY;
            g_orbitCam.Tumble(-dx * sensitivity, -dy * sensitivity);
            g_orbitCam.lastX = x;
            g_orbitCam.lastY = y;
        }
        break;
    }
    case OH_NATIVEXCOMPONENT_UP:
    case OH_NATIVEXCOMPONENT_CANCEL:
        g_orbitCam.dragging = false;
        g_orbitCam.pinching = false;
        // Double-tap: toggle between the far framing and the model centre
        {
            struct timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            const double nowMs = (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
            const float dx = x - g_orbitCam.lastTapX;
            const float dy = y - g_orbitCam.lastTapY;
            if (touchEvent.type == OH_NATIVEXCOMPONENT_UP &&
                nowMs - g_orbitCam.lastTapTimeMs < 350.0 &&
                dx*dx + dy*dy < 1600.0f) {
                if (!g_orbitCam.insideView) {
                    g_orbitCam.savedDistance = g_orbitCam.distance;
                    // Model-scale dependent: jump to 5% of the far framing
                    // (≈1.7 for the fruit bowl, ≈8 for godzilla, ≈70 for doll)
                    g_orbitCam.distance = (g_orbitCam.savedDistance > 20.f)
                                              ? g_orbitCam.savedDistance * 0.03f
                                              : 1.5f;
                    g_orbitCam.insideView = true;
                    LOGI("Camera: jump to centre (distance %.2f)", g_orbitCam.distance);
                } else {
                    g_orbitCam.distance = (g_orbitCam.savedDistance > 0.5f) ? g_orbitCam.savedDistance : 50.f;
                    g_orbitCam.insideView = false;
                    LOGI("Camera: jump back to far view");
                }
                g_orbitCam.lastTapTimeMs = 0.0;
            } else if (touchEvent.type == OH_NATIVEXCOMPONENT_UP) {
                g_orbitCam.lastTapTimeMs = nowMs;
                g_orbitCam.lastTapX = x;
                g_orbitCam.lastTapY = y;
            }
        }
        g_orbitCam.LogState("after-touch");
        break;
    default:
        break;
    }

    // Update shared camera data for RenderNode
    if (g_lume.width > 0 && g_lume.height > 0) {
        g_orbitCam.UpdateSharedCameraData((uint32_t)g_lume.width, (uint32_t)g_lume.height);
    }

    DrawFrame();
}

// ---------------------------------------------------------------------------
// NAPI module — register XComponent callbacks + drawFrame export
// ---------------------------------------------------------------------------

static napi_value NapiInitResourceManager(napi_env env, napi_callback_info info)
{
    size_t argc = 2;
    napi_value args[2] = {nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    if (argc >= 1 && args[0] != nullptr) {
        g_resMan = OH_ResourceManager_InitNativeResourceManager(env, args[0]);
        LOGI("NativeResourceManager initialized: %{public}p", g_resMan);
    }
    if (argc >= 2 && args[1] != nullptr) {
        size_t len = 0;
        napi_get_value_string_utf8(env, args[1], nullptr, 0, &len);
        g_filesDir.resize(len);
        napi_get_value_string_utf8(env, args[1], &g_filesDir[0], len + 1, &len);
        LOGI("filesDir: %{public}s", g_filesDir.c_str());
    }
    return nullptr;
}

static napi_value NapiDrawFrame(napi_env /*env*/, napi_callback_info /*info*/)
{
    DrawFrame();
    return nullptr;
}

static napi_value Init(napi_env env, napi_value exports)
{
    // Save napi_env for engine initialization later.
    g_napiEnv = env;

    // Get XComponent native object.
    napi_value xcomponentObj = nullptr;
    napi_get_named_property(env, exports, OH_NATIVE_XCOMPONENT_OBJ, &xcomponentObj);

    OH_NativeXComponent* nativeXComponent = nullptr;
    napi_unwrap(env, xcomponentObj, reinterpret_cast<void**>(&nativeXComponent));

    if (nativeXComponent) {
        static OH_NativeXComponent_Callback callback;
        callback.OnSurfaceCreated = OnSurfaceCreated;
        callback.OnSurfaceChanged = OnSurfaceChanged;
        callback.OnSurfaceDestroyed = OnSurfaceDestroyed;
        callback.DispatchTouchEvent = DispatchTouchEvent;
        OH_NativeXComponent_RegisterCallback(nativeXComponent, &callback);
        LOGI("XComponent callbacks registered");
    }

    // Export drawFrame so ArkTS can drive the render loop via setInterval.
    napi_property_descriptor desc[] = {
        {"drawFrame", nullptr, NapiDrawFrame, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"initResourceManager", nullptr, NapiInitResourceManager, nullptr, nullptr, nullptr, napi_default, nullptr},
    };
    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);

    return exports;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------
EXTERN_C_START
static napi_module s_module = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "lumeapp",
    .nm_priv = nullptr,
    .reserved = {0},
};
EXTERN_C_END

extern "C" __attribute__((constructor)) void RegisterLumeRenderModule(void)
{
    napi_module_register(&s_module);
}


