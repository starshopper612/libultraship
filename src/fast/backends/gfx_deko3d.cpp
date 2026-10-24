#ifdef ENABLE_DEKO3D
#include "fast/backends/gfx_deko3d.h"
#include <array>
#include <switch.h>
#include <deko3d.hpp>

namespace Fast {
namespace {
constexpr uint32_t kNumFb = 2;
constexpr uint32_t kCmdMemSize = 0x10000;
struct DkState {
    dk::UniqueDevice device;
    dk::UniqueQueue queue;
    dk::UniqueMemBlock fbMem;
    dk::UniqueMemBlock cmdMem;
    dk::UniqueCmdBuf cmdBuf;
    dk::Image images[kNumFb];
    dk::UniqueSwapchain swapchain;
    uint32_t width = 1280;
    uint32_t height = 720;
} gDk;
} // namespace

const char* GfxRenderingAPIDeko3D::GetName() { return "deko3d"; }
int GfxRenderingAPIDeko3D::GetMaxTextureSize() { return 16384; }
GfxClipParameters GfxRenderingAPIDeko3D::GetClipParameters() { return { true, false }; } // TODO verify
void GfxRenderingAPIDeko3D::UnloadShader(ShaderProgram*) {}
void GfxRenderingAPIDeko3D::LoadShader(ShaderProgram*) {}
void GfxRenderingAPIDeko3D::ClearShaderCache() {}
ShaderProgram* GfxRenderingAPIDeko3D::CreateAndLoadNewShader(uint64_t, uint64_t) { return nullptr; }
ShaderProgram* GfxRenderingAPIDeko3D::LookupShader(uint64_t, uint64_t) { return nullptr; }
void GfxRenderingAPIDeko3D::ShaderGetInfo(ShaderProgram*, uint8_t* numInputs, bool usedTextures[2]) {
    *numInputs = 0; usedTextures[0] = usedTextures[1] = false;
}
uint32_t GfxRenderingAPIDeko3D::NewTexture() { return 0; }
void GfxRenderingAPIDeko3D::SelectTexture(int, uint32_t) {}
void GfxRenderingAPIDeko3D::UploadTexture(const uint8_t*, uint32_t, uint32_t) {}
void GfxRenderingAPIDeko3D::SetSamplerParameters(int, bool, uint32_t, uint32_t) {}
void GfxRenderingAPIDeko3D::SetDepthTestAndMask(bool, bool) {}
void GfxRenderingAPIDeko3D::SetZmodeDecal(bool) {}
void GfxRenderingAPIDeko3D::SetViewport(int, int, int, int) {}
void GfxRenderingAPIDeko3D::SetScissor(int, int, int, int) {}
void GfxRenderingAPIDeko3D::SetUseAlpha(bool) {}
void GfxRenderingAPIDeko3D::DrawTriangles(float*, size_t, size_t) {}
void GfxRenderingAPIDeko3D::Init() {
    auto& d = gDk;
    d.device = dk::DeviceMaker{}.create();
    d.queue = dk::QueueMaker{ d.device }.setFlags(DkQueueFlags_Graphics).create();

    dk::ImageLayout layout;
    dk::ImageLayoutMaker{ d.device }
        .setFlags(DkImageFlags_UsageRender | DkImageFlags_UsagePresent | DkImageFlags_HwCompression)
        .setFormat(DkImageFormat_RGBA8_Unorm)
        .setDimensions(d.width, d.height)
        .initialize(layout);
    uint32_t sz = layout.getSize();
    uint32_t al = layout.getAlignment();
    sz = (sz + al - 1) & ~(al - 1);
    d.fbMem = dk::MemBlockMaker{ d.device, kNumFb * sz }
                  .setFlags(DkMemBlockFlags_GpuCached | DkMemBlockFlags_Image)
                  .create();

    std::array<DkImage const*, kNumFb> ptrs;
    for (uint32_t i = 0; i < kNumFb; i++) {
        d.images[i].initialize(layout, d.fbMem, i * sz);
        ptrs[i] = &d.images[i];
    }
    d.swapchain = dk::SwapchainMaker{ d.device, nwindowGetDefault(), ptrs }.create();

    d.cmdMem = dk::MemBlockMaker{ d.device, kCmdMemSize }
                   .setFlags(DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached)
                   .create();
    d.cmdBuf = dk::CmdBufMaker{ d.device }.create();
    d.cmdBuf.addMemory(d.cmdMem, 0, kCmdMemSize);
}
void GfxRenderingAPIDeko3D::OnResize() {}
void GfxRenderingAPIDeko3D::StartFrame() {}
void GfxRenderingAPIDeko3D::EndFrame() {
    auto& d = gDk;
    int slot = d.queue.acquireImage(d.swapchain);
    d.cmdBuf.clear();
    dk::ImageView target{ d.images[slot] };
    d.cmdBuf.bindRenderTargets(&target);
    d.cmdBuf.clearColor(0, DkColorMask_RGBA, 0.2f, 0.0f, 0.4f, 1.0f);
    d.queue.submitCommands(d.cmdBuf.finishList());
    d.queue.presentImage(d.swapchain, slot);
    d.queue.waitIdle(); // TODO: per-slot command memory instead of a full stall
}
void GfxRenderingAPIDeko3D::FinishRender() {}
int GfxRenderingAPIDeko3D::CreateFramebuffer() { return 0; }
void GfxRenderingAPIDeko3D::UpdateFramebufferParameters(int, uint32_t, uint32_t, uint32_t, bool, bool, bool, bool) {}
void GfxRenderingAPIDeko3D::StartDrawToFramebuffer(int, float) {}
void GfxRenderingAPIDeko3D::CopyFramebuffer(int, int, int, int, int, int, int, int, int, int) {}
void GfxRenderingAPIDeko3D::ClearFramebuffer(bool, bool) {}
void GfxRenderingAPIDeko3D::ReadFramebufferToCPU(int, uint32_t, uint32_t, uint16_t*) {}
void GfxRenderingAPIDeko3D::ResolveMSAAColorBuffer(int, int) {}
std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
GfxRenderingAPIDeko3D::GetPixelDepth(int, const std::set<std::pair<float, float>>&) { return {}; }
void* GfxRenderingAPIDeko3D::GetFramebufferTextureId(int) { return nullptr; }
void GfxRenderingAPIDeko3D::SelectTextureFb(int) {}
void GfxRenderingAPIDeko3D::DeleteTexture(uint32_t) {}
void GfxRenderingAPIDeko3D::SetTextureFilter(FilteringMode mode) { mFilter = mode; }
FilteringMode GfxRenderingAPIDeko3D::GetTextureFilter() { return mFilter; }
void GfxRenderingAPIDeko3D::SetSrgbMode() { mSrgbMode = true; }
ImTextureID GfxRenderingAPIDeko3D::GetTextureById(int) { return 0; }
void GfxRenderingAPIDeko3D::SetCurrentPrimDepth(float depth) { mCurrentPrimDepth = depth; mPrimDepthDirty = true; }
} // namespace Fast

#endif // ENABLE_DEKO3D
