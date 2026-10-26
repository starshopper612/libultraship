#ifdef ENABLE_DEKO3D
#include "fast/backends/gfx_deko3d.h"
#include <array>
#include <switch.h>
#include <deko3d.hpp>
#include <cstdarg>
#include <typeinfo>
#include <cstdlib>
#include <cstdio>

namespace Fast {
namespace {
constexpr uint32_t kNumFb = 2;
constexpr uint32_t kCmdMemSize = 0x10000;
void DkLogF(const char* fmt, ...) {
    FILE* f = fopen("sdmc:/deko3d.log", "a");
    if (!f) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vfprintf(f, fmt, ap);
    va_end(ap);
    fputc('\n', f);
    fclose(f);
}
void DkDebugCb(void*, const char* ctx, DkResult res, const char* msg) {
    DkLogF("deko3d: %s result=%d %s", ctx ? ctx : "?", (int)res, msg ? msg : "");
}
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
    d.device = dk::DeviceMaker{}.setCbDebug(DkDebugCb).create();
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
    DkLogF("Init: layout size=%u align=%u fb=%ux%u", sz, al, d.width, d.height);
    nwindowSetDimensions(nwindowGetDefault(), d.width, d.height);
    DkLogF("Init: creating swapchain");
    d.swapchain = dk::SwapchainMaker{ d.device, nwindowGetDefault(), ptrs }.create();
    DkLogF("Init: swapchain ok");

    d.cmdMem = dk::MemBlockMaker{ d.device, kCmdMemSize }
                   .setFlags(DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached)
                   .create();
    d.cmdBuf = dk::CmdBufMaker{ d.device }.create();
    d.cmdBuf.addMemory(d.cmdMem, 0, kCmdMemSize);
}
void GfxRenderingAPIDeko3D::OnResize() {}
void GfxRenderingAPIDeko3D::StartFrame() {
    static uint32_t n = 0;
    if (n < 3 || n % 600 == 0) {
        DkLogF("StartFrame #%u", n);
    }
    n++;
}
void GfxRenderingAPIDeko3D::EndFrame() {
    auto& d = gDk;
    static uint32_t n = 0;
    if (n < 3 || n % 600 == 0) {
        DkLogF("EndFrame #%u", n);
    }
    n++;
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

extern "C" {
extern char __start__;
[[noreturn]] void __real_exit(int code);
[[noreturn]] void __wrap_exit(int code) {
    uintptr_t ra = (uintptr_t)__builtin_return_address(0);
    Fast::DkLogF("exit(%d) called, return address offset=0x%lx", code, (unsigned long)(ra - (uintptr_t)&__start__));
    __real_exit(code);
}
}

extern "C" {
static void DkLogExit(const char* what, int code, uintptr_t ra) {
    MemoryInfo mi;
    u32 pi;
    uintptr_t base = 0;
    if (R_SUCCEEDED(svcQueryMemory(&mi, &pi, ra))) {
        base = (uintptr_t)mi.addr;
    }
    Fast::DkLogF("%s(%d) ra=0x%lx offset=0x%lx", what, code, (unsigned long)ra, (unsigned long)(ra - base));
}
[[noreturn]] void __real__exit(int code);
[[noreturn]] void __wrap__exit(int code) {
    DkLogExit("_exit", code, (uintptr_t)__builtin_return_address(0));
    __real__exit(code);
}
[[noreturn]] void __real___libnx_exit(int code);
[[noreturn]] void __wrap___libnx_exit(int code) {
    DkLogExit("__libnx_exit", code, (uintptr_t)__builtin_return_address(0));
    __real___libnx_exit(code);
}
[[noreturn]] void __real_svcExitProcess(void);
[[noreturn]] void __wrap_svcExitProcess(void) {
    DkLogExit("svcExitProcess", 0, (uintptr_t)__builtin_return_address(0));
    __real_svcExitProcess();
}
bool __real_appletMainLoop(void);
bool __wrap_appletMainLoop(void) {
    bool r = __real_appletMainLoop();
    if (!r) {
        DkLogExit("appletMainLoop returned false", 0, (uintptr_t)__builtin_return_address(0));
    }
    return r;
}
}
extern "C" {
[[noreturn]] void __real_abort(void);
[[noreturn]] void __wrap_abort(void) {
    DkLogExit("abort", 0, (uintptr_t)__builtin_return_address(0));
    __real_abort();
}
[[noreturn]] void __real___cxa_throw(void*, void*, void (*)(void*));
[[noreturn]] void __wrap___cxa_throw(void* obj, void* tinfo, void (*dest)(void*)) {
    static int count = 0;
    if (count++ < 20) {
        DkLogExit(tinfo ? static_cast<std::type_info*>(tinfo)->name() : "throw", 0,
                  (uintptr_t)__builtin_return_address(0));
    }
    __real___cxa_throw(obj, tinfo, dest);
}
}
#endif // ENABLE_DEKO3D
