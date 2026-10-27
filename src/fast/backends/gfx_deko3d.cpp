#ifdef ENABLE_DEKO3D
#include "fast/backends/gfx_deko3d.h"
#include "gfx_deko3d_shaders.h"
#include "gfx_deko3d_shaders_tex.h"
#include "fast/interpreter.h"
#include <map>
#include <vector>
#include <cstring>
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
constexpr uint32_t kCmdMemSize = 0x200000;
constexpr uint32_t kVtxMemSize = 0x400000;
constexpr uint32_t kCodeMemSize = 0x10000;
constexpr uint32_t kMaxTex = 4096;
constexpr uint32_t kNumSamplers = 32;
// If the image is upside down, change this to -1.0f (or back to 1.0f).
constexpr float kFlipY = 1.0f;

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
    dk::UniqueMemBlock depthMem;
    dk::UniqueMemBlock vtxMem;
    dk::UniqueMemBlock codeMem;
    dk::UniqueMemBlock cmdMem;
    dk::UniqueCmdBuf cmdBuf;
    dk::Image images[kNumFb];
    dk::Image depthImage;
    dk::Shader vsh;
    dk::Shader fsh;
    dk::Shader tvsh;
    dk::Shader tfsh;
    dk::UniqueMemBlock imgDescMem;
    dk::UniqueMemBlock smpDescMem;
    dk::UniqueSwapchain swapchain;
    uint32_t width = 1280;
    uint32_t height = 720;
} gDk;

struct Rect {
    int x = 0, y = 0, w = 1280, h = 720;
};
Rect gVp, gSc;
struct DkTex {
    dk::UniqueMemBlock mem;
    dk::Image image;
    uint32_t w = 0;
    uint32_t h = 0;
    bool linear = false;
    uint32_t cms = 0;
    uint32_t cmt = 0;
};
std::vector<DkTex> gTexs(1);
std::vector<uint32_t> gFreeTex;
std::vector<dk::UniqueMemBlock> gGrave; // freed after the GPU is idle
uint32_t gBoundTex[6] = {};
uint32_t gCurTile = 0;
bool gBlend = false;
uint32_t gTexturedDraws = 0;
// Debug switches read from sdmc:/deko3d.cfg (e.g. "notex noblend nodesc").
bool gNoTex = false;
bool gNoBlend = false;
bool gNoDesc = false;
bool gQueueDead = false;
// Per-shader vertex layout, decoded from the shader ids by Fast3D's own feature parser.
struct DkShaderInfo {
    uint8_t numInputs = 0;
    bool usedTextures[6] = {};
    bool alpha = false;
    uint32_t numFloats = 4;
    uint32_t inputOff = 4;
    uint32_t nColor = 3;
};
std::map<std::pair<uint64_t, uint64_t>, DkShaderInfo> gShaders;
DkShaderInfo* gCurShader = nullptr;
uint32_t gSkippedNoColor = 0;
uint32_t gLayoutMismatch = 0;
uint32_t gDraws = 0, gTris = 0;
uint32_t gVtxCursor = 0;
int gCurFb = 0;
int gDkSlot = 0;
bool gFrameOpen = false;
bool gDepthTest = true;
bool gZUpd = true;

uint32_t Align256(uint32_t v) { return (v + 255u) & ~255u; }

void ApplyRect(const Rect& r, bool viewport) {
    auto& d = gDk;
    int w = r.w < 1 ? 1 : r.w;
    int h = r.h < 1 ? 1 : r.h;
    int x = r.x < 0 ? 0 : r.x;
    // Fast3D gives a bottom-left origin, deko3d wants top-left.
    int y = (int)d.height - (r.y + h);
    if (y < 0) {
        y = 0;
    }
    if (viewport) {
        d.cmdBuf.setViewports(0, { { (float)x, (float)y, (float)w, (float)h, 0.0f, 1.0f } });
    } else {
        d.cmdBuf.setScissors(0, { { (uint32_t)x, (uint32_t)y, (uint32_t)w, (uint32_t)h } });
    }
}

void BeginFrame() {
    auto& d = gDk;
    if (d.queue.isInErrorState()) {
        if (!gQueueDead) {
            gQueueDead = true;
            DkLogF("queue is in error state: rendering disabled");
        }
        gFrameOpen = false;
        return;
    }
    gDraws = 0;
    gTris = 0;
    gTexturedDraws = 0;
    gVtxCursor = 0;
    gCurFb = 0;
    int slot = d.queue.acquireImage(d.swapchain);
    d.cmdBuf.clear();
    dk::ImageView target{ d.images[slot] };
    dk::ImageView depth{ d.depthImage };
    d.cmdBuf.bindRenderTargets(&target, &depth);
    if (!gNoDesc) {
        d.cmdBuf.bindImageDescriptorSet(d.imgDescMem.getGpuAddr(), kMaxTex + 1);
        d.cmdBuf.bindSamplerDescriptorSet(d.smpDescMem.getGpuAddr(), kNumSamplers);
    }
    d.cmdBuf.setViewports(0, { { 0.0f, 0.0f, (float)d.width, (float)d.height, 0.0f, 1.0f } });
    d.cmdBuf.setScissors(0, { { 0, 0, d.width, d.height } });
    d.cmdBuf.clearColor(0, DkColorMask_RGBA, 0.2f, 0.0f, 0.4f, 1.0f);
    d.cmdBuf.clearDepthStencil(true, 1.0f, 0xFF, 0);
    gFrameOpen = true;
    gDkSlot = slot;
}
} // namespace

const char* GfxRenderingAPIDeko3D::GetName() { return "deko3d"; }
int GfxRenderingAPIDeko3D::GetMaxTextureSize() { return 16384; }
GfxClipParameters GfxRenderingAPIDeko3D::GetClipParameters() { return { true, false }; } // TODO verify
void GfxRenderingAPIDeko3D::UnloadShader(ShaderProgram*) {}
void GfxRenderingAPIDeko3D::LoadShader(ShaderProgram* prg) { gCurShader = reinterpret_cast<DkShaderInfo*>(prg); }
void GfxRenderingAPIDeko3D::ClearShaderCache() {}
ShaderProgram* GfxRenderingAPIDeko3D::CreateAndLoadNewShader(uint64_t id0, uint64_t id1) {
    CCFeatures cc;
    gfx_cc_get_features(id0, id1, &cc);
    DkShaderInfo& info = gShaders[std::make_pair(id0, id1)];
    uint32_t cnt = 4;
    for (int i = 0; i < 2; i++) {
        if (cc.usedTextures[i]) {
            cnt += 2;
            for (int j = 0; j < 2; j++) {
                if (cc.clamp[i][j]) {
                    cnt += 1;
                }
            }
        }
    }
    if (cc.opt_fog) {
        cnt += 4;
    }
    if (cc.opt_grayscale) {
        cnt += 4;
    }
    info.inputOff = cnt;
    info.alpha = cc.opt_alpha;
    info.nColor = cc.opt_alpha ? 4 : 3;
    info.numInputs = cc.numInputs;
    cnt += (uint32_t)cc.numInputs * info.nColor;
    info.numFloats = cnt;
    info.usedTextures[0] = cc.usedTextures[0];
    info.usedTextures[1] = cc.usedTextures[1];
    info.usedTextures[2] = cc.used_masks[0];
    info.usedTextures[3] = cc.used_masks[1];
    info.usedTextures[4] = cc.used_blend[0];
    info.usedTextures[5] = cc.used_blend[1];
    static uint32_t n = 0;
    if (n++ < 60) {
        DkLogF("shader %llx %llx: floats=%u inputs=%u alpha=%d tex=%d%d mask=%d%d blend=%d%d fog=%d gray=%d", (unsigned long long)id0,
               (unsigned long long)id1, info.numFloats, (unsigned)info.numInputs, (int)info.alpha,
               (int)info.usedTextures[0], (int)info.usedTextures[1], (int)info.usedTextures[2],
               (int)info.usedTextures[3], (int)info.usedTextures[4], (int)info.usedTextures[5], (int)cc.opt_fog,
               (int)cc.opt_grayscale);
    }
    gCurShader = &info;
    return reinterpret_cast<ShaderProgram*>(&info);
}
ShaderProgram* GfxRenderingAPIDeko3D::LookupShader(uint64_t id0, uint64_t id1) {
    auto it = gShaders.find(std::make_pair(id0, id1));
    return it == gShaders.end() ? nullptr : reinterpret_cast<ShaderProgram*>(&it->second);
}
void GfxRenderingAPIDeko3D::ShaderGetInfo(ShaderProgram* prg, uint8_t* numInputs, bool usedTextures[2]) {
    DkShaderInfo* info = reinterpret_cast<DkShaderInfo*>(prg);
    *numInputs = info->numInputs;
    usedTextures[0] = info->usedTextures[0];
    usedTextures[1] = info->usedTextures[1];
}
uint32_t GfxRenderingAPIDeko3D::NewTexture() {
    uint32_t id;
    if (!gFreeTex.empty()) {
        id = gFreeTex.back();
        gFreeTex.pop_back();
    } else {
        id = (uint32_t)gTexs.size();
        if (id > kMaxTex) {
            DkLogF("out of texture ids");
            return 0;
        }
        gTexs.emplace_back();
    }
    return id;
}
void GfxRenderingAPIDeko3D::SelectTexture(int tile, uint32_t id) {
    if (tile >= 0 && tile < 6) {
        gBoundTex[tile] = id;
        gCurTile = (uint32_t)tile;
    }
}
void GfxRenderingAPIDeko3D::UploadTexture(const uint8_t* rgba32, uint32_t w, uint32_t h) {
    auto& d = gDk;
    uint32_t id = gBoundTex[gCurTile];
    if (id == 0 || id >= gTexs.size() || w == 0 || h == 0) {
        return;
    }
    DkTex& t = gTexs[id];
    if (t.mem) {
        gGrave.push_back(std::move(t.mem));
    }
    uint32_t stride = (w * 4 + 31u) & ~31u;
    dk::ImageLayout layout;
    dk::ImageLayoutMaker{ d.device }
        .setFlags(DkImageFlags_PitchLinear)
        .setFormat(DkImageFormat_RGBA8_Unorm)
        .setDimensions(w, h)
        .setPitchStride(stride)
        .initialize(layout);
    uint32_t al = layout.getAlignment();
    if (al < 0x1000) {
        al = 0x1000;
    }
    uint32_t msz = ((uint32_t)layout.getSize() + al - 1) & ~(al - 1);
    t.mem = dk::MemBlockMaker{ d.device, msz }
                .setFlags(DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached | DkMemBlockFlags_Image)
                .create();
    t.image.initialize(layout, t.mem, 0);
    uint8_t* dst = (uint8_t*)t.mem.getCpuAddr();
    for (uint32_t y = 0; y < h; y++) {
        memcpy(dst + (size_t)y * stride, rgba32 + (size_t)y * w * 4, (size_t)w * 4);
    }
    t.w = w;
    t.h = h;
    dk::ImageView view{ t.image };
    DkImageDescriptor* descs = (DkImageDescriptor*)d.imgDescMem.getCpuAddr();
    dkImageDescriptorInitialize(&descs[id], &view, false, false);
    static uint32_t logged = 0;
    if (logged++ < 20) {
        DkLogF("texture %u uploaded %ux%u stride=%u", id, w, h, stride);
    }
}
void GfxRenderingAPIDeko3D::SetSamplerParameters(int tile, bool linear, uint32_t cms, uint32_t cmt) {
    if (tile < 0 || tile >= 6) {
        return;
    }
    uint32_t id = gBoundTex[tile];
    if (id == 0 || id >= gTexs.size()) {
        return;
    }
    gTexs[id].linear = linear;
    gTexs[id].cms = cms;
    gTexs[id].cmt = cmt;
}
void GfxRenderingAPIDeko3D::SetDepthTestAndMask(bool depth_test, bool z_upd) {
    gDepthTest = depth_test;
    gZUpd = z_upd;
}
void GfxRenderingAPIDeko3D::SetZmodeDecal(bool) {}
void GfxRenderingAPIDeko3D::SetViewport(int x, int y, int w, int h) { gVp = { x, y, w, h }; }
void GfxRenderingAPIDeko3D::SetScissor(int x, int y, int w, int h) { gSc = { x, y, w, h }; }
void GfxRenderingAPIDeko3D::SetUseAlpha(bool a) { gBlend = a; }
void GfxRenderingAPIDeko3D::DrawTriangles(float* buf, size_t len, size_t tris) {
    gDraws++;
    gTris += (uint32_t)tris;
    auto& d = gDk;
    if (!gFrameOpen || gCurFb != 0 || tris == 0) {
        return;
    }
    uint32_t nverts = (uint32_t)tris * 3;
    uint32_t per = (uint32_t)(len / nverts);
    if (per < 4 || per > 64 || (size_t)per * nverts != len) {
        static int warned = 0;
        if (warned++ < 5) {
            DkLogF("draw skipped: len=%zu tris=%zu per=%u", len, tris, per);
        }
        return;
    }
    uint32_t nColor;
    uint32_t colOff;
    if (gCurShader && gCurShader->numFloats == per) {
        if (gCurShader->numInputs == 0) {
            gSkippedNoColor++;
            return;
        }
        nColor = gCurShader->nColor;
        colOff = gCurShader->inputOff;
    } else {
        if (gLayoutMismatch++ < 10) {
            DkLogF("layout mismatch: per=%u shader floats=%d", per, gCurShader ? (int)gCurShader->numFloats : -1);
        }
        if (per < 7) {
            return;
        }
        // fallback guess: color is the last 3 or 4 floats
        nColor = (per % 4 == 3) ? 3 : 4;
        colOff = per - nColor;
    }
    uint32_t bytes = nverts * per * 4;
    gVtxCursor = Align256(gVtxCursor);
    if (gVtxCursor + bytes > kVtxMemSize) {
        return;
    }
    float* dst = (float*)((uint8_t*)d.vtxMem.getCpuAddr() + gVtxCursor);
    memcpy(dst, buf, bytes);
    if (kFlipY < 0.0f) {
        for (uint32_t v = 0; v < nverts; v++) {
            dst[v * per + 1] = -dst[v * per + 1];
        }
    }
    DkGpuAddr addr = d.vtxMem.getGpuAddr() + gVtxCursor;
    gVtxCursor += bytes;

    uint32_t texId = gBoundTex[0];
    bool textured = !gNoTex && gCurShader && gCurShader->numFloats == per && gCurShader->usedTextures[0] && texId != 0 &&
                    texId < gTexs.size() && gTexs[texId].w != 0;
    if (textured) {
        d.cmdBuf.bindShaders(DkStageFlag_GraphicsMask, { &d.tvsh, &d.tfsh });
        gTexturedDraws++;
    } else {
        d.cmdBuf.bindShaders(DkStageFlag_GraphicsMask, { &d.vsh, &d.fsh });
    }
    dk::RasterizerState rs;
    rs.setCullMode(DkFace_None);
    d.cmdBuf.bindRasterizerState(rs);
    dk::ColorState cs;
    cs.setBlendEnable(0, gBlend && !gNoBlend);
    d.cmdBuf.bindColorState(cs);
    dk::ColorWriteState cws;
    d.cmdBuf.bindColorWriteState(cws);
    if (!gNoBlend) {
        dk::BlendState bs;
        d.cmdBuf.bindBlendStates(0, bs);
    }
    dk::DepthStencilState ds;
    ds.setDepthTestEnable(gDepthTest).setDepthWriteEnable(gZUpd).setDepthCompareOp(DkCompareOp_Less);
    d.cmdBuf.bindDepthStencilState(ds);
    std::array<DkVtxAttribState, 3> attribs{};
    attribs[0].size = DkVtxAttribSize_4x32;
    attribs[0].type = DkVtxAttribType_Float;
    uint32_t numAttribs = 2;
    if (textured) {
        // location 1 = uv0 (right after position), location 2 = color
        attribs[1].offset = 16;
        attribs[1].size = DkVtxAttribSize_2x32;
        attribs[1].type = DkVtxAttribType_Float;
        attribs[2].offset = colOff * 4;
        attribs[2].size = (nColor == 3) ? DkVtxAttribSize_3x32 : DkVtxAttribSize_4x32;
        attribs[2].type = DkVtxAttribType_Float;
        numAttribs = 3;
        const DkTex& tx = gTexs[texId];
        uint32_t sIdx = (tx.linear ? 16u : 0u) + (tx.cms & 3u) * 4u + (tx.cmt & 3u);
        std::array<DkResHandle, 1> handles = { { dkMakeTextureHandle(texId, sIdx) } };
        d.cmdBuf.bindTextures(DkStage_Fragment, 0, handles);
    } else {
        attribs[1].offset = colOff * 4;
        attribs[1].size = (nColor == 3) ? DkVtxAttribSize_3x32 : DkVtxAttribSize_4x32;
        attribs[1].type = DkVtxAttribType_Float;
    }
    if (numAttribs < 3) attribs[2] = attribs[1]; // zeroed attrib = invalid format, so give the unused slot a valid copy
    d.cmdBuf.bindVtxAttribState(attribs);
    std::array<DkVtxBufferState, 1> bufState{};
    bufState[0].stride = per * 4;
    bufState[0].divisor = 0;
    d.cmdBuf.bindVtxBufferState(bufState);
    d.cmdBuf.bindVtxBuffer(0, addr, bytes);
    ApplyRect(gVp, true);
    ApplyRect(gSc, false);
    d.cmdBuf.draw(DkPrimitive_Triangles, nverts, 1, 0, 0);
}
void GfxRenderingAPIDeko3D::Init() {
    auto& d = gDk;
    {
        char cfg[256] = {};
        FILE* cf = fopen("sdmc:/deko3d.cfg", "r");
        if (cf) {
            size_t n = fread(cfg, 1, sizeof(cfg) - 1, cf);
            cfg[n] = 0;
            fclose(cf);
        }
        gNoTex = gNoTex || strstr(cfg, "notex") != nullptr;
        gNoBlend = gNoBlend || strstr(cfg, "noblend") != nullptr;
        gNoDesc = gNoDesc || strstr(cfg, "nodesc") != nullptr;
        DkLogF("cfg: notex=%d noblend=%d nodesc=%d", (int)gNoTex, (int)gNoBlend, (int)gNoDesc);
    }
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

    // Depth buffer
    dk::ImageLayout dlayout;
    dk::ImageLayoutMaker{ d.device }
        .setFlags(DkImageFlags_UsageRender | DkImageFlags_HwCompression)
        .setFormat(DkImageFormat_Z24S8)
        .setDimensions(d.width, d.height)
        .initialize(dlayout);
    uint32_t dsz = dlayout.getSize();
    uint32_t dal = dlayout.getAlignment();
    dsz = (dsz + dal - 1) & ~(dal - 1);
    d.depthMem = dk::MemBlockMaker{ d.device, dsz }
                     .setFlags(DkMemBlockFlags_GpuCached | DkMemBlockFlags_Image)
                     .create();
    d.depthImage.initialize(dlayout, d.depthMem, 0);

    // Vertex memory
    d.vtxMem = dk::MemBlockMaker{ d.device, kVtxMemSize }
                   .setFlags(DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached)
                   .create();

    // Shaders
    d.codeMem = dk::MemBlockMaker{ d.device, kCodeMemSize }
                    .setFlags(DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached | DkMemBlockFlags_Code)
                    .create();
    uint8_t* code = (uint8_t*)d.codeMem.getCpuAddr();
    memcpy(code, kDkBasicVsh, kDkBasicVsh_size);
    memcpy(code + 0x1000, kDkBasicFsh, kDkBasicFsh_size);
    dk::ShaderMaker{ d.codeMem, 0 }.initialize(d.vsh);
    dk::ShaderMaker{ d.codeMem, 0x1000 }.initialize(d.fsh);
    memcpy(code + 0x2000, kDkTexVsh, kDkTexVsh_size);
    memcpy(code + 0x3000, kDkTexFsh, kDkTexFsh_size);
    dk::ShaderMaker{ d.codeMem, 0x2000 }.initialize(d.tvsh);
    dk::ShaderMaker{ d.codeMem, 0x3000 }.initialize(d.tfsh);
    DkLogF("Init: tex shaders tvsh=%d tfsh=%d sizes=%u/%u", (int)dkShaderIsValid(&d.tvsh), (int)dkShaderIsValid(&d.tfsh),
           (unsigned)kDkTexVsh_size, (unsigned)kDkTexFsh_size);

    uint32_t imgBytes = ((kMaxTex + 1) * 32u + 0xFFFu) & ~0xFFFu;
    d.imgDescMem = dk::MemBlockMaker{ d.device, imgBytes }
                       .setFlags(DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached)
                       .create();
    memset(d.imgDescMem.getCpuAddr(), 0, imgBytes);
    d.smpDescMem = dk::MemBlockMaker{ d.device, 0x1000 }
                       .setFlags(DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached)
                       .create();
    DkSamplerDescriptor* sd = (DkSamplerDescriptor*)d.smpDescMem.getCpuAddr();
    static const DkWrapMode wm[4] = { DkWrapMode_Repeat, DkWrapMode_MirroredRepeat, DkWrapMode_ClampToEdge,
                                      DkWrapMode_MirrorClampToEdge };
    for (uint32_t i = 0; i < kNumSamplers; i++) {
        DkFilter f = (i >= 16) ? DkFilter_Linear : DkFilter_Nearest;
        dk::Sampler smp;
        smp.setFilter(f, f).setWrapMode(wm[(i >> 2) & 3], wm[i & 3]);
        dkSamplerDescriptorInitialize(&sd[i], &smp);
    }
    DkLogF("Init: shaders vsh=%d fsh=%d depth=%u vtx=%u", (int)dkShaderIsValid(&d.vsh), (int)dkShaderIsValid(&d.fsh),
           dsz, kVtxMemSize);

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
    BeginFrame();
}
void GfxRenderingAPIDeko3D::EndFrame() {
    auto& d = gDk;
    static uint32_t n = 0;
    if (n < 3 || n % 120 == 0) {
        DkLogF("frame #%u draws=%u tris=%u vtxbytes=%u nocolor=%u mismatch=%u texdraws=%u texs=%zu", n, gDraws, gTris,
               gVtxCursor, gSkippedNoColor, gLayoutMismatch, gTexturedDraws, gTexs.size());
    }
    n++;
    if (!gFrameOpen) {
        BeginFrame();
        if (!gFrameOpen) {
            return;
        }
    }
    d.queue.submitCommands(d.cmdBuf.finishList());
    d.queue.presentImage(d.swapchain, gDkSlot);
    d.queue.waitIdle(); // TODO: per-slot command memory instead of a full stall
    gGrave.clear();
    gFrameOpen = false;
}
void GfxRenderingAPIDeko3D::FinishRender() {}
int GfxRenderingAPIDeko3D::CreateFramebuffer() { static int next = 1; DkLogF("CreateFramebuffer -> %d", next); return next++; }
void GfxRenderingAPIDeko3D::UpdateFramebufferParameters(int fb, uint32_t w, uint32_t h, uint32_t msaa, bool, bool rt, bool depth, bool) {
    static int logged = 0;
    if (logged++ < 40) {
        DkLogF("fbparams id=%d %ux%u msaa=%u rt=%d depth=%d", fb, w, h, msaa, (int)rt, (int)depth);
    }
}
void GfxRenderingAPIDeko3D::StartDrawToFramebuffer(int fb, float) { gCurFb = fb; }
void GfxRenderingAPIDeko3D::CopyFramebuffer(int, int, int, int, int, int, int, int, int, int) {}
void GfxRenderingAPIDeko3D::ClearFramebuffer(bool c, bool dp) {
    if (!gFrameOpen || gCurFb != 0) {
        return;
    }
    auto& d = gDk;
    d.cmdBuf.setScissors(0, { { 0, 0, d.width, d.height } });
    if (c) {
        d.cmdBuf.clearColor(0, DkColorMask_RGBA, 0.2f, 0.0f, 0.4f, 1.0f);
    }
    if (dp) {
        d.cmdBuf.clearDepthStencil(true, 1.0f, 0xFF, 0);
    }
}
void GfxRenderingAPIDeko3D::ReadFramebufferToCPU(int, uint32_t, uint32_t, uint16_t*) {}
void GfxRenderingAPIDeko3D::ResolveMSAAColorBuffer(int, int) {}
std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
GfxRenderingAPIDeko3D::GetPixelDepth(int, const std::set<std::pair<float, float>>& coordinates) {
    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff> res;
    for (const auto& c : coordinates) {
        res[c] = 0xFFFF; // TODO: read the real depth buffer
    }
    return res;
}
void* GfxRenderingAPIDeko3D::GetFramebufferTextureId(int) { return nullptr; }
void GfxRenderingAPIDeko3D::SelectTextureFb(int) {}
void GfxRenderingAPIDeko3D::DeleteTexture(uint32_t id) {
    if (id == 0 || id >= gTexs.size()) {
        return;
    }
    if (gTexs[id].mem) {
        gGrave.push_back(std::move(gTexs[id].mem));
    }
    gTexs[id] = DkTex();
    gFreeTex.push_back(id);
    for (auto& b : gBoundTex) {
        if (b == id) {
            b = 0;
        }
    }
}
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
