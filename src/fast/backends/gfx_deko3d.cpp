#ifdef ENABLE_DEKO3D
#include "fast/backends/gfx_deko3d.h"
#include "gfx_deko3d_shaders.h"
#include "gfx_deko3d_shaders_tex.h"
#include "fast/interpreter.h"
#include <map>
#include <set>
#include <vector>
#include <cstring>
#include <array>
#include <switch.h>
#include <deko3d.hpp>
#include <cstdarg>
#include <typeinfo>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cfloat>
#include <cstddef>
#include <string>
#include <variant>
#include <optional>
#include <memory>
#include <sys/stat.h>
#include <prism/processor.h>
#include "ship/Context.h"
#include "imgui.h"
#include "ship/resource/factory/ShaderFactory.h"

namespace Fast {
namespace {
constexpr uint32_t kNumFb = 2;
constexpr uint32_t kFrames = 2; // frames in flight (command/vertex memory is per frame)
constexpr uint32_t kCmdMemSize = 0x200000;
constexpr uint32_t kVtxMemSize = 0x400000;
constexpr uint32_t kCodeMemSize = 0x10000;
constexpr uint32_t kMaxTex = 4096;
constexpr uint32_t kNumSamplers = 32;
constexpr uint32_t kShaderMemSize = 0x800000; // precompiled exact shaders loaded from sdmc:/dkshaders
// If the image is upside down, change this to -1.0f (or back to 1.0f).
constexpr float kFlipY = 1.0f;

void DkLogF(const char* fmt, ...) {
    static FILE* f = fopen("sdmc:/deko3d.log", "a");
    if (!f) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vfprintf(f, fmt, ap);
    va_end(ap);
    fputc('\n', f);
    fflush(f);
}
void DkDebugCb(void*, const char* ctx, DkResult res, const char* msg) {
    DkLogF("deko3d: %s result=%d %s", ctx ? ctx : "?", (int)res, msg ? msg : "");
}
struct DkState {
    dk::UniqueDevice device;
    dk::UniqueQueue queue;
    dk::UniqueMemBlock fbMem;
    dk::UniqueMemBlock depthMem;
    dk::UniqueMemBlock vtxMems[kFrames];
    dk::UniqueMemBlock codeMem;
    dk::UniqueMemBlock shaderMem;
    dk::UniqueMemBlock cmdMems[kFrames];
    dk::UniqueCmdBuf cmdBufs[kFrames];
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

int gFrameIdx = 0;          // frame slot being recorded (or the last one submitted, between frames)
uint32_t gFrameNo = 0;
dk::Fence gFences[kFrames]; // signalled when that slot's last submission has finished on the GPU
bool gFenceSet[kFrames] = {};
bool gSyncFrame = false;    // cfg word "syncframe": old behaviour, waitIdle after every frame
uint32_t gReplaceWaits = 0; // texture/fbo replacements that had to stall the queue
dk::UniqueCmdBuf& CmdBuf() { return gDk.cmdBufs[gFrameIdx]; }

// Lightweight timing, reported to the log every 120 frames ("perf:" lines).
struct DkScope {
    uint64_t& acc;
    uint64_t t0;
    explicit DkScope(uint64_t& a) : acc(a), t0(armGetSystemTick()) {}
    ~DkScope() { acc += armGetSystemTick() - t0; }
};
uint64_t gTWait = 0, gTAcq = 0, gTDraw = 0, gTEnd = 0, gTFrame = 0, gLastBegin = 0;
uint32_t gPerfFrames = 0;
dk::UniqueMemBlock& VtxMem() { return gDk.vtxMems[gFrameIdx]; }

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
std::vector<dk::UniqueMemBlock> gGraves[kFrames]; // freed once that frame slot's fence has signalled
std::vector<uint32_t> gPendingFree[kFrames];      // texture ids that may be reused once the fence has signalled

// ---------------------------------------------------------------------------
// Offscreen framebuffers (v6). Fb 0 is the swapchain image. Every other id gets its own
// color (+ optional depth) target and an entry in the image descriptor set, so it can be
// drawn to and then sampled as a texture (SelectTextureFb / CopyFramebuffer).
// ---------------------------------------------------------------------------
struct DkFbo {
    dk::UniqueMemBlock colorMem;
    dk::UniqueMemBlock depthMem;
    dk::Image color;
    dk::Image depth;
    uint32_t w = 0;
    uint32_t h = 0;
    bool hasDepth = false;
    bool inited = false;
    bool dirty = false;
    uint32_t texId = 0;
    Rect vp, sc; // viewport/scissor saved while another target is bound
};
std::vector<DkFbo> gFbos(1);
uint32_t gScreenTex[kNumFb] = {}; // descriptor ids for the two swapchain images
Rect gSavedVp0, gSavedSc0;        // fb 0's viewport/scissor while an offscreen target is bound
uint32_t gCurW = 1280, gCurH = 720;
bool gCurHasDepth = true;
bool gCurValid = true;
bool gNeedBarrier = false;
bool gNoClamp = false;            // cfg word "noclamp": back to near-plane clipping
bool gDecal = false;              // Fast3D zmode decal: lequal compare + depth bias
float gDecalBias = -2.0f;         // cfg "decal=<float>" overrides (negative = toward the camera)
bool gNoFb = false;               // cfg word "nofb": old behaviour (skip draws to fb != 0)
bool gNoImGui = false;            // cfg word "noimgui": never draw the ImGui overlay
float gClr[3] = { 0.0f, 0.0f, 0.0f }; // cfg word "purple" switches the clear color back to purple
uint32_t gFbDraws = 0, gFbSkipped = 0, gSelects = 0, gCopies = 0, gResolves = 0;
std::set<uint64_t> gSeen;
bool DkFirst(uint32_t tag, uint64_t key) { return gSeen.insert(((uint64_t)tag << 56) ^ key).second; }
uint32_t gBoundTex[6] = {};
uint32_t gCurTile = 0;
bool gBlend = false;
uint32_t gTexturedDraws = 0;
// Debug switches read from sdmc:/deko3d.cfg (e.g. "notex noblend nodesc").
bool gNoTex = false;
bool gNoBlend = false;
bool gNoDesc = false;
bool gQueueDead = false;
bool gNoExact = false;
uint32_t gFrameCount = 0;
float gNoiseScale = 1.0f;
uint32_t gExactDraws = 0;
uint32_t gShaderCursor = 0;
// Per-shader vertex layout, decoded from the shader ids by Fast3D's own feature parser.
struct DkShaderInfo {
    uint8_t numInputs = 0;
    bool usedTextures[6] = {};
    bool alpha = false;
    uint32_t numFloats = 4;
    uint32_t inputOff = 4;
    uint32_t nColor = 3;
    // exact (Fast3D combiner) shaders, loaded from precompiled .dksh files
    bool exact = false;
    dk::Shader exVsh;
    dk::Shader exFsh;
    uint32_t numAttrs = 0;
    bool attrOverflow = false;
    uint8_t attrOff[16] = {};
    uint8_t attrSize[16] = {};
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

// ---------------------------------------------------------------------------
// Exact combiner shaders. The GLSL comes from Fast3D's own template (prism) and is
// dumped to sdmc:/dkshaders. Compile it with uam into .dksh files in the same folder;
// the backend loads those by shader id. Shaders without a .dksh use the generic path.
// ---------------------------------------------------------------------------
#define DK_RAND_NOISE "((random(vec3(floor(gl_FragCoord.xy * noise_scale), float(frame_count))) + 1.0) / 2.0)"

const char* DkItemToStr(uint32_t item, bool with_alpha, bool only_alpha, bool inputs_have_alpha, bool first_cycle,
                        bool hint_single_element) {
    if (!only_alpha) {
        switch (item) {
            case SHADER_0:
                return with_alpha ? "vec4(0.0, 0.0, 0.0, 0.0)" : "vec3(0.0, 0.0, 0.0)";
            case SHADER_1:
                return with_alpha ? "vec4(1.0, 1.0, 1.0, 1.0)" : "vec3(1.0, 1.0, 1.0)";
            case SHADER_INPUT_1:
                return with_alpha || !inputs_have_alpha ? "vInput1" : "vInput1.rgb";
            case SHADER_INPUT_2:
                return with_alpha || !inputs_have_alpha ? "vInput2" : "vInput2.rgb";
            case SHADER_INPUT_3:
                return with_alpha || !inputs_have_alpha ? "vInput3" : "vInput3.rgb";
            case SHADER_INPUT_4:
                return with_alpha || !inputs_have_alpha ? "vInput4" : "vInput4.rgb";
            case SHADER_TEXEL0:
                return first_cycle ? (with_alpha ? "texVal0" : "texVal0.rgb") : (with_alpha ? "texVal1" : "texVal1.rgb");
            case SHADER_TEXEL0A:
                return first_cycle
                           ? (hint_single_element ? "texVal0.a"
                                                  : (with_alpha ? "vec4(texVal0.a, texVal0.a, texVal0.a, texVal0.a)"
                                                                : "vec3(texVal0.a, texVal0.a, texVal0.a)"))
                           : (hint_single_element ? "texVal1.a"
                                                  : (with_alpha ? "vec4(texVal1.a, texVal1.a, texVal1.a, texVal1.a)"
                                                                : "vec3(texVal1.a, texVal1.a, texVal1.a)"));
            case SHADER_TEXEL1A:
                return first_cycle
                           ? (hint_single_element ? "texVal1.a"
                                                  : (with_alpha ? "vec4(texVal1.a, texVal1.a, texVal1.a, texVal1.a)"
                                                                : "vec3(texVal1.a, texVal1.a, texVal1.a)"))
                           : (hint_single_element ? "texVal0.a"
                                                  : (with_alpha ? "vec4(texVal0.a, texVal0.a, texVal0.a, texVal0.a)"
                                                                : "vec3(texVal0.a, texVal0.a, texVal0.a)"));
            case SHADER_TEXEL1:
                return first_cycle ? (with_alpha ? "texVal1" : "texVal1.rgb") : (with_alpha ? "texVal0" : "texVal0.rgb");
            case SHADER_COMBINED:
                return with_alpha ? "texel" : "texel.rgb";
            case SHADER_NOISE:
                return with_alpha ? "vec4(" DK_RAND_NOISE ", " DK_RAND_NOISE ", " DK_RAND_NOISE ", " DK_RAND_NOISE ")"
                                  : "vec3(" DK_RAND_NOISE ", " DK_RAND_NOISE ", " DK_RAND_NOISE ")";
        }
    } else {
        switch (item) {
            case SHADER_0:
                return "0.0";
            case SHADER_1:
                return "1.0";
            case SHADER_INPUT_1:
                return "vInput1.a";
            case SHADER_INPUT_2:
                return "vInput2.a";
            case SHADER_INPUT_3:
                return "vInput3.a";
            case SHADER_INPUT_4:
                return "vInput4.a";
            case SHADER_TEXEL0:
                return first_cycle ? "texVal0.a" : "texVal1.a";
            case SHADER_TEXEL0A:
                return first_cycle ? "texVal0.a" : "texVal1.a";
            case SHADER_TEXEL1A:
                return first_cycle ? "texVal1.a" : "texVal0.a";
            case SHADER_TEXEL1:
                return first_cycle ? "texVal1.a" : "texVal0.a";
            case SHADER_COMBINED:
                return "texel.a";
            case SHADER_NOISE:
                return DK_RAND_NOISE;
        }
    }
    return "";
}

bool DkGetBool(prism::ContextTypes* value) {
    if (std::holds_alternative<int>(*value)) {
        return std::get<int>(*value) == 1;
    }
    return false;
}

prism::ContextTypes* DkAppendFormula(prism::ContextTypes* _, prism::ContextTypes* a_arg, prism::ContextTypes* a_single,
                                     prism::ContextTypes* a_mult, prism::ContextTypes* a_mix,
                                     prism::ContextTypes* a_with_alpha, prism::ContextTypes* a_only_alpha,
                                     prism::ContextTypes* a_alpha, prism::ContextTypes* a_first_cycle) {
    auto c = std::get<prism::MTDArray<int>>(*a_arg);
    bool do_single = DkGetBool(a_single);
    bool do_multiply = DkGetBool(a_mult);
    bool do_mix = DkGetBool(a_mix);
    bool with_alpha = DkGetBool(a_with_alpha);
    bool only_alpha = DkGetBool(a_only_alpha);
    bool opt_alpha = DkGetBool(a_alpha);
    bool first_cycle = DkGetBool(a_first_cycle);
    std::string out = "";
    if (do_single) {
        out += DkItemToStr(c.at(only_alpha, 3), with_alpha, only_alpha, opt_alpha, first_cycle, false);
    } else if (do_multiply) {
        out += DkItemToStr(c.at(only_alpha, 0), with_alpha, only_alpha, opt_alpha, first_cycle, false);
        out += " * ";
        out += DkItemToStr(c.at(only_alpha, 2), with_alpha, only_alpha, opt_alpha, first_cycle, true);
    } else if (do_mix) {
        out += "mix(";
        out += DkItemToStr(c.at(only_alpha, 1), with_alpha, only_alpha, opt_alpha, first_cycle, false);
        out += ", ";
        out += DkItemToStr(c.at(only_alpha, 0), with_alpha, only_alpha, opt_alpha, first_cycle, false);
        out += ", ";
        out += DkItemToStr(c.at(only_alpha, 2), with_alpha, only_alpha, opt_alpha, first_cycle, true);
        out += ")";
    } else {
        out += "(";
        out += DkItemToStr(c.at(only_alpha, 0), with_alpha, only_alpha, opt_alpha, first_cycle, false);
        out += " - ";
        out += DkItemToStr(c.at(only_alpha, 1), with_alpha, only_alpha, opt_alpha, first_cycle, false);
        out += ") * ";
        out += DkItemToStr(c.at(only_alpha, 2), with_alpha, only_alpha, opt_alpha, first_cycle, true);
        out += " + ";
        out += DkItemToStr(c.at(only_alpha, 3), with_alpha, only_alpha, opt_alpha, first_cycle, false);
    }
    return new prism::ContextTypes{ out };
}

size_t gUpdateFloatsTotal = 0;
prism::ContextTypes* DkUpdateFloats(prism::ContextTypes* _, prism::ContextTypes* num) {
    gUpdateFloatsTotal += std::get<int>(*num);
    return nullptr;
}

std::optional<std::string> DkIncludeFs(const std::string& path) {
    auto init = std::make_shared<Ship::ResourceInitData>();
    init->Type = (uint32_t)Ship::ResourceType::Shader;
    init->ByteOrder = Ship::Endianness::Native;
    init->Format = RESOURCE_FORMAT_BINARY;
    auto res = std::static_pointer_cast<Ship::Shader>(
        Ship::Context::GetRawInstance()->GetResourceManager()->LoadResource(path, true, init));
    if (res == nullptr) {
        return std::nullopt;
    }
    auto inc = static_cast<std::string*>(res->GetRawPointer());
    return *inc;
}

std::string DkRunPrism(const CCFeatures& cc, prism::ContextItems ctx) {
    prism::Processor processor;
    processor.populate(ctx);
    auto init = std::make_shared<Ship::ResourceInitData>();
    init->Type = (uint32_t)Ship::ResourceType::Shader;
    init->ByteOrder = Ship::Endianness::Native;
    init->Format = RESOURCE_FORMAT_BINARY;
    const char* shaderName = Fast::gfx_get_shader(cc.shader_id);
    std::string path = "shaders/opengl/default.shader.glsl";
    if (nullptr != shaderName) {
        path = std::string(shaderName) + ".glsl";
    }
    auto res = std::static_pointer_cast<Ship::Shader>(
        Ship::Context::GetRawInstance()->GetResourceManager()->LoadResource(path, true, init));
    if (res == nullptr) {
        return std::string();
    }
    auto shader = static_cast<std::string*>(res->GetRawPointer());
    processor.load(*shader);
    processor.bind_include_loader(DkIncludeFs);
    return processor.process();
}

std::string DkBuildVs(const CCFeatures& cc) {
    gUpdateFloatsTotal = 0;
    prism::ContextItems ctx = { { "VERTEX_SHADER", true },
                                { "o_textures", M_ARRAY(cc.usedTextures, bool, 2) },
                                { "o_clamp", M_ARRAY(cc.clamp, bool, 2, 2) },
                                { "o_fog", cc.opt_fog },
                                { "o_grayscale", cc.opt_grayscale },
                                { "o_alpha", cc.opt_alpha },
                                { "o_inputs", cc.numInputs },
                                { "update_floats", (InvokeFunc)DkUpdateFloats },
                                { "GLSL_VERSION", "#version 460" },
                                { "attr", "in" },
                                { "out", "out" },
                                { "opengles", false } };
    return DkRunPrism(cc, ctx);
}

std::string DkBuildFs(const CCFeatures& cc) {
    prism::ContextItems ctx = {
        { "VERTEX_SHADER", false },
        { "o_c", M_ARRAY(cc.c, int, 2, 2, 4) },
        { "o_alpha", cc.opt_alpha },
        { "o_fog", cc.opt_fog },
        { "o_texture_edge", cc.opt_texture_edge },
        { "o_noise", cc.opt_noise },
        { "o_2cyc", cc.opt_2cyc },
        { "o_alpha_threshold", cc.opt_alpha_threshold },
        { "o_invisible", cc.opt_invisible },
        { "o_grayscale", cc.opt_grayscale },
        { "o_prim_depth", cc.opt_prim_depth },
        { "o_textures", M_ARRAY(cc.usedTextures, bool, 2) },
        { "o_masks", M_ARRAY(cc.used_masks, bool, 2) },
        { "o_blend", M_ARRAY(cc.used_blend, bool, 2) },
        { "o_clamp", M_ARRAY(cc.clamp, bool, 2, 2) },
        { "o_inputs", cc.numInputs },
        { "o_do_mix", M_ARRAY(cc.do_mix, bool, 2, 2) },
        { "o_do_single", M_ARRAY(cc.do_single, bool, 2, 2) },
        { "o_do_multiply", M_ARRAY(cc.do_multiply, bool, 2, 2) },
        { "o_color_alpha_same", M_ARRAY(cc.color_alpha_same, bool, 2) },
        { "FILTER_THREE_POINT", FILTER_THREE_POINT },
        { "FILTER_LINEAR", FILTER_LINEAR },
        { "FILTER_NONE", FILTER_NONE },
        { "srgb_mode", false },
        { "SHADER_0", SHADER_0 },
        { "SHADER_INPUT_1", SHADER_INPUT_1 },
        { "SHADER_INPUT_2", SHADER_INPUT_2 },
        { "SHADER_INPUT_3", SHADER_INPUT_3 },
        { "SHADER_INPUT_4", SHADER_INPUT_4 },
        { "SHADER_INPUT_5", SHADER_INPUT_5 },
        { "SHADER_INPUT_6", SHADER_INPUT_6 },
        { "SHADER_INPUT_7", SHADER_INPUT_7 },
        { "SHADER_TEXEL0", SHADER_TEXEL0 },
        { "SHADER_TEXEL0A", SHADER_TEXEL0A },
        { "SHADER_TEXEL1", SHADER_TEXEL1 },
        { "SHADER_TEXEL1A", SHADER_TEXEL1A },
        { "SHADER_1", SHADER_1 },
        { "SHADER_COMBINED", SHADER_COMBINED },
        { "SHADER_NOISE", SHADER_NOISE },
        { "o_three_point_filtering", false },
        { "append_formula", (InvokeFunc)DkAppendFormula },
        { "GLSL_VERSION", "#version 460" },
        { "attr", "in" },
        { "opengles", false },
        { "core_opengl", true },
        { "texture", "texture" },
        { "vOutColor", "vOutColor" },
    };
    return DkRunPrism(cc, ctx);
}

// Rewrites prism's OpenGL-style GLSL into deko3d/uam-friendly GLSL: explicit locations,
// sampler bindings 0..5, and one std140 uniform block at binding 0 for the loose uniforms.
std::string DkPatchGlsl(const std::string& src, bool vertex) {
    std::string out;
    out.reserve(src.size() + 1024);
    uint32_t inLoc = 0, outLoc = 0;
    size_t pos = 0;
    auto startsWith = [](const std::string& str, const char* p) { return str.compare(0, strlen(p), p) == 0; };
    while (pos <= src.size()) {
        size_t nl = src.find('\n', pos);
        std::string line = src.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
        pos = (nl == std::string::npos) ? src.size() + 1 : nl + 1;
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
        }
        size_t ws = 0;
        while (ws < line.size() && (line[ws] == ' ' || line[ws] == '\t')) {
            ws++;
        }
        std::string t = line.substr(ws);
        if (t.empty()) {
            continue;
        }
        char buf[64];
        if (startsWith(t, "in ")) {
            snprintf(buf, sizeof(buf), "layout (location = %u) ", inLoc++);
            out += buf + t + "\n";
            continue;
        }
        if (startsWith(t, "out vec4 vOutColor")) {
            out += "layout (location = 0) " + t + "\n";
            continue;
        }
        if (startsWith(t, "out ")) {
            snprintf(buf, sizeof(buf), "layout (location = %u) ", outLoc++);
            out += buf + t + "\n";
            continue;
        }
        if (!vertex && startsWith(t, "uniform sampler2D ")) {
            int b = 0;
            if (t.find("uTexMask0") != std::string::npos) {
                b = 2;
            } else if (t.find("uTexMask1") != std::string::npos) {
                b = 3;
            } else if (t.find("uTexBlend0") != std::string::npos) {
                b = 4;
            } else if (t.find("uTexBlend1") != std::string::npos) {
                b = 5;
            } else if (t.find("uTex1") != std::string::npos) {
                b = 1;
            }
            snprintf(buf, sizeof(buf), "layout (binding = %d) ", b);
            out += buf + t + "\n";
            continue;
        }
        if (!vertex && startsWith(t, "uniform ") &&
            (t.find("frame_count") != std::string::npos || t.find("noise_scale") != std::string::npos ||
             t.find("prim_depth") != std::string::npos || t.find("texture_width") != std::string::npos ||
             t.find("texture_height") != std::string::npos || t.find("texture_filtering") != std::string::npos)) {
            continue; // moved into the uniform block
        }
        out += t + "\n";
    }
    if (!vertex) {
        size_t vp = out.find("#version");
        size_t v = (vp == std::string::npos) ? std::string::npos : out.find('\n', vp);
        std::string blk = "layout (std140, binding = 0) uniform DkParams {\n"
                          "    int frame_count;\n"
                          "    float noise_scale;\n"
                          "    float prim_depth;\n"
                          "    int dk_pad;\n"
                          "    int texture_width[2];\n"
                          "    int texture_height[2];\n"
                          "    int texture_filtering[2];\n"
                          "};\n";
        if (v != std::string::npos) {
            out.insert(v + 1, blk);
        }
    }
    return out;
}

// Matches the std140 block above (array elements are 16 bytes apart).
struct DkParams {
    int32_t frameCount;
    float noiseScale;
    float primDepth;
    int32_t pad;
    int32_t texW[2][4];
    int32_t texH[2][4];
    int32_t texF[2][4];
};
static_assert(sizeof(DkParams) == 112, "DkParams must match the std140 block");

bool DkFileExists(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        return false;
    }
    fclose(f);
    return true;
}

void DkWriteFile(const char* path, const std::string& text) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        return;
    }
    fwrite(text.data(), 1, text.size(), f);
    fclose(f);
}

bool DkLoadDksh(const char* path, dk::Shader& sh) {
    // sdmc:/dkshaders/<name> wins (so a newer compile can override), then the copy embedded in the NRO's romfs.
    FILE* f = fopen(path, "rb");
    if (!f) {
        static const char kSd[] = "sdmc:/dkshaders/";
        if (strncmp(path, kSd, sizeof(kSd) - 1) == 0) {
            char alt[192];
            snprintf(alt, sizeof(alt), "romfs:/dkshaders/%s", path + sizeof(kSd) - 1);
            f = fopen(alt, "rb");
        }
    }
    if (!f) {
        return false;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || gShaderCursor + (uint32_t)sz > kShaderMemSize) {
        fclose(f);
        return false;
    }
    uint8_t* code = (uint8_t*)gDk.shaderMem.getCpuAddr() + gShaderCursor;
    size_t n = fread(code, 1, (size_t)sz, f);
    fclose(f);
    if (n != (size_t)sz) {
        return false;
    }
    dk::ShaderMaker{ gDk.shaderMem, gShaderCursor }.initialize(sh);
    if (!dkShaderIsValid(&sh)) {
        return false;
    }
    gShaderCursor = Align256(gShaderCursor + (uint32_t)sz);
    return true;
}

// Loads the precompiled .dksh pair for this shader if present; otherwise dumps its GLSL once.
void DkPrepareExact(uint64_t id0, uint64_t id1, const CCFeatures& cc, DkShaderInfo& info) {
    if (gNoExact || info.attrOverflow || !gDk.shaderMem) {
        return;
    }
    static bool madeDir = false;
    if (!madeDir) {
        mkdir("sdmc:/dkshaders", 0777);
        madeDir = true;
    }
    char base[96];
    snprintf(base, sizeof(base), "sdmc:/dkshaders/%016llx_%016llx", (unsigned long long)id0, (unsigned long long)id1);
    char pv[128], pf[128], dv[128], df[128];
    snprintf(pv, sizeof(pv), "%s.vert.glsl", base);
    snprintf(pf, sizeof(pf), "%s.frag.glsl", base);
    snprintf(dv, sizeof(dv), "%s.vert.dksh", base);
    snprintf(df, sizeof(df), "%s.frag.dksh", base);
    uint32_t savedCursor = gShaderCursor;
    if (DkLoadDksh(dv, info.exVsh) && DkLoadDksh(df, info.exFsh)) {
        info.exact = true;
        return;
    }
    gShaderCursor = savedCursor;
    if (DkFileExists(pf)) {
        return; // already dumped, waiting for the .dksh
    }
    try {
        std::string vs = DkBuildVs(cc);
        size_t floatsFromTemplate = 4 + gUpdateFloatsTotal;
        std::string fs = DkBuildFs(cc);
        if (vs.empty() || fs.empty()) {
            DkLogF("glsl template missing for %s", base);
            return;
        }
        if (floatsFromTemplate != info.numFloats) {
            DkLogF("float count mismatch %s: template=%zu ours=%u", base, floatsFromTemplate, info.numFloats);
        }
        DkWriteFile(pv, DkPatchGlsl(vs, true));
        DkWriteFile(pf, DkPatchGlsl(fs, false));
    } catch (...) {
        DkLogF("glsl build threw for %s", base);
    }
}

DkVtxAttribSize DkVtxSize(uint32_t n) {
    switch (n) {
        case 1:
            return DkVtxAttribSize_1x32;
        case 2:
            return DkVtxAttribSize_2x32;
        case 3:
            return DkVtxAttribSize_3x32;
        default:
            return DkVtxAttribSize_4x32;
    }
}

void ApplyRect(const Rect& r, bool viewport) {
    auto& d = gDk;
    int w = r.w < 1 ? 1 : r.w;
    int h = r.h < 1 ? 1 : r.h;
    int x = r.x < 0 ? 0 : r.x;
    // Fast3D gives a bottom-left origin, deko3d wants top-left.
    int y = (int)gCurH - (r.y + h);
    if (y < 0) {
        y = 0;
    }
    if (viewport) {
        CmdBuf().setViewports(0, { { (float)x, (float)y, (float)w, (float)h, 0.0f, 1.0f } });
    } else {
        CmdBuf().setScissors(0, { { (uint32_t)x, (uint32_t)y, (uint32_t)w, (uint32_t)h } });
    }
}

uint32_t AllocTexId() {
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

bool DkFbValid(int fb) {
    if (fb == 0) {
        return true;
    }
    return fb > 0 && (size_t)fb < gFbos.size() && (bool)gFbos[fb].colorMem;
}

bool DkFbDims(int fb, uint32_t* w, uint32_t* h) {
    if (!DkFbValid(fb)) {
        return false;
    }
    if (fb == 0) {
        *w = gDk.width;
        *h = gDk.height;
    } else {
        *w = gFbos[fb].w;
        *h = gFbos[fb].h;
    }
    return true;
}

uint32_t DkFbTexId(int fb) {
    if (!DkFbValid(fb)) {
        return 0;
    }
    return fb == 0 ? gScreenTex[gDkSlot % kNumFb] : gFbos[fb].texId;
}

void DkMakeFbo(int id, uint32_t w, uint32_t h, bool depth) {
    auto& d = gDk;
    if (id <= 0 || w == 0 || h == 0 || w > 4096 || h > 4096) {
        return;
    }
    if ((size_t)id >= gFbos.size()) {
        gFbos.resize((size_t)id + 1);
    }
    DkFbo& f = gFbos[id];
    if (f.colorMem && f.w == w && f.h == h && f.hasDepth == depth) {
        return;
    }
    if (f.colorMem) {
        d.queue.waitIdle(); // fbo is being recreated: nothing in flight may still use the old one
        gGraves[gFrameIdx].push_back(std::move(f.colorMem));
    }
    if (f.depthMem) {
        gGraves[gFrameIdx].push_back(std::move(f.depthMem));
    }
    dk::ImageLayout cl;
    dk::ImageLayoutMaker{ d.device }
        .setFlags(DkImageFlags_UsageRender)
        .setFormat(DkImageFormat_RGBA8_Unorm)
        .setDimensions(w, h)
        .initialize(cl);
    uint32_t csz = cl.getSize();
    uint32_t cal = cl.getAlignment();
    if (cal < 0x1000) {
        cal = 0x1000;
    }
    csz = (csz + cal - 1) & ~(cal - 1);
    f.colorMem = dk::MemBlockMaker{ d.device, csz }.setFlags(DkMemBlockFlags_GpuCached | DkMemBlockFlags_Image).create();
    f.color.initialize(cl, f.colorMem, 0);
    if (depth) {
        dk::ImageLayout zl;
        dk::ImageLayoutMaker{ d.device }
            .setFlags(DkImageFlags_UsageRender)
            .setFormat(DkImageFormat_Z24S8)
            .setDimensions(w, h)
            .initialize(zl);
        uint32_t zsz = zl.getSize();
        uint32_t zal = zl.getAlignment();
        if (zal < 0x1000) {
            zal = 0x1000;
        }
        zsz = (zsz + zal - 1) & ~(zal - 1);
        f.depthMem =
            dk::MemBlockMaker{ d.device, zsz }.setFlags(DkMemBlockFlags_GpuCached | DkMemBlockFlags_Image).create();
        f.depth.initialize(zl, f.depthMem, 0);
    }
    f.w = w;
    f.h = h;
    f.hasDepth = depth;
    f.inited = false;
    f.dirty = false;
    f.vp = { 0, 0, (int)w, (int)h };
    f.sc = { 0, 0, (int)w, (int)h };
    if (!f.texId) {
        f.texId = AllocTexId();
    }
    if (f.texId) {
        DkTex& t = gTexs[f.texId];
        t.image.initialize(cl, f.colorMem, 0);
        t.w = w;
        t.h = h;
        t.linear = false;
        t.cms = 2;
        t.cmt = 2;
        dk::ImageView view{ t.image };
        DkImageDescriptor* descs = (DkImageDescriptor*)d.imgDescMem.getCpuAddr();
        dkImageDescriptorInitialize(&descs[f.texId], &view, false, false);
    }
    DkLogF("fbo %d created %ux%u depth=%d tex=%u", id, w, h, (int)depth, f.texId);
}

// Binds fb as the render target (0 = swapchain). Viewport/scissor are saved and restored per target.
void DkSetTarget(int fb) {
    auto& d = gDk;
    if (gCurFb == 0) {
        gSavedVp0 = gVp;
        gSavedSc0 = gSc;
    } else if (gCurFb > 0 && (size_t)gCurFb < gFbos.size()) {
        gFbos[gCurFb].vp = gVp;
        gFbos[gCurFb].sc = gSc;
    }
    gCurFb = fb;
    if (fb == 0) {
        dk::ImageView c{ d.images[gDkSlot] };
        dk::ImageView z{ d.depthImage };
        CmdBuf().bindRenderTargets(&c, &z);
        gCurW = d.width;
        gCurH = d.height;
        gCurHasDepth = true;
        gCurValid = true;
        gVp = gSavedVp0;
        gSc = gSavedSc0;
        return;
    }
    if (!DkFbValid(fb)) {
        gCurValid = false;
        return;
    }
    DkFbo& f = gFbos[fb];
    dk::ImageView c{ f.color };
    dk::ImageView z{ f.depth };
    const DkImageView* zp = f.hasDepth ? &z : nullptr;
    CmdBuf().bindRenderTargets(&c, zp);
    gCurW = f.w;
    gCurH = f.h;
    gCurHasDepth = f.hasDepth;
    gCurValid = true;
    gVp = f.vp;
    gSc = f.sc;
    if (!f.inited) {
        f.inited = true;
        CmdBuf().setScissors(0, { { 0, 0, f.w, f.h } });
        CmdBuf().clearColor(0, DkColorMask_RGBA, 0.0f, 0.0f, 0.0f, 0.0f);
        if (f.hasDepth) {
            CmdBuf().clearDepthStencil(true, 1.0f, 0xFF, 0);
        }
    }
}

// Copies a rectangle of one framebuffer into another with a textured quad (nearest filtering).
// Coordinates are top-left based, like Fast3D's DX11/Metal paths.
void DkDrawFbQuad(int dst, int src, int sx0, int sy0, int sx1, int sy1, int dx0, int dy0, int dx1, int dy1) {
    auto& d = gDk;
    if (!gFrameOpen || dst == src) {
        return;
    }
    uint32_t sw = 0, sh = 0, dw = 0, dh = 0;
    uint32_t stid = DkFbTexId(src);
    if (!stid || !DkFbDims(src, &sw, &sh) || !DkFbDims(dst, &dw, &dh)) {
        return;
    }
    if (sx1 == sx0 || sy1 == sy0 || dx1 <= dx0 || dy1 <= dy0) {
        return;
    }
    int scx = dx0 < 0 ? 0 : dx0;
    int scy = dy0 < 0 ? 0 : dy0;
    int scx1 = dx1 > (int)dw ? (int)dw : dx1;
    int scy1 = dy1 > (int)dh ? (int)dh : dy1;
    if (scx1 <= scx || scy1 <= scy) {
        return;
    }
    int prev = gCurFb;
    bool switched = (dst != gCurFb);
    if (switched) {
        DkSetTarget(dst);
    }
    if (!gCurValid) {
        if (switched) {
            DkSetTarget(prev);
        }
        return;
    }
    if (gNeedBarrier) {
        CmdBuf().barrier(DkBarrier_Fragments, DkInvalidateFlags_Image);
        gNeedBarrier = false;
    }
    uint32_t bytes = 6 * 10 * 4;
    gVtxCursor = Align256(gVtxCursor);
    if (gVtxCursor + bytes > kVtxMemSize) {
        if (switched) {
            DkSetTarget(prev);
        }
        return;
    }
    float* v = (float*)((uint8_t*)VtxMem().getCpuAddr() + gVtxCursor);
    DkGpuAddr addr = VtxMem().getGpuAddr() + gVtxCursor;
    gVtxCursor += bytes;
    float u0 = (float)sx0 / (float)sw, u1 = (float)sx1 / (float)sw;
    float v0 = (float)sy0 / (float)sh, v1 = (float)sy1 / (float)sh;
    const float px[6] = { -1, 1, 1, -1, 1, -1 };
    const float py[6] = { -1, -1, 1, -1, 1, 1 };
    for (int i = 0; i < 6; i++) {
        float* o = v + i * 10;
        o[0] = px[i];
        o[1] = py[i];
        o[2] = 0.0f;
        o[3] = 1.0f;
        o[4] = px[i] < 0 ? u0 : u1;
        o[5] = py[i] > 0 ? v0 : v1; // +Y is the top of the viewport
        o[6] = o[7] = o[8] = o[9] = 1.0f;
    }
    CmdBuf().bindShaders(DkStageFlag_GraphicsMask, { &d.tvsh, &d.tfsh });
    dk::RasterizerState rs;
    rs.setCullMode(DkFace_None);
    CmdBuf().bindRasterizerState(rs);
    dk::ColorState cs;
    cs.setBlendEnable(0, false);
    CmdBuf().bindColorState(cs);
    dk::ColorWriteState cws;
    CmdBuf().bindColorWriteState(cws);
    dk::BlendState bs;
    CmdBuf().bindBlendStates(0, bs);
    dk::DepthStencilState ds;
    ds.setDepthTestEnable(false).setDepthWriteEnable(false).setDepthCompareOp(DkCompareOp_Always);
    CmdBuf().bindDepthStencilState(ds);
    std::array<DkVtxAttribState, 3> at{};
    at[0].offset = 0;
    at[0].size = DkVtxAttribSize_4x32;
    at[0].type = DkVtxAttribType_Float;
    at[1].offset = 16;
    at[1].size = DkVtxAttribSize_2x32;
    at[1].type = DkVtxAttribType_Float;
    at[2].offset = 24;
    at[2].size = DkVtxAttribSize_4x32;
    at[2].type = DkVtxAttribType_Float;
    CmdBuf().bindVtxAttribState(at);
    std::array<DkVtxBufferState, 1> bst{};
    bst[0].stride = 40;
    bst[0].divisor = 0;
    CmdBuf().bindVtxBufferState(bst);
    CmdBuf().bindVtxBuffer(0, addr, bytes);
    std::array<DkResHandle, 1> h = { { dkMakeTextureHandle(stid, 10) } }; // nearest, clamp-to-edge
    CmdBuf().bindTextures(DkStage_Fragment, 0, h);
    CmdBuf().setViewports(0, { { (float)dx0, (float)dy0, (float)(dx1 - dx0), (float)(dy1 - dy0), 0.0f, 1.0f } });
    CmdBuf().setScissors(0, { { (uint32_t)scx, (uint32_t)scy, (uint32_t)(scx1 - scx), (uint32_t)(scy1 - scy) } });
    CmdBuf().draw(DkPrimitive_Triangles, 6, 1, 0, 0);
    gNeedBarrier = true;
    if (dst != 0) {
        gFbos[dst].dirty = true;
    }
    if (switched) {
        DkSetTarget(prev);
    }
}

void BeginFrame() {
    auto& d = gDk;
    {
        uint64_t now = armGetSystemTick();
        if (gLastBegin) {
            gTFrame += now - gLastBegin;
            gPerfFrames++;
        }
        gLastBegin = now;
        if (gPerfFrames >= 120) {
            double toMs = 1000.0 / (double)armGetSystemTickFreq() / (double)gPerfFrames;
            double frame = (double)gTFrame * toMs, wait = (double)gTWait * toMs, acq = (double)gTAcq * toMs;
            double draw = (double)gTDraw * toMs, end = (double)gTEnd * toMs;
            DkLogF("perf: %.1f fps frame=%.2fms | fenceWait=%.2f acquire=%.2f backendDraw=%.2f end=%.2f other(game+Fast3D)=%.2f | draws=%u tris=%u",
                   1000.0 / frame, frame, wait, acq, draw, end, frame - wait - acq - draw - end, gDraws, gTris);
            gTFrame = gTWait = gTAcq = gTDraw = gTEnd = 0;
            gPerfFrames = 0;
        }
    }
    // Pick this frame's slot. Wait only for the frame that used the same slot last time, then
    // release everything that was retired while it was in flight.
    gFrameIdx = (int)(gFrameNo++ % kFrames);
    if (gFenceSet[gFrameIdx]) {
        DkScope s(gTWait);
        gFences[gFrameIdx].wait();
        gFenceSet[gFrameIdx] = false;
    }
    gGraves[gFrameIdx].clear();
    for (uint32_t id : gPendingFree[gFrameIdx]) {
        gFreeTex.push_back(id);
    }
    gPendingFree[gFrameIdx].clear();
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
    gExactDraws = 0;
    gVtxCursor = 0;
    if (gCurFb != 0) {
        gVp = gSavedVp0;
        gSc = gSavedSc0;
    }
    gCurFb = 0;
    gCurW = d.width;
    gCurH = d.height;
    gCurHasDepth = true;
    gCurValid = true;
    gNeedBarrier = false;
    gFbDraws = 0;
    gFbSkipped = 0;
    gSelects = 0;
    gCopies = 0;
    gResolves = 0;
    int slot;
    {
        DkScope s(gTAcq);
        slot = d.queue.acquireImage(d.swapchain);
    }
    CmdBuf().clear();
    dk::ImageView target{ d.images[slot] };
    dk::ImageView depth{ d.depthImage };
    CmdBuf().bindRenderTargets(&target, &depth);
    if (!gNoDesc) {
        CmdBuf().bindImageDescriptorSet(d.imgDescMem.getGpuAddr(), kMaxTex + 1);
        CmdBuf().bindSamplerDescriptorSet(d.smpDescMem.getGpuAddr(), kNumSamplers);
    }
    CmdBuf().setViewports(0, { { 0.0f, 0.0f, (float)d.width, (float)d.height, 0.0f, 1.0f } });
    CmdBuf().setScissors(0, { { 0, 0, d.width, d.height } });
    dkCmdBufSetDepthBias(CmdBuf(), gDecalBias, 0.0f, gDecalBias); // C call: the C++ wrapper is misspelled seDepthBias
    CmdBuf().clearColor(0, DkColorMask_RGBA, gClr[0], gClr[1], gClr[2], 1.0f);
    CmdBuf().clearDepthStencil(true, 1.0f, 0xFF, 0);
    gFrameOpen = true;
    gDkSlot = slot;
}
// Creates (or replaces) the image behind texture id `id` from tightly packed RGBA8 pixels.
void DkUploadRgba(uint32_t id, const uint8_t* rgba32, uint32_t w, uint32_t h) {
    auto& d = gDk;
    if (id == 0 || id >= gTexs.size() || w == 0 || h == 0) {
        return;
    }
    DkTex& t = gTexs[id];
    if (t.mem) {
        if (gReplaceWaits++ == 0 || gReplaceWaits == 100 || gReplaceWaits == 1000) {
            DkLogF("texture %u replaced while in use: queue stall #%u", id, gReplaceWaits);
        }
        d.queue.waitIdle(); // the descriptor is rewritten below; an in-flight frame may still sample the old image
        gGraves[gFrameIdx].push_back(std::move(t.mem));
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

// ---------------------------------------------------------------------------
// ImGui overlay (v9). Fast3dGui hands us ImDrawData; we copy it to CPU vectors and replay it
// into fb 0 at EndFrame, right before submit. Shaders are the usual .dksh round trip:
// the first run dumps imgui.vert.glsl / imgui.frag.glsl into sdmc:/dkshaders/.
// ---------------------------------------------------------------------------
struct DkImCmd {
    float x0, y0, x1, y1; // scissor in framebuffer pixels, top-left origin
    uint32_t tex;
    uint32_t count;
    uint32_t idxOff;
    int32_t vtxOff;
};
static_assert(sizeof(ImDrawVert) == 20, "ImDrawVert layout changed");
constexpr uint32_t kImMemSize = 0x200000;
constexpr uint32_t kImSampler = 26; // linear filter, clamp-to-edge on both axes (16 + 2*4 + 2)
std::vector<ImDrawVert> gImVerts;
std::vector<ImDrawIdx> gImIdx;
std::vector<DkImCmd> gImCmds;
float gImPos[2] = { 0.0f, 0.0f };
float gImSize[2] = { 1280.0f, 720.0f };
bool gImPending = false;
dk::UniqueMemBlock gImMems[kFrames];
dk::Shader gImVsh, gImFsh;
bool gImShadersOk = false;
uint32_t gImFontTex = 0;
uint32_t gImFrames = 0;
PadState gImPad;

const char* kImVertGlsl = "#version 460\n"
                          "layout (location = 0) in vec2 aPos;\n"
                          "layout (location = 1) in vec2 aUV;\n"
                          "layout (location = 2) in vec4 aColor;\n"
                          "layout (std140, binding = 0) uniform ImParams {\n"
                          "    vec2 uScale;\n"
                          "    vec2 uTranslate;\n"
                          "};\n"
                          "layout (location = 0) out vec2 vUV;\n"
                          "layout (location = 1) out vec4 vColor;\n"
                          "void main() {\n"
                          "    vUV = aUV;\n"
                          "    vColor = aColor;\n"
                          "    gl_Position = vec4(aPos * uScale + uTranslate, 0.0, 1.0);\n"
                          "}\n";
const char* kImFragGlsl = "#version 460\n"
                          "layout (location = 0) in vec2 vUV;\n"
                          "layout (location = 1) in vec4 vColor;\n"
                          "layout (binding = 0) uniform sampler2D uTex;\n"
                          "layout (location = 0) out vec4 vOutColor;\n"
                          "void main() {\n"
                          "    vOutColor = vColor * texture(uTex, vUV);\n"
                          "}\n";

void DkImGuiInit() {
    auto& d = gDk;
    for (uint32_t i = 0; i < kFrames; i++) {
        gImMems[i] = dk::MemBlockMaker{ d.device, kImMemSize }
                         .setFlags(DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached)
                         .create();
    }
    mkdir("sdmc:/dkshaders", 0777);
    uint32_t savedCursor = gShaderCursor;
    if (d.shaderMem && DkLoadDksh("sdmc:/dkshaders/imgui.vert.dksh", gImVsh) &&
        DkLoadDksh("sdmc:/dkshaders/imgui.frag.dksh", gImFsh)) {
        gImShadersOk = true;
        DkLogF("imgui shaders loaded");
        return;
    }
    gShaderCursor = savedCursor;
    if (!DkFileExists("sdmc:/dkshaders/imgui.vert.glsl")) {
        DkWriteFile("sdmc:/dkshaders/imgui.vert.glsl", kImVertGlsl);
    }
    if (!DkFileExists("sdmc:/dkshaders/imgui.frag.glsl")) {
        DkWriteFile("sdmc:/dkshaders/imgui.frag.glsl", kImFragGlsl);
    }
    DkLogF("imgui shaders missing: compile sdmc:/dkshaders/imgui.{vert,frag}.glsl with uam and copy the .dksh files back");
}

// Draws the data copied by GfxDeko3dImGuiRender into fb 0. Called from EndFrame.
void DkImGuiReplay() {
    if (!gImPending) {
        return;
    }
    gImPending = false;
    auto& d = gDk;
    bool ok = !gNoImGui && gImShadersOk && gFrameOpen && gImMems[gFrameIdx] && !gImCmds.empty();
    if (ok) {
        size_t vbytes = gImVerts.size() * sizeof(ImDrawVert);
        size_t ibytes = gImIdx.size() * sizeof(ImDrawIdx);
        uint32_t iOff = Align256((uint32_t)vbytes);
        uint32_t uOff = Align256(iOff + (uint32_t)ibytes);
        if (uOff + 256 > kImMemSize) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                DkLogF("imgui: draw data too large (%zu verts, %zu idx), skipped", gImVerts.size(), gImIdx.size());
            }
            ok = false;
        } else {
            uint8_t* base = (uint8_t*)gImMems[gFrameIdx].getCpuAddr();
            memcpy(base, gImVerts.data(), vbytes);
            memcpy(base + iOff, gImIdx.data(), ibytes);
            float* u = (float*)(base + uOff);
            memset(u, 0, 256);
            u[0] = 2.0f / gImSize[0];
            u[1] = -2.0f / gImSize[1];
            u[2] = -1.0f - gImPos[0] * u[0];
            u[3] = 1.0f - gImPos[1] * u[1];
            DkGpuAddr gpu = gImMems[gFrameIdx].getGpuAddr();

            if (gCurFb != 0) {
                DkSetTarget(0);
            }
            CmdBuf().bindShaders(DkStageFlag_GraphicsMask, { &gImVsh, &gImFsh });
            CmdBuf().bindUniformBuffer(DkStage_Vertex, 0, gpu + uOff, 256);
            dk::RasterizerState rs;
            rs.setCullMode(DkFace_None);
            CmdBuf().bindRasterizerState(rs);
            dk::ColorState cs;
            cs.setBlendEnable(0, true);
            CmdBuf().bindColorState(cs);
            dk::ColorWriteState cws;
            CmdBuf().bindColorWriteState(cws);
            dk::BlendState bs;
            bs.setColorBlendOp(DkBlendOp_Add)
                .setSrcColorBlendFactor(DkBlendFactor_SrcAlpha)
                .setDstColorBlendFactor(DkBlendFactor_InvSrcAlpha)
                .setAlphaBlendOp(DkBlendOp_Add)
                .setSrcAlphaBlendFactor(DkBlendFactor_One)
                .setDstAlphaBlendFactor(DkBlendFactor_InvSrcAlpha);
            CmdBuf().bindBlendStates(0, bs);
            dk::DepthStencilState ds;
            ds.setDepthTestEnable(false).setDepthWriteEnable(false).setDepthCompareOp(DkCompareOp_Always);
            CmdBuf().bindDepthStencilState(ds);
            std::array<DkVtxAttribState, 3> at{};
            at[0].offset = offsetof(ImDrawVert, pos);
            at[0].size = DkVtxAttribSize_2x32;
            at[0].type = DkVtxAttribType_Float;
            at[1].offset = offsetof(ImDrawVert, uv);
            at[1].size = DkVtxAttribSize_2x32;
            at[1].type = DkVtxAttribType_Float;
            at[2].offset = offsetof(ImDrawVert, col);
            at[2].size = DkVtxAttribSize_4x8;
            at[2].type = DkVtxAttribType_Unorm;
            CmdBuf().bindVtxAttribState(at);
            std::array<DkVtxBufferState, 1> bst{};
            bst[0].stride = sizeof(ImDrawVert);
            bst[0].divisor = 0;
            CmdBuf().bindVtxBufferState(bst);
            CmdBuf().bindVtxBuffer(0, gpu, (uint32_t)vbytes);
            CmdBuf().bindIdxBuffer(sizeof(ImDrawIdx) == 2 ? DkIdxFormat_Uint16 : DkIdxFormat_Uint32, gpu + iOff);
            CmdBuf().setViewports(0, { { 0.0f, 0.0f, (float)d.width, (float)d.height, 0.0f, 1.0f } });
            CmdBuf().barrier(DkBarrier_Fragments, DkInvalidateFlags_Image);
            gNeedBarrier = false;

            uint32_t lastTex = 0xFFFFFFFFu;
            uint32_t drawn = 0;
            for (const DkImCmd& c : gImCmds) {
                if (c.tex == 0 || c.tex >= gTexs.size() || gTexs[c.tex].w == 0) {
                    continue;
                }
                int x0 = (int)(c.x0 < 0.0f ? 0.0f : c.x0);
                int y0 = (int)(c.y0 < 0.0f ? 0.0f : c.y0);
                int x1 = (int)(c.x1 > (float)d.width ? (float)d.width : c.x1);
                int y1 = (int)(c.y1 > (float)d.height ? (float)d.height : c.y1);
                if (x1 <= x0 || y1 <= y0) {
                    continue;
                }
                if (c.tex != lastTex) {
                    std::array<DkResHandle, 1> h = { { dkMakeTextureHandle(c.tex, kImSampler) } };
                    CmdBuf().bindTextures(DkStage_Fragment, 0, h);
                    lastTex = c.tex;
                }
                CmdBuf().setScissors(0, { { (uint32_t)x0, (uint32_t)y0, (uint32_t)(x1 - x0), (uint32_t)(y1 - y0) } });
                CmdBuf().drawIndexed(DkPrimitive_Triangles, c.count, 1, c.idxOff, c.vtxOff, 0);
                drawn++;
            }
            gNeedBarrier = true;
            if (gImFrames < 3 || gImFrames % 600 == 0) {
                DkLogF("imgui replay #%u: %zu verts %zu idx %zu cmds drawn=%u", gImFrames, gImVerts.size(), gImIdx.size(),
                       gImCmds.size(), drawn);
            }
            gImFrames++;
        }
    }
    gImVerts.clear();
    gImIdx.clear();
    gImCmds.clear();
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
    {
        uint32_t na = 0;
        uint32_t off = 0;
        auto addAttr = [&](uint32_t sz) {
            if (na < 16) {
                info.attrOff[na] = (uint8_t)off;
                info.attrSize[na] = (uint8_t)sz;
                na++;
            } else {
                info.attrOverflow = true;
            }
            off += sz;
        };
        addAttr(4);
        for (int i = 0; i < 2; i++) {
            if (cc.usedTextures[i]) {
                addAttr(2);
                for (int j = 0; j < 2; j++) {
                    if (cc.clamp[i][j]) {
                        addAttr(1);
                    }
                }
            }
        }
        if (cc.opt_fog) {
            addAttr(4);
        }
        if (cc.opt_grayscale) {
            addAttr(4);
        }
        for (int k = 0; k < cc.numInputs; k++) {
            addAttr(info.nColor);
        }
        info.numAttrs = na;
        if (off != info.numFloats) {
            info.attrOverflow = true;
        }
    }
    info.exact = false;
    DkPrepareExact(id0, id1, cc, info);
    static uint32_t n = 0;
    if (n++ < 200) {
        DkLogF("shader %llx %llx: floats=%u inputs=%u alpha=%d tex=%d%d mask=%d%d blend=%d%d fog=%d gray=%d exact=%d", (unsigned long long)id0,
               (unsigned long long)id1, info.numFloats, (unsigned)info.numInputs, (int)info.alpha,
               (int)info.usedTextures[0], (int)info.usedTextures[1], (int)info.usedTextures[2],
               (int)info.usedTextures[3], (int)info.usedTextures[4], (int)info.usedTextures[5], (int)cc.opt_fog,
               (int)cc.opt_grayscale, (int)info.exact);
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
    DkUploadRgba(gBoundTex[gCurTile], rgba32, w, h);
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
void GfxRenderingAPIDeko3D::SetZmodeDecal(bool decal) { gDecal = decal; }
void GfxRenderingAPIDeko3D::SetViewport(int x, int y, int w, int h) { gVp = { x, y, w, h }; }
void GfxRenderingAPIDeko3D::SetScissor(int x, int y, int w, int h) { gSc = { x, y, w, h }; }
void GfxRenderingAPIDeko3D::SetUseAlpha(bool a) { gBlend = a; }
void GfxRenderingAPIDeko3D::DrawTriangles(float* buf, size_t len, size_t tris) {
    DkScope perfScope(gTDraw);
    gDraws++;
    gTris += (uint32_t)tris;
    auto& d = gDk;
    if (!gFrameOpen || tris == 0) {
        return;
    }
    if (gCurFb != 0 && (gNoFb || !gCurValid)) {
        gFbSkipped++;
        if (DkFirst(4, (uint64_t)gCurFb)) {
            DkLogF("skipped draw to fb %d (valid=%d nofb=%d) frame=%u", gCurFb, (int)gCurValid, (int)gNoFb, gFrameCount);
        }
        return;
    }
    gNeedBarrier = true;
    if (gCurFb != 0) {
        gFbDraws++;
        if ((size_t)gCurFb < gFbos.size()) {
            gFbos[gCurFb].dirty = true;
        }
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
    DkShaderInfo* sh = gCurShader;
    bool exact = false;
    if (sh && sh->exact && sh->numFloats == per) {
        exact = true;
        for (int sIdx = 0; sIdx < 6; sIdx++) {
            if (sh->usedTextures[sIdx]) {
                uint32_t tid = gBoundTex[sIdx];
                if (tid == 0 || tid >= gTexs.size() || gTexs[tid].w == 0) {
                    exact = false;
                }
            }
        }
    }
    uint32_t nColor = 3;
    uint32_t colOff = 0;
    if (!exact) {
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
    }
    uint32_t bytes = nverts * per * 4;
    gVtxCursor = Align256(gVtxCursor);
    if (gVtxCursor + bytes > kVtxMemSize) {
        return;
    }
    float* dst = (float*)((uint8_t*)VtxMem().getCpuAddr() + gVtxCursor);
    memcpy(dst, buf, bytes);
    if (kFlipY < 0.0f) {
        for (uint32_t v = 0; v < nverts; v++) {
            dst[v * per + 1] = -dst[v * per + 1];
        }
    }
    DkGpuAddr addr = VtxMem().getGpuAddr() + gVtxCursor;
    gVtxCursor += bytes;

    if (exact) {
        uint32_t uboOff = Align256(gVtxCursor);
        if (uboOff + 256 > kVtxMemSize) {
            return;
        }
        gVtxCursor = uboOff + 256;
        DkParams* prm = (DkParams*)((uint8_t*)VtxMem().getCpuAddr() + uboOff);
        memset(prm, 0, 256);
        prm->frameCount = (int32_t)gFrameCount;
        prm->noiseScale = gNoiseScale;
        prm->primDepth = mCurrentPrimDepth;
        for (int t = 0; t < 2; t++) {
            uint32_t tid = gBoundTex[t];
            if (sh->usedTextures[t] && tid != 0 && tid < gTexs.size()) {
                prm->texW[t][0] = (int32_t)gTexs[tid].w;
                prm->texH[t][0] = (int32_t)gTexs[tid].h;
            }
        }
        CmdBuf().bindShaders(DkStageFlag_GraphicsMask, { &sh->exVsh, &sh->exFsh });
        CmdBuf().bindUniformBuffer(DkStage_Fragment, 0, VtxMem().getGpuAddr() + uboOff, 256);
        bool anyTex = false;
        for (uint32_t t = 0; t < 6; t++) {
            if (!sh->usedTextures[t]) {
                continue;
            }
            anyTex = true;
            uint32_t tid = gBoundTex[t];
            const DkTex& tx = gTexs[tid];
            uint32_t smpIdx = (tx.linear ? 16u : 0u) + (tx.cms & 3u) * 4u + (tx.cmt & 3u);
            std::array<DkResHandle, 1> h = { { dkMakeTextureHandle(tid, smpIdx) } };
            CmdBuf().bindTextures(DkStage_Fragment, t, h);
        }
        if (anyTex) {
            gTexturedDraws++;
        }
        gExactDraws++;
        dk::RasterizerState rs2;
        rs2.setCullMode(DkFace_None);
        rs2.setDepthBiasEnable(gDecal);
        rs2.setDepthClampEnable(!gNoClamp);
        CmdBuf().bindRasterizerState(rs2);
        dk::ColorState cs2;
        cs2.setBlendEnable(0, gBlend && !gNoBlend);
        CmdBuf().bindColorState(cs2);
        dk::ColorWriteState cws2;
        CmdBuf().bindColorWriteState(cws2);
        if (!gNoBlend) {
            dk::BlendState bs2;
            CmdBuf().bindBlendStates(0, bs2);
        }
        dk::DepthStencilState ds2;
        ds2.setDepthTestEnable(gDepthTest && gCurHasDepth).setDepthWriteEnable(gZUpd && gCurHasDepth).setDepthCompareOp(gDecal ? DkCompareOp_Lequal : DkCompareOp_Less);
        CmdBuf().bindDepthStencilState(ds2);
        // Every slot gets a valid format: unused slots copy the last real attribute.
        std::array<DkVtxAttribState, 16> at{};
        for (uint32_t k = 0; k < 16; k++) {
            uint32_t src = k < sh->numAttrs ? k : sh->numAttrs - 1;
            at[k].offset = (uint32_t)sh->attrOff[src] * 4u;
            at[k].size = DkVtxSize(sh->attrSize[src]);
            at[k].type = DkVtxAttribType_Float;
        }
        CmdBuf().bindVtxAttribState(at);
        std::array<DkVtxBufferState, 1> bst{};
        bst[0].stride = per * 4;
        bst[0].divisor = 0;
        CmdBuf().bindVtxBufferState(bst);
        CmdBuf().bindVtxBuffer(0, addr, bytes);
        ApplyRect(gVp, true);
        ApplyRect(gSc, false);
        CmdBuf().draw(DkPrimitive_Triangles, nverts, 1, 0, 0);
        return;
    }

    uint32_t texId = gBoundTex[0];
    bool textured = !gNoTex && gCurShader && gCurShader->numFloats == per && gCurShader->usedTextures[0] && texId != 0 &&
                    texId < gTexs.size() && gTexs[texId].w != 0;
    if (textured) {
        CmdBuf().bindShaders(DkStageFlag_GraphicsMask, { &d.tvsh, &d.tfsh });
        gTexturedDraws++;
    } else {
        CmdBuf().bindShaders(DkStageFlag_GraphicsMask, { &d.vsh, &d.fsh });
    }
    dk::RasterizerState rs;
    rs.setCullMode(DkFace_None);
    rs.setDepthBiasEnable(gDecal);
    rs.setDepthClampEnable(!gNoClamp);
    CmdBuf().bindRasterizerState(rs);
    dk::ColorState cs;
    cs.setBlendEnable(0, gBlend && !gNoBlend);
    CmdBuf().bindColorState(cs);
    dk::ColorWriteState cws;
    CmdBuf().bindColorWriteState(cws);
    if (!gNoBlend) {
        dk::BlendState bs;
        CmdBuf().bindBlendStates(0, bs);
    }
    dk::DepthStencilState ds;
    ds.setDepthTestEnable(gDepthTest && gCurHasDepth).setDepthWriteEnable(gZUpd && gCurHasDepth).setDepthCompareOp(gDecal ? DkCompareOp_Lequal : DkCompareOp_Less);
    CmdBuf().bindDepthStencilState(ds);
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
        CmdBuf().bindTextures(DkStage_Fragment, 0, handles);
    } else {
        attribs[1].offset = colOff * 4;
        attribs[1].size = (nColor == 3) ? DkVtxAttribSize_3x32 : DkVtxAttribSize_4x32;
        attribs[1].type = DkVtxAttribType_Float;
    }
    if (numAttribs < 3) attribs[2] = attribs[1]; // zeroed attrib = invalid format, so give the unused slot a valid copy
    CmdBuf().bindVtxAttribState(attribs);
    std::array<DkVtxBufferState, 1> bufState{};
    bufState[0].stride = per * 4;
    bufState[0].divisor = 0;
    CmdBuf().bindVtxBufferState(bufState);
    CmdBuf().bindVtxBuffer(0, addr, bytes);
    ApplyRect(gVp, true);
    ApplyRect(gSc, false);
    CmdBuf().draw(DkPrimitive_Triangles, nverts, 1, 0, 0);
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
        gNoExact = gNoExact || strstr(cfg, "noexact") != nullptr;
        gNoFb = gNoFb || strstr(cfg, "nofb") != nullptr;
        gNoClamp = gNoClamp || strstr(cfg, "noclamp") != nullptr;
        gNoImGui = gNoImGui || strstr(cfg, "noimgui") != nullptr;
        gSyncFrame = gSyncFrame || strstr(cfg, "syncframe") != nullptr;
        if (const char* dc = strstr(cfg, "decal=")) {
            gDecalBias = (float)atof(dc + 6);
        }
        if (strstr(cfg, "purple") != nullptr) {
            gClr[0] = 0.2f;
            gClr[1] = 0.0f;
            gClr[2] = 0.4f;
        }
        DkLogF("deko3d backend v12 decal bias=%.2f", gDecalBias);
        DkLogF("cfg: notex=%d noblend=%d nodesc=%d noexact=%d nofb=%d", (int)gNoTex, (int)gNoBlend, (int)gNoDesc,
               (int)gNoExact, (int)gNoFb);
    }
    {
        Result rr = romfsInit(); // precompiled shaders embedded in the NRO live in romfs:/dkshaders
        DkLogF("romfsInit -> 0x%x", (unsigned)rr);
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
    for (uint32_t i = 0; i < kFrames; i++) {
        d.vtxMems[i] = dk::MemBlockMaker{ d.device, kVtxMemSize }
                           .setFlags(DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached)
                           .create();
    }

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
    d.shaderMem = dk::MemBlockMaker{ d.device, kShaderMemSize }
                      .setFlags(DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached | DkMemBlockFlags_Code)
                      .create();
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

    for (uint32_t i = 0; i < kFrames; i++) {
        d.cmdMems[i] = dk::MemBlockMaker{ d.device, kCmdMemSize }
                           .setFlags(DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached)
                           .create();
        d.cmdBufs[i] = dk::CmdBufMaker{ d.device }.create();
        d.cmdBufs[i].addMemory(d.cmdMems[i], 0, kCmdMemSize);
    }

    // Sampler-side descriptors for the two swapchain images, so fb 0 can be a copy source.
    {
        DkImageDescriptor* descs = (DkImageDescriptor*)d.imgDescMem.getCpuAddr();
        for (uint32_t i = 0; i < kNumFb; i++) {
            uint32_t id = AllocTexId();
            if (!id) {
                continue;
            }
            gScreenTex[i] = id;
            gTexs[id].w = d.width;
            gTexs[id].h = d.height;
            dk::ImageView v{ d.images[i] };
            dkImageDescriptorInitialize(&descs[id], &v, false, false);
        }
        gSavedVp0 = { 0, 0, (int)d.width, (int)d.height };
        gSavedSc0 = gSavedVp0;
        DkLogF("Init: screen tex ids %u %u", gScreenTex[0], gScreenTex[1]);
    }
    DkImGuiInit();
}
void GfxRenderingAPIDeko3D::OnResize() {}
void GfxRenderingAPIDeko3D::StartFrame() {
    static uint32_t n = 0;
    if (n < 3 || n % 600 == 0) {
        DkLogF("StartFrame #%u", n);
    }
    n++;
    gFrameCount++;
    BeginFrame();
}
void GfxRenderingAPIDeko3D::EndFrame() {
    auto& d = gDk;
    DkScope perfScope(gTEnd);
    static uint32_t n = 0;
    if (n < 3 || n % 120 == 0) {
        DkLogF("frame #%u draws=%u tris=%u vtxbytes=%u nocolor=%u mismatch=%u texdraws=%u exactdraws=%u texs=%zu fbdraws=%u fbskip=%u sel=%u copy=%u resolve=%u", n,
               gDraws, gTris, gVtxCursor, gSkippedNoColor, gLayoutMismatch, gTexturedDraws, gExactDraws, gTexs.size(),
               gFbDraws, gFbSkipped, gSelects, gCopies, gResolves);
    }
    n++;
    if (!gFrameOpen) {
        BeginFrame();
        if (!gFrameOpen) {
            return;
        }
    }
    DkImGuiReplay();
    d.queue.submitCommands(CmdBuf().finishList());
    d.queue.presentImage(d.swapchain, gDkSlot);
    d.queue.signalFence(gFences[gFrameIdx], true);
    gFenceSet[gFrameIdx] = true;
    if (gSyncFrame) {
        d.queue.waitIdle();
    }
    gFrameOpen = false;
}
void GfxRenderingAPIDeko3D::FinishRender() {}
int GfxRenderingAPIDeko3D::CreateFramebuffer() {
    static int next = 1;
    int id = next++;
    if ((size_t)id >= gFbos.size()) {
        gFbos.resize((size_t)id + 1);
    }
    DkLogF("CreateFramebuffer -> %d", id);
    return id;
}
void GfxRenderingAPIDeko3D::UpdateFramebufferParameters(int fb, uint32_t w, uint32_t h, uint32_t msaa, bool, bool rt, bool depth, bool) {
    static int logged = 0;
    if (logged++ < 40) {
        DkLogF("fbparams id=%d %ux%u msaa=%u rt=%d depth=%d", fb, w, h, msaa, (int)rt, (int)depth);
    }
    if (fb <= 0) {
        return; // fb 0 is the swapchain
    }
    DkMakeFbo(fb, w, h, depth);
}
void GfxRenderingAPIDeko3D::StartDrawToFramebuffer(int fb, float noiseScale) {
    gNoiseScale = noiseScale;
    if (!gFrameOpen || gNoFb) {
        gCurFb = fb;
        return;
    }
    if (DkFirst(1, ((uint64_t)(uint32_t)fb << 32) ^ (uint64_t)(uint32_t)(noiseScale * 100.0f))) {
        uint32_t w = 0, h = 0;
        DkFbDims(fb, &w, &h);
        DkLogF("StartDrawToFramebuffer fb=%d size=%ux%u noise=%.2f frame=%u", fb, w, h, noiseScale, gFrameCount);
    }
    if (fb != gCurFb) {
        DkSetTarget(fb);
    }
}
void GfxRenderingAPIDeko3D::CopyFramebuffer(int dst, int src, int sx0, int sy0, int sx1, int sy1, int dx0, int dy0,
                                            int dx1, int dy1) {
    gCopies++;
    uint64_t key = 1469598103934665603ull;
    const int vals[10] = { dst, src, sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1 };
    for (int v : vals) {
        key = (key ^ (uint64_t)(uint32_t)v) * 1099511628211ull;
    }
    if (DkFirst(3, key)) {
        DkLogF("CopyFramebuffer dst=%d src=%d src=(%d,%d)-(%d,%d) dst=(%d,%d)-(%d,%d) frame=%u", dst, src, sx0, sy0,
               sx1, sy1, dx0, dy0, dx1, dy1, gFrameCount);
    }
    if (gNoFb) {
        return;
    }
    DkDrawFbQuad(dst, src, sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1);
}
void GfxRenderingAPIDeko3D::ClearFramebuffer(bool c, bool dp) {
    if (!gFrameOpen || (gCurFb != 0 && (gNoFb || !gCurValid))) {
        return;
    }
    auto& d = gDk;
    CmdBuf().setScissors(0, { { 0, 0, gCurW, gCurH } });
    if (c) {
        CmdBuf().clearColor(0, DkColorMask_RGBA, gClr[0], gClr[1], gClr[2], 1.0f);
    }
    if (dp && gCurHasDepth) {
        CmdBuf().clearDepthStencil(true, 1.0f, 0xFF, 0);
    }
    gNeedBarrier = true;
    if (gCurFb != 0 && (size_t)gCurFb < gFbos.size()) {
        gFbos[gCurFb].dirty = true;
    }
}
void GfxRenderingAPIDeko3D::ReadFramebufferToCPU(int, uint32_t, uint32_t, uint16_t*) {}
void GfxRenderingAPIDeko3D::ResolveMSAAColorBuffer(int dst, int src) {
    gResolves++;
    if (DkFirst(5, ((uint64_t)(uint32_t)dst << 32) ^ (uint64_t)(uint32_t)src)) {
        DkLogF("ResolveMSAAColorBuffer dst=%d src=%d frame=%u", dst, src, gFrameCount);
    }
    if (gNoFb || !gFrameOpen || dst == src) {
        return;
    }
    // msaa is always 1 here, so a resolve is a plain copy of whatever was drawn since the last one.
    bool dirty = (src == 0) || ((size_t)src < gFbos.size() && gFbos[src].dirty);
    if (!dirty) {
        return;
    }
    uint32_t sw = 0, sh = 0, dw = 0, dh = 0;
    if (!DkFbDims(src, &sw, &sh) || !DkFbDims(dst, &dw, &dh)) {
        return;
    }
    DkDrawFbQuad(dst, src, 0, 0, (int)sw, (int)sh, 0, 0, (int)dw, (int)dh);
    if (src != 0) {
        gFbos[src].dirty = false;
    }
}
std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
GfxRenderingAPIDeko3D::GetPixelDepth(int, const std::set<std::pair<float, float>>& coordinates) {
    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff> res;
    for (const auto& c : coordinates) {
        res[c] = 0xFFFF; // TODO: read the real depth buffer
    }
    return res;
}
void* GfxRenderingAPIDeko3D::GetFramebufferTextureId(int) { return nullptr; }
void GfxRenderingAPIDeko3D::SelectTextureFb(int fb) {
    gSelects++;
    if (gNoFb) {
        return;
    }
    uint32_t id = DkFbTexId(fb);
    if (DkFirst(2, (uint64_t)(uint32_t)fb)) {
        DkLogF("SelectTextureFb fb=%d tex=%u frame=%u", fb, id, gFrameCount);
    }
    if (!id) {
        return;
    }
    gBoundTex[0] = id; // Fast3D always samples framebuffers on tile 0
    gCurTile = 0;
    if (gFrameOpen && gNeedBarrier) {
        CmdBuf().barrier(DkBarrier_Fragments, DkInvalidateFlags_Image);
        gNeedBarrier = false;
    }
}
void GfxRenderingAPIDeko3D::DeleteTexture(uint32_t id) {
    if (id == 0 || id >= gTexs.size()) {
        return;
    }
    if (gTexs[id].mem) {
        gGraves[gFrameIdx].push_back(std::move(gTexs[id].mem));
    }
    gTexs[id] = DkTex();
    gPendingFree[gFrameIdx].push_back(id);
    for (auto& b : gBoundTex) {
        if (b == id) {
            b = 0;
        }
    }
}
void GfxRenderingAPIDeko3D::SetTextureFilter(FilteringMode mode) { mFilter = mode; }
FilteringMode GfxRenderingAPIDeko3D::GetTextureFilter() { return mFilter; }
void GfxRenderingAPIDeko3D::SetSrgbMode() { mSrgbMode = true; }
ImTextureID GfxRenderingAPIDeko3D::GetTextureById(int id) { return (ImTextureID)(uintptr_t)id; }
void GfxRenderingAPIDeko3D::SetCurrentPrimDepth(float depth) { mCurrentPrimDepth = depth; mPrimDepthDirty = true; }

// Called from Fast3dGui::ImGuiBackendNewFrame, before ImGui::NewFrame().
void GfxDeko3dImGuiNewFrame() {
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2((float)gDk.width, (float)gDk.height);
    io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);

    static bool first = true;
    if (first) {
        first = false;
        io.BackendPlatformName = "deko3d-libnx";
        io.BackendRendererName = "deko3d";
        io.BackendFlags |= ImGuiBackendFlags_HasGamepad | ImGuiBackendFlags_RendererHasVtxOffset;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
        padInitializeDefault(&gImPad);
        hidInitializeTouchScreen();
    }

    // Time step
    static uint64_t lastTick = 0;
    uint64_t now = armGetSystemTick();
    if (lastTick != 0) {
        float dt = (float)((double)(now - lastTick) / (double)armGetSystemTickFreq());
        io.DeltaTime = dt > 0.0001f ? (dt > 0.25f ? 0.25f : dt) : 0.0001f;
    }
    lastTick = now;

    // Font atlas -> a real texture. Re-uploaded whenever the atlas is rebuilt or its id was reset.
    if (!io.Fonts->IsBuilt() || gImFontTex == 0 || (uintptr_t)io.Fonts->TexID != (uintptr_t)gImFontTex) {
        unsigned char* px = nullptr;
        int w = 0, h = 0;
        io.Fonts->GetTexDataAsRGBA32(&px, &w, &h);
        if (gImFontTex == 0) {
            gImFontTex = AllocTexId();
        }
        if (gImFontTex && px && w > 0 && h > 0) {
            DkUploadRgba(gImFontTex, px, (uint32_t)w, (uint32_t)h);
            io.Fonts->SetTexID((ImTextureID)(uintptr_t)gImFontTex);
            DkLogF("imgui font atlas %dx%d -> texture %u", w, h, gImFontTex);
        }
    }

    // Gamepad -> ImGui nav keys. Switch A/B are the right/bottom face buttons, so A confirms.
    padUpdate(&gImPad);
    u64 b = padGetButtons(&gImPad);
    struct Map {
        u64 mask;
        ImGuiKey key;
    };
    static const Map kMap[] = {
        { HidNpadButton_A, ImGuiKey_GamepadFaceDown },   { HidNpadButton_B, ImGuiKey_GamepadFaceRight },
        { HidNpadButton_X, ImGuiKey_GamepadFaceUp },     { HidNpadButton_Y, ImGuiKey_GamepadFaceLeft },
        { HidNpadButton_Left, ImGuiKey_GamepadDpadLeft }, { HidNpadButton_Right, ImGuiKey_GamepadDpadRight },
        { HidNpadButton_Up, ImGuiKey_GamepadDpadUp },    { HidNpadButton_Down, ImGuiKey_GamepadDpadDown },
        { HidNpadButton_L, ImGuiKey_GamepadL1 },         { HidNpadButton_R, ImGuiKey_GamepadR1 },
        { HidNpadButton_ZL, ImGuiKey_GamepadL2 },        { HidNpadButton_ZR, ImGuiKey_GamepadR2 },
        { HidNpadButton_StickL, ImGuiKey_GamepadL3 },    { HidNpadButton_StickR, ImGuiKey_GamepadR3 },
        { HidNpadButton_Minus, ImGuiKey_GamepadBack },   { HidNpadButton_Plus, ImGuiKey_GamepadStart },
    };
    for (const Map& m : kMap) {
        bool down = (b & m.mask) != 0;
        io.AddKeyAnalogEvent(m.key, down, down ? 1.0f : 0.0f);
    }
    HidAnalogStickState ls = padGetStickPos(&gImPad, 0);
    float lx = (float)ls.x / 32767.0f, ly = (float)ls.y / 32767.0f;
    io.AddKeyAnalogEvent(ImGuiKey_GamepadLStickLeft, lx < -0.1f, lx < 0.0f ? -lx : 0.0f);
    io.AddKeyAnalogEvent(ImGuiKey_GamepadLStickRight, lx > 0.1f, lx > 0.0f ? lx : 0.0f);
    io.AddKeyAnalogEvent(ImGuiKey_GamepadLStickUp, ly > 0.1f, ly > 0.0f ? ly : 0.0f);
    io.AddKeyAnalogEvent(ImGuiKey_GamepadLStickDown, ly < -0.1f, ly < 0.0f ? -ly : 0.0f);

    // Touch screen -> left mouse button.
    static bool touching = false;
    HidTouchScreenState ts = {};
    size_t n = hidGetTouchScreenStates(&ts, 1);
    if (n > 0 && ts.count > 0) {
        io.AddMousePosEvent((float)ts.touches[0].x, (float)ts.touches[0].y);
        io.AddMouseButtonEvent(0, true);
        touching = true;
    } else if (touching) {
        io.AddMouseButtonEvent(0, false);
        io.AddMousePosEvent(-FLT_MAX, -FLT_MAX);
        touching = false;
    }
}

// Called from Fast3dGui::ImGuiRenderDrawData. Only copies; the GPU work happens in EndFrame.
void GfxDeko3dImGuiRender(ImDrawData* data) {
    gImVerts.clear();
    gImIdx.clear();
    gImCmds.clear();
    gImPending = true;
    if (!data || data->CmdListsCount == 0) {
        return;
    }
    gImPos[0] = data->DisplayPos.x;
    gImPos[1] = data->DisplayPos.y;
    gImSize[0] = data->DisplaySize.x > 1.0f ? data->DisplaySize.x : 1.0f;
    gImSize[1] = data->DisplaySize.y > 1.0f ? data->DisplaySize.y : 1.0f;
    gImVerts.reserve((size_t)data->TotalVtxCount);
    gImIdx.reserve((size_t)data->TotalIdxCount);
    const ImVec2 clipOff = data->DisplayPos;
    const ImVec2 clipScale = data->FramebufferScale;
    for (int n = 0; n < data->CmdListsCount; n++) {
        const ImDrawList* list = data->CmdLists[n];
        uint32_t vbase = (uint32_t)gImVerts.size();
        uint32_t ibase = (uint32_t)gImIdx.size();
        gImVerts.insert(gImVerts.end(), list->VtxBuffer.Data, list->VtxBuffer.Data + list->VtxBuffer.Size);
        gImIdx.insert(gImIdx.end(), list->IdxBuffer.Data, list->IdxBuffer.Data + list->IdxBuffer.Size);
        for (int i = 0; i < list->CmdBuffer.Size; i++) {
            const ImDrawCmd& pc = list->CmdBuffer[i];
            if (pc.UserCallback != nullptr || pc.ElemCount == 0) {
                continue; // callbacks (and ResetRenderState) are not supported
            }
            DkImCmd c;
            c.x0 = (pc.ClipRect.x - clipOff.x) * clipScale.x;
            c.y0 = (pc.ClipRect.y - clipOff.y) * clipScale.y;
            c.x1 = (pc.ClipRect.z - clipOff.x) * clipScale.x;
            c.y1 = (pc.ClipRect.w - clipOff.y) * clipScale.y;
            c.tex = (uint32_t)(uintptr_t)pc.GetTexID();
            c.count = pc.ElemCount;
            c.idxOff = ibase + pc.IdxOffset;
            c.vtxOff = (int32_t)(vbase + pc.VtxOffset);
            gImCmds.push_back(c);
        }
    }
}
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
