#include "fast/backends/gfx_deko3d.h"

namespace Fast {
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
void GfxRenderingAPIDeko3D::Init() {}
void GfxRenderingAPIDeko3D::OnResize() {}
void GfxRenderingAPIDeko3D::StartFrame() {}
void GfxRenderingAPIDeko3D::EndFrame() {}
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
