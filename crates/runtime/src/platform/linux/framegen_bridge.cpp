// Optional Linux FSR3 bridge. Built by scripts/linux-build-framegen.py, not by Cargo.
// The pinned MIT SDK owns the interpolation/optical-flow passes; Regaliant owns WSI.
#include <vulkan/vulkan.h>
#include <FidelityFX/host/ffx_frameinterpolation.h>
#include <FidelityFX/host/ffx_opticalflow.h>
#include <FidelityFX/host/backends/vk/ffx_vk.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include "framegen_motion.hpp"

struct Image { uint64_t image; uint32_t format, width, height; };
struct Params {
    Image color, hudless, depth, motion, output, prepared_motion, motion_status;
    float reprojection[16];
    float jitter_x, jitter_y, near_plane, far_plane, fov, milliseconds;
    uint32_t reset, linear;
};
static_assert(sizeof(Image)==24 && sizeof(Params)==264, "Rust bridge ABI");
struct State {
    MotionPrepare motion;
    FfxInterface backend{};
    FfxFrameInterpolationContext fi{};
    FfxOpticalflowContext of{};
    void* scratch = nullptr;
    uint32_t shared = 0;
    bool shared_ok = false, fi_ok = false, of_ok = false;
    FfxResourceInternal resources[5]{};
    uint32_t made = 0;
    uint64_t frame = 0;
    ~State() {
        if (fi_ok) ffxFrameInterpolationContextDestroy(&fi);
        if (of_ok) ffxOpticalflowContextDestroy(&of);
        for (uint32_t i=0; i<made; ++i) backend.fpDestroyResource(&backend,resources[i],shared);
        if (shared_ok) backend.fpDestroyBackendContext(&backend,shared);
        std::free(scratch);
    }
    FfxResource get(uint32_t i) { return backend.fpGetResource(&backend,resources[i]); }
};
static FfxResource resource(const Image& i, const wchar_t* name) {
    FfxResourceDescription d{};
    d.type=FFX_RESOURCE_TYPE_TEXTURE2D;
    d.format=ffxGetSurfaceFormatVK(static_cast<VkFormat>(i.format));
    d.width=i.width;d.height=i.height;d.depth=1;d.mipCount=1;
    d.usage=FFX_RESOURCE_USAGE_UAV;
    return ffxGetResourceVK(reinterpret_cast<void*>(i.image),d,name,FFX_RESOURCE_STATE_UNORDERED_ACCESS);
}
// The Linux SDK omits the Windows swapchain proxy but its backend still installs this
// callback. Our raw FI/OF API never calls it; requesting proxy ownership is unsupported.
extern "C" FfxErrorCode ffxSetFrameGenerationConfigToSwapchainVK(FfxFrameGenerationConfig const*) {
    return FFX_ERROR_INCOMPLETE_INTERFACE;
}
extern "C" uint32_t regaliant_fsrfg_abi() { return 2; }
extern "C" int regaliant_fsrfg_create(VkInstance instance, VkPhysicalDevice physical, VkDevice device,
    uint32_t width,uint32_t height,uint32_t render_width,uint32_t render_height, uint32_t infinite, void** result) {
    *result=nullptr;
    std::fprintf(stderr,"FSR3 bridge: create output %ux%u, render %ux%u\n",width,height,render_width,render_height);
    std::fflush(stderr);
    auto invoke=[](const char* stage, auto call) {
        std::fprintf(stderr,"FSR3 bridge: begin %s\n",stage);std::fflush(stderr);
        int e=call();
        std::fprintf(stderr,"FSR3 bridge: end %s, code %d\n",stage,e);std::fflush(stderr);
        return e;
    };
    try {
        auto s=std::make_unique<State>();
        if (auto e=s->motion.create(device)) return 1100+int(e);
        VkDeviceContext dc{device,physical,vkGetDeviceProcAddr,vkGetInstanceProcAddr,instance};
        auto bytes=ffxGetScratchMemorySizeVK(physical,3);
        if (posix_memalign(&s->scratch,64,bytes)) return -1;
        std::memset(s->scratch,0,bytes);
        int e=invoke("backend interface", [&] { return ffxGetInterfaceVK(&s->backend,ffxGetDeviceVK(&dc),s->scratch,bytes,3); });
        if (e) return 100+e;
        e=invoke("shared context", [&] { return s->backend.fpCreateBackendContext(&s->backend,FFX_EFFECT_SHAREDRESOURCES,nullptr,&s->shared); });
        if (e) return 200+e;
        s->shared_ok=true;
        FfxOpticalflowContextDescription of{};of.backendInterface=s->backend;of.resolution={width,height};
        if ((e=invoke("optical flow context", [&] { return ffxOpticalflowContextCreate(&s->of,&of); }))) return 300+e;
        s->of_ok=true;
        FfxFrameInterpolationContextDescription fi{};
        fi.backendInterface=s->backend;fi.maxRenderSize={render_width,render_height};fi.displaySize={width,height};
        fi.backBufferFormat=fi.previousInterpolationSourceFormat=FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT;
        fi.flags=FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INVERTED;
        if(infinite) fi.flags|=FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INFINITE;
        if ((e=invoke("interpolation context", [&] { return ffxFrameInterpolationContextCreate(&s->fi,&fi); }))) return 400+e;
        s->fi_ok=true;
        FfxOpticalflowSharedResourceDescriptions o{};FfxFrameInterpolationSharedResourceDescriptions f{};
        if ((e=invoke("optical flow resources", [&] { return ffxOpticalflowGetSharedResourceDescriptions(&s->of,&o); }))) return 500+e;
        if ((e=invoke("interpolation resources", [&] { return ffxFrameInterpolationGetSharedResourceDescriptions(&s->fi,&f); }))) return 600+e;
        FfxCreateResourceDescription descs[]={o.opticalFlowVector,o.opticalFlowSCD,f.dilatedDepth,f.dilatedMotionVectors,f.reconstructedPrevNearestDepth};
        for (auto& desc:descs) {
            if ((e=invoke("shared image", [&] { return s->backend.fpCreateResource(&s->backend,&desc,s->shared,&s->resources[s->made]); }))) return 700+e;
            ++s->made;
        }
        *result=s.release();
        std::fprintf(stderr,"FSR3 bridge: context ready\n");std::fflush(stderr);return 0;
    } catch (...) { return -2; }
}
// All external images enter/leave GENERAL. Caller waits for this dispatch before reuse/destroy.
extern "C" int regaliant_fsrfg_dispatch(void* state,VkCommandBuffer cmd,const Params* p) {
    auto& s=*static_cast<State*>(state);
    int e;
    if((e=s.motion.record(cmd,std::array<Image,4>{p->motion,p->depth,p->prepared_motion,p->motion_status},
                          p->reprojection,p->jitter_x,p->jitter_y))) return 1200+e;
    FfxFrameInterpolationPrepareDescription prep{};
    prep.commandList=ffxGetCommandListVK(cmd);prep.renderSize={p->depth.width,p->depth.height};
    prep.jitterOffset={p->jitter_x,p->jitter_y};
    // Title vectors are screen fractions, current minus previous. FSR wants pixels toward previous.
    prep.motionVectorScale={-float(p->motion.width),-float(p->motion.height)};
    prep.frameTimeDelta=p->milliseconds;prep.cameraNear=p->near_plane;prep.cameraFar=p->far_plane;
    prep.cameraFovAngleVertical=p->fov;prep.viewSpaceToMetersFactor=1.0f;prep.frameID=s.frame;
    prep.depth=resource(p->depth,L"Regaliant depth");prep.motionVectors=resource(p->prepared_motion,L"Regaliant valid motion");
    prep.dilatedDepth=s.get(2);prep.dilatedMotionVectors=s.get(3);prep.reconstructedPrevDepth=s.get(4);
    if ((e=ffxFrameInterpolationPrepare(&s.fi,&prep))) return 800+e;
    FfxOpticalflowDispatchDescription of{};
    of.commandList=prep.commandList;of.color=resource(p->hudless,L"Regaliant pre-UI");
    of.opticalFlowVector=s.get(0);of.opticalFlowSCD=s.get(1);of.reset=p->reset||s.frame==0;
    of.backbufferTransferFunction=p->linear?FFX_BACKBUFFER_TRANSFER_FUNCTION_SCRGB:FFX_BACKBUFFER_TRANSFER_FUNCTION_SRGB;
    of.minMaxLuminance={0,1};
    if ((e=ffxOpticalflowContextDispatch(&s.of,&of))) return 900+e;
    FfxFrameInterpolationDispatchDescription fi{};
    fi.commandList=prep.commandList;fi.displaySize={p->color.width,p->color.height};fi.renderSize=prep.renderSize;
    fi.currentBackBuffer=resource(p->color,L"Regaliant final");fi.currentBackBuffer_HUDLess=of.color;
    fi.output=resource(p->output,L"Regaliant generated");
    fi.interpolationRect={0,0,int32_t(p->color.width),int32_t(p->color.height)};
    fi.opticalFlowVector=s.get(0);fi.opticalFlowSceneChangeDetection=s.get(1);
    fi.opticalFlowBlockSize=8;fi.opticalFlowBufferSize={(p->color.width+7)/8,(p->color.height+7)/8};
    fi.opticalFlowScale={1.0f/p->color.width,1.0f/p->color.height};
    fi.cameraNear=p->near_plane;fi.cameraFar=p->far_plane;fi.cameraFovAngleVertical=p->fov;
    fi.viewSpaceToMetersFactor=1;fi.frameTimeDelta=p->milliseconds;fi.reset=of.reset;fi.frameID=s.frame++;
    fi.backBufferTransferFunction=static_cast<FfxBackbufferTransferFunction>(of.backbufferTransferFunction);
    fi.minMaxLuminance[0]=0;fi.minMaxLuminance[1]=1;
    fi.dilatedDepth=s.get(2);fi.dilatedMotionVectors=s.get(3);fi.reconstructedPrevDepth=s.get(4);
    if ((e=ffxFrameInterpolationDispatch(&s.fi,&fi))) return 1000+e;
    return 0;
}
extern "C" void regaliant_fsrfg_destroy(void* state) { delete static_cast<State*>(state); }
