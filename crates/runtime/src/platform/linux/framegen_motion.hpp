// Built alongside the optional bridge; no changes to SDK shaders or guest targets.
#pragma once
#include <vulkan/vulkan.h>
#include <array>
#include "framegen_motion.h" // glslangValidator --vn regaliant_motion_spv

struct MotionPrepare {
    VkDevice device{};
    VkDescriptorSetLayout descriptors{};
    VkPipelineLayout layout{};
    VkPipeline pipeline{};
    VkDescriptorPool pool{};
    VkDescriptorSet set{};
    std::array<VkImageView,4> views{};
    ~MotionPrepare() {
        if (!device) return;
        for (auto v:views) if(v) vkDestroyImageView(device,v,nullptr);
        if(pool) vkDestroyDescriptorPool(device,pool,nullptr);
        if(pipeline) vkDestroyPipeline(device,pipeline,nullptr);
        if(layout) vkDestroyPipelineLayout(device,layout,nullptr);
        if(descriptors) vkDestroyDescriptorSetLayout(device,descriptors,nullptr);
    }
    VkResult create(VkDevice d) {
        device=d;
        std::array<VkDescriptorSetLayoutBinding,4> bindings{};
        for(uint32_t i=0;i<4;++i) bindings[i]={i,i==1?VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo di{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        di.bindingCount=4;di.pBindings=bindings.data();
        VkResult e=vkCreateDescriptorSetLayout(d,&di,nullptr,&descriptors);if(e) return e;
        VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT,0,72};
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        li.setLayoutCount=1;li.pSetLayouts=&descriptors;li.pushConstantRangeCount=1;li.pPushConstantRanges=&pc;
        e=vkCreatePipelineLayout(d,&li,nullptr,&layout);if(e) return e;
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        mi.codeSize=sizeof(regaliant_motion_spv);mi.pCode=regaliant_motion_spv;
        VkShaderModule module{};e=vkCreateShaderModule(d,&mi,nullptr,&module);if(e) return e;
        VkComputePipelineCreateInfo pi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pi.layout=layout;pi.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pi.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;pi.stage.module=module;pi.stage.pName="main";
        e=vkCreateComputePipelines(d,VK_NULL_HANDLE,1,&pi,nullptr,&pipeline);
        vkDestroyShaderModule(d,module,nullptr);if(e) return e;
        VkDescriptorPoolSize sizes[]={{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,3},{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,1}};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets=1;dpi.poolSizeCount=2;dpi.pPoolSizes=sizes;
        e=vkCreateDescriptorPool(d,&dpi,nullptr,&pool);if(e) return e;
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool=pool;ai.descriptorSetCount=1;ai.pSetLayouts=&descriptors;
        return vkAllocateDescriptorSets(d,&ai,&set);
    }
    // The caller waits the previous generation fence before updating descriptors or views.
    template<class Image> VkResult record(VkCommandBuffer cmd,const std::array<Image,4>& images,
                                          const float* reprojection,float jx,float jy) {
        std::array<VkDescriptorImageInfo,4> info{};
        std::array<VkWriteDescriptorSet,4> writes{};
        for(uint32_t i=0;i<4;++i) {
            if(views[i]) { vkDestroyImageView(device,views[i],nullptr); views[i]=VK_NULL_HANDLE; }
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image=reinterpret_cast<VkImage>(images[i].image);vi.viewType=VK_IMAGE_VIEW_TYPE_2D;
            vi.format=static_cast<VkFormat>(images[i].format);vi.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
            auto e=vkCreateImageView(device,&vi,nullptr,&views[i]);if(e) return e;
            info[i]={VK_NULL_HANDLE,views[i],VK_IMAGE_LAYOUT_GENERAL};
            writes[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};writes[i].dstSet=set;writes[i].dstBinding=i;
            writes[i].descriptorCount=1;writes[i].descriptorType=i==1?VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[i].pImageInfo=&info[i];
        }
        vkUpdateDescriptorSets(device,4,writes.data(),0,nullptr);
        float pc[18];for(int i=0;i<16;++i) pc[i]=reprojection[i];pc[16]=jx;pc[17]=jy;
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);
        vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&set,0,nullptr);
        vkCmdPushConstants(cmd,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(pc),pc);
        vkCmdDispatch(cmd,(images[0].width+7)/8,(images[0].height+7)/8,1);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,1,&barrier,0,nullptr,0,nullptr);
        return VK_SUCCESS;
    }
};
