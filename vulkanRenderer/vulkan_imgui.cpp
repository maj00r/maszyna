/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include <vulkan/vulkan.h>

#include "stdafx.h"  // pulls in imgui.h + STL

#include "vulkan_imgui.h"

#include <cstring>

#include "utilities/Logs.h"

#include "imgui_vert_spv.h"
#include "imgui_frag_spv.h"

namespace {

void log_error(const std::string &msg) {
  ErrorLog("vulkan_imgui: " + msg);
}

uint32_t find_memory_type(VkPhysicalDevice pd, uint32_t type_bits,
                          VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(pd, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
    if ((type_bits & (1u << i)) &&
        (mp.memoryTypes[i].propertyFlags & props) == props) {
      return i;
    }
  }
  return UINT32_MAX;
}

bool create_buffer(VkPhysicalDevice pd, VkDevice dev, VkDeviceSize size,
                   VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
                   VkBuffer &buf, VkDeviceMemory &mem) {
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = size;
  bi.usage = usage;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(dev, &bi, nullptr, &buf) != VK_SUCCESS) return false;

  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(dev, buf, &req);

  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = find_memory_type(pd, req.memoryTypeBits, props);
  if (ai.memoryTypeIndex == UINT32_MAX) return false;
  if (vkAllocateMemory(dev, &ai, nullptr, &mem) != VK_SUCCESS) return false;
  vkBindBufferMemory(dev, buf, mem, 0);
  return true;
}

}  // namespace

bool vulkan_imgui_renderer::Init() {
  if (m_ctx.device == VK_NULL_HANDLE) {
    log_error("Init called without a Vulkan context.");
    return false;
  }
  if (!create_font_texture()) return false;
  if (!create_pipeline()) return false;
  m_ready = true;
  return true;
}

bool vulkan_imgui_renderer::create_font_texture() {
  ImGuiIO &io = ImGui::GetIO();
  unsigned char *pixels = nullptr;
  int width = 0, height = 0;
  io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
  const VkDeviceSize upload_size =
      static_cast<VkDeviceSize>(width) * height * 4;

  // Staging buffer.
  VkBuffer staging = VK_NULL_HANDLE;
  VkDeviceMemory staging_mem = VK_NULL_HANDLE;
  if (!create_buffer(m_ctx.physical_device, m_ctx.device, upload_size,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     staging, staging_mem)) {
    log_error("font staging buffer allocation failed.");
    return false;
  }
  void *mapped = nullptr;
  vkMapMemory(m_ctx.device, staging_mem, 0, upload_size, 0, &mapped);
  std::memcpy(mapped, pixels, static_cast<size_t>(upload_size));
  vkUnmapMemory(m_ctx.device, staging_mem);

  // Device-local image.
  VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ici.imageType = VK_IMAGE_TYPE_2D;
  ici.format = VK_FORMAT_R8G8B8A8_UNORM;
  ici.extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
  ici.mipLevels = 1;
  ici.arrayLayers = 1;
  ici.samples = VK_SAMPLE_COUNT_1_BIT;
  ici.tiling = VK_IMAGE_TILING_OPTIMAL;
  ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (vkCreateImage(m_ctx.device, &ici, nullptr, &m_font_image) != VK_SUCCESS) {
    log_error("font image creation failed.");
    return false;
  }

  VkMemoryRequirements req;
  vkGetImageMemoryRequirements(m_ctx.device, m_font_image, &req);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = find_memory_type(m_ctx.physical_device,
                                        req.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (vkAllocateMemory(m_ctx.device, &ai, nullptr, &m_font_memory) !=
      VK_SUCCESS) {
    log_error("font image memory allocation failed.");
    return false;
  }
  vkBindImageMemory(m_ctx.device, m_font_image, m_font_memory, 0);

  // One-time upload: transition, copy, transition to shader-read.
  VkCommandBufferAllocateInfo cbai{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = m_ctx.command_pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  vkAllocateCommandBuffers(m_ctx.device, &cbai, &cmd);

  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &bi);

  VkImageMemoryBarrier to_dst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  to_dst.srcAccessMask = 0;
  to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_dst.image = m_font_image;
  to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                       1, &to_dst);

  VkBufferImageCopy region{};
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.imageExtent = {static_cast<uint32_t>(width),
                        static_cast<uint32_t>(height), 1};
  vkCmdCopyBufferToImage(cmd, staging, m_font_image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  VkImageMemoryBarrier to_read{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  to_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  to_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  to_read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  to_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  to_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_read.image = m_font_image;
  to_read.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &to_read);

  vkEndCommandBuffer(cmd);
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;
  vkQueueSubmit(m_ctx.queue, 1, &submit, VK_NULL_HANDLE);
  vkQueueWaitIdle(m_ctx.queue);
  vkFreeCommandBuffers(m_ctx.device, m_ctx.command_pool, 1, &cmd);
  vkDestroyBuffer(m_ctx.device, staging, nullptr);
  vkFreeMemory(m_ctx.device, staging_mem, nullptr);

  // View + sampler.
  VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vci.image = m_font_image;
  vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vci.format = VK_FORMAT_R8G8B8A8_UNORM;
  vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCreateImageView(m_ctx.device, &vci, nullptr, &m_font_view);

  VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  sci.magFilter = VK_FILTER_LINEAR;
  sci.minFilter = VK_FILTER_LINEAR;
  sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sci.minLod = -1000.f;
  sci.maxLod = 1000.f;
  sci.maxAnisotropy = 1.f;
  vkCreateSampler(m_ctx.device, &sci, nullptr, &m_font_sampler);

  // Descriptor set layout / pool / set.
  VkDescriptorSetLayoutBinding binding{};
  binding.binding = 0;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  binding.descriptorCount = 1;
  binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  VkDescriptorSetLayoutCreateInfo dlci{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dlci.bindingCount = 1;
  dlci.pBindings = &binding;
  vkCreateDescriptorSetLayout(m_ctx.device, &dlci, nullptr, &m_set_layout);

  VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
  VkDescriptorPoolCreateInfo dpci{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &pool_size;
  vkCreateDescriptorPool(m_ctx.device, &dpci, nullptr, &m_descriptor_pool);

  VkDescriptorSetAllocateInfo dsai{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = m_descriptor_pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &m_set_layout;
  vkAllocateDescriptorSets(m_ctx.device, &dsai, &m_descriptor_set);

  VkDescriptorImageInfo image_info{};
  image_info.sampler = m_font_sampler;
  image_info.imageView = m_font_view;
  image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = m_descriptor_set;
  write.dstBinding = 0;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  write.pImageInfo = &image_info;
  vkUpdateDescriptorSets(m_ctx.device, 1, &write, 0, nullptr);

  io.Fonts->SetTexID((ImTextureID)(intptr_t)m_descriptor_set);
  return true;
}

bool vulkan_imgui_renderer::create_pipeline() {
  VkShaderModuleCreateInfo vmci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  vmci.codeSize = sizeof(imgui_vert_spv);
  vmci.pCode = imgui_vert_spv;
  VkShaderModule vert = VK_NULL_HANDLE;
  vkCreateShaderModule(m_ctx.device, &vmci, nullptr, &vert);

  VkShaderModuleCreateInfo fmci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  fmci.codeSize = sizeof(imgui_frag_spv);
  fmci.pCode = imgui_frag_spv;
  VkShaderModule frag = VK_NULL_HANDLE;
  vkCreateShaderModule(m_ctx.device, &fmci, nullptr, &frag);

  VkPipelineShaderStageCreateInfo stages[2] = {
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vert;
  stages[0].pName = "main";
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = frag;
  stages[1].pName = "main";

  VkVertexInputBindingDescription binding{};
  binding.binding = 0;
  binding.stride = sizeof(ImDrawVert);
  binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  VkVertexInputAttributeDescription attrs[3]{};
  attrs[0] = {0, 0, VK_FORMAT_R32G32_SFLOAT, IM_OFFSETOF(ImDrawVert, pos)};
  attrs[1] = {1, 0, VK_FORMAT_R32G32_SFLOAT, IM_OFFSETOF(ImDrawVert, uv)};
  attrs[2] = {2, 0, VK_FORMAT_R8G8B8A8_UNORM, IM_OFFSETOF(ImDrawVert, col)};

  VkPipelineVertexInputStateCreateInfo vertex_input{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vertex_input.vertexBindingDescriptionCount = 1;
  vertex_input.pVertexBindingDescriptions = &binding;
  vertex_input.vertexAttributeDescriptionCount = 3;
  vertex_input.pVertexAttributeDescriptions = attrs;

  VkPipelineInputAssemblyStateCreateInfo input_assembly{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

  VkPipelineViewportStateCreateInfo viewport_state{
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  viewport_state.viewportCount = 1;
  viewport_state.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo raster{
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  raster.polygonMode = VK_POLYGON_MODE_FILL;
  raster.cullMode = VK_CULL_MODE_NONE;
  raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  raster.lineWidth = 1.0f;

  VkPipelineMultisampleStateCreateInfo multisample{
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineColorBlendAttachmentState blend{};
  blend.blendEnable = VK_TRUE;
  blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  blend.colorBlendOp = VK_BLEND_OP_ADD;
  blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  blend.alphaBlendOp = VK_BLEND_OP_ADD;
  blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  VkPipelineColorBlendStateCreateInfo color_blend{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  color_blend.attachmentCount = 1;
  color_blend.pAttachments = &blend;

  VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamic_state{
      VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dynamic_state.dynamicStateCount = 2;
  dynamic_state.pDynamicStates = dyn;

  VkPushConstantRange pc_range{};
  pc_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  pc_range.offset = 0;
  pc_range.size = sizeof(float) * 4;  // scale (vec2) + translate (vec2)

  VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  lci.setLayoutCount = 1;
  lci.pSetLayouts = &m_set_layout;
  lci.pushConstantRangeCount = 1;
  lci.pPushConstantRanges = &pc_range;
  if (vkCreatePipelineLayout(m_ctx.device, &lci, nullptr, &m_pipeline_layout) !=
      VK_SUCCESS) {
    log_error("imgui pipeline layout creation failed.");
    return false;
  }

  VkPipelineRenderingCreateInfo rendering_ci{
      VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rendering_ci.colorAttachmentCount = 1;
  rendering_ci.pColorAttachmentFormats = &m_ctx.color_format;

  VkGraphicsPipelineCreateInfo pci{
      VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  pci.pNext = &rendering_ci;
  pci.stageCount = 2;
  pci.pStages = stages;
  pci.pVertexInputState = &vertex_input;
  pci.pInputAssemblyState = &input_assembly;
  pci.pViewportState = &viewport_state;
  pci.pRasterizationState = &raster;
  pci.pMultisampleState = &multisample;
  pci.pColorBlendState = &color_blend;
  pci.pDynamicState = &dynamic_state;
  pci.layout = m_pipeline_layout;
  pci.renderPass = VK_NULL_HANDLE;

  VkResult res = vkCreateGraphicsPipelines(m_ctx.device, VK_NULL_HANDLE, 1, &pci,
                                           nullptr, &m_pipeline);
  vkDestroyShaderModule(m_ctx.device, vert, nullptr);
  vkDestroyShaderModule(m_ctx.device, frag, nullptr);
  if (res != VK_SUCCESS) {
    log_error("imgui pipeline creation failed.");
    return false;
  }
  return true;
}

bool vulkan_imgui_renderer::ensure_buffers(VkDeviceSize vtx_size,
                                           VkDeviceSize idx_size) {
  frame_buffers &fb = m_buffers[m_buffer_index];
  if (fb.vertex == VK_NULL_HANDLE || fb.vertex_size < vtx_size) {
    if (fb.vertex) vkDestroyBuffer(m_ctx.device, fb.vertex, nullptr);
    if (fb.vertex_memory) vkFreeMemory(m_ctx.device, fb.vertex_memory, nullptr);
    VkDeviceSize alloc = vtx_size + (vtx_size / 2) + 4096;
    if (!create_buffer(m_ctx.physical_device, m_ctx.device, alloc,
                       VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       fb.vertex, fb.vertex_memory)) {
      return false;
    }
    fb.vertex_size = alloc;
  }
  if (fb.index == VK_NULL_HANDLE || fb.index_size < idx_size) {
    if (fb.index) vkDestroyBuffer(m_ctx.device, fb.index, nullptr);
    if (fb.index_memory) vkFreeMemory(m_ctx.device, fb.index_memory, nullptr);
    VkDeviceSize alloc = idx_size + (idx_size / 2) + 4096;
    if (!create_buffer(m_ctx.physical_device, m_ctx.device, alloc,
                       VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       fb.index, fb.index_memory)) {
      return false;
    }
    fb.index_size = alloc;
  }
  return true;
}

void vulkan_imgui_renderer::Render() {
  if (!m_ready || m_current_cmd == VK_NULL_HANDLE) return;

  ImDrawData *draw_data = ImGui::GetDrawData();
  if (draw_data == nullptr || draw_data->TotalVtxCount == 0) return;

  const VkDeviceSize vtx_size =
      static_cast<VkDeviceSize>(draw_data->TotalVtxCount) * sizeof(ImDrawVert);
  const VkDeviceSize idx_size =
      static_cast<VkDeviceSize>(draw_data->TotalIdxCount) * sizeof(ImDrawIdx);
  if (!ensure_buffers(vtx_size, idx_size)) return;

  frame_buffers &fb = m_buffers[m_buffer_index];

  // Upload vertices/indices.
  ImDrawVert *vtx_dst = nullptr;
  ImDrawIdx *idx_dst = nullptr;
  vkMapMemory(m_ctx.device, fb.vertex_memory, 0, vtx_size, 0,
              reinterpret_cast<void **>(&vtx_dst));
  vkMapMemory(m_ctx.device, fb.index_memory, 0, idx_size, 0,
              reinterpret_cast<void **>(&idx_dst));
  for (int n = 0; n < draw_data->CmdListsCount; ++n) {
    const ImDrawList *cl = draw_data->CmdLists[n];
    std::memcpy(vtx_dst, cl->VtxBuffer.Data,
                cl->VtxBuffer.Size * sizeof(ImDrawVert));
    std::memcpy(idx_dst, cl->IdxBuffer.Data,
                cl->IdxBuffer.Size * sizeof(ImDrawIdx));
    vtx_dst += cl->VtxBuffer.Size;
    idx_dst += cl->IdxBuffer.Size;
  }
  vkUnmapMemory(m_ctx.device, fb.vertex_memory);
  vkUnmapMemory(m_ctx.device, fb.index_memory);

  VkCommandBuffer cmd = m_current_cmd;

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          m_pipeline_layout, 0, 1, &m_descriptor_set, 0,
                          nullptr);

  VkDeviceSize offset = 0;
  vkCmdBindVertexBuffers(cmd, 0, 1, &fb.vertex, &offset);
  vkCmdBindIndexBuffer(cmd, fb.index, 0,
                       sizeof(ImDrawIdx) == 2 ? VK_INDEX_TYPE_UINT16
                                              : VK_INDEX_TYPE_UINT32);

  VkViewport viewport{};
  viewport.width = static_cast<float>(m_current_extent.width);
  viewport.height = static_cast<float>(m_current_extent.height);
  viewport.minDepth = 0.f;
  viewport.maxDepth = 1.f;
  vkCmdSetViewport(cmd, 0, 1, &viewport);

  const float scale[2] = {2.0f / draw_data->DisplaySize.x,
                          2.0f / draw_data->DisplaySize.y};
  const float translate[2] = {-1.0f - draw_data->DisplayPos.x * scale[0],
                              -1.0f - draw_data->DisplayPos.y * scale[1]};
  vkCmdPushConstants(cmd, m_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                     sizeof(float) * 2, scale);
  vkCmdPushConstants(cmd, m_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
                     sizeof(float) * 2, sizeof(float) * 2, translate);

  const ImVec2 clip_off = draw_data->DisplayPos;
  int global_vtx = 0;
  int global_idx = 0;
  for (int n = 0; n < draw_data->CmdListsCount; ++n) {
    const ImDrawList *cl = draw_data->CmdLists[n];
    for (int c = 0; c < cl->CmdBuffer.Size; ++c) {
      const ImDrawCmd *pcmd = &cl->CmdBuffer[c];
      if (pcmd->UserCallback != nullptr) {
        pcmd->UserCallback(cl, pcmd);
      } else {
        // Clip rectangle -> scissor.
        float clip_x = pcmd->ClipRect.x - clip_off.x;
        float clip_y = pcmd->ClipRect.y - clip_off.y;
        float clip_w = pcmd->ClipRect.z - clip_off.x;
        float clip_h = pcmd->ClipRect.w - clip_off.y;
        if (clip_x < 0.f) clip_x = 0.f;
        if (clip_y < 0.f) clip_y = 0.f;

        VkRect2D scissor{};
        scissor.offset = {static_cast<int32_t>(clip_x),
                          static_cast<int32_t>(clip_y)};
        scissor.extent = {static_cast<uint32_t>(clip_w - clip_x),
                          static_cast<uint32_t>(clip_h - clip_y)};
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        vkCmdDrawIndexed(cmd, pcmd->ElemCount, 1,
                         global_idx + pcmd->IdxOffset,
                         global_vtx + pcmd->VtxOffset, 0);
      }
    }
    global_idx += cl->IdxBuffer.Size;
    global_vtx += cl->VtxBuffer.Size;
  }

  m_buffer_index = (m_buffer_index + 1) % kFrames;
}

void vulkan_imgui_renderer::Shutdown() {
  if (m_ctx.device == VK_NULL_HANDLE) return;
  vkDeviceWaitIdle(m_ctx.device);

  for (auto &fb : m_buffers) {
    if (fb.vertex) vkDestroyBuffer(m_ctx.device, fb.vertex, nullptr);
    if (fb.vertex_memory) vkFreeMemory(m_ctx.device, fb.vertex_memory, nullptr);
    if (fb.index) vkDestroyBuffer(m_ctx.device, fb.index, nullptr);
    if (fb.index_memory) vkFreeMemory(m_ctx.device, fb.index_memory, nullptr);
    fb = {};
  }

  if (m_pipeline) vkDestroyPipeline(m_ctx.device, m_pipeline, nullptr);
  if (m_pipeline_layout)
    vkDestroyPipelineLayout(m_ctx.device, m_pipeline_layout, nullptr);
  if (m_descriptor_pool)
    vkDestroyDescriptorPool(m_ctx.device, m_descriptor_pool, nullptr);
  if (m_set_layout)
    vkDestroyDescriptorSetLayout(m_ctx.device, m_set_layout, nullptr);
  if (m_font_view) vkDestroyImageView(m_ctx.device, m_font_view, nullptr);
  if (m_font_sampler) vkDestroySampler(m_ctx.device, m_font_sampler, nullptr);
  if (m_font_image) vkDestroyImage(m_ctx.device, m_font_image, nullptr);
  if (m_font_memory) vkFreeMemory(m_ctx.device, m_font_memory, nullptr);

  m_pipeline = VK_NULL_HANDLE;
  m_pipeline_layout = VK_NULL_HANDLE;
  m_descriptor_pool = VK_NULL_HANDLE;
  m_set_layout = VK_NULL_HANDLE;
  m_font_view = VK_NULL_HANDLE;
  m_font_sampler = VK_NULL_HANDLE;
  m_font_image = VK_NULL_HANDLE;
  m_font_memory = VK_NULL_HANDLE;
  m_ready = false;

  // Make a second Shutdown() (the UI layer also calls it, in unspecified
  // order relative to the renderer) a safe no-op.
  m_ctx.device = VK_NULL_HANDLE;
}
