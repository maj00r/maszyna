/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#pragma once

#include <vulkan/vulkan.h>

#include <array>

#include "rendering/renderer.h"  // imgui_renderer

// Hand-written Dear ImGui rendering backend for Vulkan (the bundled ImGui 1.73
// ships no Vulkan backend). Integrates with the renderer's dynamic-rendering
// pass: the owner sets the active command buffer for the frame, then the
// engine's UI layer calls Render(), which records the ImGui draw data into it.

struct vulkan_imgui_context {
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  VkCommandPool command_pool = VK_NULL_HANDLE;
  VkFormat color_format = VK_FORMAT_UNDEFINED;
};

class vulkan_imgui_renderer : public imgui_renderer {
 public:
  void set_context(const vulkan_imgui_context &ctx) { m_ctx = ctx; }
  // Called by the renderer each frame, inside an active dynamic-rendering pass,
  // right before the engine's render_ui() drives Render(). A null command
  // buffer makes Render() a no-op (frame skipped / minimized).
  void set_current_frame(VkCommandBuffer cmd, VkExtent2D extent) {
    m_current_cmd = cmd;
    m_current_extent = extent;
  }

  bool Init() override;       // build font texture + pipeline
  void Shutdown() override;
  void BeginFrame() override {}
  void Render() override;     // record ImGui draw data into the current cmd buf

 private:
  bool create_font_texture();
  bool create_pipeline();
  bool ensure_buffers(VkDeviceSize vtx_size, VkDeviceSize idx_size);

  vulkan_imgui_context m_ctx{};
  bool m_ready = false;

  VkCommandBuffer m_current_cmd = VK_NULL_HANDLE;
  VkExtent2D m_current_extent{0, 0};

  VkSampler m_font_sampler = VK_NULL_HANDLE;
  VkImage m_font_image = VK_NULL_HANDLE;
  VkDeviceMemory m_font_memory = VK_NULL_HANDLE;
  VkImageView m_font_view = VK_NULL_HANDLE;

  VkDescriptorSetLayout m_set_layout = VK_NULL_HANDLE;
  VkDescriptorPool m_descriptor_pool = VK_NULL_HANDLE;
  VkDescriptorSet m_descriptor_set = VK_NULL_HANDLE;

  VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;
  VkPipeline m_pipeline = VK_NULL_HANDLE;

  // One vertex/index buffer per frame in flight, grown on demand.
  static constexpr size_t kFrames = 2;
  struct frame_buffers {
    VkBuffer vertex = VK_NULL_HANDLE;
    VkDeviceMemory vertex_memory = VK_NULL_HANDLE;
    VkDeviceSize vertex_size = 0;
    VkBuffer index = VK_NULL_HANDLE;
    VkDeviceMemory index_memory = VK_NULL_HANDLE;
    VkDeviceSize index_size = 0;
  };
  std::array<frame_buffers, kFrames> m_buffers{};
  size_t m_buffer_index = 0;
};
