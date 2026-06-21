/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#pragma once

// Native Vulkan renderer for EU07, built directly on the Vulkan C API.
//
// This is the ground-up replacement for the retired NVRHI-based renderer.
// It is being grown in stages; at this stage it implements the full
// gfx_renderer interface (geometry/material/texture management mirror the
// null_renderer placeholder) and a real Vulkan bring-up that clears and
// presents the swap chain every frame.

#include <vulkan/vulkan.h>

#include <optional>
#include <string>
#include <vector>

#include "model/Texture.h"
#include "model/material.h"
#include "rendering/geometrybank.h"
#include "rendering/renderer.h"

struct GLFWwindow;
class vulkan_imgui_renderer;

// Shared, renderer-owned state the geometry banks need: the device (for buffer
// allocation in create_) and the command buffer currently being recorded (for
// draw_). The renderer sets current_cmd while recording the scene pass.
struct vulkan_geometry_context {
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkCommandBuffer current_cmd = VK_NULL_HANDLE;
  // Pipelines the bank binds per chunk depending on primitive type (both share
  // the renderer's pipeline layout, so the pushed MVP stays valid).
  VkPipeline pipeline_triangles = VK_NULL_HANDLE;
  VkPipeline pipeline_strips = VK_NULL_HANDLE;
};

// GPU-backed geometry bank: uploads each chunk's basic_vertex/index data to
// VkBuffers in create_, and records bind+draw into the current command buffer
// in draw_.
class vulkan_geometrybank : public gfx::geometry_bank {
 public:
  explicit vulkan_geometrybank(const vulkan_geometry_context *ctx)
      : m_ctx(ctx) {}
  ~vulkan_geometrybank() override;

 private:
  void create_(gfx::geometry_handle const &Geometry) override;
  void replace_(gfx::geometry_handle const &Geometry) override;
  std::size_t draw_(gfx::geometry_handle const &Geometry,
                    gfx::stream_units const &Units,
                    unsigned int const Streams) override;
  void release_() override {}

  void upload(gfx::geometry_handle const &Geometry);

  struct gpu_chunk {
    VkBuffer vbuf = VK_NULL_HANDLE;
    VkDeviceMemory vmem = VK_NULL_HANDLE;
    VkBuffer ibuf = VK_NULL_HANDLE;
    VkDeviceMemory imem = VK_NULL_HANDLE;
    uint32_t vertex_count = 0;
    uint32_t index_count = 0;
    unsigned int type = 0;
  };

  const vulkan_geometry_context *m_ctx = nullptr;
  std::vector<gpu_chunk> m_gpu;  // indexed by chunk-1
};

class vulkan_renderer : public gfx_renderer {
 public:
  vulkan_renderer() = default;
  ~vulkan_renderer() override;

  // --- core ---------------------------------------------------------------
  bool Init(GLFWwindow *Window) override;
  bool AddViewport(const global_settings::extraviewport_config &conf) override {
    return false;
  }
  void Shutdown() override;
  bool Render() override;
  void SwapBuffers() override;
  float Framerate() override { return m_framerate; }

  // --- geometry -----------------------------------------------------------
  gfx::geometrybank_handle Create_Bank() override {
    return m_geometry.register_bank(
        std::make_unique<vulkan_geometrybank>(&m_geo_ctx));
  }
  gfx::geometry_handle Insert(gfx::index_array &Indices,
                              gfx::vertex_array &Vertices,
                              gfx::userdata_array &Userdata,
                              gfx::geometrybank_handle const &Geometry,
                              int const Type) override {
    return m_geometry.create_chunk(Indices, Vertices, Userdata, Geometry, Type);
  }
  gfx::geometry_handle Insert(gfx::vertex_array &Vertices,
                              gfx::userdata_array &Userdata,
                              gfx::geometrybank_handle const &Geometry,
                              int const Type) override {
    gfx::calculate_tangents(Vertices, gfx::index_array(), Type);
    return m_geometry.create_chunk(Vertices, Userdata, Geometry, Type);
  }
  bool Replace(gfx::vertex_array &Vertices, gfx::userdata_array &Userdata,
               gfx::geometry_handle const &Geometry, int const Type,
               const std::size_t Offset = 0) override {
    gfx::calculate_tangents(Vertices, gfx::index_array(), Type);
    return m_geometry.replace(Vertices, Userdata, Geometry, Offset);
  }
  bool Append(gfx::vertex_array &Vertices, gfx::userdata_array &Userdata,
              gfx::geometry_handle const &Geometry, int const Type) override {
    gfx::calculate_tangents(Vertices, gfx::index_array(), Type);
    return m_geometry.append(Vertices, Userdata, Geometry);
  }
  gfx::index_array const &Indices(
      gfx::geometry_handle const &Geometry) const override {
    return m_geometry.indices(Geometry);
  }
  gfx::vertex_array const &Vertices(
      gfx::geometry_handle const &Geometry) const override {
    return m_geometry.vertices(Geometry);
  }
  gfx::userdata_array const &UserData(
      gfx::geometry_handle const &Geometry) const override {
    return m_geometry.userdata(Geometry);
  }

  // --- materials ----------------------------------------------------------
  material_handle Fetch_Material(std::string const &Filename,
                                 bool const Loadnow = true) override {
    m_materials.push_back(std::make_shared<opengl_material>());
    m_materials.back()->name = Filename;
    return m_materials.size();
  }
  void Bind_Material(material_handle const Material, TSubModel const *sm = nullptr,
                     lighting_data const *lighting = nullptr) override {}
  // Handle 0 (null_handle) and out-of-range handles must not throw: many call
  // sites do Material(h)->method() without checking h first. Hand back a safe
  // default material instead of indexing past the end.
  IMaterial const *Material(material_handle const Material) const override {
    if (Material == null_handle ||
        static_cast<std::size_t>(Material) > m_materials.size()) {
      return &m_default_material;
    }
    return m_materials[Material - 1].get();
  }

  // --- shaders ------------------------------------------------------------
  std::shared_ptr<gl::program> Fetch_Shader(std::string const &name) override {
    return nullptr;
  }

  // --- textures -----------------------------------------------------------
  texture_handle Fetch_Texture(std::string const &Filename,
                               bool const Loadnow = true,
                               GLint format_hint = GL_SRGB_ALPHA) override {
    return 0;
  }
  void Bind_Texture(texture_handle const Texture) override {}
  void Bind_Texture(std::size_t const Unit,
                    texture_handle const Texture) override {}
  // Until the sampling pipeline exists, hand back the shared no-op texture so
  // scene loading can query dimensions/state without crashing.
  ITexture &Texture(texture_handle const Texture) override {
    return *ITexture::null_texture();
  }
  ITexture const &Texture(texture_handle const Texture) const override {
    return *ITexture::null_texture();
  }

  // --- picking / camera ---------------------------------------------------
  void Pick_Control_Callback(
      std::function<void(TSubModel const *, const glm::vec2)> Callback)
      override {}
  void Pick_Node_Callback(
      std::function<void(scene::basic_node *)> Callback) override {}
  TSubModel const *Pick_Control() const override { return nullptr; }
  scene::basic_node const *Pick_Node() const override { return nullptr; }
  glm::dvec3 Mouse_Position() const override { return glm::dvec3(); }

  // --- maintenance --------------------------------------------------------
  void Update(double const Deltatime) override;
  void Update_Pick_Control() override {}
  void Update_Pick_Node() override {}
  glm::dvec3 Update_Mouse_Position() override { return glm::dvec3(); }
  bool Debug_Ui_State(std::optional<bool>) override { return false; }

  // --- debug --------------------------------------------------------------
  std::string const &info_times() const override { return m_info_times; }
  std::string const &info_stats() const override { return m_info_stats; }
  imgui_renderer *GetImguiRenderer() override;
  void MakeScreenshot() override {}

 private:
  // Vulkan bring-up (implemented in vulkan_renderer.cpp).
  bool create_instance();
  bool create_surface();
  bool pick_physical_device();
  bool create_device();
  bool create_swapchain();
  void destroy_swapchain();
  bool create_image_views();
  bool create_depth_resources();
  void destroy_depth_resources();
  bool create_world_pipeline(VkPrimitiveTopology topology, VkPipeline &out);
  bool create_test_geometry();
  VkShaderModule create_shader_module(const uint32_t *code, size_t size_bytes);
  bool create_frame_resources();
  void recreate_swapchain();

  struct frame_sync {
    VkSemaphore image_available = VK_NULL_HANDLE;
    VkSemaphore render_finished = VK_NULL_HANDLE;
    VkFence in_flight = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  };

  GLFWwindow *m_window = nullptr;

  VkInstance m_instance = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT m_debug_messenger = VK_NULL_HANDLE;
  VkSurfaceKHR m_surface = VK_NULL_HANDLE;
  VkPhysicalDevice m_physical_device = VK_NULL_HANDLE;
  VkDevice m_device = VK_NULL_HANDLE;

  uint32_t m_graphics_family = UINT32_MAX;
  uint32_t m_present_family = UINT32_MAX;
  VkQueue m_graphics_queue = VK_NULL_HANDLE;
  VkQueue m_present_queue = VK_NULL_HANDLE;

  VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
  VkFormat m_swapchain_format = VK_FORMAT_UNDEFINED;
  VkExtent2D m_swapchain_extent = {0, 0};
  std::vector<VkImage> m_swapchain_images;
  std::vector<VkImageView> m_swapchain_image_views;

  // Depth buffer (recreated with the swap chain).
  VkFormat m_depth_format = VK_FORMAT_D32_SFLOAT;
  VkImage m_depth_image = VK_NULL_HANDLE;
  VkDeviceMemory m_depth_memory = VK_NULL_HANDLE;
  VkImageView m_depth_view = VK_NULL_HANDLE;

  VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;
  VkPipeline m_pipeline_triangles = VK_NULL_HANDLE;
  VkPipeline m_pipeline_strips = VK_NULL_HANDLE;

  // Shared state handed to every geometry bank (device + per-frame cmd buffer).
  vulkan_geometry_context m_geo_ctx;

  // Temporary world-anchored test cube, now routed through a real GPU geometry
  // bank to validate that path. Replaced by scene traversal next.
  gfx::geometry_handle m_test_geometry{};
  glm::dvec3 m_world_anchor{0.0};
  bool m_anchor_set = false;

  VkCommandPool m_command_pool = VK_NULL_HANDLE;
  std::vector<frame_sync> m_frames;
  uint32_t m_frame_index = 0;
  uint32_t m_acquired_image = 0;
  bool m_frame_acquired = false;

  bool m_enable_validation = false;
  bool m_vsync = true;

  static constexpr uint32_t kMaxFramesInFlight = 2;

  // gfx_renderer bookkeeping
  std::unique_ptr<vulkan_imgui_renderer> m_imgui;

  gfx::geometrybank_manager m_geometry;
  std::vector<std::shared_ptr<opengl_material>> m_materials;
  opengl_material m_default_material;  // returned for null/invalid handles
  std::string m_info_times;
  std::string m_info_stats;
  float m_framerate = 60.f;
};
