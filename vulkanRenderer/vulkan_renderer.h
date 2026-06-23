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
#include <set>
#include <string>
#include <vector>

#include "model/Texture.h"
#include "model/material.h"
#include "rendering/geometrybank.h"
#include "rendering/renderer.h"

struct GLFWwindow;
class vulkan_imgui_renderer;
class TSubModel;
class TDynamicObject;
class TAnimModel;

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
  VkPipeline pipeline_fans = VK_NULL_HANDLE;
  // Translucent variants (depth-write off), bound when translucent_mode set.
  VkPipeline translucent_triangles = VK_NULL_HANDLE;
  VkPipeline translucent_strips = VK_NULL_HANDLE;
  VkPipeline translucent_fans = VK_NULL_HANDLE;
  // Pick pipelines (flat ID colour), bound instead when pick_mode is set.
  VkPipeline pick_pipeline_triangles = VK_NULL_HANDLE;
  VkPipeline pick_pipeline_strips = VK_NULL_HANDLE;
  VkPipeline pick_pipeline_fans = VK_NULL_HANDLE;
  // Shadow (sun depth pre-pass) pipelines, bound when shadow_mode is set.
  VkPipeline shadow_pipeline_triangles = VK_NULL_HANDLE;
  VkPipeline shadow_pipeline_strips = VK_NULL_HANDLE;
  VkPipeline shadow_pipeline_fans = VK_NULL_HANDLE;
  // Deferred geometry-pass pipelines (write the G-buffer), bound when
  // gbuffer_mode is set.
  VkPipeline gbuffer_pipeline_triangles = VK_NULL_HANDLE;
  VkPipeline gbuffer_pipeline_strips = VK_NULL_HANDLE;
  VkPipeline gbuffer_pipeline_fans = VK_NULL_HANDLE;
  bool pick_mode = false;
  bool translucent_mode = false;
  bool shadow_mode = false;
  bool gbuffer_mode = false;
  // Within a shadow pass, restrict to the player cab (cab-light shadow map).
  bool cab_only = false;
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

// Lightweight ITexture returned by vulkan_renderer::Texture(). Only the name
// and dimensions matter to the engine (e.g. material_manager::create reads the
// name; an empty name makes it discard the material). Actual sampling goes
// through the descriptor sets, not this object.
class vulkan_itexture : public ITexture {
 public:
  std::string m_name;
  int m_width = 1;
  int m_height = 1;
  bool m_alpha = true;
  std::size_t m_id = 0;

  bool create(bool = false) override { return true; }
  int get_width() const override { return m_width; }
  int get_height() const override { return m_height; }
  std::size_t get_id() const override { return m_id; }
  void release() override {}
  void make_stub() override {}
  std::string_view get_traits() const override { return {}; }
  std::string_view get_name() const override { return m_name; }
  std::string_view get_type() const override { return {}; }
  bool is_stub() const override { return false; }
  bool get_has_alpha() const override { return m_alpha; }
  bool get_is_ready() const override { return true; }
  void set_components_hint(int) override {}
  void make_from_memory(std::size_t, std::size_t, const uint8_t *) override {}
  void update_from_memory(std::size_t, std::size_t, const uint8_t *) override {}
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
  // Use the engine's material manager: it parses .mat files and fills texture
  // handles via Fetch_Texture below. Its GL-only validation is skipped because
  // Global.GfxRenderer == "vulkan".
  material_handle Fetch_Material(std::string const &Filename,
                                 bool const Loadnow = true) override {
    return m_material_manager.create(Filename, Loadnow);
  }
  void Bind_Material(material_handle const Material, TSubModel const *sm = nullptr,
                     lighting_data const *lighting = nullptr) override {}
  IMaterial const *Material(material_handle const Material) const override {
    return &m_material_manager.material(Material);
  }

  // --- shaders ------------------------------------------------------------
  std::shared_ptr<gl::program> Fetch_Shader(std::string const &name) override {
    return nullptr;
  }

  // --- textures -----------------------------------------------------------
  texture_handle Fetch_Texture(std::string const &Filename,
                               bool const Loadnow = true,
                               GLint format_hint = GL_SRGB_ALPHA) override;
  void Bind_Texture(texture_handle const Texture) override {}
  void Bind_Texture(std::size_t const Unit,
                    texture_handle const Texture) override {}
  ITexture &Texture(texture_handle const Texture) override {
    if (Texture != null_handle &&
        static_cast<std::size_t>(Texture) <= m_itextures.size()) {
      return m_itextures[Texture - 1];
    }
    return *ITexture::null_texture();
  }
  ITexture const &Texture(texture_handle const Texture) const override {
    if (Texture != null_handle &&
        static_cast<std::size_t>(Texture) <= m_itextures.size()) {
      return m_itextures[Texture - 1];
    }
    return *ITexture::null_texture();
  }

  // --- picking / camera ---------------------------------------------------
  void Pick_Control_Callback(
      std::function<void(TSubModel const *, const glm::vec2)> Callback)
      override {
    m_pick_callbacks.push_back(std::move(Callback));
  }
  void Pick_Node_Callback(
      std::function<void(scene::basic_node *)> Callback) override {}
  TSubModel const *Pick_Control() const override { return m_pick_control; }
  scene::basic_node const *Pick_Node() const override { return nullptr; }
  glm::dvec3 Mouse_Position() const override { return glm::dvec3(); }

  // --- maintenance --------------------------------------------------------
  void Update(double const Deltatime) override;
  void Update_Pick_Control() override;
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
  bool create_command_pool();
  bool create_default_texture();
  bool create_flat_normal();
  bool create_world_pipeline(VkPrimitiveTopology topology, bool depth_write,
                             VkPipeline &out);
  bool create_light_layout();  // set 1: per-frame light/scene UBO
  bool create_shadow_resources();  // shadow map image/view/sampler
  bool create_shadow_pipeline(VkPrimitiveTopology topology, VkPipeline &out);
  void destroy_shadow();
  // Deferred shading: geometry-pass pipelines (write the G-buffer) + the
  // fullscreen lighting pass that consumes it.
  bool create_gbuffer_pipeline(VkPrimitiveTopology topology, VkPipeline &out);
  bool create_deferred_light_pipeline();
  void destroy_deferred();
  bool create_pick_pipeline(VkPrimitiveTopology topology, VkPipeline &out);
  bool create_pick_resources();
  void destroy_pick_resources();
  // Gradient skydome (vertex-coloured dome from simulation::Environment).
  bool create_sky_pipeline();
  void render_skydome(const glm::mat4 &proj, const glm::mat4 &rot,
                      VkCommandBuffer cmd);
  void destroy_sky();
  void pick_submodel(TSubModel *sm, const glm::mat4 &parent,
                     const glm::mat4 &rot, const glm::mat4 &proj,
                     VkCommandBuffer cmd, uint32_t &index);
  bool create_test_geometry();
  // Recursively draw a model's submodel tree (camera-relative). parent is the
  // accumulated model matrix; rot/proj are the camera rotation and projection;
  // skins points to the model's replacable-skin handles (or null) for
  // submodels with m_material < 0.
  void render_submodel(TSubModel *sm, const glm::mat4 &parent,
                       const glm::mat4 &rot, const glm::mat4 &proj,
                       const material_handle *skins, bool translucent_pass,
                       float interior, VkCommandBuffer cmd,
                       bool instrument = false);
  // Traverses the region + the player's consist for one pass, pushing the
  // camera-relative model matrices. Used by the main colour pass and the sun
  // shadow depth pass (which sets m_geo_ctx.shadow_mode).
  void draw_scene(bool translucent_pass, const glm::dvec3 &campos,
                  const glm::mat4 &rot, const glm::mat4 &proj,
                  const std::set<TDynamicObject *> &consist,
                  TDynamicObject *player, VkCommandBuffer cmd);
  VkShaderModule create_shader_module(const uint32_t *code, size_t size_bytes);
  bool create_frame_resources();
  void recreate_swapchain();

  struct frame_sync {
    VkSemaphore image_available = VK_NULL_HANDLE;
    VkSemaphore render_finished = VK_NULL_HANDLE;
    VkFence in_flight = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    // Per-frame light/scene uniform buffer (set 1) + its descriptor.
    VkBuffer light_ubo = VK_NULL_HANDLE;
    VkDeviceMemory light_ubo_memory = VK_NULL_HANDLE;
    void *light_ubo_mapped = nullptr;
    VkDescriptorSet light_descriptor = VK_NULL_HANDLE;
  };

  // Fills frame's light/scene UBO (set 1) from the sun + simulation::Lights.
  void update_lights(const glm::mat4 &viewproj, const glm::dvec3 &campos,
                     frame_sync &frame);

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

  // Depth buffer (recreated with the swap chain). Sized at the supersampled
  // render extent, not the swapchain extent.
  VkFormat m_depth_format = VK_FORMAT_D32_SFLOAT;
  VkImage m_depth_image = VK_NULL_HANDLE;
  VkDeviceMemory m_depth_memory = VK_NULL_HANDLE;
  VkImageView m_depth_view = VK_NULL_HANDLE;

  // SSAA: the scene renders to this offscreen colour target at m_render_extent
  // (= swapchain extent * m_ssaa_scale per axis), then is blitted/downscaled to
  // the swapchain (the UI is drawn afterwards at native resolution). scale 1.0
  // disables supersampling. 2.0 = 2x2 = 4x samples per pixel.
  float m_ssaa_scale = 2.0f;
  VkExtent2D m_render_extent = {0, 0};
  VkImage m_ssaa_color = VK_NULL_HANDLE;
  VkDeviceMemory m_ssaa_memory = VK_NULL_HANDLE;
  VkImageView m_ssaa_view = VK_NULL_HANDLE;

  // Deferred shading G-buffer (geometry pass writes these at m_render_extent;
  // the fullscreen lighting pass samples them). All recreated with the swapchain.
  //   albedo:   rgb albedo, a emission strength
  //   normal:   rgb world-space normal, a cab-interior flag
  //   position: rgb camera-relative position
  struct gbuffer_target {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
  };
  gbuffer_target m_gbuf_albedo;
  gbuffer_target m_gbuf_normal;
  gbuffer_target m_gbuf_position;
  VkSampler m_gbuffer_sampler = VK_NULL_HANDLE;             // nearest, clamp
  VkDescriptorSetLayout m_gbuffer_set_layout = VK_NULL_HANDLE;  // 3 samplers
  VkDescriptorPool m_gbuffer_pool = VK_NULL_HANDLE;
  VkDescriptorSet m_gbuffer_descriptor = VK_NULL_HANDLE;
  // Geometry-pass pipelines (write the G-buffer) + the fullscreen lighting pass.
  VkPipeline m_gbuffer_pipeline_triangles = VK_NULL_HANDLE;
  VkPipeline m_gbuffer_pipeline_strips = VK_NULL_HANDLE;
  VkPipeline m_gbuffer_pipeline_fans = VK_NULL_HANDLE;
  VkPipelineLayout m_deferred_light_layout = VK_NULL_HANDLE;
  VkPipeline m_deferred_light_pipeline = VK_NULL_HANDLE;

  VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;
  VkPipeline m_pipeline_triangles = VK_NULL_HANDLE;
  VkPipeline m_pipeline_strips = VK_NULL_HANDLE;
  VkPipeline m_pipeline_fans = VK_NULL_HANDLE;
  VkPipeline m_pipeline_triangles_blend = VK_NULL_HANDLE;  // depth-write off
  VkPipeline m_pipeline_strips_blend = VK_NULL_HANDLE;
  VkPipeline m_pipeline_fans_blend = VK_NULL_HANDLE;
  // When false, the whole scene draws in a single opaque pass. When true,
  // opaque first then a separate depth-write-off translucent pass so glass and
  // other translucent surfaces blend over what's behind without occluding it.
  bool m_two_pass_translucency = true;
  // Submodel animation (RaAnimation): gauges, levers, wheels, pantograph...
  bool m_submodel_animations = true;
  // Cab instrument backlight: the submodel (subtree) the switch makes visible
  // (btInstrumentLight.on_submodel()). Only that subtree glows on demand (even in
  // daylight); the rest of the cab keeps the normal darkness-gated emission, so
  // flipping the switch lights the gauges, not the whole cab. Per-frame.
  const TSubModel *m_instrument_submodel = nullptr;

  // Texturing: a shared sampler + descriptor set layout (set 0 = combined
  // image sampler) and a default 1x1 white texture bound when a draw has no
  // material texture. Real per-material textures plug into the same path.
  VkSampler m_sampler = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_texture_set_layout = VK_NULL_HANDLE;
  VkDescriptorPool m_descriptor_pool = VK_NULL_HANDLE;
  // Set 1: per-frame light/scene uniform buffer (sun + dynamic lights) +
  // the sun shadow map (binding 1).
  VkDescriptorSetLayout m_light_set_layout = VK_NULL_HANDLE;
  VkDescriptorPool m_light_pool = VK_NULL_HANDLE;
  // Cascaded sun shadow map: a depth texture array (one layer per cascade),
  // sampled with comparison/PCF. Per-layer views render each cascade; the array
  // view is sampled in the world shader.
  static constexpr uint32_t kShadowCascades = 3;          // sun cascades
  static constexpr uint32_t kCabShadowLayer = 3;          // cab-light layer
  static constexpr uint32_t kShadowLayers = kShadowCascades + 1;
  VkExtent2D m_shadow_extent = {2048, 2048};
  VkImage m_shadow_image = VK_NULL_HANDLE;
  VkDeviceMemory m_shadow_memory = VK_NULL_HANDLE;
  VkImageView m_shadow_layer_views[kShadowLayers] = {};
  VkImageView m_shadow_array_view = VK_NULL_HANDLE;
  VkSampler m_shadow_sampler = VK_NULL_HANDLE;       // comparison (PCF)
  VkSampler m_shadow_sampler_raw = VK_NULL_HANDLE;   // plain depth (PCSS blocker)
  VkPipeline m_pipeline_shadow_triangles = VK_NULL_HANDLE;
  VkPipeline m_pipeline_shadow_strips = VK_NULL_HANDLE;
  VkPipeline m_pipeline_shadow_fans = VK_NULL_HANDLE;
  VkImage m_white_image = VK_NULL_HANDLE;
  VkDeviceMemory m_white_memory = VK_NULL_HANDLE;
  VkImageView m_white_view = VK_NULL_HANDLE;
  VkDescriptorSet m_white_descriptor = VK_NULL_HANDLE;
  // Flat normal/height default (rg=0.5 -> normal (0,0,1), b=0 -> no parallax)
  // bound to set 2 for materials that have no normal map.
  VkImage m_flat_normal_image = VK_NULL_HANDLE;
  VkDeviceMemory m_flat_normal_memory = VK_NULL_HANDLE;
  VkImageView m_flat_normal_view = VK_NULL_HANDLE;
  VkDescriptorSet m_flat_normal_descriptor = VK_NULL_HANDLE;

  // Control picking: an offscreen ID target (recreated with the swap chain),
  // pick pipelines (flat colour), a 1-pixel readback buffer, the queued
  // pick callbacks, the last picked control, and the per-pick submodel table.
  VkPipelineLayout m_pick_layout = VK_NULL_HANDLE;
  VkPipeline m_pick_pipeline_triangles = VK_NULL_HANDLE;
  VkPipeline m_pick_pipeline_strips = VK_NULL_HANDLE;
  VkPipeline m_pick_pipeline_fans = VK_NULL_HANDLE;
  VkFormat m_pick_format = VK_FORMAT_R8G8B8A8_UNORM;
  VkImage m_pick_color_image = VK_NULL_HANDLE;
  VkDeviceMemory m_pick_color_memory = VK_NULL_HANDLE;
  VkImageView m_pick_color_view = VK_NULL_HANDLE;
  VkImage m_pick_depth_image = VK_NULL_HANDLE;
  VkDeviceMemory m_pick_depth_memory = VK_NULL_HANDLE;
  VkImageView m_pick_depth_view = VK_NULL_HANDLE;
  VkBuffer m_pick_readback = VK_NULL_HANDLE;
  VkDeviceMemory m_pick_readback_memory = VK_NULL_HANDLE;
  std::vector<std::function<void(TSubModel const *, const glm::vec2)>>
      m_pick_callbacks;
  TSubModel const *m_pick_control = nullptr;
  std::vector<TSubModel const *> m_pick_submodels;

  // Gradient skydome: a flat-colour pipeline + position/colour/index buffers
  // filled from simulation::Environment.skydome(). Colours are host-visible and
  // re-uploaded when the simulation marks the dome dirty.
  VkPipelineLayout m_sky_layout = VK_NULL_HANDLE;
  VkPipeline m_sky_pipeline = VK_NULL_HANDLE;
  VkBuffer m_sky_vertex = VK_NULL_HANDLE;
  VkDeviceMemory m_sky_vertex_memory = VK_NULL_HANDLE;
  VkBuffer m_sky_color = VK_NULL_HANDLE;
  VkDeviceMemory m_sky_color_memory = VK_NULL_HANDLE;
  VkBuffer m_sky_index = VK_NULL_HANDLE;
  VkDeviceMemory m_sky_index_memory = VK_NULL_HANDLE;
  uint32_t m_sky_index_count = 0;
  bool m_sky_ready = false;

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
  material_manager m_material_manager;

  // GPU textures indexed by (handle - 1); handle 0 (null) maps to the white
  // default. Loaded on demand by Fetch_Texture, deduped by filename.
  struct gpu_texture {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDescriptorSet descriptor = VK_NULL_HANDLE;
  };
  std::vector<gpu_texture> m_textures;
  // Parallel to m_textures (deque keeps element addresses stable across growth,
  // so Texture() can return references safely). Holds the ITexture views.
  std::deque<vulkan_itexture> m_itextures;
  std::unordered_map<std::string, texture_handle> m_texture_map;
  // Returns the descriptor set for a material's diffuse texture, or the white
  // default if there is none.
  VkDescriptorSet material_texture_descriptor(material_handle material) const;
  // Normal/height map descriptor for a material (set 2), or the flat-normal
  // default when the material has no normal map.
  VkDescriptorSet material_normal_descriptor(material_handle material) const;
  // True when the material uses a parallax shader (declares a height_scale
  // param), i.e. its normal map's blue channel is a height map -> safe to POM.
  bool material_has_parallax(material_handle material) const;
  // Binds a material's texture (set 0) and pushes its opacity for the frame's
  // draw that follows.
  void bind_material(material_handle material, VkCommandBuffer cmd);
  std::string m_info_times;
  std::string m_info_stats;
  float m_framerate = 60.f;
};
