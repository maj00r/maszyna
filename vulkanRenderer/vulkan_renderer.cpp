/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// Vulkan must be visible before stdafx.h pulls in <GLFW/glfw3.h> (which is
// included there with GLFW_INCLUDE_NONE); otherwise GLFW won't declare its
// Vulkan helpers such as glfwCreateWindowSurface.
#include <vulkan/vulkan.h>

#include "stdafx.h"

#include "vulkan_renderer.h"
#include "vulkan_imgui.h"

#include <algorithm>
#include <cstring>
#include <set>

#include <fstream>

#include "stb/stb_image.h"  // declarations; implemented in engine's stb_image.c

#include "application/application.h"
#include "model/AnimModel.h"  // TAnimModel
#include "model/Model3d.h"
#include "model/Texture.h"  // texture_manager::find_on_disk
#include "rendering/lightarray.h"  // light_array (dynamic light sources)
#include "scene/scene.h"
#include "simulation/simulationenvironment.h"  // simulation::Environment, CSkyDome
#include "utilities/Globals.h"
#include "utilities/Logs.h"
#include "vehicle/Train.h"
#include "world/Track.h"

// Defined in the engine; forward-declared here to avoid pulling simulation.h's
// heavy transitive includes into this TU.
namespace simulation {
extern scene::basic_region *Region;
extern TTrain *Train;
extern light_array Lights;
}  // namespace simulation

// SPIR-V embedded by the build (glslangValidator --vn). Each header defines a
// `const uint32_t <name>[]` array.
#include "world_vert_spv.h"
#include "world_frag_spv.h"
#include "pick_vert_spv.h"
#include "pick_frag_spv.h"
#include "skydome_vert_spv.h"
#include "skydome_frag_spv.h"
#include "shadow_vert_spv.h"
#include "gbuffer_frag_spv.h"
#include "deferred_light_vert_spv.h"
#include "deferred_light_frag_spv.h"

namespace {

void log_info(const std::string &msg) {
  WriteLog("vulkan_renderer: " + msg);
}
void log_error(const std::string &msg) {
  ErrorLog("vulkan_renderer: " + msg);
}

const char *vk_result_str(VkResult r) {
  switch (r) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
    case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
    default: return "VK_ERROR_<other>";
  }
}

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*types*/,
    const VkDebugUtilsMessengerCallbackDataEXT *data, void * /*user*/) {
  if (data == nullptr || data->pMessage == nullptr) return VK_FALSE;
  if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    log_error(std::string("[validation] ") + data->pMessage);
  } else {
    log_info(std::string("[validation] ") + data->pMessage);
  }
  return VK_FALSE;
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

}  // namespace

#define VK_CHECK(expr)                                                       \
  do {                                                                       \
    const VkResult _res = (expr);                                            \
    if (_res != VK_SUCCESS) {                                                \
      log_error(std::string("call failed (") + vk_result_str(_res) +         \
                "): " #expr);                                                \
      return false;                                                          \
    }                                                                        \
  } while (0)

imgui_renderer *vulkan_renderer::GetImguiRenderer() { return m_imgui.get(); }

// ---------------------------------------------------------------------------
// Factory entry point (registered from register.cpp, compiled into eu07)
// ---------------------------------------------------------------------------

std::unique_ptr<gfx_renderer> create_vulkan_renderer() {
  return std::unique_ptr<vulkan_renderer>(new vulkan_renderer());
}

// ---------------------------------------------------------------------------
// Lifetime
// ---------------------------------------------------------------------------

vulkan_renderer::~vulkan_renderer() { Shutdown(); }

bool vulkan_renderer::Init(GLFWwindow *Window) {
  m_window = Window;
  m_enable_validation = Global.gfx_gldebug;
  m_vsync = Global.VSync;
  m_imgui = std::make_unique<vulkan_imgui_renderer>();

  if (!glfwVulkanSupported()) {
    log_error("GLFW reports Vulkan is not supported on this system.");
    return false;
  }

  if (!create_instance()) return false;
  if (!create_surface()) return false;
  if (!pick_physical_device()) return false;
  if (!create_device()) return false;
  if (!create_swapchain()) return false;
  if (!create_command_pool()) return false;
  if (!create_default_texture()) return false;  // also creates the set layout
  if (!create_flat_normal()) return false;      // set-2 default (no normal map)
  if (!create_bindless()) return false;         // big texture array (set 0)
  if (!create_light_layout()) return false;  // set 1 (light/scene UBO + shadow)
  if (!create_shadow_resources()) return false;  // sun shadow map
  if (!create_world_pipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, true,
                             m_pipeline_triangles))
    return false;
  if (!create_world_pipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, true,
                             m_pipeline_strips))
    return false;
  if (!create_world_pipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN, true,
                             m_pipeline_fans))
    return false;
  if (!create_world_pipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, false,
                             m_pipeline_triangles_blend))
    return false;
  if (!create_world_pipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, false,
                             m_pipeline_strips_blend))
    return false;
  if (!create_world_pipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN, false,
                             m_pipeline_fans_blend))
    return false;
  if (!create_pick_pipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                            m_pick_pipeline_triangles))
    return false;
  if (!create_pick_pipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
                            m_pick_pipeline_strips))
    return false;
  if (!create_pick_pipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,
                            m_pick_pipeline_fans))
    return false;
  if (!create_sky_pipeline()) return false;
  // Shadow pipelines share the world pipeline layout (created above).
  if (!create_shadow_pipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                              m_pipeline_shadow_triangles))
    return false;
  if (!create_shadow_pipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
                              m_pipeline_shadow_strips))
    return false;
  if (!create_shadow_pipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,
                              m_pipeline_shadow_fans))
    return false;
  // Deferred geometry-pass + lighting pipelines.
  if (!create_gbuffer_pipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                               m_gbuffer_pipeline_triangles))
    return false;
  if (!create_gbuffer_pipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
                               m_gbuffer_pipeline_strips))
    return false;
  if (!create_gbuffer_pipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,
                               m_gbuffer_pipeline_fans))
    return false;
  if (!create_deferred_light_pipeline()) return false;
  m_geo_ctx.physical_device = m_physical_device;
  m_geo_ctx.device = m_device;
  m_geo_ctx.pipeline_triangles = m_pipeline_triangles;
  m_geo_ctx.pipeline_strips = m_pipeline_strips;
  m_geo_ctx.pipeline_fans = m_pipeline_fans;
  m_geo_ctx.translucent_triangles = m_pipeline_triangles_blend;
  m_geo_ctx.translucent_strips = m_pipeline_strips_blend;
  m_geo_ctx.translucent_fans = m_pipeline_fans_blend;
  m_geo_ctx.pick_pipeline_triangles = m_pick_pipeline_triangles;
  m_geo_ctx.pick_pipeline_strips = m_pick_pipeline_strips;
  m_geo_ctx.pick_pipeline_fans = m_pick_pipeline_fans;
  m_geo_ctx.shadow_pipeline_triangles = m_pipeline_shadow_triangles;
  m_geo_ctx.shadow_pipeline_strips = m_pipeline_shadow_strips;
  m_geo_ctx.shadow_pipeline_fans = m_pipeline_shadow_fans;
  m_geo_ctx.gbuffer_pipeline_triangles = m_gbuffer_pipeline_triangles;
  m_geo_ctx.gbuffer_pipeline_strips = m_gbuffer_pipeline_strips;
  m_geo_ctx.gbuffer_pipeline_fans = m_gbuffer_pipeline_fans;
  {
    // 1-pixel readback buffer for control picking.
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = 4;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(m_device, &bi, nullptr, &m_pick_readback));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(m_device, m_pick_readback, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_memory_type(
        m_physical_device, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(m_device, &ai, nullptr, &m_pick_readback_memory));
    vkBindBufferMemory(m_device, m_pick_readback, m_pick_readback_memory, 0);
  }
  if (!create_test_geometry()) return false;
  if (!create_frame_resources()) return false;

  // Hand the ImGui backend everything it needs to build its font texture and
  // pipeline when the UI layer later calls its Init().
  vulkan_imgui_context imgui_ctx{};
  imgui_ctx.physical_device = m_physical_device;
  imgui_ctx.device = m_device;
  imgui_ctx.queue = m_graphics_queue;
  imgui_ctx.command_pool = m_command_pool;
  imgui_ctx.color_format = m_swapchain_format;
  m_imgui->set_context(imgui_ctx);

  log_info("initialized.");
  return true;
}

void vulkan_renderer::Shutdown() {
  if (m_device != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(m_device);

    // Destroy ImGui GPU resources while the device is still alive.
    if (m_imgui) m_imgui->Shutdown();

    for (auto &f : m_frames) {
      if (f.image_available) vkDestroySemaphore(m_device, f.image_available, nullptr);
      if (f.render_finished) vkDestroySemaphore(m_device, f.render_finished, nullptr);
      if (f.in_flight) vkDestroyFence(m_device, f.in_flight, nullptr);
      if (f.light_ubo) vkDestroyBuffer(m_device, f.light_ubo, nullptr);
      if (f.light_ubo_memory) vkFreeMemory(m_device, f.light_ubo_memory, nullptr);
      if (f.instance_buffer)
        vkDestroyBuffer(m_device, f.instance_buffer, nullptr);
      if (f.instance_memory)
        vkFreeMemory(m_device, f.instance_memory, nullptr);
    }
    m_frames.clear();
    if (m_light_pool) vkDestroyDescriptorPool(m_device, m_light_pool, nullptr);
    if (m_light_set_layout)
      vkDestroyDescriptorSetLayout(m_device, m_light_set_layout, nullptr);
    m_light_pool = VK_NULL_HANDLE;
    m_light_set_layout = VK_NULL_HANDLE;

    if (m_command_pool) {
      vkDestroyCommandPool(m_device, m_command_pool, nullptr);
      m_command_pool = VK_NULL_HANDLE;
    }

    if (m_pipeline_triangles) {
      vkDestroyPipeline(m_device, m_pipeline_triangles, nullptr);
      m_pipeline_triangles = VK_NULL_HANDLE;
    }
    if (m_pipeline_strips) {
      vkDestroyPipeline(m_device, m_pipeline_strips, nullptr);
      m_pipeline_strips = VK_NULL_HANDLE;
    }
    if (m_pipeline_triangles_blend) {
      vkDestroyPipeline(m_device, m_pipeline_triangles_blend, nullptr);
      m_pipeline_triangles_blend = VK_NULL_HANDLE;
    }
    if (m_pipeline_strips_blend) {
      vkDestroyPipeline(m_device, m_pipeline_strips_blend, nullptr);
      m_pipeline_strips_blend = VK_NULL_HANDLE;
    }
    if (m_pipeline_fans) {
      vkDestroyPipeline(m_device, m_pipeline_fans, nullptr);
      m_pipeline_fans = VK_NULL_HANDLE;
    }
    if (m_pipeline_fans_blend) {
      vkDestroyPipeline(m_device, m_pipeline_fans_blend, nullptr);
      m_pipeline_fans_blend = VK_NULL_HANDLE;
    }
    destroy_sky();
    destroy_shadow();
    destroy_deferred();
    if (m_pipeline_layout) {
      vkDestroyPipelineLayout(m_device, m_pipeline_layout, nullptr);
      m_pipeline_layout = VK_NULL_HANDLE;
    }
    if (m_pick_pipeline_triangles)
      vkDestroyPipeline(m_device, m_pick_pipeline_triangles, nullptr);
    if (m_pick_pipeline_strips)
      vkDestroyPipeline(m_device, m_pick_pipeline_strips, nullptr);
    if (m_pick_pipeline_fans)
      vkDestroyPipeline(m_device, m_pick_pipeline_fans, nullptr);
    if (m_pick_layout)
      vkDestroyPipelineLayout(m_device, m_pick_layout, nullptr);
    if (m_pick_readback) vkDestroyBuffer(m_device, m_pick_readback, nullptr);
    if (m_pick_readback_memory)
      vkFreeMemory(m_device, m_pick_readback_memory, nullptr);
    m_pick_pipeline_triangles = VK_NULL_HANDLE;
    m_pick_pipeline_strips = VK_NULL_HANDLE;
    m_pick_pipeline_fans = VK_NULL_HANDLE;
    m_pick_layout = VK_NULL_HANDLE;
    m_pick_readback = VK_NULL_HANDLE;
    m_pick_readback_memory = VK_NULL_HANDLE;
    if (m_descriptor_pool) {
      vkDestroyDescriptorPool(m_device, m_descriptor_pool, nullptr);
      m_descriptor_pool = VK_NULL_HANDLE;
    }
    if (m_texture_set_layout) {
      vkDestroyDescriptorSetLayout(m_device, m_texture_set_layout, nullptr);
      m_texture_set_layout = VK_NULL_HANDLE;
    }
    if (m_bindless_pool) {
      vkDestroyDescriptorPool(m_device, m_bindless_pool, nullptr);
      m_bindless_pool = VK_NULL_HANDLE;
    }
    if (m_bindless_layout) {
      vkDestroyDescriptorSetLayout(m_device, m_bindless_layout, nullptr);
      m_bindless_layout = VK_NULL_HANDLE;
    }
    if (m_sampler) {
      vkDestroySampler(m_device, m_sampler, nullptr);
      m_sampler = VK_NULL_HANDLE;
    }
    if (m_white_view) {
      vkDestroyImageView(m_device, m_white_view, nullptr);
      m_white_view = VK_NULL_HANDLE;
    }
    if (m_white_image) {
      vkDestroyImage(m_device, m_white_image, nullptr);
      m_white_image = VK_NULL_HANDLE;
    }
    if (m_white_memory) {
      vkFreeMemory(m_device, m_white_memory, nullptr);
      m_white_memory = VK_NULL_HANDLE;
    }
    if (m_flat_normal_view) {
      vkDestroyImageView(m_device, m_flat_normal_view, nullptr);
      m_flat_normal_view = VK_NULL_HANDLE;
    }
    if (m_flat_normal_image) {
      vkDestroyImage(m_device, m_flat_normal_image, nullptr);
      m_flat_normal_image = VK_NULL_HANDLE;
    }
    if (m_flat_normal_memory) {
      vkFreeMemory(m_device, m_flat_normal_memory, nullptr);
      m_flat_normal_memory = VK_NULL_HANDLE;
    }

    destroy_swapchain();

    vkDestroyDevice(m_device, nullptr);
    m_device = VK_NULL_HANDLE;
    // Geometry banks outlive Shutdown (destroyed with m_geometry); null the
    // shared device so their destructors skip freeing on a dead device.
    m_geo_ctx.device = VK_NULL_HANDLE;
  }

  if (m_surface) {
    vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    m_surface = VK_NULL_HANDLE;
  }

  if (m_debug_messenger) {
    auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT"));
    if (destroy) destroy(m_instance, m_debug_messenger, nullptr);
    m_debug_messenger = VK_NULL_HANDLE;
  }

  if (m_instance) {
    vkDestroyInstance(m_instance, nullptr);
    m_instance = VK_NULL_HANDLE;
  }
}

// ---------------------------------------------------------------------------
// Instance
// ---------------------------------------------------------------------------

bool vulkan_renderer::create_instance() {
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "EU07";
  app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app.pEngineName = "MaSzyna";
  app.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app.apiVersion = VK_API_VERSION_1_3;

  uint32_t glfw_count = 0;
  const char **glfw_ext = glfwGetRequiredInstanceExtensions(&glfw_count);
  if (glfw_ext == nullptr) {
    log_error("glfwGetRequiredInstanceExtensions returned null.");
    return false;
  }
  std::vector<const char *> extensions(glfw_ext, glfw_ext + glfw_count);

  std::vector<const char *> layers;
  if (m_enable_validation) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    layers.push_back("VK_LAYER_KHRONOS_validation");
  }

  VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ci.pApplicationInfo = &app;
  ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  ci.ppEnabledExtensionNames = extensions.data();
  ci.enabledLayerCount = static_cast<uint32_t>(layers.size());
  ci.ppEnabledLayerNames = layers.data();

  VK_CHECK(vkCreateInstance(&ci, nullptr, &m_instance));

  if (m_enable_validation) {
    auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT"));
    if (create) {
      VkDebugUtilsMessengerCreateInfoEXT dci{
          VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
      dci.messageSeverity =
          VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
          VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
          VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
      dci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
      dci.pfnUserCallback = debug_callback;
      create(m_instance, &dci, nullptr, &m_debug_messenger);
      log_info("validation layer enabled.");
    } else {
      log_error("validation requested but debug-utils messenger unavailable.");
    }
  }

  return true;
}

// ---------------------------------------------------------------------------
// Surface / device
// ---------------------------------------------------------------------------

bool vulkan_renderer::create_surface() {
  VK_CHECK(glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface));
  return true;
}

bool vulkan_renderer::pick_physical_device() {
  uint32_t count = 0;
  vkEnumeratePhysicalDevices(m_instance, &count, nullptr);
  if (count == 0) {
    log_error("no Vulkan physical devices found.");
    return false;
  }
  std::vector<VkPhysicalDevice> devices(count);
  vkEnumeratePhysicalDevices(m_instance, &count, devices.data());

  auto find_families = [&](VkPhysicalDevice dev, uint32_t &graphics,
                           uint32_t &present) -> bool {
    graphics = UINT32_MAX;
    present = UINT32_MAX;
    uint32_t qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, nullptr);
    std::vector<VkQueueFamilyProperties> props(qcount);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, props.data());
    for (uint32_t i = 0; i < qcount; ++i) {
      if (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        if (graphics == UINT32_MAX) graphics = i;
      }
      VkBool32 supports_present = VK_FALSE;
      vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, m_surface, &supports_present);
      if (supports_present && present == UINT32_MAX) present = i;
    }
    return graphics != UINT32_MAX && present != UINT32_MAX;
  };

  VkPhysicalDevice fallback = VK_NULL_HANDLE;
  uint32_t fb_graphics = UINT32_MAX, fb_present = UINT32_MAX;
  for (VkPhysicalDevice dev : devices) {
    uint32_t g, p;
    if (!find_families(dev, g, p)) continue;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(dev, &props);

    if (fallback == VK_NULL_HANDLE) {
      fallback = dev;
      fb_graphics = g;
      fb_present = p;
    }
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      m_physical_device = dev;
      m_graphics_family = g;
      m_present_family = p;
      log_info(std::string("selected discrete GPU: ") + props.deviceName);
      return true;
    }
  }

  if (fallback != VK_NULL_HANDLE) {
    m_physical_device = fallback;
    m_graphics_family = fb_graphics;
    m_present_family = fb_present;
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(fallback, &props);
    log_info(std::string("selected GPU: ") + props.deviceName);
    return true;
  }

  log_error("no GPU with graphics+present queues supporting the surface.");
  return false;
}

bool vulkan_renderer::create_device() {
  std::set<uint32_t> unique_families = {m_graphics_family, m_present_family};
  float priority = 1.f;
  std::vector<VkDeviceQueueCreateInfo> queue_cis;
  for (uint32_t family : unique_families) {
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    queue_cis.push_back(qci);
  }

  const char *device_extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  VkPhysicalDeviceFeatures features{};
  features.samplerAnisotropy = VK_TRUE;  // anisotropic texture filtering

  // Vulkan 1.2 descriptor indexing for bindless textures: one large, partially
  // bound sampler array updated after bind, indexed per draw by a push constant.
  VkPhysicalDeviceVulkan12Features features12{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  features12.descriptorIndexing = VK_TRUE;
  features12.runtimeDescriptorArray = VK_TRUE;
  features12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
  features12.descriptorBindingPartiallyBound = VK_TRUE;
  features12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;

  // Vulkan 1.3 dynamic rendering lets us render straight to swap-chain image
  // views without VkRenderPass/VkFramebuffer objects.
  VkPhysicalDeviceVulkan13Features features13{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  features13.dynamicRendering = VK_TRUE;
  features13.synchronization2 = VK_TRUE;
  features13.pNext = &features12;

  VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  ci.pNext = &features13;
  ci.queueCreateInfoCount = static_cast<uint32_t>(queue_cis.size());
  ci.pQueueCreateInfos = queue_cis.data();
  ci.enabledExtensionCount = 1;
  ci.ppEnabledExtensionNames = device_extensions;
  ci.pEnabledFeatures = &features;

  VK_CHECK(vkCreateDevice(m_physical_device, &ci, nullptr, &m_device));

  vkGetDeviceQueue(m_device, m_graphics_family, 0, &m_graphics_queue);
  vkGetDeviceQueue(m_device, m_present_family, 0, &m_present_queue);
  return true;
}

// ---------------------------------------------------------------------------
// Swap chain
// ---------------------------------------------------------------------------

bool vulkan_renderer::create_swapchain() {
  VkSurfaceCapabilitiesKHR caps;
  VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physical_device,
                                                     m_surface, &caps));

  uint32_t fmt_count = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(m_physical_device, m_surface, &fmt_count,
                                       nullptr);
  std::vector<VkSurfaceFormatKHR> formats(fmt_count);
  vkGetPhysicalDeviceSurfaceFormatsKHR(m_physical_device, m_surface, &fmt_count,
                                       formats.data());
  if (formats.empty()) {
    log_error("surface reports no formats.");
    return false;
  }

  VkSurfaceFormatKHR chosen = formats[0];
  for (const auto &f : formats) {
    if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
        f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      chosen = f;
      break;
    }
  }
  m_swapchain_format = chosen.format;

  uint32_t pm_count = 0;
  vkGetPhysicalDeviceSurfacePresentModesKHR(m_physical_device, m_surface,
                                            &pm_count, nullptr);
  std::vector<VkPresentModeKHR> present_modes(pm_count);
  vkGetPhysicalDeviceSurfacePresentModesKHR(m_physical_device, m_surface,
                                            &pm_count, present_modes.data());

  // FIFO is always available; prefer mailbox when vsync is off.
  VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
  if (!m_vsync) {
    for (auto pm : present_modes) {
      if (pm == VK_PRESENT_MODE_MAILBOX_KHR) {
        present_mode = pm;
        break;
      }
      if (pm == VK_PRESENT_MODE_IMMEDIATE_KHR) present_mode = pm;
    }
  }

  if (caps.currentExtent.width != UINT32_MAX) {
    m_swapchain_extent = caps.currentExtent;
  } else {
    int w = 0, h = 0;
    glfwGetFramebufferSize(m_window, &w, &h);
    m_swapchain_extent.width =
        std::clamp(static_cast<uint32_t>(w), caps.minImageExtent.width,
                   caps.maxImageExtent.width);
    m_swapchain_extent.height =
        std::clamp(static_cast<uint32_t>(h), caps.minImageExtent.height,
                   caps.maxImageExtent.height);
  }

  if (m_swapchain_extent.width == 0 || m_swapchain_extent.height == 0) {
    // Window is minimized; defer creation.
    return false;
  }

  uint32_t image_count = caps.minImageCount + 1;
  if (caps.maxImageCount > 0 && image_count > caps.maxImageCount) {
    image_count = caps.maxImageCount;
  }

  VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  ci.surface = m_surface;
  ci.minImageCount = image_count;
  ci.imageFormat = chosen.format;
  ci.imageColorSpace = chosen.colorSpace;
  ci.imageExtent = m_swapchain_extent;
  ci.imageArrayLayers = 1;
  ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                  VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  ci.preTransform = caps.currentTransform;
  ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  ci.presentMode = present_mode;
  ci.clipped = VK_TRUE;
  ci.oldSwapchain = VK_NULL_HANDLE;

  uint32_t families[] = {m_graphics_family, m_present_family};
  if (m_graphics_family != m_present_family) {
    ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    ci.queueFamilyIndexCount = 2;
    ci.pQueueFamilyIndices = families;
  } else {
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }

  VK_CHECK(vkCreateSwapchainKHR(m_device, &ci, nullptr, &m_swapchain));

  uint32_t actual = 0;
  vkGetSwapchainImagesKHR(m_device, m_swapchain, &actual, nullptr);
  m_swapchain_images.resize(actual);
  vkGetSwapchainImagesKHR(m_device, m_swapchain, &actual,
                          m_swapchain_images.data());

  log_info("swap chain created: " + std::to_string(m_swapchain_extent.width) +
           "x" + std::to_string(m_swapchain_extent.height) + ", " +
           std::to_string(actual) + " images.");

  if (!create_image_views()) return false;
  if (!create_depth_resources()) return false;
  return create_pick_resources();
}

bool vulkan_renderer::create_image_views() {
  m_swapchain_image_views.resize(m_swapchain_images.size());
  for (size_t i = 0; i < m_swapchain_images.size(); ++i) {
    VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    ci.image = m_swapchain_images[i];
    ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ci.format = m_swapchain_format;
    ci.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                     VK_COMPONENT_SWIZZLE_IDENTITY,
                     VK_COMPONENT_SWIZZLE_IDENTITY};
    ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ci.subresourceRange.levelCount = 1;
    ci.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(m_device, &ci, nullptr,
                               &m_swapchain_image_views[i]));
  }
  return true;
}

void vulkan_renderer::destroy_swapchain() {
  destroy_depth_resources();
  destroy_pick_resources();

  for (VkImageView view : m_swapchain_image_views) {
    if (view) vkDestroyImageView(m_device, view, nullptr);
  }
  m_swapchain_image_views.clear();

  if (m_swapchain) {
    vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
    m_swapchain = VK_NULL_HANDLE;
  }
  m_swapchain_images.clear();
}

VkShaderModule vulkan_renderer::create_shader_module(const uint32_t *code,
                                                     size_t size_bytes) {
  VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  ci.codeSize = size_bytes;
  ci.pCode = code;
  VkShaderModule module = VK_NULL_HANDLE;
  if (vkCreateShaderModule(m_device, &ci, nullptr, &module) != VK_SUCCESS) {
    log_error("vkCreateShaderModule failed.");
    return VK_NULL_HANDLE;
  }
  return module;
}

namespace {
constexpr int kMaxLights = 8;
constexpr int kCascades = 3;  // must match vulkan_renderer::kShadowCascades
// Instanced draws: total mat4 slots in the per-frame instance buffer (slot 0 is
// reserved as identity for the non-instanced path), and the per-batch cap.
constexpr uint32_t kInstanceCapacity = 16384;
constexpr uint32_t kMaxInstancesPerBatch = 256;
// std140-compatible layout shared with world.vert/world.frag (set 1).
struct GpuLight {
  glm::vec4 pos;    // xyz: camera-relative position, w: 1 = spot
  glm::vec4 dir;    // xyz: spot direction, w: range
  glm::vec4 color;  // rgb: colour * intensity, a: cos(outer cone)
  glm::vec4 extra;  // x: cos(inner cone)
};
struct LightUBO {
  glm::mat4 viewproj;
  glm::vec4 sun_dir;
  glm::vec4 sun_color;
  glm::vec4 ambient;
  glm::vec4 interior_light;            // cab interior glow (colour * level)
  glm::mat4 lightspace[kCascades + 1]; // [0..2] sun cascades, [3] cab light
  glm::vec4 cascade_splits;            // .xyz: far distance of each cascade
  glm::vec4 cab_light;                 // xyz: camera-relative pos, w: enable
  glm::vec4 fog;                       // rgb: fog colour, a: 1/range (density)
  glm::ivec4 count;
  GpuLight lights[kMaxLights];
};
}  // namespace

bool vulkan_renderer::create_light_layout() {
  VkDescriptorSetLayoutBinding bindings[4]{};
  bindings[0].binding = 0;  // light/scene UBO
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  bindings[0].descriptorCount = 1;
  bindings[0].stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  bindings[1].binding = 1;  // shadow map (comparison sampler, PCF)
  bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  bindings[1].descriptorCount = 1;
  bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  bindings[2].binding = 2;  // shadow map (plain sampler, PCSS blocker search)
  bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  bindings[2].descriptorCount = 1;
  bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  bindings[3].binding = 3;  // per-instance root matrices (instanced draws)
  bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[3].descriptorCount = 1;
  bindings[3].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  VkDescriptorSetLayoutCreateInfo dlci{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dlci.bindingCount = 4;
  dlci.pBindings = bindings;
  VK_CHECK(
      vkCreateDescriptorSetLayout(m_device, &dlci, nullptr, &m_light_set_layout));

  VkDescriptorPoolSize ps[3] = {
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kMaxFramesInFlight},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxFramesInFlight * 2},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kMaxFramesInFlight}};
  VkDescriptorPoolCreateInfo dpci{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = kMaxFramesInFlight;
  dpci.poolSizeCount = 3;
  dpci.pPoolSizes = ps;
  VK_CHECK(vkCreateDescriptorPool(m_device, &dpci, nullptr, &m_light_pool));
  return true;
}

bool vulkan_renderer::create_shadow_resources() {
  VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ici.imageType = VK_IMAGE_TYPE_2D;
  ici.format = m_depth_format;
  ici.extent = {m_shadow_extent.width, m_shadow_extent.height, 1};
  ici.mipLevels = 1;
  ici.arrayLayers = kShadowLayers;  // sun cascades + cab-light layer
  ici.samples = VK_SAMPLE_COUNT_1_BIT;
  ici.tiling = VK_IMAGE_TILING_OPTIMAL;
  ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
              VK_IMAGE_USAGE_SAMPLED_BIT;
  ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VK_CHECK(vkCreateImage(m_device, &ici, nullptr, &m_shadow_image));
  VkMemoryRequirements req;
  vkGetImageMemoryRequirements(m_device, m_shadow_image, &req);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = find_memory_type(m_physical_device, req.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VK_CHECK(vkAllocateMemory(m_device, &ai, nullptr, &m_shadow_memory));
  vkBindImageMemory(m_device, m_shadow_image, m_shadow_memory, 0);

  // One 2D view per layer (render targets: sun cascades + cab light).
  for (uint32_t i = 0; i < kShadowLayers; ++i) {
    VkImageViewCreateInfo lvci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    lvci.image = m_shadow_image;
    lvci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    lvci.format = m_depth_format;
    lvci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, i, 1};
    VK_CHECK(
        vkCreateImageView(m_device, &lvci, nullptr, &m_shadow_layer_views[i]));
  }
  // Array view (sampled in the world shader as sampler2DArrayShadow).
  VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vci.image = m_shadow_image;
  vci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  vci.format = m_depth_format;
  vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, kShadowLayers};
  VK_CHECK(vkCreateImageView(m_device, &vci, nullptr, &m_shadow_array_view));

  // Comparison sampler -> hardware PCF via sampler2DArrayShadow.
  VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  sci.magFilter = VK_FILTER_LINEAR;
  sci.minFilter = VK_FILTER_LINEAR;
  sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sci.compareEnable = VK_TRUE;
  sci.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
  sci.maxLod = 1.f;
  VK_CHECK(vkCreateSampler(m_device, &sci, nullptr, &m_shadow_sampler));

  // Plain (non-comparison) sampler so the lighting pass can read raw shadow
  // depth for the PCSS blocker search.
  sci.compareEnable = VK_FALSE;
  sci.magFilter = VK_FILTER_NEAREST;
  sci.minFilter = VK_FILTER_NEAREST;
  VK_CHECK(vkCreateSampler(m_device, &sci, nullptr, &m_shadow_sampler_raw));
  return true;
}

void vulkan_renderer::destroy_shadow() {
  if (m_pipeline_shadow_triangles)
    vkDestroyPipeline(m_device, m_pipeline_shadow_triangles, nullptr);
  if (m_pipeline_shadow_strips)
    vkDestroyPipeline(m_device, m_pipeline_shadow_strips, nullptr);
  if (m_pipeline_shadow_fans)
    vkDestroyPipeline(m_device, m_pipeline_shadow_fans, nullptr);
  if (m_shadow_sampler) vkDestroySampler(m_device, m_shadow_sampler, nullptr);
  if (m_shadow_sampler_raw)
    vkDestroySampler(m_device, m_shadow_sampler_raw, nullptr);
  for (auto &v : m_shadow_layer_views)
    if (v) {
      vkDestroyImageView(m_device, v, nullptr);
      v = VK_NULL_HANDLE;
    }
  if (m_shadow_array_view)
    vkDestroyImageView(m_device, m_shadow_array_view, nullptr);
  if (m_shadow_image) vkDestroyImage(m_device, m_shadow_image, nullptr);
  if (m_shadow_memory) vkFreeMemory(m_device, m_shadow_memory, nullptr);
  m_pipeline_shadow_triangles = VK_NULL_HANDLE;
  m_pipeline_shadow_strips = VK_NULL_HANDLE;
  m_pipeline_shadow_fans = VK_NULL_HANDLE;
  m_shadow_sampler = VK_NULL_HANDLE;
  m_shadow_sampler_raw = VK_NULL_HANDLE;
  m_shadow_array_view = VK_NULL_HANDLE;
  m_shadow_image = VK_NULL_HANDLE;
  m_shadow_memory = VK_NULL_HANDLE;
}

void vulkan_renderer::destroy_deferred() {
  VkPipeline pipes[] = {m_gbuffer_pipeline_triangles, m_gbuffer_pipeline_strips,
                        m_gbuffer_pipeline_fans, m_deferred_light_pipeline};
  for (VkPipeline p : pipes)
    if (p) vkDestroyPipeline(m_device, p, nullptr);
  if (m_deferred_light_layout)
    vkDestroyPipelineLayout(m_device, m_deferred_light_layout, nullptr);
  if (m_gbuffer_sampler) vkDestroySampler(m_device, m_gbuffer_sampler, nullptr);
  if (m_gbuffer_pool)
    vkDestroyDescriptorPool(m_device, m_gbuffer_pool, nullptr);
  if (m_gbuffer_set_layout)
    vkDestroyDescriptorSetLayout(m_device, m_gbuffer_set_layout, nullptr);
  m_gbuffer_pipeline_triangles = VK_NULL_HANDLE;
  m_gbuffer_pipeline_strips = VK_NULL_HANDLE;
  m_gbuffer_pipeline_fans = VK_NULL_HANDLE;
  m_deferred_light_pipeline = VK_NULL_HANDLE;
  m_deferred_light_layout = VK_NULL_HANDLE;
  m_gbuffer_sampler = VK_NULL_HANDLE;
  m_gbuffer_pool = VK_NULL_HANDLE;
  m_gbuffer_set_layout = VK_NULL_HANDLE;
}

void vulkan_renderer::update_lights(const glm::mat4 &viewproj,
                                    const glm::dvec3 &campos, frame_sync &frame) {
  LightUBO ubo{};
  ubo.viewproj = viewproj;
  ubo.sun_dir = glm::vec4(Global.DayLight.direction, 0.f);
  ubo.sun_color = Global.DayLight.diffuse;
  ubo.ambient = Global.DayLight.ambient;
  // Distance fog: blend to the horizon colour over the sim's visibility range
  // (fFogEnd shrinks in fog/rain weather, so fog appears then).
  ubo.fog = glm::vec4(glm::vec3(Global.FogColor),
                      std::max(1.f, static_cast<float>(Global.fFogEnd)));
  // Cab interior glow (only applied to cab geometry, via the per-draw flag).
  // Like the GL renderer it fades in as the scene darkens, so it doesn't blow
  // out the (often brightly painted) cab in daylight.
  // It stays on in daylight too (just dimmer), so it reacts alongside the sun
  // rather than only at night; the floor keeps it visible without blowing out
  // the brightly painted cab in full daylight.
  const float cab_darkness =
      glm::clamp(1.25f - static_cast<float>(Global.fLuminance), 0.f, 1.f);
  const float cab_dim = 0.4f + 0.6f * cab_darkness;  // 0.4 (day) .. 1.0 (night)
  float cab_level = 0.f;
  glm::vec3 interior(0.f);
  if (simulation::Train != nullptr) {
    if (TDynamicObject *p = simulation::Train->Dynamic()) {
      cab_level = p->InteriorLightLevel;
      interior = p->InteriorLight * cab_level * cab_dim;
    }
  }
  ubo.interior_light = glm::vec4(interior, 0.f);
  // Instrument backlight: the subtree the switch makes visible. Only it gets
  // on-demand glow (lit gauges) even in daylight; the rest of the cab keeps the
  // darkness-gated emission, so the switch lights the gauges, not the whole cab.
  m_instrument_submodel =
      (simulation::Train != nullptr)
          ? simulation::Train->btInstrumentLight.on_submodel()
          : nullptr;
  // Cascaded sun light-space matrices: nested camera-relative ortho boxes, the
  // near cascade tight (crisp near/cab shadows), the far one wide for coverage.
  {
    glm::vec3 sundir = glm::normalize(glm::vec3(Global.DayLight.direction));
    if (glm::length(sundir) < 0.001f) sundir = glm::vec3(0.f, -1.f, 0.f);
    glm::vec3 up =
        std::abs(sundir.y) > 0.95f ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);
    const float far_dist[kCascades] = {35.f, 130.f, 400.f};
    ubo.cascade_splits =
        glm::vec4(far_dist[0], far_dist[1], far_dist[2], 0.f);
    for (int i = 0; i < kCascades; ++i) {
      const float R = far_dist[i];          // half-extent of this cascade
      const float depth = 2.f * R + 200.f;  // room for tall casters
      glm::mat4 lview =
          glm::lookAtRH(-sundir * (depth * 0.5f), glm::vec3(0.f), up);
      glm::mat4 lproj = glm::orthoRH_ZO(-R, R, -R, R, 0.f, depth);
      lproj[1][1] *= -1.f;  // Vulkan clip Y
      ubo.lightspace[i] = lproj * lview;
    }
  }
  // Cab-light shadow: a perspective depth map from the ceiling lamp above the
  // console (camera-relative, only in cab view). Anchored with the vehicle's own
  // axes (up/front) so it stays put as the player looks around, instead of
  // floating above the eye. The cab light is otherwise a flat fill; this lets
  // cab objects occlude it, i.e. cast shadows from it.
  {
    const bool enable = !FreeFlyModeFlag && cab_level > 0.01f;
    glm::vec3 lamp(0.f, 0.85f, 0.f);  // fallback: above the eye
    glm::vec3 down(0.f, -1.f, 0.f), fwd(0.f, 0.f, 1.f);
    if (simulation::Train != nullptr) {
      if (TDynamicObject *p = simulation::Train->Dynamic()) {
        const glm::vec3 up = glm::vec3(p->VectorUp());
        fwd = glm::vec3(p->VectorFront());
        // Pick the cab end the driver actually occupies.
        if (glm::dot(campos - p->vPosition, p->VectorFront()) < 0.0) fwd = -fwd;
        lamp = up * 0.7f + fwd * 0.55f;  // up to the ceiling, forward to console
        down = -up;
      }
    }
    ubo.cab_light = glm::vec4(lamp, enable ? 1.f : 0.f);
    glm::mat4 lview = glm::lookAtRH(lamp, lamp + down, fwd);
    glm::mat4 lproj =
        glm::perspectiveRH_ZO(glm::radians(150.f), 1.f, 0.05f, 6.f);
    lproj[1][1] *= -1.f;  // Vulkan clip Y
    ubo.lightspace[kCascades] = lproj * lview;  // layer 3
  }
  // Gather active dynamic lights (vehicle head/marker lights), camera-relative.
  int n = 0;
  for (auto const &rec : simulation::Lights.data) {
    if (n >= kMaxLights) break;
    if (rec.intensity <= 0.01f) continue;
    glm::vec3 dir = rec.direction;
    if (glm::length(dir) < 0.001f) dir = glm::vec3(0.f, 0.f, 1.f);
    GpuLight &gl = ubo.lights[n];
    gl.pos = glm::vec4(glm::vec3(rec.position - campos), 1.f);  // spot
    gl.dir = glm::vec4(glm::normalize(dir), 80.f);              // range 80 m
    gl.color = glm::vec4(rec.color * rec.intensity,
                         std::cos(glm::radians(35.f)));  // outer cone
    gl.extra = glm::vec4(std::cos(glm::radians(20.f)), 0.f, 0.f, 0.f);  // inner
    ++n;
  }
  ubo.count = glm::ivec4(n, 0, 0, 0);
  if (frame.light_ubo_mapped != nullptr)
    std::memcpy(frame.light_ubo_mapped, &ubo, sizeof(ubo));
}

bool vulkan_renderer::create_shadow_pipeline(VkPrimitiveTopology topology,
                                             VkPipeline &out) {
  VkShaderModule vert =
      create_shader_module(shadow_vert_spv, sizeof(shadow_vert_spv));
  if (!vert) return false;
  VkPipelineShaderStageCreateInfo stage{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
  stage.module = vert;
  stage.pName = "main";

  VkVertexInputBindingDescription vtx_binding{};
  vtx_binding.stride = sizeof(gfx::basic_vertex);
  vtx_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
  VkVertexInputAttributeDescription vtx_attrs[4]{};
  vtx_attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT,
                  offsetof(gfx::basic_vertex, position)};
  vtx_attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT,
                  offsetof(gfx::basic_vertex, normal)};
  vtx_attrs[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT,
                  offsetof(gfx::basic_vertex, texture)};
  vtx_attrs[3] = {3, 0, VK_FORMAT_R32G32B32A32_SFLOAT,
                  offsetof(gfx::basic_vertex, tangent)};
  VkPipelineVertexInputStateCreateInfo vertex_input{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vertex_input.vertexBindingDescriptionCount = 1;
  vertex_input.pVertexBindingDescriptions = &vtx_binding;
  vertex_input.vertexAttributeDescriptionCount = 4;
  vertex_input.pVertexAttributeDescriptions = vtx_attrs;

  VkPipelineInputAssemblyStateCreateInfo input_assembly{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  input_assembly.topology = topology;

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
  raster.depthBiasEnable = VK_TRUE;
  raster.depthBiasConstantFactor = 1.25f;
  raster.depthBiasSlopeFactor = 1.75f;

  VkPipelineMultisampleStateCreateInfo multisample{
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineColorBlendStateCreateInfo color_blend{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  color_blend.attachmentCount = 0;  // depth only

  VkPipelineDepthStencilStateCreateInfo depth_stencil{
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  depth_stencil.depthTestEnable = VK_TRUE;
  depth_stencil.depthWriteEnable = VK_TRUE;
  depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

  VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamic_state{
      VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dynamic_state.dynamicStateCount = 2;
  dynamic_state.pDynamicStates = dyn;

  VkFormat no_color = VK_FORMAT_UNDEFINED;
  VkPipelineRenderingCreateInfo rendering_ci{
      VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rendering_ci.colorAttachmentCount = 0;
  rendering_ci.pColorAttachmentFormats = &no_color;
  rendering_ci.depthAttachmentFormat = m_depth_format;

  VkGraphicsPipelineCreateInfo pci{
      VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  pci.pNext = &rendering_ci;
  pci.stageCount = 1;
  pci.pStages = &stage;
  pci.pVertexInputState = &vertex_input;
  pci.pInputAssemblyState = &input_assembly;
  pci.pViewportState = &viewport_state;
  pci.pRasterizationState = &raster;
  pci.pMultisampleState = &multisample;
  pci.pColorBlendState = &color_blend;
  pci.pDepthStencilState = &depth_stencil;
  pci.pDynamicState = &dynamic_state;
  pci.layout = m_pipeline_layout;  // shared with the world pipeline
  pci.renderPass = VK_NULL_HANDLE;

  VkResult res = vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pci,
                                           nullptr, &out);
  vkDestroyShaderModule(m_device, vert, nullptr);
  if (res != VK_SUCCESS) {
    log_error("shadow pipeline creation failed.");
    return false;
  }
  return true;
}

bool vulkan_renderer::create_world_pipeline(VkPrimitiveTopology topology,
                                            bool depth_write, VkPipeline &out) {
  VkShaderModule vert =
      create_shader_module(world_vert_spv, sizeof(world_vert_spv));
  VkShaderModule frag =
      create_shader_module(world_frag_spv, sizeof(world_frag_spv));
  if (!vert || !frag) return false;

  VkPipelineShaderStageCreateInfo stages[2] = {
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vert;
  stages[0].pName = "main";
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = frag;
  stages[1].pName = "main";

  VkVertexInputBindingDescription vtx_binding{};
  vtx_binding.binding = 0;
  vtx_binding.stride = sizeof(gfx::basic_vertex);
  vtx_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
  VkVertexInputAttributeDescription vtx_attrs[4]{};
  vtx_attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT,
                  offsetof(gfx::basic_vertex, position)};
  vtx_attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT,
                  offsetof(gfx::basic_vertex, normal)};
  vtx_attrs[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT,
                  offsetof(gfx::basic_vertex, texture)};
  vtx_attrs[3] = {3, 0, VK_FORMAT_R32G32B32A32_SFLOAT,
                  offsetof(gfx::basic_vertex, tangent)};
  VkPipelineVertexInputStateCreateInfo vertex_input{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vertex_input.vertexBindingDescriptionCount = 1;
  vertex_input.pVertexBindingDescriptions = &vtx_binding;
  vertex_input.vertexAttributeDescriptionCount = 4;
  vertex_input.pVertexAttributeDescriptions = vtx_attrs;

  VkPipelineInputAssemblyStateCreateInfo input_assembly{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  input_assembly.topology = topology;

  VkPipelineViewportStateCreateInfo viewport_state{
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  viewport_state.viewportCount = 1;
  viewport_state.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo raster{
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  raster.polygonMode = VK_POLYGON_MODE_FILL;
  // Engine geometry is CCW-front, cull-back (opengl33renderer.cpp:419), and that
  // CCW winding carries through to the framebuffer here. Opaque culls back faces;
  // translucent (depth_write off, e.g. glass) stays double-sided so it shows
  // from both sides.
  raster.cullMode = depth_write ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
  raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  raster.lineWidth = 1.0f;

  VkPipelineMultisampleStateCreateInfo multisample{
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineColorBlendAttachmentState blend_attachment{};
  // Standard alpha blending. Opaque texels (alpha 1) blend to themselves, so
  // this is a no-op for them; translucent texels (glass, shadow decals) blend.
  blend_attachment.blendEnable = VK_TRUE;
  blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
  blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
  blend_attachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  VkPipelineColorBlendStateCreateInfo color_blend{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  color_blend.attachmentCount = 1;
  color_blend.pAttachments = &blend_attachment;

  VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                     VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamic_state{
      VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dynamic_state.dynamicStateCount = 2;
  dynamic_state.pDynamicStates = dynamic_states;

  VkPipelineDepthStencilStateCreateInfo depth_stencil{
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  depth_stencil.depthTestEnable = VK_TRUE;
  depth_stencil.depthWriteEnable = depth_write ? VK_TRUE : VK_FALSE;
  depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

  // Vertex push constant: mat4 local model matrix (offset 0) + misc
  // (.x alpha-test threshold, .y emission, offset 64) = 80 bytes. The
  // view-projection, sun and dynamic lights live in the set-1 UBO.
  VkPushConstantRange pc_range{};
  pc_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  pc_range.offset = 0;
  pc_range.size = sizeof(float) * 24;  // + uGloss + uInstanceBase (@92)

  if (m_pipeline_layout == VK_NULL_HANDLE) {
    // set 0 = bindless texture array, set 1 = light UBO + shadows. Materials are
    // selected by pushed slot indices, not per-material descriptor binds.
    const VkDescriptorSetLayout set_layouts[2] = {m_bindless_layout,
                                                  m_light_set_layout};
    VkPipelineLayoutCreateInfo layout_ci{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_ci.setLayoutCount = 2;
    layout_ci.pSetLayouts = set_layouts;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges = &pc_range;
    if (vkCreatePipelineLayout(m_device, &layout_ci, nullptr,
                               &m_pipeline_layout) != VK_SUCCESS) {
      log_error("vkCreatePipelineLayout failed.");
      return false;
    }
  }

  // Dynamic rendering: declare the color + depth attachment formats instead of
  // a VkRenderPass.
  VkPipelineRenderingCreateInfo rendering_ci{
      VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rendering_ci.colorAttachmentCount = 1;
  rendering_ci.pColorAttachmentFormats = &m_swapchain_format;
  rendering_ci.depthAttachmentFormat = m_depth_format;

  VkGraphicsPipelineCreateInfo pipeline_ci{
      VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  pipeline_ci.pNext = &rendering_ci;
  pipeline_ci.stageCount = 2;
  pipeline_ci.pStages = stages;
  pipeline_ci.pVertexInputState = &vertex_input;
  pipeline_ci.pInputAssemblyState = &input_assembly;
  pipeline_ci.pViewportState = &viewport_state;
  pipeline_ci.pRasterizationState = &raster;
  pipeline_ci.pMultisampleState = &multisample;
  pipeline_ci.pColorBlendState = &color_blend;
  pipeline_ci.pDepthStencilState = &depth_stencil;
  pipeline_ci.pDynamicState = &dynamic_state;
  pipeline_ci.layout = m_pipeline_layout;
  pipeline_ci.renderPass = VK_NULL_HANDLE;

  VkResult res = vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1,
                                           &pipeline_ci, nullptr, &out);

  vkDestroyShaderModule(m_device, vert, nullptr);
  vkDestroyShaderModule(m_device, frag, nullptr);

  if (res != VK_SUCCESS) {
    log_error(std::string("vkCreateGraphicsPipelines failed: ") +
              vk_result_str(res));
    return false;
  }
  return true;
}

bool vulkan_renderer::create_gbuffer_pipeline(VkPrimitiveTopology topology,
                                              VkPipeline &out) {
  VkShaderModule vert =
      create_shader_module(world_vert_spv, sizeof(world_vert_spv));
  VkShaderModule frag =
      create_shader_module(gbuffer_frag_spv, sizeof(gbuffer_frag_spv));
  if (!vert || !frag) return false;

  VkPipelineShaderStageCreateInfo stages[2] = {
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vert;
  stages[0].pName = "main";
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = frag;
  stages[1].pName = "main";

  VkVertexInputBindingDescription vtx_binding{};
  vtx_binding.binding = 0;
  vtx_binding.stride = sizeof(gfx::basic_vertex);
  vtx_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
  VkVertexInputAttributeDescription vtx_attrs[4]{};
  vtx_attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT,
                  offsetof(gfx::basic_vertex, position)};
  vtx_attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT,
                  offsetof(gfx::basic_vertex, normal)};
  vtx_attrs[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT,
                  offsetof(gfx::basic_vertex, texture)};
  vtx_attrs[3] = {3, 0, VK_FORMAT_R32G32B32A32_SFLOAT,
                  offsetof(gfx::basic_vertex, tangent)};
  VkPipelineVertexInputStateCreateInfo vertex_input{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vertex_input.vertexBindingDescriptionCount = 1;
  vertex_input.pVertexBindingDescriptions = &vtx_binding;
  vertex_input.vertexAttributeDescriptionCount = 4;
  vertex_input.pVertexAttributeDescriptions = vtx_attrs;

  VkPipelineInputAssemblyStateCreateInfo input_assembly{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  input_assembly.topology = topology;

  VkPipelineViewportStateCreateInfo viewport_state{
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  viewport_state.viewportCount = 1;
  viewport_state.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo raster{
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  raster.polygonMode = VK_POLYGON_MODE_FILL;
  raster.cullMode = VK_CULL_MODE_BACK_BIT;
  raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  raster.lineWidth = 1.0f;

  VkPipelineMultisampleStateCreateInfo multisample{
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  // Three G-buffer outputs, no blending (raw writes).
  VkPipelineColorBlendAttachmentState ba[3] = {};
  for (int i = 0; i < 3; ++i) {
    ba[i].blendEnable = VK_FALSE;
    ba[i].colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  }
  VkPipelineColorBlendStateCreateInfo color_blend{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  color_blend.attachmentCount = 3;
  color_blend.pAttachments = ba;

  VkPipelineDepthStencilStateCreateInfo depth_stencil{
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  depth_stencil.depthTestEnable = VK_TRUE;
  depth_stencil.depthWriteEnable = VK_TRUE;
  depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS;

  VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamic_state{
      VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dynamic_state.dynamicStateCount = 2;
  dynamic_state.pDynamicStates = dyn;

  const VkFormat color_formats[3] = {
      m_gbuf_albedo.format, m_gbuf_normal.format, m_gbuf_position.format};
  VkPipelineRenderingCreateInfo rendering_ci{
      VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rendering_ci.colorAttachmentCount = 3;
  rendering_ci.pColorAttachmentFormats = color_formats;
  rendering_ci.depthAttachmentFormat = m_depth_format;

  VkGraphicsPipelineCreateInfo pipeline_ci{
      VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  pipeline_ci.pNext = &rendering_ci;
  pipeline_ci.stageCount = 2;
  pipeline_ci.pStages = stages;
  pipeline_ci.pVertexInputState = &vertex_input;
  pipeline_ci.pInputAssemblyState = &input_assembly;
  pipeline_ci.pViewportState = &viewport_state;
  pipeline_ci.pRasterizationState = &raster;
  pipeline_ci.pMultisampleState = &multisample;
  pipeline_ci.pColorBlendState = &color_blend;
  pipeline_ci.pDepthStencilState = &depth_stencil;
  pipeline_ci.pDynamicState = &dynamic_state;
  pipeline_ci.layout = m_pipeline_layout;
  VkResult res = vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1,
                                           &pipeline_ci, nullptr, &out);
  vkDestroyShaderModule(m_device, vert, nullptr);
  vkDestroyShaderModule(m_device, frag, nullptr);
  if (res != VK_SUCCESS) {
    log_error("gbuffer pipeline creation failed.");
    return false;
  }
  return true;
}

bool vulkan_renderer::create_deferred_light_pipeline() {
  VkShaderModule vert = create_shader_module(deferred_light_vert_spv,
                                             sizeof(deferred_light_vert_spv));
  VkShaderModule frag = create_shader_module(deferred_light_frag_spv,
                                             sizeof(deferred_light_frag_spv));
  if (!vert || !frag) return false;

  if (m_deferred_light_layout == VK_NULL_HANDLE) {
    const VkDescriptorSetLayout sets[2] = {m_gbuffer_set_layout,
                                           m_light_set_layout};
    VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    lci.setLayoutCount = 2;
    lci.pSetLayouts = sets;
    VK_CHECK(vkCreatePipelineLayout(m_device, &lci, nullptr,
                                    &m_deferred_light_layout));
  }

  VkPipelineShaderStageCreateInfo stages[2] = {
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vert;
  stages[0].pName = "main";
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = frag;
  stages[1].pName = "main";

  VkPipelineVertexInputStateCreateInfo vertex_input{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};  // none

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

  VkPipelineColorBlendAttachmentState ba{};
  ba.blendEnable = VK_FALSE;
  ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  VkPipelineColorBlendStateCreateInfo color_blend{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  color_blend.attachmentCount = 1;
  color_blend.pAttachments = &ba;

  VkPipelineDepthStencilStateCreateInfo depth_stencil{
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  depth_stencil.depthTestEnable = VK_FALSE;
  depth_stencil.depthWriteEnable = VK_FALSE;

  VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamic_state{
      VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dynamic_state.dynamicStateCount = 2;
  dynamic_state.pDynamicStates = dyn;

  VkPipelineRenderingCreateInfo rendering_ci{
      VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rendering_ci.colorAttachmentCount = 1;
  rendering_ci.pColorAttachmentFormats = &m_swapchain_format;
  // Shares the composite scope (which has a depth attachment for the forward
  // translucent draws), so declare the format even though we don't test/write.
  rendering_ci.depthAttachmentFormat = m_depth_format;

  VkGraphicsPipelineCreateInfo pipeline_ci{
      VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  pipeline_ci.pNext = &rendering_ci;
  pipeline_ci.stageCount = 2;
  pipeline_ci.pStages = stages;
  pipeline_ci.pVertexInputState = &vertex_input;
  pipeline_ci.pInputAssemblyState = &input_assembly;
  pipeline_ci.pViewportState = &viewport_state;
  pipeline_ci.pRasterizationState = &raster;
  pipeline_ci.pMultisampleState = &multisample;
  pipeline_ci.pColorBlendState = &color_blend;
  pipeline_ci.pDepthStencilState = &depth_stencil;
  pipeline_ci.pDynamicState = &dynamic_state;
  pipeline_ci.layout = m_deferred_light_layout;
  VkResult res = vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1,
                                           &pipeline_ci, nullptr,
                                           &m_deferred_light_pipeline);
  vkDestroyShaderModule(m_device, vert, nullptr);
  vkDestroyShaderModule(m_device, frag, nullptr);
  if (res != VK_SUCCESS) {
    log_error("deferred lighting pipeline creation failed.");
    return false;
  }
  return true;
}

bool vulkan_renderer::create_depth_resources() {
  // Supersampled render extent: swapchain * scale per axis, clamped so we don't
  // exceed the device's max 2D image dimension.
  const float scale = m_ssaa_scale > 0.f ? m_ssaa_scale : 1.f;
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(m_physical_device, &props);
  const uint32_t maxdim = props.limits.maxImageDimension2D;
  m_render_extent.width = std::min(
      maxdim, static_cast<uint32_t>(m_swapchain_extent.width * scale + 0.5f));
  m_render_extent.height = std::min(
      maxdim, static_cast<uint32_t>(m_swapchain_extent.height * scale + 0.5f));

  VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ici.imageType = VK_IMAGE_TYPE_2D;
  ici.format = m_depth_format;
  ici.extent = {m_render_extent.width, m_render_extent.height, 1};
  ici.mipLevels = 1;
  ici.arrayLayers = 1;
  ici.samples = VK_SAMPLE_COUNT_1_BIT;
  ici.tiling = VK_IMAGE_TILING_OPTIMAL;
  ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VK_CHECK(vkCreateImage(m_device, &ici, nullptr, &m_depth_image));

  VkMemoryRequirements req;
  vkGetImageMemoryRequirements(m_device, m_depth_image, &req);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = find_memory_type(m_physical_device, req.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VK_CHECK(vkAllocateMemory(m_device, &ai, nullptr, &m_depth_memory));
  vkBindImageMemory(m_device, m_depth_image, m_depth_memory, 0);

  VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vci.image = m_depth_image;
  vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vci.format = m_depth_format;
  vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
  VK_CHECK(vkCreateImageView(m_device, &vci, nullptr, &m_depth_view));

  // SSAA colour target (scene renders here, then is downscaled to the swapchain).
  VkImageCreateInfo cci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  cci.imageType = VK_IMAGE_TYPE_2D;
  cci.format = m_swapchain_format;
  cci.extent = {m_render_extent.width, m_render_extent.height, 1};
  cci.mipLevels = 1;
  cci.arrayLayers = 1;
  cci.samples = VK_SAMPLE_COUNT_1_BIT;
  cci.tiling = VK_IMAGE_TILING_OPTIMAL;
  cci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  cci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  cci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VK_CHECK(vkCreateImage(m_device, &cci, nullptr, &m_ssaa_color));
  vkGetImageMemoryRequirements(m_device, m_ssaa_color, &req);
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = find_memory_type(m_physical_device, req.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VK_CHECK(vkAllocateMemory(m_device, &ai, nullptr, &m_ssaa_memory));
  vkBindImageMemory(m_device, m_ssaa_color, m_ssaa_memory, 0);
  VkImageViewCreateInfo cvi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  cvi.image = m_ssaa_color;
  cvi.viewType = VK_IMAGE_VIEW_TYPE_2D;
  cvi.format = m_swapchain_format;
  cvi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VK_CHECK(vkCreateImageView(m_device, &cvi, nullptr, &m_ssaa_view));

  log_info("SSAA render extent: " + std::to_string(m_render_extent.width) + "x" +
           std::to_string(m_render_extent.height) + " (scale " +
           std::to_string(scale) + ")");

  // --- Deferred G-buffer (recreated with the swapchain) ---
  auto make_gbuf = [&](VkFormat fmt, gbuffer_target &t) -> bool {
    t.format = fmt;
    VkImageCreateInfo gi{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    gi.imageType = VK_IMAGE_TYPE_2D;
    gi.format = fmt;
    gi.extent = {m_render_extent.width, m_render_extent.height, 1};
    gi.mipLevels = 1;
    gi.arrayLayers = 1;
    gi.samples = VK_SAMPLE_COUNT_1_BIT;
    gi.tiling = VK_IMAGE_TILING_OPTIMAL;
    gi.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    gi.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(m_device, &gi, nullptr, &t.image) != VK_SUCCESS)
      return false;
    VkMemoryRequirements r;
    vkGetImageMemoryRequirements(m_device, t.image, &r);
    VkMemoryAllocateInfo a{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    a.allocationSize = r.size;
    a.memoryTypeIndex = find_memory_type(m_physical_device, r.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(m_device, &a, nullptr, &t.memory) != VK_SUCCESS)
      return false;
    vkBindImageMemory(m_device, t.image, t.memory, 0);
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = t.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = fmt;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return vkCreateImageView(m_device, &vi, nullptr, &t.view) == VK_SUCCESS;
  };
  if (!make_gbuf(VK_FORMAT_R8G8B8A8_UNORM, m_gbuf_albedo)) return false;
  if (!make_gbuf(VK_FORMAT_R16G16B16A16_SFLOAT, m_gbuf_normal)) return false;
  if (!make_gbuf(VK_FORMAT_R16G16B16A16_SFLOAT, m_gbuf_position)) return false;

  // One-time: sampler + descriptor set layout + pool + the descriptor itself.
  if (m_gbuffer_set_layout == VK_NULL_HANDLE) {
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = VK_FILTER_NEAREST;
    si.minFilter = VK_FILTER_NEAREST;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(m_device, &si, nullptr, &m_gbuffer_sampler));

    VkDescriptorSetLayoutBinding b[3] = {};
    for (int i = 0; i < 3; ++i) {
      b[i].binding = i;
      b[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      b[i].descriptorCount = 1;
      b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = 3;
    lci.pBindings = b;
    VK_CHECK(vkCreateDescriptorSetLayout(m_device, &lci, nullptr,
                                         &m_gbuffer_set_layout));

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1;
    pci.poolSizeCount = 1;
    pci.pPoolSizes = &ps;
    VK_CHECK(vkCreateDescriptorPool(m_device, &pci, nullptr, &m_gbuffer_pool));

    VkDescriptorSetAllocateInfo dsai{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = m_gbuffer_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &m_gbuffer_set_layout;
    VK_CHECK(vkAllocateDescriptorSets(m_device, &dsai, &m_gbuffer_descriptor));
  }

  // (Re)point the descriptor at the current G-buffer views.
  VkDescriptorImageInfo gii[3] = {};
  const VkImageView views[3] = {m_gbuf_albedo.view, m_gbuf_normal.view,
                                m_gbuf_position.view};
  VkWriteDescriptorSet gw[3] = {};
  for (int i = 0; i < 3; ++i) {
    gii[i].sampler = m_gbuffer_sampler;
    gii[i].imageView = views[i];
    gii[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    gw[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    gw[i].dstSet = m_gbuffer_descriptor;
    gw[i].dstBinding = i;
    gw[i].descriptorCount = 1;
    gw[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    gw[i].pImageInfo = &gii[i];
  }
  vkUpdateDescriptorSets(m_device, 3, gw, 0, nullptr);
  return true;
}

void vulkan_renderer::destroy_depth_resources() {
  if (m_depth_view) {
    vkDestroyImageView(m_device, m_depth_view, nullptr);
    m_depth_view = VK_NULL_HANDLE;
  }
  if (m_depth_image) {
    vkDestroyImage(m_device, m_depth_image, nullptr);
    m_depth_image = VK_NULL_HANDLE;
  }
  if (m_depth_memory) {
    vkFreeMemory(m_device, m_depth_memory, nullptr);
    m_depth_memory = VK_NULL_HANDLE;
  }
  if (m_ssaa_view) {
    vkDestroyImageView(m_device, m_ssaa_view, nullptr);
    m_ssaa_view = VK_NULL_HANDLE;
  }
  if (m_ssaa_color) {
    vkDestroyImage(m_device, m_ssaa_color, nullptr);
    m_ssaa_color = VK_NULL_HANDLE;
  }
  if (m_ssaa_memory) {
    vkFreeMemory(m_device, m_ssaa_memory, nullptr);
    m_ssaa_memory = VK_NULL_HANDLE;
  }
  for (gbuffer_target *t : {&m_gbuf_albedo, &m_gbuf_normal, &m_gbuf_position}) {
    if (t->view) vkDestroyImageView(m_device, t->view, nullptr);
    if (t->image) vkDestroyImage(m_device, t->image, nullptr);
    if (t->memory) vkFreeMemory(m_device, t->memory, nullptr);
    *t = gbuffer_target{};
  }
}

bool vulkan_renderer::create_test_geometry() {
  // Build a unit cube as gfx::basic_vertex and push it through the real
  // geometry-bank path (Create_Bank + Insert), so draw time exercises the GPU
  // bank used by scene geometry.
  const float s = 1.0f;
  const glm::vec3 c[8] = {{-s, -s, -s}, {s, -s, -s}, {s, s, -s}, {-s, s, -s},
                          {-s, -s, s},  {s, -s, s},  {s, s, s},  {-s, s, s}};
  struct face {
    int a, b, c, d;
    glm::vec3 normal;
  };
  const face faces[6] = {
      {0, 1, 2, 3, {0.f, 0.f, -1.f}}, {5, 4, 7, 6, {0.f, 0.f, 1.f}},
      {4, 0, 3, 7, {-1.f, 0.f, 0.f}}, {1, 5, 6, 2, {1.f, 0.f, 0.f}},
      {3, 2, 6, 7, {0.f, 1.f, 0.f}},  {4, 5, 1, 0, {0.f, -1.f, 0.f}}};

  gfx::vertex_array verts;
  verts.reserve(36);
  auto push = [&](int idx, const glm::vec3 &n) {
    verts.emplace_back(c[idx], n, glm::vec2(0.f));
  };
  for (const auto &f : faces) {
    push(f.a, f.normal);
    push(f.b, f.normal);
    push(f.c, f.normal);
    push(f.a, f.normal);
    push(f.c, f.normal);
    push(f.d, f.normal);
  }
  gfx::userdata_array userdata(verts.size());

  const gfx::geometrybank_handle bank = Create_Bank();
  m_test_geometry = Insert(verts, userdata, bank, GL_TRIANGLES);
  return true;
}

// ---------------------------------------------------------------------------
// GPU geometry bank
// ---------------------------------------------------------------------------

vulkan_geometrybank::~vulkan_geometrybank() {
  if (m_ctx == nullptr || m_ctx->device == VK_NULL_HANDLE) return;
  for (auto &g : m_gpu) {
    if (g.vbuf) vkDestroyBuffer(m_ctx->device, g.vbuf, nullptr);
    if (g.vmem) vkFreeMemory(m_ctx->device, g.vmem, nullptr);
    if (g.ibuf) vkDestroyBuffer(m_ctx->device, g.ibuf, nullptr);
    if (g.imem) vkFreeMemory(m_ctx->device, g.imem, nullptr);
  }
}

namespace {
bool make_device_buffer(const vulkan_geometry_context &ctx, const void *data,
                        VkDeviceSize size, VkBufferUsageFlags usage,
                        VkBuffer &buf, VkDeviceMemory &mem) {
  if (size == 0) return true;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = size;
  bi.usage = usage;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(ctx.device, &bi, nullptr, &buf) != VK_SUCCESS) return false;
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(ctx.device, buf, &req);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = find_memory_type(ctx.physical_device, req.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (ai.memoryTypeIndex == UINT32_MAX) return false;
  if (vkAllocateMemory(ctx.device, &ai, nullptr, &mem) != VK_SUCCESS)
    return false;
  vkBindBufferMemory(ctx.device, buf, mem, 0);
  void *mapped = nullptr;
  vkMapMemory(ctx.device, mem, 0, size, 0, &mapped);
  std::memcpy(mapped, data, static_cast<size_t>(size));
  vkUnmapMemory(ctx.device, mem);
  return true;
}
}  // namespace

void vulkan_geometrybank::upload(gfx::geometry_handle const &Geometry) {
  if (m_ctx == nullptr || m_ctx->device == VK_NULL_HANDLE) return;
  const uint32_t index = Geometry.chunk - 1;
  if (index >= m_gpu.size()) m_gpu.resize(index + 1);
  gpu_chunk &g = m_gpu[index];

  // Free any previous buffers (replace case).
  if (g.vbuf) vkDestroyBuffer(m_ctx->device, g.vbuf, nullptr);
  if (g.vmem) vkFreeMemory(m_ctx->device, g.vmem, nullptr);
  if (g.ibuf) vkDestroyBuffer(m_ctx->device, g.ibuf, nullptr);
  if (g.imem) vkFreeMemory(m_ctx->device, g.imem, nullptr);
  g = {};

  auto const &c = chunk(Geometry);
  g.type = c.type;
  g.vertex_count = static_cast<uint32_t>(c.vertices.size());
  g.index_count = static_cast<uint32_t>(c.indices.size());

  make_device_buffer(*m_ctx, c.vertices.data(),
                     c.vertices.size() * sizeof(gfx::basic_vertex),
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, g.vbuf, g.vmem);
  if (!c.indices.empty()) {
    make_device_buffer(*m_ctx, c.indices.data(),
                       c.indices.size() * sizeof(gfx::basic_index),
                       VK_BUFFER_USAGE_INDEX_BUFFER_BIT, g.ibuf, g.imem);
  }
}

void vulkan_geometrybank::create_(gfx::geometry_handle const &Geometry) {
  upload(Geometry);
}

void vulkan_geometrybank::replace_(gfx::geometry_handle const &Geometry) {
  upload(Geometry);
}

std::size_t vulkan_geometrybank::draw_(gfx::geometry_handle const &Geometry,
                                       gfx::stream_units const & /*Units*/,
                                       unsigned int const /*Streams*/) {
  if (m_ctx == nullptr || m_ctx->current_cmd == VK_NULL_HANDLE) return 0;
  const uint32_t index = Geometry.chunk - 1;
  if (index >= m_gpu.size()) return 0;
  const gpu_chunk &g = m_gpu[index];
  if (g.vbuf == VK_NULL_HANDLE) return 0;

  // Pick the pipeline matching the chunk's primitive type (triangle fans and
  // other types are skipped for now). In pick mode use the ID-colour pipelines.
  VkPipeline pipe = VK_NULL_HANDLE;
  if (g.type == GL_TRIANGLES) {
    pipe = m_ctx->shadow_mode        ? m_ctx->shadow_pipeline_triangles
           : m_ctx->gbuffer_mode     ? m_ctx->gbuffer_pipeline_triangles
           : m_ctx->pick_mode        ? m_ctx->pick_pipeline_triangles
           : m_ctx->translucent_mode ? m_ctx->translucent_triangles
                                     : m_ctx->pipeline_triangles;
  } else if (g.type == GL_TRIANGLE_STRIP) {
    pipe = m_ctx->shadow_mode        ? m_ctx->shadow_pipeline_strips
           : m_ctx->gbuffer_mode     ? m_ctx->gbuffer_pipeline_strips
           : m_ctx->pick_mode        ? m_ctx->pick_pipeline_strips
           : m_ctx->translucent_mode ? m_ctx->translucent_strips
                                     : m_ctx->pipeline_strips;
  } else if (g.type == GL_TRIANGLE_FAN) {
    pipe = m_ctx->shadow_mode        ? m_ctx->shadow_pipeline_fans
           : m_ctx->gbuffer_mode     ? m_ctx->gbuffer_pipeline_fans
           : m_ctx->pick_mode        ? m_ctx->pick_pipeline_fans
           : m_ctx->translucent_mode ? m_ctx->translucent_fans
                                     : m_ctx->pipeline_fans;
  }
  if (pipe == VK_NULL_HANDLE) return 0;
  vkCmdBindPipeline(m_ctx->current_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);

  VkDeviceSize offset = 0;
  vkCmdBindVertexBuffers(m_ctx->current_cmd, 0, 1, &g.vbuf, &offset);
  const uint32_t instances = m_ctx->instance_count;  // >1 for batched draws
  if (g.index_count > 0) {
    vkCmdBindIndexBuffer(m_ctx->current_cmd, g.ibuf, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(m_ctx->current_cmd, g.index_count, instances, 0, 0, 0);
    return (g.index_count / 3) * instances;
  }
  vkCmdDraw(m_ctx->current_cmd, g.vertex_count, instances, 0, 0);
  return (g.vertex_count / 3) * instances;
}

void vulkan_renderer::recreate_swapchain() {
  if (m_device == VK_NULL_HANDLE) return;

  // Wait until the window has a non-zero size again.
  int w = 0, h = 0;
  glfwGetFramebufferSize(m_window, &w, &h);
  while ((w == 0 || h == 0) && !glfwWindowShouldClose(m_window)) {
    glfwWaitEvents();
    glfwGetFramebufferSize(m_window, &w, &h);
  }

  vkDeviceWaitIdle(m_device);
  destroy_swapchain();
  create_swapchain();
}

// ---------------------------------------------------------------------------
// Per-frame resources
// ---------------------------------------------------------------------------

bool vulkan_renderer::create_default_texture() {
  // Shared sampler.
  {
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.maxLod = VK_LOD_CLAMP_NONE;
    // Anisotropic filtering (needs the mip chain to do much): sharpens textures
    // viewed at grazing angles (rails, ground, long surfaces).
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(m_physical_device, &props);
    sci.anisotropyEnable = VK_TRUE;
    sci.maxAnisotropy = std::min(16.f, props.limits.maxSamplerAnisotropy);
    VK_CHECK(vkCreateSampler(m_device, &sci, nullptr, &m_sampler));
  }

  // Descriptor set layout (set 0 = combined image sampler, fragment stage).
  {
    VkDescriptorSetLayoutBinding b{};
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo ci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ci.bindingCount = 1;
    ci.pBindings = &b;
    VK_CHECK(vkCreateDescriptorSetLayout(m_device, &ci, nullptr,
                                         &m_texture_set_layout));
  }

  // Descriptor pool sized for many textures (per-material sets land in step B).
  {
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4096};
    VkDescriptorPoolCreateInfo ci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    ci.maxSets = 4096;
    ci.poolSizeCount = 1;
    ci.pPoolSizes = &ps;
    VK_CHECK(vkCreateDescriptorPool(m_device, &ci, nullptr, &m_descriptor_pool));
  }

  // 1x1 white image, uploaded via a staging buffer.
  const uint8_t white[4] = {255, 255, 255, 255};
  VkBuffer staging = VK_NULL_HANDLE;
  VkDeviceMemory staging_mem = VK_NULL_HANDLE;
  {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = sizeof(white);
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(m_device, &bi, nullptr, &staging));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(m_device, staging, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_memory_type(
        m_physical_device, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(m_device, &ai, nullptr, &staging_mem));
    vkBindBufferMemory(m_device, staging, staging_mem, 0);
    void *mapped = nullptr;
    vkMapMemory(m_device, staging_mem, 0, sizeof(white), 0, &mapped);
    std::memcpy(mapped, white, sizeof(white));
    vkUnmapMemory(m_device, staging_mem);
  }

  VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ici.imageType = VK_IMAGE_TYPE_2D;
  ici.format = VK_FORMAT_R8G8B8A8_UNORM;
  ici.extent = {1, 1, 1};
  ici.mipLevels = 1;
  ici.arrayLayers = 1;
  ici.samples = VK_SAMPLE_COUNT_1_BIT;
  ici.tiling = VK_IMAGE_TILING_OPTIMAL;
  ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VK_CHECK(vkCreateImage(m_device, &ici, nullptr, &m_white_image));
  {
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(m_device, m_white_image, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_memory_type(m_physical_device, req.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(m_device, &ai, nullptr, &m_white_memory));
    vkBindImageMemory(m_device, m_white_image, m_white_memory, 0);
  }

  // One-time upload.
  VkCommandBufferAllocateInfo cbai{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = m_command_pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  vkAllocateCommandBuffers(m_device, &cbai, &cmd);
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
  to_dst.image = m_white_image;
  to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                       1, &to_dst);
  VkBufferImageCopy region{};
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.imageExtent = {1, 1, 1};
  vkCmdCopyBufferToImage(cmd, staging, m_white_image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  VkImageMemoryBarrier to_read = to_dst;
  to_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  to_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  to_read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  to_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &to_read);
  vkEndCommandBuffer(cmd);
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;
  vkQueueSubmit(m_graphics_queue, 1, &submit, VK_NULL_HANDLE);
  vkQueueWaitIdle(m_graphics_queue);
  vkFreeCommandBuffers(m_device, m_command_pool, 1, &cmd);
  vkDestroyBuffer(m_device, staging, nullptr);
  vkFreeMemory(m_device, staging_mem, nullptr);

  VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vci.image = m_white_image;
  vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vci.format = VK_FORMAT_R8G8B8A8_UNORM;
  vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VK_CHECK(vkCreateImageView(m_device, &vci, nullptr, &m_white_view));

  VkDescriptorSetAllocateInfo dsai{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = m_descriptor_pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &m_texture_set_layout;
  VK_CHECK(vkAllocateDescriptorSets(m_device, &dsai, &m_white_descriptor));
  VkDescriptorImageInfo info{};
  info.sampler = m_sampler;
  info.imageView = m_white_view;
  info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = m_white_descriptor;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  write.pImageInfo = &info;
  vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
  return true;
}

bool vulkan_renderer::create_command_pool() {
  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = m_graphics_family;
  VK_CHECK(vkCreateCommandPool(m_device, &pci, nullptr, &m_command_pool));
  return true;
}

bool vulkan_renderer::create_frame_resources() {
  m_frames.resize(kMaxFramesInFlight);
  for (auto &f : m_frames) {
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VK_CHECK(vkCreateSemaphore(m_device, &sci, nullptr, &f.image_available));
    VK_CHECK(vkCreateSemaphore(m_device, &sci, nullptr, &f.render_finished));

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VK_CHECK(vkCreateFence(m_device, &fci, nullptr, &f.in_flight));

    VkCommandBufferAllocateInfo ai{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = m_command_pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(m_device, &ai, &f.command_buffer));

    // Per-frame light/scene UBO (host-visible, persistently mapped) + its set.
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = sizeof(LightUBO);
    bi.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(m_device, &bi, nullptr, &f.light_ubo));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(m_device, f.light_ubo, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = find_memory_type(
        m_physical_device, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(m_device, &mai, nullptr, &f.light_ubo_memory));
    vkBindBufferMemory(m_device, f.light_ubo, f.light_ubo_memory, 0);
    vkMapMemory(m_device, f.light_ubo_memory, 0, sizeof(LightUBO), 0,
                &f.light_ubo_mapped);

    // Per-frame instance matrix buffer (host-visible, persistently mapped). The
    // instanced-draw flush writes one region per batch and indexes it via the
    // uInstanceBase push constant.
    const VkDeviceSize inst_size =
        static_cast<VkDeviceSize>(kInstanceCapacity) * sizeof(glm::mat4);
    VkBufferCreateInfo ibi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ibi.size = inst_size;
    ibi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    ibi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(m_device, &ibi, nullptr, &f.instance_buffer));
    VkMemoryRequirements ireq;
    vkGetBufferMemoryRequirements(m_device, f.instance_buffer, &ireq);
    VkMemoryAllocateInfo imai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    imai.allocationSize = ireq.size;
    imai.memoryTypeIndex = find_memory_type(
        m_physical_device, ireq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(m_device, &imai, nullptr, &f.instance_memory));
    vkBindBufferMemory(m_device, f.instance_buffer, f.instance_memory, 0);
    vkMapMemory(m_device, f.instance_memory, 0, inst_size, 0, &f.instance_mapped);

    VkDescriptorSetAllocateInfo dsai{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = m_light_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &m_light_set_layout;
    VK_CHECK(vkAllocateDescriptorSets(m_device, &dsai, &f.light_descriptor));
    VkDescriptorBufferInfo dbi{f.light_ubo, 0, sizeof(LightUBO)};
    VkDescriptorImageInfo sii{};
    sii.sampler = m_shadow_sampler;
    sii.imageView = m_shadow_array_view;
    sii.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo sir{};  // raw depth for the PCSS blocker search
    sir.sampler = m_shadow_sampler_raw;
    sir.imageView = m_shadow_array_view;
    sir.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    VkDescriptorBufferInfo idbi{f.instance_buffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet writes[4] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
    writes[0].dstSet = f.light_descriptor;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].pBufferInfo = &dbi;
    writes[1].dstSet = f.light_descriptor;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = &sii;
    writes[2].dstSet = f.light_descriptor;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].pImageInfo = &sir;
    writes[3].dstSet = f.light_descriptor;
    writes[3].dstBinding = 3;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[3].pBufferInfo = &idbi;
    vkUpdateDescriptorSets(m_device, 4, writes, 0, nullptr);
  }
  return true;
}

// ---------------------------------------------------------------------------
// Frame loop
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Textures
// ---------------------------------------------------------------------------

namespace {
struct decoded_image {
  std::vector<uint8_t> pixels;  // mip chain, concatenated mip 0..mip_levels-1
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t mip_levels = 1;  // how many mips are present in `pixels`
  VkFormat format = VK_FORMAT_UNDEFINED;
};

// Minimal DDS reader: DXT1/3/5 mip 0 -> BC block data (Vulkan samples BC
// natively, no decode). Other DDS variants are rejected (caller falls back).
bool decode_dds(const std::string &path, decoded_image &out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
  if (b.size() < 128 || std::memcmp(b.data(), "DDS ", 4) != 0) return false;
  auto rd32 = [&](size_t off) {
    uint32_t v;
    std::memcpy(&v, b.data() + off, 4);
    return v;
  };
  const uint32_t height = rd32(12);
  const uint32_t width = rd32(16);
  char fourcc[5] = {0};
  std::memcpy(fourcc, b.data() + 84, 4);
  uint32_t blockbytes = 0;
  if (std::strcmp(fourcc, "DXT1") == 0) {
    out.format = VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    blockbytes = 8;
  } else if (std::strcmp(fourcc, "DXT3") == 0) {
    out.format = VK_FORMAT_BC2_UNORM_BLOCK;
    blockbytes = 16;
  } else if (std::strcmp(fourcc, "DXT5") == 0) {
    out.format = VK_FORMAT_BC3_UNORM_BLOCK;
    blockbytes = 16;
  } else {
    return false;
  }
  // Read the whole mip chain the file carries (BC can't be downscaled on the
  // GPU, so we rely on the precomputed mips in the .dds).
  uint32_t mipcount = rd32(28);  // dwMipMapCount
  if (mipcount == 0) mipcount = 1;
  size_t total = 0;
  uint32_t levels = 0;
  for (uint32_t i = 0; i < mipcount; ++i) {
    const uint32_t w = std::max(1u, width >> i);
    const uint32_t h = std::max(1u, height >> i);
    const size_t sz =
        static_cast<size_t>(std::max(1u, (w + 3) / 4)) *
        std::max(1u, (h + 3) / 4) * blockbytes;
    if (b.size() < 128 + total + sz) break;  // truncated; stop at what we have
    total += sz;
    ++levels;
    if (w == 1 && h == 1) break;
  }
  if (levels == 0) return false;
  out.width = width;
  out.height = height;
  out.mip_levels = levels;
  out.pixels.assign(b.begin() + 128, b.begin() + 128 + total);
  return true;
}

bool decode_stb(const std::string &path, decoded_image &out) {
  int x = 0, y = 0, n = 0;
  stbi_set_flip_vertically_on_load(0);
  uint8_t *img = stbi_load(path.c_str(), &x, &y, &n, 4);
  if (img == nullptr) return false;
  out.width = static_cast<uint32_t>(x);
  out.height = static_cast<uint32_t>(y);
  out.format = VK_FORMAT_R8G8B8A8_UNORM;
  out.pixels.assign(img, img + static_cast<size_t>(x) * y * 4);
  stbi_image_free(img);
  return true;
}

bool make_texture_image(VkPhysicalDevice pd, VkDevice dev, VkCommandPool pool,
                        VkQueue queue, const decoded_image &src, VkImage &image,
                        VkDeviceMemory &mem, VkImageView &view) {
  if (src.pixels.empty() || src.width == 0 || src.height == 0) return false;

  // Staging buffer.
  VkBuffer staging = VK_NULL_HANDLE;
  VkDeviceMemory staging_mem = VK_NULL_HANDLE;
  {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = src.pixels.size();
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(dev, &bi, nullptr, &staging));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, staging, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_memory_type(
        pd, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(dev, &ai, nullptr, &staging_mem));
    vkBindBufferMemory(dev, staging, staging_mem, 0);
    void *mapped = nullptr;
    vkMapMemory(dev, staging_mem, 0, src.pixels.size(), 0, &mapped);
    std::memcpy(mapped, src.pixels.data(), src.pixels.size());
    vkUnmapMemory(dev, staging_mem);
  }

  // Mip levels: use the chain provided in the file (DDS); otherwise generate a
  // full chain by blitting (only for blittable/uncompressed formats).
  const bool blittable = (src.format == VK_FORMAT_R8G8B8A8_UNORM);
  uint32_t full_levels = 1;
  for (uint32_t d = std::max(src.width, src.height); d > 1; d >>= 1) ++full_levels;
  uint32_t mip_levels;
  bool generate;
  if (src.mip_levels > 1) {
    mip_levels = src.mip_levels;  // precomputed chain in the file
    generate = false;
  } else if (blittable) {
    mip_levels = full_levels;
    generate = true;
  } else {
    mip_levels = 1;
    generate = false;
  }
  auto level_bytes = [&](uint32_t lvl) -> size_t {
    const uint32_t w = std::max(1u, src.width >> lvl);
    const uint32_t h = std::max(1u, src.height >> lvl);
    if (blittable) return static_cast<size_t>(w) * h * 4;
    const uint32_t bb =
        (src.format == VK_FORMAT_BC1_RGBA_UNORM_BLOCK) ? 8u : 16u;
    return static_cast<size_t>(std::max(1u, (w + 3) / 4)) *
           std::max(1u, (h + 3) / 4) * bb;
  };

  VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ici.imageType = VK_IMAGE_TYPE_2D;
  ici.format = src.format;
  ici.extent = {src.width, src.height, 1};
  ici.mipLevels = mip_levels;
  ici.arrayLayers = 1;
  ici.samples = VK_SAMPLE_COUNT_1_BIT;
  ici.tiling = VK_IMAGE_TILING_OPTIMAL;
  ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
              VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VK_CHECK(vkCreateImage(dev, &ici, nullptr, &image));
  {
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev, image, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_memory_type(pd, req.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(dev, &ai, nullptr, &mem));
    vkBindImageMemory(dev, image, mem, 0);
  }

  VkCommandBufferAllocateInfo cbai{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  vkAllocateCommandBuffers(dev, &cbai, &cmd);
  VkCommandBufferBeginInfo bbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &bbi);

  const uint32_t upload_levels = generate ? 1u : mip_levels;
  VkImageMemoryBarrier to_dst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_dst.image = image;
  to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mip_levels, 0, 1};
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                       1, &to_dst);

  // Copy the uploaded mip(s) from the staging buffer.
  std::vector<VkBufferImageCopy> copies;
  size_t off = 0;
  for (uint32_t i = 0; i < upload_levels; ++i) {
    VkBufferImageCopy r{};
    r.bufferOffset = off;
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1};
    r.imageExtent = {std::max(1u, src.width >> i), std::max(1u, src.height >> i),
                     1};
    copies.push_back(r);
    off += level_bytes(i);
  }
  vkCmdCopyBufferToImage(cmd, staging, image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         static_cast<uint32_t>(copies.size()), copies.data());

  if (generate) {
    // Blit-generate the chain: each level is a linear downscale of the previous.
    int32_t mw = static_cast<int32_t>(src.width);
    int32_t mh = static_cast<int32_t>(src.height);
    for (uint32_t i = 1; i < mip_levels; ++i) {
      VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      b.image = image;
      b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 1, 0, 1};
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                           nullptr, 1, &b);
      const int32_t nw = mw > 1 ? mw / 2 : 1;
      const int32_t nh = mh > 1 ? mh / 2 : 1;
      VkImageBlit blit{};
      blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1};
      blit.srcOffsets[1] = {mw, mh, 1};
      blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1};
      blit.dstOffsets[1] = {nw, nh, 1};
      vkCmdBlitImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                     VK_FILTER_LINEAR);
      mw = nw;
      mh = nh;
    }
  }

  // Everything -> shader-read. Levels written as DST (or, when generated, all
  // but the last are SRC) need the right old layout, so do it in two ranges.
  VkImageMemoryBarrier to_read[2] = {};
  uint32_t nbarriers = 0;
  if (generate && mip_levels > 1) {
    to_read[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    to_read[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    to_read[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    to_read[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_read[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    to_read[0].image = image;
    to_read[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mip_levels - 1,
                                   0, 1};
    to_read[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    to_read[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_read[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    to_read[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_read[1].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    to_read[1].image = image;
    to_read[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip_levels - 1, 1,
                                   0, 1};
    nbarriers = 2;
  } else {
    to_read[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    to_read[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_read[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    to_read[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_read[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    to_read[0].image = image;
    to_read[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mip_levels, 0,
                                   1};
    nbarriers = 1;
  }
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, nbarriers, to_read);
  vkEndCommandBuffer(cmd);
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;
  vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue);
  vkFreeCommandBuffers(dev, pool, 1, &cmd);
  vkDestroyBuffer(dev, staging, nullptr);
  vkFreeMemory(dev, staging_mem, nullptr);

  VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vci.image = image;
  vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vci.format = src.format;
  vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mip_levels, 0, 1};
  VK_CHECK(vkCreateImageView(dev, &vci, nullptr, &view));
  return true;
}
}  // namespace

bool vulkan_renderer::create_flat_normal() {
  // Flat normal/height default for materials without a normal map: rg=0.5 ->
  // tangent normal (0,0,1), b=0 -> zero parallax height.
  decoded_image flat;
  flat.width = 1;
  flat.height = 1;
  flat.format = VK_FORMAT_R8G8B8A8_UNORM;
  flat.pixels = {128, 128, 0, 255};
  if (!make_texture_image(m_physical_device, m_device, m_command_pool,
                          m_graphics_queue, flat, m_flat_normal_image,
                          m_flat_normal_memory, m_flat_normal_view))
    return false;
  VkDescriptorSetAllocateInfo nai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  nai.descriptorPool = m_descriptor_pool;
  nai.descriptorSetCount = 1;
  nai.pSetLayouts = &m_texture_set_layout;
  VK_CHECK(vkAllocateDescriptorSets(m_device, &nai, &m_flat_normal_descriptor));
  VkDescriptorImageInfo ninfo{};
  ninfo.sampler = m_sampler;
  ninfo.imageView = m_flat_normal_view;
  ninfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  VkWriteDescriptorSet nwrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  nwrite.dstSet = m_flat_normal_descriptor;
  nwrite.descriptorCount = 1;
  nwrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  nwrite.pImageInfo = &ninfo;
  vkUpdateDescriptorSets(m_device, 1, &nwrite, 0, nullptr);
  return true;
}

bool vulkan_renderer::create_bindless() {
  VkDescriptorSetLayoutBinding b{};
  b.binding = 0;
  b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  b.descriptorCount = kBindlessTextures;
  b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  VkDescriptorBindingFlags bflags =
      VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
      VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
  VkDescriptorSetLayoutBindingFlagsCreateInfo bf{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
  bf.bindingCount = 1;
  bf.pBindingFlags = &bflags;
  VkDescriptorSetLayoutCreateInfo lci{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  lci.pNext = &bf;
  lci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
  lci.bindingCount = 1;
  lci.pBindings = &b;
  VK_CHECK(
      vkCreateDescriptorSetLayout(m_device, &lci, nullptr, &m_bindless_layout));

  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          kBindlessTextures};
  VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pci.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
  pci.maxSets = 1;
  pci.poolSizeCount = 1;
  pci.pPoolSizes = &ps;
  VK_CHECK(vkCreateDescriptorPool(m_device, &pci, nullptr, &m_bindless_pool));

  VkDescriptorSetAllocateInfo dsai{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = m_bindless_pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &m_bindless_layout;
  VK_CHECK(vkAllocateDescriptorSets(m_device, &dsai, &m_bindless_set));

  bindless_write(kBindlessWhiteSlot, m_white_view);
  bindless_write(kBindlessFlatNormalSlot, m_flat_normal_view);
  return true;
}

void vulkan_renderer::bindless_write(uint32_t slot, VkImageView view) {
  if (view == VK_NULL_HANDLE || m_bindless_set == VK_NULL_HANDLE) return;
  VkDescriptorImageInfo ii{};
  ii.sampler = m_sampler;
  ii.imageView = view;
  ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  w.dstSet = m_bindless_set;
  w.dstBinding = 0;
  w.dstArrayElement = slot;
  w.descriptorCount = 1;
  w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  w.pImageInfo = &ii;
  vkUpdateDescriptorSets(m_device, 1, &w, 0, nullptr);
}

texture_handle vulkan_renderer::Fetch_Texture(std::string const &Filename,
                                              bool const /*Loadnow*/,
                                              GLint /*format_hint*/) {
  if (Filename.empty()) return null_handle;
  if (auto it = m_texture_map.find(Filename); it != m_texture_map.end()) {
    return it->second;
  }

  texture_handle handle = null_handle;
  const auto located = texture_manager::find_on_disk(Filename);
  if (located.first.empty()) {
    WriteLog("vk-tex: not found on disk: " + Filename);
  }
  if (!located.first.empty()) {
    const std::string path = located.first + located.second;
    decoded_image img;
    bool ok = false;
    if (located.second == ".dds") {
      ok = decode_dds(path, img);
    } else {
      ok = decode_stb(path, img);
    }
    if (!ok) {
      WriteLog("vk-tex: decode failed (" + located.second + "): " + path);
    }
    if (ok) {
      gpu_texture tex;
      if (make_texture_image(m_physical_device, m_device, m_command_pool,
                             m_graphics_queue, img, tex.image, tex.memory,
                             tex.view)) {
        VkDescriptorSetAllocateInfo dsai{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = m_descriptor_pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &m_texture_set_layout;
        if (vkAllocateDescriptorSets(m_device, &dsai, &tex.descriptor) ==
            VK_SUCCESS) {
          VkDescriptorImageInfo info{};
          info.sampler = m_sampler;
          info.imageView = tex.view;
          info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
          VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
          write.dstSet = tex.descriptor;
          write.descriptorCount = 1;
          write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
          write.pImageInfo = &info;
          vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
          m_textures.push_back(tex);
          vulkan_itexture view;
          view.m_name = Filename;
          view.m_width = static_cast<int>(img.width);
          view.m_height = static_cast<int>(img.height);
          view.m_alpha = (img.format != VK_FORMAT_BC1_RGBA_UNORM_BLOCK);
          // get_id() doubles as the ImGui texture id: hand back the descriptor
          // set so reinterpret_cast<ImTextureID>(tex.get_id()) binds the right
          // texture (the ImGui backend binds it per draw command). The set is
          // bind-compatible with the ImGui pipeline (identical set-0 layout).
          view.m_id = reinterpret_cast<std::size_t>(tex.descriptor);
          m_itextures.push_back(view);
          handle = static_cast<texture_handle>(m_textures.size());  // 1-based
          // Register into the bindless array at slot (handle + 1); draws sample
          // it by that index (slots 0/1 are the white/flat-normal defaults).
          const uint32_t slot = static_cast<uint32_t>(handle) + 1;
          if (slot < kBindlessTextures) bindless_write(slot, tex.view);
        }
      }
    }
  }

  m_texture_map[Filename] = handle;
  return handle;
}

void vulkan_renderer::bind_material(material_handle material,
                                    VkCommandBuffer cmd) {
  // Bindless: no per-material descriptor binds. Push the diffuse/normal slot
  // indices into the big texture array plus the material misc. uMisc.x is the
  // alpha-test threshold (0 for blended atlases, ~0.5 for cutout); uMisc.w is
  // the detail mode (0 none, 1 normal map, 2 normal map + parallax).
  float alpha_ref = 0.f;
  int32_t diff = static_cast<int32_t>(kBindlessWhiteSlot);
  int32_t norm = static_cast<int32_t>(kBindlessFlatNormalSlot);
  float detail_mode = 0.f;
  float gloss = 16.f;
  if (material != null_handle) {
    const opengl_material &mat = m_material_manager.material(material);
    alpha_ref = mat.get_or_guess_opacity();
    gloss = mat.glossiness;
    const texture_handle dt = mat.GetTexture(0);
    if (dt != null_handle &&
        static_cast<uint32_t>(dt) + 1 < kBindlessTextures)
      diff = static_cast<int32_t>(dt) + 1;
    const texture_handle nt = mat.GetTexture(1);
    if (nt != null_handle &&
        static_cast<uint32_t>(nt) + 1 < kBindlessTextures) {
      norm = static_cast<int32_t>(nt) + 1;
      detail_mode = material_has_parallax(material) ? 2.f : 1.f;
    }
  }
  const glm::vec4 misc(alpha_ref, 0.f, 0.f, detail_mode);
  vkCmdPushConstants(cmd, m_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
                     sizeof(float) * 16, sizeof(misc), &misc);  // uMisc @ 64
  const int32_t idx[2] = {diff, norm};
  vkCmdPushConstants(cmd, m_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
                     sizeof(float) * 20, sizeof(idx), idx);  // uTex @ 80
  vkCmdPushConstants(cmd, m_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
                     sizeof(float) * 22, sizeof(gloss), &gloss);  // uGloss @ 88
}

VkDescriptorSet vulkan_renderer::material_texture_descriptor(
    material_handle material) const {
  if (material != null_handle) {
    const texture_handle t = m_material_manager.material(material).GetTexture(0);
    if (t != null_handle && static_cast<size_t>(t) <= m_textures.size()) {
      const VkDescriptorSet d = m_textures[t - 1].descriptor;
      if (d != VK_NULL_HANDLE) return d;
    }
  }
  return m_white_descriptor;
}

VkDescriptorSet vulkan_renderer::material_normal_descriptor(
    material_handle material) const {
  if (material != null_handle) {
    // Slot 1 is the normal/height map in MaSzyna's material convention.
    const texture_handle t = m_material_manager.material(material).GetTexture(1);
    if (t != null_handle && static_cast<size_t>(t) <= m_textures.size()) {
      const VkDescriptorSet d = m_textures[t - 1].descriptor;
      if (d != VK_NULL_HANDLE) return d;
    }
  }
  return m_flat_normal_descriptor;
}

bool vulkan_renderer::material_has_parallax(material_handle material) const {
  if (material == null_handle) return false;
  const opengl_material &mat = m_material_manager.material(material);
  // Parallax material variants declare a height_scale parameter; their normal
  // map stores height in the blue channel. Plain normalmaps don't -> no POM
  // (so we never misread a conventional normal map's .b = z as a height).
  return mat.shader && mat.shader->param_conf.count("height_scale") > 0;
}

// ---------------------------------------------------------------------------
// Control picking
// ---------------------------------------------------------------------------

bool vulkan_renderer::create_pick_pipeline(VkPrimitiveTopology topology,
                                           VkPipeline &out) {
  VkShaderModule vert =
      create_shader_module(pick_vert_spv, sizeof(pick_vert_spv));
  VkShaderModule frag =
      create_shader_module(pick_frag_spv, sizeof(pick_frag_spv));
  if (!vert || !frag) return false;

  VkPipelineShaderStageCreateInfo stages[2] = {
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vert;
  stages[0].pName = "main";
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = frag;
  stages[1].pName = "main";

  VkVertexInputBindingDescription vtx_binding{};
  vtx_binding.stride = sizeof(gfx::basic_vertex);
  vtx_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
  VkVertexInputAttributeDescription vtx_attrs[4]{};
  vtx_attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT,
                  offsetof(gfx::basic_vertex, position)};
  vtx_attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT,
                  offsetof(gfx::basic_vertex, normal)};
  vtx_attrs[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT,
                  offsetof(gfx::basic_vertex, texture)};
  vtx_attrs[3] = {3, 0, VK_FORMAT_R32G32B32A32_SFLOAT,
                  offsetof(gfx::basic_vertex, tangent)};
  VkPipelineVertexInputStateCreateInfo vertex_input{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vertex_input.vertexBindingDescriptionCount = 1;
  vertex_input.pVertexBindingDescriptions = &vtx_binding;
  vertex_input.vertexAttributeDescriptionCount = 4;
  vertex_input.pVertexAttributeDescriptions = vtx_attrs;

  VkPipelineInputAssemblyStateCreateInfo input_assembly{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  input_assembly.topology = topology;

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
  blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  VkPipelineColorBlendStateCreateInfo color_blend{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  color_blend.attachmentCount = 1;
  color_blend.pAttachments = &blend;

  VkPipelineDepthStencilStateCreateInfo depth_stencil{
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  depth_stencil.depthTestEnable = VK_TRUE;
  depth_stencil.depthWriteEnable = VK_TRUE;
  depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

  VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamic_state{
      VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dynamic_state.dynamicStateCount = 2;
  dynamic_state.pDynamicStates = dyn;

  if (m_pick_layout == VK_NULL_HANDLE) {
    VkPushConstantRange pc_range{};
    pc_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pc_range.offset = 0;
    pc_range.size = sizeof(float) * 20;  // mat4 + vec4
    VkPipelineLayoutCreateInfo lci{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    lci.pushConstantRangeCount = 1;
    lci.pPushConstantRanges = &pc_range;
    VK_CHECK(vkCreatePipelineLayout(m_device, &lci, nullptr, &m_pick_layout));
  }

  VkPipelineRenderingCreateInfo rendering_ci{
      VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rendering_ci.colorAttachmentCount = 1;
  rendering_ci.pColorAttachmentFormats = &m_pick_format;
  rendering_ci.depthAttachmentFormat = m_depth_format;

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
  pci.pDepthStencilState = &depth_stencil;
  pci.pDynamicState = &dynamic_state;
  pci.layout = m_pick_layout;
  pci.renderPass = VK_NULL_HANDLE;

  VkResult res = vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pci,
                                           nullptr, &out);
  vkDestroyShaderModule(m_device, vert, nullptr);
  vkDestroyShaderModule(m_device, frag, nullptr);
  if (res != VK_SUCCESS) {
    log_error("pick pipeline creation failed.");
    return false;
  }
  return true;
}

bool vulkan_renderer::create_pick_resources() {
  auto make_image = [&](VkFormat format, VkImageUsageFlags usage,
                        VkImageAspectFlags aspect, VkImage &image,
                        VkDeviceMemory &mem, VkImageView &view) -> bool {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {m_swapchain_extent.width, m_swapchain_extent.height, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(m_device, &ici, nullptr, &image));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(m_device, image, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_memory_type(m_physical_device, req.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(m_device, &ai, nullptr, &mem));
    vkBindImageMemory(m_device, image, mem, 0);
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format;
    vci.subresourceRange = {aspect, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(m_device, &vci, nullptr, &view));
    return true;
  };
  if (!make_image(m_pick_format,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT, m_pick_color_image,
                  m_pick_color_memory, m_pick_color_view))
    return false;
  if (!make_image(m_depth_format, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                  VK_IMAGE_ASPECT_DEPTH_BIT, m_pick_depth_image,
                  m_pick_depth_memory, m_pick_depth_view))
    return false;
  return true;
}

void vulkan_renderer::destroy_pick_resources() {
  if (m_pick_color_view) vkDestroyImageView(m_device, m_pick_color_view, nullptr);
  if (m_pick_color_image) vkDestroyImage(m_device, m_pick_color_image, nullptr);
  if (m_pick_color_memory) vkFreeMemory(m_device, m_pick_color_memory, nullptr);
  if (m_pick_depth_view) vkDestroyImageView(m_device, m_pick_depth_view, nullptr);
  if (m_pick_depth_image) vkDestroyImage(m_device, m_pick_depth_image, nullptr);
  if (m_pick_depth_memory) vkFreeMemory(m_device, m_pick_depth_memory, nullptr);
  m_pick_color_view = VK_NULL_HANDLE;
  m_pick_color_image = VK_NULL_HANDLE;
  m_pick_color_memory = VK_NULL_HANDLE;
  m_pick_depth_view = VK_NULL_HANDLE;
  m_pick_depth_image = VK_NULL_HANDLE;
  m_pick_depth_memory = VK_NULL_HANDLE;
}

void vulkan_renderer::pick_submodel(TSubModel *sm, const glm::mat4 &parent,
                                    const glm::mat4 &rot, const glm::mat4 &proj,
                                    VkCommandBuffer cmd, uint32_t &index) {
  if (sm == nullptr) return;
  if (sm->iVisible && TSubModel::fSquareDist >= sm->fSquareMinDist &&
      TSubModel::fSquareDist < sm->fSquareMaxDist) {
    glm::mat4 local = parent;
    if (sm->iFlags & 0xC000) {
      if (sm->fMatrix != nullptr)
        local = parent * glm::make_mat4(sm->fMatrix->readArray());
      if (m_submodel_animations && sm->b_aAnim != TAnimType::at_None)
        sm->RaAnimation(local, sm->b_aAnim);
    }
    if (sm->eType < TP_ROTATOR) {
      ++index;
      m_pick_submodels.push_back(sm);
      struct {
        glm::mat4 mvp;
        glm::vec4 color;
      } pc;
      pc.mvp = proj * rot * local;
      pc.color = glm::vec4(((index >> 16) & 0xff) / 255.f,
                           ((index >> 8) & 0xff) / 255.f,
                           (index & 0xff) / 255.f, 1.f);
      vkCmdPushConstants(cmd, m_pick_layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                         sizeof(pc), &pc);
      m_geometry.draw(sm->m_geometry.handle);
    }
    if (sm->Child != nullptr)
      pick_submodel(sm->Child, local, rot, proj, cmd, index);
  }
  if (sm->Next != nullptr) pick_submodel(sm->Next, parent, rot, proj, cmd, index);
}

void vulkan_renderer::Update_Pick_Control() {
  if (m_pick_callbacks.empty()) return;  // pick only when something requested it
  if (m_swapchain == VK_NULL_HANDLE || m_pick_color_image == VK_NULL_HANDLE ||
      simulation::Train == nullptr) {
    m_pick_callbacks.clear();
    return;
  }
  TDynamicObject *player = simulation::Train->Dynamic();
  if (player == nullptr || player->mdKabina == nullptr ||
      player->mdKabina->Root == nullptr) {
    m_pick_callbacks.clear();
    return;
  }

  m_pick_submodels.clear();

  VkCommandBufferAllocateInfo cbai{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = m_command_pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  vkAllocateCommandBuffers(m_device, &cbai, &cmd);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &bi);

  auto barrier = [&](VkImage img, VkImageAspectFlags aspect, VkImageLayout from,
                     VkImageLayout to, VkAccessFlags src, VkAccessFlags dst,
                     VkPipelineStageFlags ss, VkPipelineStageFlags ds) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = src;
    b.dstAccessMask = dst;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {aspect, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, ss, ds, 0, 0, nullptr, 0, nullptr, 1, &b);
  };

  barrier(m_pick_color_image, VK_IMAGE_ASPECT_COLOR_BIT,
          VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0,
          VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
          VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
  barrier(m_pick_depth_image, VK_IMAGE_ASPECT_DEPTH_BIT,
          VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, 0,
          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
          VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
          VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT);

  VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  color.imageView = m_pick_color_view;
  color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  color.clearValue.color = {{0.f, 0.f, 0.f, 0.f}};
  VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  depth.imageView = m_pick_depth_view;
  depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
  depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depth.clearValue.depthStencil = {1.0f, 0};
  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea.extent = m_swapchain_extent;
  ri.layerCount = 1;
  ri.colorAttachmentCount = 1;
  ri.pColorAttachments = &color;
  ri.pDepthAttachment = &depth;
  vkCmdBeginRendering(cmd, &ri);

  VkViewport vp{0.f,
                0.f,
                static_cast<float>(m_swapchain_extent.width),
                static_cast<float>(m_swapchain_extent.height),
                0.f,
                1.f};
  vkCmdSetViewport(cmd, 0, 1, &vp);
  VkRect2D sc{{0, 0}, m_swapchain_extent};
  vkCmdSetScissor(cmd, 0, 1, &sc);

  glm::dmat4 view_d(1.0);
  Global.pCamera.SetMatrix(view_d);
  const glm::mat4 rot = glm::mat4(glm::mat3(view_d));
  glm::mat4 proj = glm::perspectiveFovRH_ZO(
      glm::radians(static_cast<float>(Global.FieldOfView)),
      static_cast<float>(m_swapchain_extent.width),
      static_cast<float>(m_swapchain_extent.height), 0.1f, 5000.f);
  proj[1][1] *= -1.f;
  const glm::mat4 vm =
      glm::translate(glm::mat4(1.f), glm::vec3(player->vPosition - Global.pCamera.Pos)) *
      glm::mat4(player->mMatrix);

  m_geo_ctx.current_cmd = cmd;
  m_geo_ctx.pick_mode = true;
  TSubModel::fSquareDist = 0.f;
  uint32_t index = 0;
  pick_submodel(player->mdKabina->Root, vm, rot, proj, cmd, index);
  m_geo_ctx.pick_mode = false;
  m_geo_ctx.current_cmd = VK_NULL_HANDLE;

  vkCmdEndRendering(cmd);

  barrier(m_pick_color_image, VK_IMAGE_ASPECT_COLOR_BIT,
          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
          VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
          VK_PIPELINE_STAGE_TRANSFER_BIT);

  glm::ivec2 cursor = glm::ivec2(Global.cursor_pos);
  cursor = glm::clamp(
      cursor, glm::ivec2(0),
      glm::ivec2(static_cast<int>(m_swapchain_extent.width) - 1,
                 static_cast<int>(m_swapchain_extent.height) - 1));
  VkBufferImageCopy region{};
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.imageOffset = {cursor.x, cursor.y, 0};
  region.imageExtent = {1, 1, 1};
  vkCmdCopyImageToBuffer(cmd, m_pick_color_image,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_pick_readback,
                         1, &region);

  vkEndCommandBuffer(cmd);
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;
  vkQueueSubmit(m_graphics_queue, 1, &submit, VK_NULL_HANDLE);
  vkQueueWaitIdle(m_graphics_queue);
  vkFreeCommandBuffers(m_device, m_command_pool, 1, &cmd);

  uint8_t px[4] = {0, 0, 0, 0};
  void *mapped = nullptr;
  vkMapMemory(m_device, m_pick_readback_memory, 0, 4, 0, &mapped);
  std::memcpy(px, mapped, 4);
  vkUnmapMemory(m_device, m_pick_readback_memory);

  const uint32_t picked = px[2] + (px[1] * 256u) + (px[0] * 256u * 256u);
  m_pick_control = (picked > 0 && picked <= m_pick_submodels.size())
                       ? m_pick_submodels[picked - 1]
                       : nullptr;

  for (auto &cb : m_pick_callbacks) cb(m_pick_control, glm::vec2(0.f));
  m_pick_callbacks.clear();
}

// ---------------------------------------------------------------------------
// Gradient skydome
// ---------------------------------------------------------------------------

bool vulkan_renderer::create_sky_pipeline() {
  VkShaderModule vert =
      create_shader_module(skydome_vert_spv, sizeof(skydome_vert_spv));
  VkShaderModule frag =
      create_shader_module(skydome_frag_spv, sizeof(skydome_frag_spv));
  if (!vert || !frag) return false;

  VkPipelineShaderStageCreateInfo stages[2] = {
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vert;
  stages[0].pName = "main";
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = frag;
  stages[1].pName = "main";

  // Two vertex buffers: position (binding 0) and colour (binding 1).
  VkVertexInputBindingDescription bindings[2]{};
  bindings[0] = {0, sizeof(glm::vec3), VK_VERTEX_INPUT_RATE_VERTEX};
  bindings[1] = {1, sizeof(glm::vec3), VK_VERTEX_INPUT_RATE_VERTEX};
  VkVertexInputAttributeDescription attrs[2]{};
  attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0};
  attrs[1] = {1, 1, VK_FORMAT_R32G32B32_SFLOAT, 0};
  VkPipelineVertexInputStateCreateInfo vertex_input{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vertex_input.vertexBindingDescriptionCount = 2;
  vertex_input.pVertexBindingDescriptions = bindings;
  vertex_input.vertexAttributeDescriptionCount = 2;
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
  blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  VkPipelineColorBlendStateCreateInfo color_blend{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  color_blend.attachmentCount = 1;
  color_blend.pAttachments = &blend;

  // No depth interaction: the dome is drawn first and the world overwrites it.
  VkPipelineDepthStencilStateCreateInfo depth_stencil{
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  depth_stencil.depthTestEnable = VK_FALSE;
  depth_stencil.depthWriteEnable = VK_FALSE;

  VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamic_state{
      VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dynamic_state.dynamicStateCount = 2;
  dynamic_state.pDynamicStates = dyn;

  VkPushConstantRange pc_range{};
  pc_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  pc_range.offset = 0;
  pc_range.size = sizeof(float) * 16;  // mat4 MVP
  VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  lci.pushConstantRangeCount = 1;
  lci.pPushConstantRanges = &pc_range;
  VK_CHECK(vkCreatePipelineLayout(m_device, &lci, nullptr, &m_sky_layout));

  VkPipelineRenderingCreateInfo rendering_ci{
      VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rendering_ci.colorAttachmentCount = 1;
  rendering_ci.pColorAttachmentFormats = &m_swapchain_format;
  rendering_ci.depthAttachmentFormat = m_depth_format;

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
  pci.pDepthStencilState = &depth_stencil;
  pci.pDynamicState = &dynamic_state;
  pci.layout = m_sky_layout;
  pci.renderPass = VK_NULL_HANDLE;

  VkResult res = vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pci,
                                           nullptr, &m_sky_pipeline);
  vkDestroyShaderModule(m_device, vert, nullptr);
  vkDestroyShaderModule(m_device, frag, nullptr);
  if (res != VK_SUCCESS) {
    log_error("skydome pipeline creation failed.");
    return false;
  }
  return true;
}

void vulkan_renderer::render_skydome(const glm::mat4 &proj, const glm::mat4 &rot,
                                     VkCommandBuffer cmd) {
  if (m_sky_pipeline == VK_NULL_HANDLE) return;
  CSkyDome &dome = simulation::Environment.skydome();
  const auto &verts = dome.vertices();
  auto &cols = dome.colors();
  const auto &idx = dome.indices();
  if (verts.empty() || idx.empty() || cols.size() != verts.size()) return;

  const VkDeviceSize csize = cols.size() * sizeof(glm::vec3);
  if (!m_sky_ready) {
    // Static position + index buffers (host-visible; uploaded once).
    make_device_buffer(m_geo_ctx, verts.data(), verts.size() * sizeof(glm::vec3),
                       VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, m_sky_vertex,
                       m_sky_vertex_memory);
    make_device_buffer(m_geo_ctx, idx.data(), idx.size() * sizeof(std::uint16_t),
                       VK_BUFFER_USAGE_INDEX_BUFFER_BIT, m_sky_index,
                       m_sky_index_memory);
    // Host-visible colour buffer, re-uploaded when the dome is dirty.
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = csize;
    bi.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(m_device, &bi, nullptr, &m_sky_color) != VK_SUCCESS)
      return;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(m_device, m_sky_color, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_memory_type(
        m_physical_device, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (vkAllocateMemory(m_device, &ai, nullptr, &m_sky_color_memory) !=
        VK_SUCCESS)
      return;
    vkBindBufferMemory(m_device, m_sky_color, m_sky_color_memory, 0);
    m_sky_index_count = static_cast<uint32_t>(idx.size());
    m_sky_ready = true;
    dome.is_dirty() = true;  // force the first colour upload below
  }
  if (dome.is_dirty()) {
    void *mapped = nullptr;
    vkMapMemory(m_device, m_sky_color_memory, 0, csize, 0, &mapped);
    std::memcpy(mapped, cols.data(), static_cast<size_t>(csize));
    vkUnmapMemory(m_device, m_sky_color_memory);
    dome.is_dirty() = false;
  }

  // Camera-centred dome, scaled to sit far away (matches the GL renderer's 500m).
  const glm::mat4 mvp =
      proj * rot * glm::scale(glm::mat4(1.f), glm::vec3(500.f));

  // Match the supersampled render extent (the scene pass renders to the SSAA
  // target, not the swapchain) so the subsequent draws keep the full viewport.
  VkViewport vp{0.f,
                0.f,
                static_cast<float>(m_render_extent.width),
                static_cast<float>(m_render_extent.height),
                0.f,
                1.f};
  vkCmdSetViewport(cmd, 0, 1, &vp);
  VkRect2D sc{{0, 0}, m_render_extent};
  vkCmdSetScissor(cmd, 0, 1, &sc);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_sky_pipeline);
  vkCmdPushConstants(cmd, m_sky_layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                     sizeof(mvp), &mvp);
  VkBuffer bufs[2] = {m_sky_vertex, m_sky_color};
  VkDeviceSize offs[2] = {0, 0};
  vkCmdBindVertexBuffers(cmd, 0, 2, bufs, offs);
  vkCmdBindIndexBuffer(cmd, m_sky_index, 0, VK_INDEX_TYPE_UINT16);
  vkCmdDrawIndexed(cmd, m_sky_index_count, 1, 0, 0, 0);
}

void vulkan_renderer::destroy_sky() {
  if (m_sky_pipeline) vkDestroyPipeline(m_device, m_sky_pipeline, nullptr);
  if (m_sky_layout) vkDestroyPipelineLayout(m_device, m_sky_layout, nullptr);
  if (m_sky_vertex) vkDestroyBuffer(m_device, m_sky_vertex, nullptr);
  if (m_sky_vertex_memory) vkFreeMemory(m_device, m_sky_vertex_memory, nullptr);
  if (m_sky_color) vkDestroyBuffer(m_device, m_sky_color, nullptr);
  if (m_sky_color_memory) vkFreeMemory(m_device, m_sky_color_memory, nullptr);
  if (m_sky_index) vkDestroyBuffer(m_device, m_sky_index, nullptr);
  if (m_sky_index_memory) vkFreeMemory(m_device, m_sky_index_memory, nullptr);
  m_sky_pipeline = VK_NULL_HANDLE;
  m_sky_layout = VK_NULL_HANDLE;
  m_sky_vertex = VK_NULL_HANDLE;
  m_sky_vertex_memory = VK_NULL_HANDLE;
  m_sky_color = VK_NULL_HANDLE;
  m_sky_color_memory = VK_NULL_HANDLE;
  m_sky_index = VK_NULL_HANDLE;
  m_sky_index_memory = VK_NULL_HANDLE;
  m_sky_ready = false;
}

void vulkan_renderer::render_submodel(TSubModel *sm, const glm::mat4 &parent,
                                      const glm::mat4 &rot,
                                      const glm::mat4 &proj,
                                      const material_handle *skins,
                                      bool translucent_pass, float interior,
                                      VkCommandBuffer cmd, bool instrument) {
  if (sm == nullptr) return;
  // Inside the instrument-backlight subtree? (root or inherited from a parent.)
  const bool is_instr = instrument || (sm == m_instrument_submodel);

  if (sm->iVisible && TSubModel::fSquareDist >= sm->fSquareMinDist &&
      TSubModel::fSquareDist < sm->fSquareMaxDist) {
    glm::mat4 local = parent;
    if (sm->iFlags & 0xC000) {
      if (sm->fMatrix != nullptr)
        local = parent * glm::make_mat4(sm->fMatrix->readArray());
      // Submodel animation (gauges, levers, switches, wheels, clocks...).
      if (m_submodel_animations && sm->b_aAnim != TAnimType::at_None)
        sm->RaAnimation(local, sm->b_aAnim);
    }
    // Skip the legacy fake ground-shadow decal ("cien"): an opaque-ish dark
    // quad on the railhead that would cover the track. Real shadows are a
    // future shadow-mapping pass.
    if (sm->eType < TP_ROTATOR && sm->pName != "cien") {
      // m_material: >0 direct handle, 0 none, <0 replacable skin index.
      material_handle mh = null_handle;
      if (sm->m_material > 0) {
        mh = sm->m_material;
      } else if (sm->m_material < 0 && skins != nullptr) {
        mh = skins[-sm->m_material];
      }
      // Pass selection matches the engine's own split: iAlpha marks which of a
      // submodel's texture layers are opaque (mask 0x1F) vs translucent
      // (0x1F0000). Only genuinely translucent layers (e.g. glass) go to the
      // depth-write-off pass; alpha-cutout bodies stay opaque so they keep
      // writing depth and don't go see-through.
      bool draw_this;
      if (m_two_pass_translucency) {
        draw_this = translucent_pass
                        ? ((sm->iAlpha & sm->iFlags & 0x1F0000) != 0)
                        : ((sm->iAlpha & sm->iFlags & 0x1F) != 0);
      } else {
        draw_this = !translucent_pass;  // single pass draws everything once
      }
      if (draw_this) {
        // Push the camera-relative model matrix (offset 0); the UBO viewproj
        // projects it. Lighting is per-fragment from the same UBO.
        vkCmdPushConstants(cmd, m_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                           sizeof(local), &local);
        if (!m_geo_ctx.shadow_mode) {
          bind_material(mh, cmd);  // pushes uMisc (.x threshold, .y emission = 0)
          // Self-illumination: gauges/indicators/lit panels glow once the scene
          // is dark enough (engine gate: f4Emision.a > 0 && fLuminance < fLight),
          // or whenever this is instrument-backlight geometry (only visible while
          // the switch is on), so flipping it lights the gauges in any light.
          const bool lit = (Global.fLuminance < sm->fLight) || is_instr;
          const float emission = (sm->f4Emision.a > 0.f && lit) ? sm->f4Emision.a
                                                                : 0.f;
          // uMisc.y = emission, uMisc.z = cab-interior flag (offset 68).
          const float misc_yz[2] = {emission, interior};
          vkCmdPushConstants(cmd, m_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
                             sizeof(float) * 17, sizeof(misc_yz), misc_yz);
        }
        // Shadow pass: depth only, no material/uMisc (cascade index in uMisc.w
        // is pushed once per cascade and must survive).
        m_geometry.draw(sm->m_geometry.handle);
      }
    }
    if (sm->Child != nullptr)
      // Children are inside this submodel's subtree (inherit is_instr); siblings
      // (Next) are not, so they only carry the inherited `instrument` flag.
      render_submodel(sm->Child, local, rot, proj, skins, translucent_pass,
                      interior, cmd, is_instr);
  }
  if (sm->Next != nullptr)
    render_submodel(sm->Next, parent, rot, proj, skins, translucent_pass,
                    interior, cmd, instrument);
}

void vulkan_renderer::render_instanced_bucket(
    TModel3d *model, const material_handle *skins,
    const std::vector<TAnimModel *> &instances, const glm::dvec3 &campos,
    const glm::mat4 &rot, const glm::mat4 &proj, bool translucent_pass,
    const glm::vec4 *fplanes, VkCommandBuffer cmd) {
  if (model == nullptr || model->Root == nullptr || instances.empty()) return;
  if (m_instance_mapped == nullptr) return;

  // Build the camera-relative root matrix for each visible instance, matching
  // render_instance's transform (translate * rotate per axis; no scale).
  m_instance_matrices.clear();
  float closest = 1e30f;
  for (auto *inst : instances) {
    if (inst == nullptr || !inst->m_visible) continue;
    const glm::dvec3 off = inst->location() - campos;
    const glm::vec3 c(off);
    if (fplanes != nullptr) {
      const float r = static_cast<float>(inst->radius());
      bool in = true;
      for (int i = 0; i < 6; ++i)
        if (glm::dot(glm::vec3(fplanes[i]), c) + fplanes[i].w < -r) {
          in = false;
          break;
        }
      if (!in) continue;
    }
    glm::mat4 m = glm::translate(glm::mat4(1.f), c);
    const glm::vec3 ang = inst->vAngle;
    if (ang.y != 0.f) m = glm::rotate(m, glm::radians(ang.y), glm::vec3(0, 1, 0));
    if (ang.x != 0.f) m = glm::rotate(m, glm::radians(ang.x), glm::vec3(1, 0, 0));
    if (ang.z != 0.f) m = glm::rotate(m, glm::radians(ang.z), glm::vec3(0, 0, 1));
    m_instance_matrices.push_back(m);
    closest = std::min(
        closest, glm::length2(c / static_cast<float>(Global.ZoomFactor)));
  }
  if (m_instance_matrices.empty()) return;

  TSubModel::fSquareDist = closest;  // shared global, drives submodel LOD gating
  const glm::mat4 identity(1.f);
  auto *const slots = static_cast<glm::mat4 *>(m_instance_mapped);

  const size_t total = m_instance_matrices.size();
  size_t done = 0;
  while (done < total) {
    size_t batch = std::min<size_t>(total - done, kMaxInstancesPerBatch);
    if (m_instance_cursor + batch > kInstanceCapacity)
      batch = kInstanceCapacity - m_instance_cursor;  // buffer full this frame
    if (batch == 0) break;
    const uint32_t base = m_instance_cursor;
    std::memcpy(slots + base, m_instance_matrices.data() + done,
                batch * sizeof(glm::mat4));
    m_instance_cursor += static_cast<uint32_t>(batch);

    // uInstanceBase @ offset 92: the shader reads inst.m[base + gl_InstanceIndex].
    const int32_t ibase = static_cast<int32_t>(base);
    vkCmdPushConstants(cmd, m_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
                       sizeof(float) * 23, sizeof(ibase), &ibase);
    m_geo_ctx.instance_count = static_cast<uint32_t>(batch);
    render_submodel(model->Root, identity, rot, proj, skins, translucent_pass,
                    0.f, cmd);
    done += batch;
  }

  // Restore the non-instanced defaults for subsequent draws in this pass.
  m_geo_ctx.instance_count = 1;
  const int32_t zero = 0;
  vkCmdPushConstants(cmd, m_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
                     sizeof(float) * 23, sizeof(zero), &zero);
}

void vulkan_renderer::draw_scene(bool translucent_pass, const glm::dvec3 &campos,
                                const glm::mat4 &rot, const glm::mat4 &proj,
                                const std::set<TDynamicObject *> &consist,
                                TDynamicObject *player, VkCommandBuffer cmd) {
  m_geo_ctx.translucent_mode = translucent_pass;

  // Default every draw in this pass to the non-instanced path (identity slot 0)
  // until the instanced flush overrides uInstanceBase for its batches.
  {
    const int32_t zero = 0;
    vkCmdPushConstants(cmd, m_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
                       sizeof(float) * 23, sizeof(zero), &zero);
  }

  // The world shader takes the camera-relative model matrix in the push
  // constant (offset 0); the UBO viewproj turns it into clip space.
  auto push_group_mvp = [&](glm::dvec3 const &center) {
    const glm::mat4 model =
        glm::translate(glm::mat4(1.f), glm::vec3(center - campos));
    vkCmdPushConstants(cmd, m_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                       sizeof(model), &model);
  };
  // Shadow pass is depth-only: skip material binds so the per-cascade uMisc.w
  // push (cascade index) survives; push_group_mvp still sets the local matrix.
  auto bind_mat = [&](material_handle material) {
    if (!m_geo_ctx.shadow_mode) bind_material(material, cmd);
  };

  auto render_instance = [&](TAnimModel *inst, bool tp) {
    if (inst == nullptr || inst->pModel == nullptr ||
        inst->pModel->Root == nullptr)
      return;
    const glm::dvec3 pos = inst->location() - campos;
    const glm::vec3 ang = inst->vAngle;
    glm::mat4 model = glm::translate(glm::mat4(1.f), glm::vec3(pos));
    if (ang.y != 0.f)
      model = glm::rotate(model, glm::radians(ang.y), glm::vec3(0, 1, 0));
    if (ang.x != 0.f)
      model = glm::rotate(model, glm::radians(ang.x), glm::vec3(1, 0, 0));
    if (ang.z != 0.f)
      model = glm::rotate(model, glm::radians(ang.z), glm::vec3(0, 0, 1));
    TSubModel::iInstance = reinterpret_cast<std::uintptr_t>(inst);
    TSubModel::fSquareDist =
        glm::length2(glm::vec3(pos) / static_cast<float>(Global.ZoomFactor));
    const material_data *md = inst->Material();
    const material_handle *skins =
        (md != nullptr) ? md->replacable_skins : nullptr;
    render_submodel(inst->pModel->Root, model, rot, proj, skins, tp, 0.f, cmd);
  };

  auto render_vehicle = [&](TDynamicObject *veh, bool with_cab, bool tp) {
    if (veh == nullptr) return;
    TSubModel::iInstance = reinterpret_cast<std::uintptr_t>(veh);
    TSubModel::fSquareDist = glm::length2(glm::vec3(veh->vPosition - campos) /
                                          static_cast<float>(Global.ZoomFactor));
    const glm::mat4 vm =
        glm::translate(glm::mat4(1.f), glm::vec3(veh->vPosition - campos)) *
        glm::mat4(veh->mMatrix);
    const material_data *md = veh->Material();
    const material_handle *skins =
        (md != nullptr) ? md->replacable_skins : nullptr;
    const float interior = with_cab ? 1.f : 0.f;
    if (veh->mdLowPolyInt && veh->mdLowPolyInt->Root)
      render_submodel(veh->mdLowPolyInt->Root, vm, rot, proj, skins, tp,
                      interior, cmd);
    if (veh->mdModel && veh->mdModel->Root)
      render_submodel(veh->mdModel->Root, vm, rot, proj, skins, tp, 0.f, cmd);
    if (veh->mdLoad && veh->mdLoad->Root) {
      const glm::mat4 lm =
          glm::translate(vm, glm::vec3(0.f, veh->LoadOffset, 0.f));
      render_submodel(veh->mdLoad->Root, lm, rot, proj, skins, tp, 0.f, cmd);
    }
    if (with_cab && veh->mdKabina && veh->mdKabina->Root)
      render_submodel(veh->mdKabina->Root, vm, rot, proj, skins, tp, interior,
                      cmd);
  };

  // Cab-light shadow pass: only the player vehicle (its cab/body occlude the
  // lamp); nothing else is relevant to the cab interior.
  if (m_geo_ctx.cab_only) {
    if (player != nullptr) render_vehicle(player, true, false);
    return;
  }

  // Camera frustum (camera-relative space) for culling the colour passes; the
  // shadow pass keeps every caster (off-screen objects cast shadows into view).
  const bool do_cull = !m_geo_ctx.shadow_mode;
  glm::vec4 fplanes[6];
  if (do_cull) {
    const glm::mat4 vp = proj * rot;
    auto prow = [&](int i) {
      return glm::vec4(vp[0][i], vp[1][i], vp[2][i], vp[3][i]);
    };
    const glm::vec4 r3 = prow(3);
    fplanes[0] = r3 + prow(0);  // left
    fplanes[1] = r3 - prow(0);  // right
    fplanes[2] = r3 + prow(1);  // bottom
    fplanes[3] = r3 - prow(1);  // top
    fplanes[4] = prow(2);       // near (Vulkan ZO depth)
    fplanes[5] = r3 - prow(2);  // far
    for (auto &p : fplanes) {
      const float len = glm::length(glm::vec3(p));
      if (len > 0.f) p /= len;
    }
  }
  auto visible = [&](const glm::dvec3 &center, double radius) -> bool {
    if (!do_cull) return true;
    const glm::vec3 c = glm::vec3(center - campos);
    const float r = static_cast<float>(radius);
    for (auto const &p : fplanes)
      if (glm::dot(glm::vec3(p), c) + p.w < -r) return false;
    return true;
  };

  // Merge every visible cell's instance buckets keyed by (model, skins); the
  // batched draws are issued once per unique key after the region loop, so the
  // same model spread across many cells collapses into a few instanced draws.
  scene::basic_cell::instance_bucket_map framebuckets;

  if (scene::basic_region *region = simulation::Region) {
    const int side = scene::EU07_REGIONSIDESECTIONCOUNT;
    const int half =
        static_cast<int>(std::ceil(2000.0 / scene::EU07_SECTIONSIZE));
    const int cx = static_cast<int>(
        std::floor(campos.x / scene::EU07_SECTIONSIZE + side / 2));
    const int cz = static_cast<int>(
        std::floor(campos.z / scene::EU07_SECTIONSIZE + side / 2));
    for (int z = cz - half; z <= cz + half; ++z) {
      if (z < 0 || z >= side) continue;
      for (int x = cx - half; x <= cx + half; ++x) {
        if (x < 0 || x >= side) continue;
        scene::basic_section *section =
            region->get_section(static_cast<size_t>(z) * side + x);
        if (section == nullptr) continue;
        section->create_geometry();
        if (!visible(section->m_area.center, section->m_area.radius)) continue;
        if (!translucent_pass && !section->m_shapes.empty()) {
          push_group_mvp(section->m_area.center);
          for (auto const &shape : section->m_shapes) {
            bind_mat(shape.data().material);
            m_geometry.draw(shape.data().geometry);
          }
        }
        for (auto &cell : section->m_cells) {
          if (!cell.m_active) continue;
          if (!visible(cell.m_area.center, cell.m_area.radius)) continue;
          push_group_mvp(cell.m_area.center);
          if (!translucent_pass) {
            for (auto const &shape : cell.m_shapesopaque) {
              bind_mat(shape.data().material);
              m_geometry.draw(shape.data().geometry);
            }
            for (auto *track : cell.m_paths) {
              if (track == nullptr || !track->m_visible) continue;
              if (track->m_material1 != null_handle) {
                bind_mat(track->m_material1);
                for (auto const &g : track->Geometry1) m_geometry.draw(g);
              }
              if (track->m_material2 != null_handle) {
                bind_mat(track->m_material2);
                for (auto const &g : track->Geometry2) m_geometry.draw(g);
              }
              if (track->SwitchExtension &&
                  track->SwitchExtension->m_material3 != null_handle) {
                bind_mat(track->SwitchExtension->m_material3);
                m_geometry.draw(track->SwitchExtension->Geometry3);
              }
            }
            for (auto const &bucket : cell.m_instancebuckets_opaque) {
              auto &dst = framebuckets[bucket.first];
              dst.insert(dst.end(), bucket.second.begin(), bucket.second.end());
            }
            for (auto *inst : cell.m_instancesopaque)
              render_instance(inst, false);
          } else {
            for (auto const &shape : cell.m_shapestranslucent) {
              bind_mat(shape.data().material);
              m_geometry.draw(shape.data().geometry);
            }
            for (auto *inst : cell.m_instancetranslucent)
              render_instance(inst, true);
          }
        }
      }
    }
  }

  // Flush the accumulated opaque instance buckets as batched draws. The shadow
  // pass keeps every caster (fplanes = null); the colour passes frustum-cull
  // each instance. (Translucent instances are not bucketed, so this is a no-op
  // there.)
  for (auto const &b : framebuckets)
    render_instanced_bucket(b.first.pModel, b.first.skins.data(), b.second,
                            campos, rot, proj, translucent_pass,
                            do_cull ? fplanes : nullptr, cmd);

  for (TDynamicObject *v : consist)
    render_vehicle(v, v == player, translucent_pass);
}

bool vulkan_renderer::Render() {
  // Resolve any queued control-pick request (on mouse click) before the frame.
  Update_Pick_Control();

  // The application opened an ImGui frame in begin_ui_frame(); the engine
  // relies on the renderer driving render_ui() (which calls ImGui::Render())
  // exactly once per frame, or the next ImGui::NewFrame() asserts. On the
  // normal path we do it inside the dynamic-rendering pass so the UI is drawn;
  // on skipped frames we still call it (with no command buffer) just to end
  // the ImGui frame.
  if (m_swapchain == VK_NULL_HANDLE) {
    recreate_swapchain();
    if (m_swapchain == VK_NULL_HANDLE) {
      Application.render_ui();  // still minimized: end the ImGui frame anyway
      return true;
    }
  }

  frame_sync &frame = m_frames[m_frame_index];

  vkWaitForFences(m_device, 1, &frame.in_flight, VK_TRUE, UINT64_MAX);

  VkResult acquire = vkAcquireNextImageKHR(
      m_device, m_swapchain, UINT64_MAX, frame.image_available,
      VK_NULL_HANDLE, &m_acquired_image);
  if (acquire == VK_ERROR_OUT_OF_DATE_KHR) {
    recreate_swapchain();
    Application.render_ui();  // end the ImGui frame even though we skip drawing
    return true;
  }
  if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR) {
    log_error(std::string("vkAcquireNextImageKHR failed: ") +
              vk_result_str(acquire));
    return false;
  }

  vkResetFences(m_device, 1, &frame.in_flight);
  vkResetCommandBuffer(frame.command_buffer, 0);

  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(frame.command_buffer, &bi);

  VkImage image = m_swapchain_images[m_acquired_image];
  VkImageView view = m_swapchain_image_views[m_acquired_image];
  VkImageSubresourceRange range{};
  range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  range.levelCount = 1;
  range.layerCount = 1;

  // --- camera + per-frame lights (needed by both the shadow and main pass) ---
  glm::dmat4 view_d(1.0);
  Global.pCamera.SetMatrix(view_d);
  const glm::mat4 rot = glm::mat4(glm::mat3(view_d));
  glm::mat4 proj = glm::perspectiveFovRH_ZO(
      glm::radians(static_cast<float>(Global.FieldOfView)),
      static_cast<float>(m_swapchain_extent.width),
      static_cast<float>(m_swapchain_extent.height), 0.1f, 5000.f);
  proj[1][1] *= -1.f;  // flip Y for Vulkan clip space
  const glm::dvec3 campos = Global.pCamera.Pos;
  const glm::mat4 viewproj = proj * rot;

  m_geo_ctx.current_cmd = frame.command_buffer;
  update_lights(viewproj, campos, frame);

  // Reset the per-frame instance buffer: slot 0 is identity (used by every
  // non-instanced draw via uInstanceBase = 0); batched draws allocate from 1.
  m_instance_mapped = frame.instance_mapped;
  if (m_instance_mapped != nullptr) {
    *static_cast<glm::mat4 *>(m_instance_mapped) = glm::mat4(1.f);
    m_instance_cursor = 1;
  }

  // Gather the player's consist once; both passes draw the same set, and
  // submodel animation is advanced exactly once per frame.
  std::set<TDynamicObject *> consist;
  TDynamicObject *player = nullptr;
  if (simulation::Train != nullptr) {
    player = simulation::Train->Dynamic();
    if (player != nullptr) {
      consist.insert(player);
      for (TDynamicObject *v = player->NextConnected();
           v != nullptr && consist.insert(v).second; v = v->NextConnected()) {
      }
      for (TDynamicObject *v = player->PrevConnected();
           v != nullptr && consist.insert(v).second; v = v->PrevConnected()) {
      }
    }
  }
  TTrack::fetch_default_profiles();
  for (TDynamicObject *v : consist)
    if (v != nullptr) v->ABuLittleUpdate(0.0);

  // Bind set 0 (white) + set 1 (light UBO + shadow map); they persist into the
  // shadow and main passes (rebound after the sky, whose layout differs). Also
  // seed uMisc so draws before the first bind_material() are well-defined.
  auto bind_world_sets = [&]() {
    // set 0 = bindless texture array (materials select slots via push); set 1 =
    // per-frame light UBO + shadow maps.
    vkCmdBindDescriptorSets(frame.command_buffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline_layout, 0,
                            1, &m_bindless_set, 0, nullptr);
    vkCmdBindDescriptorSets(frame.command_buffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline_layout, 1,
                            1, &frame.light_descriptor, 0, nullptr);
  };
  bind_world_sets();
  const glm::vec4 misc_default(0.f);
  vkCmdPushConstants(frame.command_buffer, m_pipeline_layout,
                     VK_SHADER_STAGE_VERTEX_BIT, sizeof(float) * 16,
                     sizeof(misc_default), &misc_default);
  const int32_t idx_default[2] = {static_cast<int32_t>(kBindlessWhiteSlot),
                                  static_cast<int32_t>(kBindlessFlatNormalSlot)};
  vkCmdPushConstants(frame.command_buffer, m_pipeline_layout,
                     VK_SHADER_STAGE_VERTEX_BIT, sizeof(float) * 20,
                     sizeof(idx_default), idx_default);
  const float gloss_default = 16.f;
  vkCmdPushConstants(frame.command_buffer, m_pipeline_layout,
                     VK_SHADER_STAGE_VERTEX_BIT, sizeof(float) * 22,
                     sizeof(gloss_default), &gloss_default);

  // --- cascaded sun shadow depth passes (camera-relative, same frame) ---
  {
    VkImageMemoryBarrier sm{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    sm.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    sm.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    sm.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    sm.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    sm.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    sm.image = m_shadow_image;
    sm.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, kShadowLayers};
    vkCmdPipelineBarrier(frame.command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &sm);

    const VkViewport svp{0.f, 0.f, static_cast<float>(m_shadow_extent.width),
                         static_cast<float>(m_shadow_extent.height), 0.f, 1.f};
    const VkRect2D ssc{{0, 0}, m_shadow_extent};
    m_geo_ctx.shadow_mode = true;
    // Layers 0..2 are sun cascades (full scene); layer 3 is the cab lamp (only
    // the player cab is rendered, so cab objects occlude the lamp).
    for (uint32_t c = 0; c < kShadowLayers; ++c) {
      VkRenderingAttachmentInfo sdepth{
          VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
      sdepth.imageView = m_shadow_layer_views[c];
      sdepth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
      sdepth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      sdepth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      sdepth.clearValue.depthStencil = {1.0f, 0};
      VkRenderingInfo sri{VK_STRUCTURE_TYPE_RENDERING_INFO};
      sri.renderArea.extent = m_shadow_extent;
      sri.layerCount = 1;
      sri.pDepthAttachment = &sdepth;
      vkCmdBeginRendering(frame.command_buffer, &sri);
      vkCmdSetViewport(frame.command_buffer, 0, 1, &svp);
      vkCmdSetScissor(frame.command_buffer, 0, 1, &ssc);
      // uMisc.w = layer index; the shadow vertex selects lightspace[c]. The
      // shadow pass skips material binds, so this push persists across draws.
      const float cf = static_cast<float>(c);
      vkCmdPushConstants(frame.command_buffer, m_pipeline_layout,
                         VK_SHADER_STAGE_VERTEX_BIT, sizeof(float) * 19,
                         sizeof(float), &cf);
      m_geo_ctx.cab_only = (c == kCabShadowLayer);
      draw_scene(false, campos, rot, proj, consist, player,
                 frame.command_buffer);
      vkCmdEndRendering(frame.command_buffer);
    }
    m_geo_ctx.cab_only = false;
    m_geo_ctx.shadow_mode = false;

    sm.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    sm.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    sm.oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    sm.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(frame.command_buffer,
                         VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &sm);
  }

  const glm::vec3 fog = Global.FogColor;
  const VkImage gimgs[3] = {m_gbuf_albedo.image, m_gbuf_normal.image,
                            m_gbuf_position.image};
  const VkImageView gviews[3] = {m_gbuf_albedo.view, m_gbuf_normal.view,
                                 m_gbuf_position.view};
  const VkImageSubresourceRange depth_range{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0,
                                            1};

  // ---- Deferred geometry pass: fill the G-buffer (+ opaque depth) ----
  {
    VkImageMemoryBarrier b[4] = {};
    for (int i = 0; i < 3; ++i) {
      b[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      b[i].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      b[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      b[i].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      b[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      b[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      b[i].image = gimgs[i];
      b[i].subresourceRange = range;
    }
    b[3].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b[3].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    b[3].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b[3].newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    b[3].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b[3].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b[3].image = m_depth_image;
    b[3].subresourceRange = depth_range;
    vkCmdPipelineBarrier(frame.command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                         0, 0, nullptr, 0, nullptr, 4, b);

    VkRenderingAttachmentInfo gcol[3] = {};
    for (int i = 0; i < 3; ++i) {
      gcol[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
      gcol[i].imageView = gviews[i];
      gcol[i].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      gcol[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      gcol[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      gcol[i].clearValue.color = {{0.f, 0.f, 0.f, 0.f}};
    }
    VkRenderingAttachmentInfo gdepth{
        VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    gdepth.imageView = m_depth_view;
    gdepth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    gdepth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    gdepth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;  // composite pass reads it
    gdepth.clearValue.depthStencil = {1.0f, 0};
    VkRenderingInfo gri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    gri.renderArea.extent = m_render_extent;
    gri.layerCount = 1;
    gri.colorAttachmentCount = 3;
    gri.pColorAttachments = gcol;
    gri.pDepthAttachment = &gdepth;
    vkCmdBeginRendering(frame.command_buffer, &gri);
    const VkViewport gvp{0.f, 0.f, static_cast<float>(m_render_extent.width),
                         static_cast<float>(m_render_extent.height), 0.f, 1.f};
    vkCmdSetViewport(frame.command_buffer, 0, 1, &gvp);
    const VkRect2D gsc{{0, 0}, m_render_extent};
    vkCmdSetScissor(frame.command_buffer, 0, 1, &gsc);
    bind_world_sets();
    m_geo_ctx.gbuffer_mode = true;
    draw_scene(false, campos, rot, proj, consist, player, frame.command_buffer);
    m_geo_ctx.gbuffer_mode = false;
    vkCmdEndRendering(frame.command_buffer);

    // G-buffers -> shader-readable; opaque depth -> readable for translucent.
    VkImageMemoryBarrier rd[4] = {};
    for (int i = 0; i < 3; ++i) {
      rd[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      rd[i].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      rd[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      rd[i].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      rd[i].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      rd[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      rd[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      rd[i].image = gimgs[i];
      rd[i].subresourceRange = range;
    }
    rd[3].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    rd[3].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    rd[3].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    rd[3].oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    rd[3].newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    rd[3].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    rd[3].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    rd[3].image = m_depth_image;
    rd[3].subresourceRange = depth_range;
    vkCmdPipelineBarrier(frame.command_buffer,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                             VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                         0, 0, nullptr, 0, nullptr, 4, rd);
  }

  // m_ssaa_color UNDEFINED -> COLOR_ATTACHMENT for the composite pass.
  VkImageMemoryBarrier to_color{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  to_color.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  to_color.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  to_color.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  to_color.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_color.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_color.image = m_ssaa_color;
  to_color.subresourceRange = range;
  vkCmdPipelineBarrier(frame.command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                       nullptr, 0, nullptr, 1, &to_color);

  // ---- Composite: sky -> deferred lighting (opaque) -> forward translucent ----
  {
    VkRenderingAttachmentInfo color_attachment{
        VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    color_attachment.imageView = m_ssaa_view;
    color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.clearValue.color = {{fog.r, fog.g, fog.b, 1.0f}};

    VkRenderingAttachmentInfo depth_attachment{
        VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depth_attachment.imageView = m_depth_view;
    depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;  // opaque depth
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea.extent = m_render_extent;
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &color_attachment;
    ri.pDepthAttachment = &depth_attachment;
    vkCmdBeginRendering(frame.command_buffer, &ri);
    const VkViewport vp{0.f, 0.f, static_cast<float>(m_render_extent.width),
                        static_cast<float>(m_render_extent.height), 0.f, 1.f};
    vkCmdSetViewport(frame.command_buffer, 0, 1, &vp);
    const VkRect2D sc{{0, 0}, m_render_extent};
    vkCmdSetScissor(frame.command_buffer, 0, 1, &sc);

    // Sky fills the background.
    render_skydome(proj, rot, frame.command_buffer);

    // Deferred lighting: shade the opaque G-buffer (discards background -> keeps
    // the sky) with a single fullscreen triangle.
    vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      m_deferred_light_pipeline);
    vkCmdBindDescriptorSets(frame.command_buffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_deferred_light_layout, 0, 1, &m_gbuffer_descriptor,
                            0, nullptr);
    vkCmdBindDescriptorSets(frame.command_buffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_deferred_light_layout, 1, 1,
                            &frame.light_descriptor, 0, nullptr);
    vkCmdDraw(frame.command_buffer, 3, 1, 0, 0);

    // Forward translucent (glass) over the lit opaque, depth-tested against it.
    bind_world_sets();
    if (m_two_pass_translucency)
      draw_scene(true, campos, rot, proj, consist, player, frame.command_buffer);
    m_geo_ctx.translucent_mode = false;
    vkCmdEndRendering(frame.command_buffer);
  }

  m_geo_ctx.current_cmd = VK_NULL_HANDLE;

  // SSAA resolve: downscale the supersampled scene onto the swapchain image with
  // a linear blit (this is the antialiasing), then draw the UI over it at native
  // resolution so text/lines stay crisp.
  {
    VkImageMemoryBarrier pre[2] = {};
    pre[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    pre[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    pre[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    pre[0].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    pre[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    pre[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre[0].image = m_ssaa_color;
    pre[0].subresourceRange = range;
    pre[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    pre[1].srcAccessMask = 0;
    pre[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    pre[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    pre[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    pre[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre[1].image = image;
    pre[1].subresourceRange = range;
    vkCmdPipelineBarrier(frame.command_buffer,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                         2, pre);

    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1] = {static_cast<int32_t>(m_render_extent.width),
                          static_cast<int32_t>(m_render_extent.height), 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[1] = {static_cast<int32_t>(m_swapchain_extent.width),
                          static_cast<int32_t>(m_swapchain_extent.height), 1};
    vkCmdBlitImage(frame.command_buffer, m_ssaa_color,
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                   VK_FILTER_LINEAR);

    VkImageMemoryBarrier toc{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toc.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toc.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    toc.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toc.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    toc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toc.image = image;
    toc.subresourceRange = range;
    vkCmdPipelineBarrier(frame.command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &toc);
  }

  // UI pass: draw ImGui onto the swapchain at native resolution, keeping the
  // downscaled scene (loadOp LOAD). render_ui() -> ImGui::Render() records here.
  VkRenderingAttachmentInfo ui_color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  ui_color.imageView = view;
  ui_color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  ui_color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
  ui_color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  VkRenderingInfo ui_info{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ui_info.renderArea.extent = m_swapchain_extent;
  ui_info.layerCount = 1;
  ui_info.colorAttachmentCount = 1;
  ui_info.pColorAttachments = &ui_color;
  vkCmdBeginRendering(frame.command_buffer, &ui_info);
  if (m_imgui)
    m_imgui->set_current_frame(frame.command_buffer, m_swapchain_extent);
  Application.render_ui();
  if (m_imgui) m_imgui->set_current_frame(VK_NULL_HANDLE, {});
  vkCmdEndRendering(frame.command_buffer);

  // COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC for presentation.
  VkImageMemoryBarrier to_present{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  to_present.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  to_present.dstAccessMask = 0;
  to_present.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  to_present.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_present.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_present.image = image;
  to_present.subresourceRange = range;
  vkCmdPipelineBarrier(frame.command_buffer,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &to_present);

  vkEndCommandBuffer(frame.command_buffer);

  VkPipelineStageFlags wait_stage =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.waitSemaphoreCount = 1;
  submit.pWaitSemaphores = &frame.image_available;
  submit.pWaitDstStageMask = &wait_stage;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &frame.command_buffer;
  submit.signalSemaphoreCount = 1;
  submit.pSignalSemaphores = &frame.render_finished;

  if (vkQueueSubmit(m_graphics_queue, 1, &submit, frame.in_flight) !=
      VK_SUCCESS) {
    log_error("vkQueueSubmit failed.");
    return false;
  }

  m_frame_acquired = true;
  return true;
}

void vulkan_renderer::SwapBuffers() {
  if (!m_frame_acquired || m_swapchain == VK_NULL_HANDLE) return;

  frame_sync &frame = m_frames[m_frame_index];

  VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
  present.waitSemaphoreCount = 1;
  present.pWaitSemaphores = &frame.render_finished;
  present.swapchainCount = 1;
  present.pSwapchains = &m_swapchain;
  present.pImageIndices = &m_acquired_image;

  VkResult res = vkQueuePresentKHR(m_present_queue, &present);
  if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR) {
    recreate_swapchain();
  } else if (res != VK_SUCCESS) {
    log_error(std::string("vkQueuePresentKHR failed: ") + vk_result_str(res));
  }

  m_frame_index = (m_frame_index + 1) % kMaxFramesInFlight;
  m_frame_acquired = false;
}

void vulkan_renderer::Update(double const Deltatime) {
  if (Deltatime > 0.0) m_framerate = static_cast<float>(1.0 / Deltatime);
}
