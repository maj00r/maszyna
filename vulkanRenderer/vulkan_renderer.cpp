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

#include "application/application.h"
#include "model/Model3d.h"
#include "scene/scene.h"
#include "utilities/Globals.h"
#include "utilities/Logs.h"
#include "vehicle/Train.h"
#include "world/Track.h"

// Defined in the engine; forward-declared here to avoid pulling simulation.h's
// heavy transitive includes into this TU.
namespace simulation {
extern scene::basic_region *Region;
extern TTrain *Train;
}  // namespace simulation

// SPIR-V embedded by the build (glslangValidator --vn). Each header defines a
// `const uint32_t <name>[]` array.
#include "world_vert_spv.h"
#include "world_frag_spv.h"

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
  if (!create_world_pipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                             m_pipeline_triangles))
    return false;
  if (!create_world_pipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
                             m_pipeline_strips))
    return false;
  m_geo_ctx.physical_device = m_physical_device;
  m_geo_ctx.device = m_device;
  m_geo_ctx.pipeline_triangles = m_pipeline_triangles;
  m_geo_ctx.pipeline_strips = m_pipeline_strips;
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
    }
    m_frames.clear();

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
    if (m_pipeline_layout) {
      vkDestroyPipelineLayout(m_device, m_pipeline_layout, nullptr);
      m_pipeline_layout = VK_NULL_HANDLE;
    }
    if (m_descriptor_pool) {
      vkDestroyDescriptorPool(m_device, m_descriptor_pool, nullptr);
      m_descriptor_pool = VK_NULL_HANDLE;
    }
    if (m_texture_set_layout) {
      vkDestroyDescriptorSetLayout(m_device, m_texture_set_layout, nullptr);
      m_texture_set_layout = VK_NULL_HANDLE;
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

  // Vulkan 1.3 dynamic rendering lets us render straight to swap-chain image
  // views without VkRenderPass/VkFramebuffer objects.
  VkPhysicalDeviceVulkan13Features features13{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  features13.dynamicRendering = VK_TRUE;
  features13.synchronization2 = VK_TRUE;

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
  return create_depth_resources();
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

bool vulkan_renderer::create_world_pipeline(VkPrimitiveTopology topology,
                                            VkPipeline &out) {
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
  raster.cullMode = VK_CULL_MODE_NONE;
  raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  raster.lineWidth = 1.0f;

  VkPipelineMultisampleStateCreateInfo multisample{
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineColorBlendAttachmentState blend_attachment{};
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
  depth_stencil.depthWriteEnable = VK_TRUE;
  depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

  // mat4 MVP push constant for the vertex stage.
  VkPushConstantRange pc_range{};
  pc_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  pc_range.offset = 0;
  pc_range.size = sizeof(float) * 16;

  if (m_pipeline_layout == VK_NULL_HANDLE) {
    VkPipelineLayoutCreateInfo layout_ci{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_ci.setLayoutCount = 1;
    layout_ci.pSetLayouts = &m_texture_set_layout;
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

bool vulkan_renderer::create_depth_resources() {
  VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ici.imageType = VK_IMAGE_TYPE_2D;
  ici.format = m_depth_format;
  ici.extent = {m_swapchain_extent.width, m_swapchain_extent.height, 1};
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
  // other types are skipped for now).
  VkPipeline pipe = VK_NULL_HANDLE;
  if (g.type == GL_TRIANGLES) {
    pipe = m_ctx->pipeline_triangles;
  } else if (g.type == GL_TRIANGLE_STRIP) {
    pipe = m_ctx->pipeline_strips;
  }
  if (pipe == VK_NULL_HANDLE) return 0;
  vkCmdBindPipeline(m_ctx->current_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);

  VkDeviceSize offset = 0;
  vkCmdBindVertexBuffers(m_ctx->current_cmd, 0, 1, &g.vbuf, &offset);
  if (g.index_count > 0) {
    vkCmdBindIndexBuffer(m_ctx->current_cmd, g.ibuf, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(m_ctx->current_cmd, g.index_count, 1, 0, 0, 0);
    return g.index_count / 3;
  }
  vkCmdDraw(m_ctx->current_cmd, g.vertex_count, 1, 0, 0);
  return g.vertex_count / 3;
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
    sci.maxAnisotropy = 1.f;
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
  }
  return true;
}

// ---------------------------------------------------------------------------
// Frame loop
// ---------------------------------------------------------------------------

void vulkan_renderer::render_submodel(TSubModel *sm, const glm::mat4 &parent,
                                      const glm::mat4 &rot,
                                      const glm::mat4 &proj,
                                      VkCommandBuffer cmd) {
  if (sm == nullptr) return;

  if (sm->iVisible && TSubModel::fSquareDist >= sm->fSquareMinDist &&
      TSubModel::fSquareDist < sm->fSquareMaxDist) {
    glm::mat4 local = parent;
    if ((sm->iFlags & 0xC000) && sm->fMatrix != nullptr) {
      local = parent * glm::make_mat4(sm->fMatrix->readArray());
    }
    // Mesh submodels (eType < TP_ROTATOR) carry drawable geometry. Animations
    // and alpha sorting are ignored for now.
    if (sm->eType < TP_ROTATOR) {
      const glm::mat4 mvp = proj * rot * local;
      vkCmdPushConstants(cmd, m_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                         sizeof(mvp), &mvp);
      m_geometry.draw(sm->m_geometry.handle);
    }
    if (sm->Child != nullptr) render_submodel(sm->Child, local, rot, proj, cmd);
  }
  if (sm->Next != nullptr) render_submodel(sm->Next, parent, rot, proj, cmd);
}

bool vulkan_renderer::Render() {
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

  // UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL for rendering.
  VkImageMemoryBarrier to_color{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  to_color.srcAccessMask = 0;
  to_color.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  to_color.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  to_color.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  to_color.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_color.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_color.image = image;
  to_color.subresourceRange = range;
  vkCmdPipelineBarrier(frame.command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                       nullptr, 0, nullptr, 1, &to_color);

  // UNDEFINED -> DEPTH_ATTACHMENT_OPTIMAL for the depth buffer.
  VkImageMemoryBarrier to_depth{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  to_depth.srcAccessMask = 0;
  to_depth.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  to_depth.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  to_depth.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
  to_depth.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_depth.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_depth.image = m_depth_image;
  to_depth.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(frame.command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0, 0, nullptr,
                       0, nullptr, 1, &to_depth);

  VkRenderingAttachmentInfo color_attachment{
      VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  color_attachment.imageView = view;
  color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  color_attachment.clearValue.color = {{0.06f, 0.10f, 0.18f, 1.0f}};

  VkRenderingAttachmentInfo depth_attachment{
      VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  depth_attachment.imageView = m_depth_view;
  depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
  depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depth_attachment.clearValue.depthStencil = {1.0f, 0};

  VkRenderingInfo rendering_info{VK_STRUCTURE_TYPE_RENDERING_INFO};
  rendering_info.renderArea.offset = {0, 0};
  rendering_info.renderArea.extent = m_swapchain_extent;
  rendering_info.layerCount = 1;
  rendering_info.colorAttachmentCount = 1;
  rendering_info.pColorAttachments = &color_attachment;
  rendering_info.pDepthAttachment = &depth_attachment;

  vkCmdBeginRendering(frame.command_buffer, &rendering_info);

  VkViewport viewport{};
  viewport.x = 0.f;
  viewport.y = 0.f;
  viewport.width = static_cast<float>(m_swapchain_extent.width);
  viewport.height = static_cast<float>(m_swapchain_extent.height);
  viewport.minDepth = 0.f;
  viewport.maxDepth = 1.f;
  vkCmdSetViewport(frame.command_buffer, 0, 1, &viewport);

  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = m_swapchain_extent;
  vkCmdSetScissor(frame.command_buffer, 0, 1, &scissor);

  // Camera-relative scene rendering: rotation-only view, and each section/cell
  // group is translated by (group_center - camera_pos). Geometry is stored
  // relative to its section/cell centre, matching the engine's scheme.
  glm::dmat4 view_d(1.0);
  Global.pCamera.SetMatrix(view_d);
  const glm::mat4 rot = glm::mat4(glm::mat3(view_d));
  glm::mat4 proj = glm::perspectiveFovRH_ZO(
      glm::radians(static_cast<float>(Global.FieldOfView)),
      static_cast<float>(m_swapchain_extent.width),
      static_cast<float>(m_swapchain_extent.height), 0.1f, 5000.f);
  proj[1][1] *= -1.f;  // flip Y for Vulkan clip space
  const glm::dvec3 campos = Global.pCamera.Pos;

  // The geometry bank binds the right pipeline per chunk (triangles/strips);
  // both share the layout, so the per-group MVP push below stays valid.
  m_geo_ctx.current_cmd = frame.command_buffer;

  // Bind the default (white) texture for set 0; persists across the pipeline
  // binds the bank does, since both pipelines share this layout. Per-material
  // textures will rebind this set per draw in step B.
  vkCmdBindDescriptorSets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          m_pipeline_layout, 0, 1, &m_white_descriptor, 0,
                          nullptr);

  auto push_group_mvp = [&](glm::dvec3 const &center) {
    const glm::mat4 model =
        glm::translate(glm::mat4(1.f), glm::vec3(center - campos));
    const glm::mat4 mvp = proj * rot * model;
    vkCmdPushConstants(frame.command_buffer, m_pipeline_layout,
                       VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(mvp), &mvp);
  };

  // Building track geometry needs the default rail profiles loaded; ensure
  // they exist (idempotent) before any create_geometry() touches them.
  TTrack::fetch_default_profiles();

  // Walk the region sections around the camera and draw their opaque scenery.
  // No frustum culling yet; a fixed range keeps the section count bounded.
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

        if (!section->m_shapes.empty()) {
          push_group_mvp(section->m_area.center);
          for (auto const &shape : section->m_shapes) {
            m_geometry.draw(shape.data().geometry);
          }
        }
        for (auto &cell : section->m_cells) {
          if (!cell.m_active || cell.m_shapesopaque.empty()) continue;
          push_group_mvp(cell.m_area.center);
          for (auto const &shape : cell.m_shapesopaque) {
            m_geometry.draw(shape.data().geometry);
          }
        }
      }
    }
  }

  // Player vehicle: render its main model so the locomotive shows.
  if (simulation::Train != nullptr) {
    TDynamicObject *veh = simulation::Train->Dynamic();
    if (veh != nullptr && veh->mdModel != nullptr &&
        veh->mdModel->Root != nullptr) {
      TSubModel::fSquareDist = 0.f;
      veh->ABuLittleUpdate(0.0);
      const glm::mat4 vehicle_model =
          glm::translate(glm::mat4(1.f), glm::vec3(veh->vPosition - campos)) *
          glm::mat4(veh->mMatrix);
      render_submodel(veh->mdModel->Root, vehicle_model, rot, proj,
                      frame.command_buffer);
    }
  }

  m_geo_ctx.current_cmd = VK_NULL_HANDLE;

  // Draw the UI into the same pass: render_ui() -> ImGui::Render() -> the
  // ImGui backend records its draw data into this command buffer.
  if (m_imgui) {
    m_imgui->set_current_frame(frame.command_buffer, m_swapchain_extent);
  }
  Application.render_ui();
  if (m_imgui) {
    m_imgui->set_current_frame(VK_NULL_HANDLE, {});
  }

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
