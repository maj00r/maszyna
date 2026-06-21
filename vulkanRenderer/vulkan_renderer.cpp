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
#include <set>

#include "application/application.h"
#include "utilities/Globals.h"
#include "utilities/Logs.h"

// SPIR-V embedded by the build (glslangValidator --vn). Each header defines a
// `const uint32_t <name>[]` array.
#include "triangle_vert_spv.h"
#include "triangle_frag_spv.h"

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
  if (!create_pipeline()) return false;
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

    if (m_pipeline) {
      vkDestroyPipeline(m_device, m_pipeline, nullptr);
      m_pipeline = VK_NULL_HANDLE;
    }
    if (m_pipeline_layout) {
      vkDestroyPipelineLayout(m_device, m_pipeline_layout, nullptr);
      m_pipeline_layout = VK_NULL_HANDLE;
    }

    destroy_swapchain();

    vkDestroyDevice(m_device, nullptr);
    m_device = VK_NULL_HANDLE;
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

  return create_image_views();
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

bool vulkan_renderer::create_pipeline() {
  VkShaderModule vert =
      create_shader_module(triangle_vert_spv, sizeof(triangle_vert_spv));
  VkShaderModule frag =
      create_shader_module(triangle_frag_spv, sizeof(triangle_frag_spv));
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

  VkPipelineVertexInputStateCreateInfo vertex_input{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};

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

  VkPipelineLayoutCreateInfo layout_ci{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  if (vkCreatePipelineLayout(m_device, &layout_ci, nullptr,
                             &m_pipeline_layout) != VK_SUCCESS) {
    log_error("vkCreatePipelineLayout failed.");
    return false;
  }

  // Dynamic rendering: declare the color attachment format instead of a
  // VkRenderPass.
  VkPipelineRenderingCreateInfo rendering_ci{
      VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rendering_ci.colorAttachmentCount = 1;
  rendering_ci.pColorAttachmentFormats = &m_swapchain_format;

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
  pipeline_ci.pDynamicState = &dynamic_state;
  pipeline_ci.layout = m_pipeline_layout;
  pipeline_ci.renderPass = VK_NULL_HANDLE;

  VkResult res = vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1,
                                           &pipeline_ci, nullptr, &m_pipeline);

  vkDestroyShaderModule(m_device, vert, nullptr);
  vkDestroyShaderModule(m_device, frag, nullptr);

  if (res != VK_SUCCESS) {
    log_error(std::string("vkCreateGraphicsPipelines failed: ") +
              vk_result_str(res));
    return false;
  }
  return true;
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

bool vulkan_renderer::create_frame_resources() {
  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = m_graphics_family;
  VK_CHECK(vkCreateCommandPool(m_device, &pci, nullptr, &m_command_pool));

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

  VkRenderingAttachmentInfo color_attachment{
      VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  color_attachment.imageView = view;
  color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  color_attachment.clearValue.color = {{0.06f, 0.10f, 0.18f, 1.0f}};

  VkRenderingInfo rendering_info{VK_STRUCTURE_TYPE_RENDERING_INFO};
  rendering_info.renderArea.offset = {0, 0};
  rendering_info.renderArea.extent = m_swapchain_extent;
  rendering_info.layerCount = 1;
  rendering_info.colorAttachmentCount = 1;
  rendering_info.pColorAttachments = &color_attachment;

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

  vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    m_pipeline);
  vkCmdDraw(frame.command_buffer, 3, 1, 0, 0);

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
