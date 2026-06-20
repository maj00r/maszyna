#include "utilities/Logs.h"
#include "nvrenderer/nvrenderer.h"

namespace {
std::unique_ptr<gfx_renderer> create_nvrenderer_for_vulkan() {
  return std::make_unique<NvRenderer>(NvRenderer::Api::Vulkan);
}
}  // namespace

// The NVRHI renderer is now a Vulkan-only renderer, available under the
// canonical "vulkan" name. The historical "experimental" names are kept as
// aliases so that existing configurations keep working.
bool register_manul_vulkan =
    gfx_renderer_factory::get_instance()->register_backend(
        "vulkan", &create_nvrenderer_for_vulkan);
bool register_manul_vulkan_alias =
    gfx_renderer_factory::get_instance()->register_backend(
        "experimental_vk", &create_nvrenderer_for_vulkan);
bool register_manul_alias =
    gfx_renderer_factory::get_instance()->register_backend(
        "experimental", &create_nvrenderer_for_vulkan);
