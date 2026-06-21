/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// This translation unit is compiled directly into the main executable (added
// to it as an INTERFACE source of the vulkanRenderer target). That guarantees
// the static registration below is not dropped by the linker, the way it would
// be if it lived inside the static library and nothing referenced it.

#include "rendering/renderer.h"

// Defined in the vulkanRenderer library; deliberately keeps Vulkan headers out
// of the executable's translation units.
std::unique_ptr<gfx_renderer> create_vulkan_renderer();

namespace {
bool register_vulkan_renderer =
    gfx_renderer_factory::get_instance()->register_backend(
        "vulkan", &create_vulkan_renderer);
}  // namespace
