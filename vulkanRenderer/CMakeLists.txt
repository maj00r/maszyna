# Native Vulkan renderer (raw Vulkan API), replacement for the retired
# NVRHI-based renderer.

find_package(Vulkan REQUIRED)

# --- shader toolchain (Vulkan SDK) -----------------------------------------
find_program(GLSLANG_VALIDATOR
        NAMES glslangValidator
        HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
        DOC "glslangValidator from the Vulkan SDK")
if (NOT GLSLANG_VALIDATOR)
    message(FATAL_ERROR
            "glslangValidator not found. Install the Vulkan SDK and make sure "
            "VULKAN_SDK is set (or glslangValidator is on PATH).")
endif ()

set(SHADER_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${SHADER_GEN_DIR}")

# Compiles a GLSL shader to a SPIR-V C array header (<vn>.h) embedded in the
# binary, so there are no runtime shader files to ship or locate.
set(SHADER_HEADERS "")
function(embed_spirv_shader SRC VN)
    set(out "${SHADER_GEN_DIR}/${VN}.h")
    add_custom_command(
            OUTPUT "${out}"
            COMMAND "${GLSLANG_VALIDATOR}" -V "${CMAKE_CURRENT_SOURCE_DIR}/${SRC}"
                    --vn ${VN} -o "${out}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${SRC}"
            COMMENT "Compiling SPIR-V (embedded): ${SRC}"
            VERBATIM)
    set(SHADER_HEADERS "${SHADER_HEADERS};${out}" PARENT_SCOPE)
endfunction()

embed_spirv_shader("shaders/world.vert" world_vert_spv)
embed_spirv_shader("shaders/world.frag" world_frag_spv)
embed_spirv_shader("shaders/imgui.vert" imgui_vert_spv)
embed_spirv_shader("shaders/imgui.frag" imgui_frag_spv)
embed_spirv_shader("shaders/pick.vert" pick_vert_spv)
embed_spirv_shader("shaders/pick.frag" pick_frag_spv)
embed_spirv_shader("shaders/skydome.vert" skydome_vert_spv)
embed_spirv_shader("shaders/skydome.frag" skydome_frag_spv)
embed_spirv_shader("shaders/shadow.vert" shadow_vert_spv)
embed_spirv_shader("shaders/gbuffer.frag" gbuffer_frag_spv)
embed_spirv_shader("shaders/deferred_light.vert" deferred_light_vert_spv)
embed_spirv_shader("shaders/deferred_light.frag" deferred_light_frag_spv)

add_custom_target(vulkanRenderer_shaders DEPENDS ${SHADER_HEADERS})

add_library(vulkanRenderer STATIC
        "vulkan_renderer.cpp"
        "vulkan_imgui.cpp")

add_dependencies(vulkanRenderer vulkanRenderer_shaders)
target_include_directories(vulkanRenderer PRIVATE "${SHADER_GEN_DIR}")

target_compile_features(vulkanRenderer PRIVATE cxx_std_20)

# Same core-engine include set the previous renderer needed so that
# model/material/texture/geometry headers resolve.
target_include_directories(vulkanRenderer PRIVATE
        "${GLOBAL_ROOT_DIR}"
        "${GLOBAL_INCLUDE_DIR}"
        "${IMGUI_DIR}"
        "${GLM_INCLUDE_DIR}"
        "${GLOBAL_ROOT_DIR}/McZapkie"
        "${OPENAL_INCLUDE_DIR}"
        "${DEPS_DIR}/glad/include"
        "${PYTHON_INCLUDE_DIR}"
        "${libserialport_INCLUDE_DIR}"
        "${LUAJIT_INCLUDE_DIR}")

# GLM_*/_USE_MATH_DEFINES/NOMINMAX are already set by stdafx.h. ${DEFINITIONS}
# carries the global_settings size guard and must match the rest of the engine.
target_compile_definitions(vulkanRenderer PRIVATE
        ${DEFINITIONS})

target_link_libraries(vulkanRenderer PUBLIC Vulkan::Vulkan)
target_link_libraries(vulkanRenderer PRIVATE glfw)

# Compile the factory registration directly into the consuming executable so
# the static initializer is never stripped by the linker.
target_sources(vulkanRenderer INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/register.cpp")
