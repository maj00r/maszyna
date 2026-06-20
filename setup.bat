@echo off
echo Preparing vcpkg packages
cd ref\vcpkg
call bootstrap-vcpkg.bat
vcpkg install directx-dxc:x64-windows
rem Vulkan headers + loader for the Vulkan renderer (find_package(Vulkan))
vcpkg install vulkan-headers:x64-windows vulkan-loader:x64-windows
cd ..\..
@echo ON
