# Install

## Intel oneAPI Base Toolkit
Download offline installer
```
# from "Installation from the Command Line" Section
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/96aa5993-5b22-4a9b-91ab-da679f422594/intel-oneapi-base-toolkit-2025.0.0.885_offline.sh

# will install in /opt/intel (otherwise ~/intel if not sudo)
sudo sh ./intel-oneapi-base-toolkit-2025.0.0.885_offline.sh -a --silent --cli --eula accept
```
- https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=oneapi-toolkit&oneapi-toolkit-os=linux&oneapi-lin=offline

## oneAPI for AMD
```
wget https://developer.codeplay.com/api/v1/products/download?product=oneapi&variant=amd
sudo sh oneapi-for-amd-gpus-2025.0.0-rocm-6.1.0-linux.sh
```
- https://developer.codeplay.com/products/oneapi/amd/download

## oneMKL for AMD
```
# HIPTARGET
export HIPTARGET=gfx1100

# oneMKL
git clone https://github.com/oneapi-src/oneMKL
cd oneMKL
# Find your HIPTARGET with rocminfo, under the key 'Name:'
cmake -B buildWithrocBLAS -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx -DENABLE_MKLGPU_BACKEND=OFF -DENABLE_MKLCPU_BACKEND=OFF -DENABLE_ROCBLAS_BACKEND=ON -DHIPTARGETS=${HIPTARGET} -DTARGET_DOMAINS=blas
cmake --build buildWithrocBLAS --config Release
```
- https://github.com/ggerganov/llama.cpp/blob/master/docs/backend/SYCL.md#linux

```
ℤ ❯ cmake -B buildWithrocBLAS -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx -DENABLE_MKLGPU_BACKEND=OFF -DENABLE_MKLCPU_BACKEND=OFF -DENABLE_ROCBLAS_BACKEND=ON -DHIPTARGETS=${HIPTARGET} -DTARGET_DOMAINS=blas
-- CMAKE_BUILD_TYPE: None, set to Release by default
-- The CXX compiler identification is IntelLLVM 2025.0.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /opt/intel/oneapi/compiler/2025.0/bin/icpx - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- TARGET_DOMAINS: blas
-- Looking for dpc++
-- Performing Test is_dpcpp
-- Performing Test is_dpcpp - Success
-- Performing Test dpcpp_supports_nvptx64
-- Performing Test dpcpp_supports_nvptx64 - Success
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE
-- The C compiler identification is IntelLLVM 2025.0.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /opt/intel/oneapi/compiler/2025.0/bin/icx - skipped
-- Detecting C compile features
-- Detecting C compile features - done
CMake Deprecation Warning at deps/googletest/CMakeLists.txt:53 (cmake_minimum_required):
  Compatibility with CMake < 3.10 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value.  Or, use the <min>...<max> syntax
  to tell CMake that the project requires at least <min> but has been updated
  to work with policies introduced by <max> or earlier.


CMake Warning (dev) at deps/googletest/cmake/internal_utils.cmake:245 (find_package):
  Policy CMP0148 is not set: The FindPythonInterp and FindPythonLibs modules
  are removed.  Run "cmake --help-policy CMP0148" for policy details.  Use
  the cmake_policy command to set the policy and suppress this warning.

Call Stack (most recent call first):
  deps/googletest/CMakeLists.txt:88 (include)
This warning is for project developers.  Use -Wno-dev to suppress it.

-- Found PythonInterp: /home/lhl/mambaforge/bin/python (found version "3.11.10")
-- Found CBLAS: /usr/lib/libcblas.so
-- Found CBLAS: /usr/lib/libblas.so
-- Found CBLAS: /usr/include
-- ONEAPI_DEVICE_SELECTOR will be set to the following value(s): [hip:gpu] for run-time dispatching examples
-- Configuring done (1.7s)
-- Generating done (0.1s)
CMake Warning:
  Manually-specified variables were not used by the project:

    HIPTARGETS


-- Build files have been written to: /home/lhl/ai/oneMKL/buildWithrocBLAS
```