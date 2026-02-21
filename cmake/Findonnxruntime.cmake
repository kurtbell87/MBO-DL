# Find onnxruntime pre-built release (C API)
#
# Looks for the pre-built binary release from GitHub, e.g.:
#   https://github.com/microsoft/onnxruntime/releases
#
# Inputs:
#   onnxruntime_DIR or CMAKE_PREFIX_PATH pointing to extraction root
#
# Outputs:
#   onnxruntime_FOUND
#   onnxruntime::onnxruntime imported target

if(TARGET onnxruntime::onnxruntime)
    return()
endif()

find_path(onnxruntime_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS
        ${onnxruntime_DIR}/include
    PATH_SUFFIXES include
)

find_library(onnxruntime_LIBRARY
    NAMES onnxruntime
    PATHS
        ${onnxruntime_DIR}/lib
    PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(onnxruntime
    REQUIRED_VARS onnxruntime_LIBRARY onnxruntime_INCLUDE_DIR
)

if(onnxruntime_FOUND AND NOT TARGET onnxruntime::onnxruntime)
    add_library(onnxruntime::onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION "${onnxruntime_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_INCLUDE_DIR}"
    )
endif()
