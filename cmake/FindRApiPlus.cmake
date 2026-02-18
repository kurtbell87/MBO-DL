# FindRApiPlus.cmake
# Locates the Rithmic R | API+ C++ library (pre-compiled static archives).
#
# Searches (in order):
#   1. RAPI_ROOT (CMake variable or environment variable)
#   2. $HOME/.local/rapi/<latest-version>
#
# Provides an IMPORTED target:
#   RApiPlus::RApiPlus
#
# Sets:
#   RApiPlus_FOUND
#   RApiPlus_INCLUDE_DIR
#   RApiPlus_LIBRARY_DIR
#   RApiPlus_SSL_CERT       - path to the rithmic_ssl_cert_auth_params PEM file
#   RApiPlus_VERSION

# --- Resolve the root directory ---
if(NOT RAPI_ROOT)
    set(RAPI_ROOT "$ENV{RAPI_ROOT}")
endif()

if(NOT RAPI_ROOT)
    # Auto-detect: pick the newest version directory under ~/.local/rapi
    file(GLOB _rapi_versions "$ENV{HOME}/.local/rapi/*")
    if(_rapi_versions)
        list(SORT _rapi_versions COMPARE NATURAL ORDER DESCENDING)
        list(GET _rapi_versions 0 RAPI_ROOT)
    endif()
endif()

# --- Locate the header ---
find_path(RApiPlus_INCLUDE_DIR
    NAMES RApiPlus.h
    PATHS "${RAPI_ROOT}/include"
    NO_DEFAULT_PATH
)

# --- Determine platform library subdirectory ---
if(APPLE)
    execute_process(COMMAND uname -m OUTPUT_VARIABLE _arch OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_arch STREQUAL "arm64")
        set(_rapi_libdir "${RAPI_ROOT}/darwin-20.6-arm64/lib")
    else()
        set(_rapi_libdir "${RAPI_ROOT}/darwin-10/lib")
    endif()
elseif(UNIX)
    execute_process(COMMAND uname -r OUTPUT_VARIABLE _kver OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_kver VERSION_GREATER_EQUAL "4.18")
        set(_rapi_libdir "${RAPI_ROOT}/linux-gnu-4.18-x86_64/lib")
    else()
        set(_rapi_libdir "${RAPI_ROOT}/linux-gnu-3.10.0-x86_64/lib")
    endif()
endif()

set(RApiPlus_LIBRARY_DIR "${_rapi_libdir}" CACHE PATH "R|API+ library directory")

# --- Locate individual static archives ---
set(_rapi_lib_names
    RApiPlus-optimize
    OmneStreamEngine-optimize
    OmneChannel-optimize
    OmneEngine-optimize
    _api-optimize
    _apipoll-stubs-optimize
    _kit-optimize
    ssl
    crypto
)

set(_rapi_libs "")
foreach(_name ${_rapi_lib_names})
    find_library(_lib_${_name}
        NAMES ${_name}
        PATHS "${_rapi_libdir}"
        NO_DEFAULT_PATH
    )
    if(_lib_${_name})
        list(APPEND _rapi_libs "${_lib_${_name}}")
    endif()
endforeach()

# --- SSL certificate ---
set(RApiPlus_SSL_CERT "${RAPI_ROOT}/etc/rithmic_ssl_cert_auth_params"
    CACHE FILEPATH "Rithmic SSL certificate file")

# --- Extract version from path ---
get_filename_component(_rapi_dirname "${RAPI_ROOT}" NAME)
set(RApiPlus_VERSION "${_rapi_dirname}" CACHE STRING "R|API+ version")

# --- Standard find_package handling ---
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RApiPlus
    REQUIRED_VARS RApiPlus_INCLUDE_DIR RApiPlus_LIBRARY_DIR _rapi_libs RApiPlus_SSL_CERT
    VERSION_VAR RApiPlus_VERSION
)

# --- Create the IMPORTED target ---
if(RApiPlus_FOUND AND NOT TARGET RApiPlus::RApiPlus)
    add_library(RApiPlus::RApiPlus INTERFACE IMPORTED)
    set_target_properties(RApiPlus::RApiPlus PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${RApiPlus_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${_rapi_libs};z;pthread"
        INTERFACE_COMPILE_DEFINITIONS "_REENTRANT"
    )
endif()
