file (GLOB HEADERS "*.h")
file (GLOB SOURCES "[^_]*.cpp")

if(MVSTEX_GPU)
    list(APPEND SOURCES cuda/visibility_kernel.cu)
    list(APPEND HEADERS cuda/visibility_gpu.h)
endif()

set(LIBRARY tex)
add_library(${LIBRARY} STATIC ${SOURCES})
set_property(TARGET ${LIBRARY} PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
add_dependencies(${LIBRARY} ext_rayint ext_eigen ext_mapmap)
include_directories(${TBB_INCLUDE_DIRS})
target_link_libraries(${LIBRARY} ${MVE_MVE_LIBRARY} ${MVE_UTIL_LIBRARY} ${JPEG_LIBRARIES} ${PNG_LIBRARIES} ${TIFF_LIBRARIES} TBB::tbb)
if(MVSTEX_GPU)
    set_property(TARGET ${LIBRARY} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    # Mixing nvcc-compiled objects with whole-program IPO/LTO is a known
    # source of first-build toolchain breakage; turn it off for this target
    # only while the CUDA path is new/unproven. Revisit once it's been
    # built successfully a few times.
    set_property(TARGET ${LIBRARY} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    target_link_libraries(${LIBRARY} CUDA::cudart)
endif()
install(TARGETS ${LIBRARY} ARCHIVE DESTINATION lib)
