file (GLOB HEADERS "*.h")
file (GLOB SOURCES "[^_]*.cpp")

if(MVSTEX_GPU)
    list(APPEND SOURCES cuda/visibility_kernel.cu)
    list(APPEND HEADERS cuda/visibility_gpu.h)
    list(APPEND SOURCES cuda/edge_padding_kernel.cu)
    list(APPEND HEADERS cuda/edge_padding_gpu.h)
endif()

# poisson_blending.cpp uses Eigen::SimplicialLDLT/AMDOrdering (Eigen's
# SparseCholesky module, LGPL until Eigen 3.4+), which the project-wide
# -DEIGEN_MPL2_ONLY (set above) blocks. -U re-undefines the macro for
# just this one translation unit rather than lifting the restriction
# fork-wide -- see docs/EIGEN_COMPLIANCE_NOTE.md for why this is safe
# for this project and what it means for anyone using this file outside
# of it.
set_source_files_properties(poisson_blending.cpp PROPERTIES
    COMPILE_OPTIONS "-UEIGEN_MPL2_ONLY")

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
