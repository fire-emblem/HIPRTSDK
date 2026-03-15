cmake_minimum_required(VERSION 3.19)
project(HIPRTSDKTutorials LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(HIPRT_SOURCE_DIR "/data/HIPRT" CACHE PATH "Path to the HIPRT source/build tree")
set(HIPRT_LIB_DIR "${HIPRT_SOURCE_DIR}/dist/bin/Release" CACHE PATH "Directory containing libhiprt*.so")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tutorials/dist/bin/Release")

find_library(HIPRT_LIBRARY
	NAMES hiprt0300164 hiprt0300164.so libhiprt0300164.so
	PATHS "${HIPRT_LIB_DIR}"
	NO_DEFAULT_PATH
)

if(NOT HIPRT_LIBRARY)
	message(FATAL_ERROR "Failed to find HIPRT library under ${HIPRT_LIB_DIR}")
endif()

set(TUTORIAL_COMMON_SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/tutorials/common/TutorialBase.cpp
)

set(TUTORIAL_COMMON_INCLUDES
	${CMAKE_CURRENT_SOURCE_DIR}
	${HIPRT_SOURCE_DIR}
	${CUDAToolkit_INCLUDE_DIRS}
)

function(add_hiprtsdk_tutorial target source)
	add_executable(${target} ${source} ${TUTORIAL_COMMON_SOURCES})
	target_include_directories(${target} PRIVATE ${TUTORIAL_COMMON_INCLUDES})
	target_compile_definitions(${target} PRIVATE
		HIPRTSDK_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
		HIPRT_ROOT_DIRECTORY="${HIPRT_SOURCE_DIR}"
		HIPRT_CUDA_INCLUDE_DIR="${CUDAToolkit_INCLUDE_DIRS}"
	)
	target_link_libraries(${target} PRIVATE
		${HIPRT_LIBRARY}
		CUDA::cudart
		CUDA::cuda_driver
		CUDA::nvrtc
	)
endfunction()

add_hiprtsdk_tutorial(00_context_creation64 ${CMAKE_CURRENT_SOURCE_DIR}/tutorials/00_context_creation/main.cpp)
add_hiprtsdk_tutorial(01_geom_intersection64 ${CMAKE_CURRENT_SOURCE_DIR}/tutorials/01_geom_intersection/main.cpp)
add_hiprtsdk_tutorial(02_scene_intersection64 ${CMAKE_CURRENT_SOURCE_DIR}/tutorials/02_scene_intersection/main.cpp)
add_hiprtsdk_tutorial(03_custom_intersection64 ${CMAKE_CURRENT_SOURCE_DIR}/tutorials/03_custom_intersection/main.cpp)
add_hiprtsdk_tutorial(04_compaction64 ${CMAKE_CURRENT_SOURCE_DIR}/tutorials/04_compaction/main.cpp)
add_hiprtsdk_tutorial(05_global_stack64 ${CMAKE_CURRENT_SOURCE_DIR}/tutorials/05_global_stack/main.cpp)
add_hiprtsdk_tutorial(06_dynamic_stack64 ${CMAKE_CURRENT_SOURCE_DIR}/tutorials/06_dynamic_stack/main.cpp)
add_hiprtsdk_tutorial(07_custom_bvh_import64 ${CMAKE_CURRENT_SOURCE_DIR}/tutorials/07_custom_bvh_import/main.cpp)
add_hiprtsdk_tutorial(08_ambient_occlusion64 ${CMAKE_CURRENT_SOURCE_DIR}/tutorials/08_ambient_occlusion/main.cpp)
add_hiprtsdk_tutorial(12_cutout64 ${CMAKE_CURRENT_SOURCE_DIR}/tutorials/12_cutout/main.cpp)
add_hiprtsdk_tutorial(13_concurrent_scene_build64 ${CMAKE_CURRENT_SOURCE_DIR}/tutorials/13_concurrent_scene_build/main.cpp)
add_hiprtsdk_tutorial(14_batch_build64 ${CMAKE_CURRENT_SOURCE_DIR}/tutorials/14_batch_build/main.cpp)
add_hiprtsdk_tutorial(15_multi_level_instancing64 ${CMAKE_CURRENT_SOURCE_DIR}/tutorials/15_multi_level_instancing/main.cpp)
