cmake_minimum_required(VERSION 3.26)
project(phi2)

set(ORT_HOME /nakersha_data/Develop/install/onnxruntime)

###############################################################################
## file globbing ##############################################################
###############################################################################

# these instructions search the directory tree when CMake is
# invoked and put all files that match the pattern in the variables
# `sources` and `data`
file(GLOB_RECURSE sources      main.cpp)
#file(GLOB_RECURSE sources_test src/test/*.cpp)
#file(GLOB_RECURSE data resources/*)
# you can use set(sources src/main.cpp) etc if you don't want to
# use globbing to find files automatically

###############################################################################
## target definitions #########################################################
###############################################################################

# add the data to the target, so it becomes visible in some IDE
add_executable(example ${sources})

# just for example add some compiler flags
target_compile_options(example PUBLIC -std=c++20)

# this lets me include files relative to the root source directory with a <> pair
target_include_directories(example PUBLIC ${ORT_HOME}/include)

# this copies all resource files in the build directory
# we need this, because we want to work with paths relative to the executable
#file(COPY ${data} DESTINATION resources)

###############################################################################
## dependencies ###############################################################
###############################################################################

# this defines the variables Boost_LIBRARIES that contain all library names
# that we need to link to
#find_package(Boost 1.36.0 COMPONENTS filesystem system REQUIRED)

target_link_directories(example PUBLIC ${ORT_HOME}/lib)

target_link_libraries(example PUBLIC
  onnxruntime-genai
  onnxruntime
  # here you can add any library dependencies
)
