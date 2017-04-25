cmake_minimum_required(VERSION 2.8.9)


project(test_proj)

option(USE_SYSTEM_LIBRARIES "instead of downloading and building external libraries into tensorflowlib, just link to them" ON)



# External dependencies
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external)

# Location where external projects will be downloaded
set (DOWNLOAD_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/downloads"
     CACHE PATH "Location where external projects will be downloaded.")
mark_as_advanced(DOWNLOAD_LOCATION)


#Bring the headers into the project
include_directories(include)

# set the path to the library folder (optional)
link_directories(/usr/local/lib)

# wildcard additions:
file(GLOB SOURCES "src/*.cpp")

# executable is test_proj
add_executable(test_proj ${SOURCES})



if (USE_SYSTEM_LIBRARIES)
  find_library(JPEGLIB jpeg)
  if(NOT JPEGLIB)
    message(STATUS "libjpeg library not found, so sorry")
  else()
    message(STATUS "found libjpeg shared library (libjpeg.so) in local system. Linking to it")
    target_link_libraries (test_proj libjpeg)
  endif()
else() # no system libraries
  message(STATUS "libjpeg will be built")
  include(jpeg)
  set(test_proj_EXTERNAL_LIBRARIES
    ${jpeg_STATIC_LIBRARIES}
  )
  set(test_proj_EXTERNAL_DEPENDENCIES
  jpeg_copy_headers_to_destination
  )
  include_directories(
    # Source and generated code.
    ${test_proj_source_dir}
    ${CMAKE_CURRENT_BINARY_DIR}
    # External dependencies.
    ${jpeg_INCLUDE_DIR}
  )
endif()
