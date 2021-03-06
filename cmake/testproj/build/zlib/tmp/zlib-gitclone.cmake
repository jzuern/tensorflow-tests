if("50893291621658f355bc5b4d450a8d06a563053d" STREQUAL "")
  message(FATAL_ERROR "Tag for git checkout should not be empty.")
endif()

set(run 0)

if("/home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build/zlib/src/zlib-stamp/zlib-gitinfo.txt" IS_NEWER_THAN "/home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build/zlib/src/zlib-stamp/zlib-gitclone-lastrun.txt")
  set(run 1)
endif()

if(NOT run)
  message(STATUS "Avoiding repeated git clone, stamp file is up to date: '/home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build/zlib/src/zlib-stamp/zlib-gitclone-lastrun.txt'")
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E remove_directory "/home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build/zlib/src/zlib"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build/zlib/src/zlib'")
endif()

# try the clone 3 times incase there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/bin/git" clone --origin "origin" "https://github.com/madler/zlib" "zlib"
    WORKING_DIRECTORY "/home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build/zlib/src"
    RESULT_VARIABLE error_code
    )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once:
          ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/madler/zlib'")
endif()

execute_process(
  COMMAND "/usr/bin/git" checkout 50893291621658f355bc5b4d450a8d06a563053d
  WORKING_DIRECTORY "/home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build/zlib/src/zlib"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: '50893291621658f355bc5b4d450a8d06a563053d'")
endif()

execute_process(
  COMMAND "/usr/bin/git" submodule init 
  WORKING_DIRECTORY "/home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build/zlib/src/zlib"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to init submodules in: '/home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build/zlib/src/zlib'")
endif()

execute_process(
  COMMAND "/usr/bin/git" submodule update --recursive 
  WORKING_DIRECTORY "/home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build/zlib/src/zlib"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build/zlib/src/zlib'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy
    "/home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build/zlib/src/zlib-stamp/zlib-gitinfo.txt"
    "/home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build/zlib/src/zlib-stamp/zlib-gitclone-lastrun.txt"
  WORKING_DIRECTORY "/home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build/zlib/src/zlib"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build/zlib/src/zlib-stamp/zlib-gitclone-lastrun.txt'")
endif()

