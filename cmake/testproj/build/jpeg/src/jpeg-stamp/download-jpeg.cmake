if(EXISTS "/home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build/downloads/jpegsrc.v9a.tar.gz")
  file("SHA256" "/home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build/downloads/jpegsrc.v9a.tar.gz" hash_value)
  if("x${hash_value}" STREQUAL "x3a753ea48d917945dd54a2d97de388aa06ca2eb1066cbfdc6652036349fe05a7")
    return()
  endif()
endif()
message(STATUS "downloading...
     src='http://www.ijg.org/files/jpegsrc.v9a.tar.gz'
     dst='/home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build/downloads/jpegsrc.v9a.tar.gz'
     timeout='none'")




file(DOWNLOAD
  "http://www.ijg.org/files/jpegsrc.v9a.tar.gz"
  "/home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build/downloads/jpegsrc.v9a.tar.gz"
  SHOW_PROGRESS
  # no TIMEOUT
  STATUS status
  LOG log)

list(GET status 0 status_code)
list(GET status 1 status_string)

if(NOT status_code EQUAL 0)
  message(FATAL_ERROR "error: downloading 'http://www.ijg.org/files/jpegsrc.v9a.tar.gz' failed
  status_code: ${status_code}
  status_string: ${status_string}
  log: ${log}
")
endif()

message(STATUS "downloading... done")
