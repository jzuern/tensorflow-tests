set(file "/home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build/downloads/jpegsrc.v9a.tar.gz")
message(STATUS "verifying file...
     file='${file}'")
set(expect_value "3a753ea48d917945dd54a2d97de388aa06ca2eb1066cbfdc6652036349fe05a7")
set(attempt 0)
set(succeeded 0)
while(${attempt} LESS 3 OR ${attempt} EQUAL 3 AND NOT ${succeeded})
  file(SHA256 "${file}" actual_value)
  if("${actual_value}" STREQUAL "${expect_value}")
    set(succeeded 1)
  elseif(${attempt} LESS 3)
    message(STATUS "SHA256 hash of ${file}
does not match expected value
  expected: ${expect_value}
    actual: ${actual_value}
Retrying download.
")
    file(REMOVE "${file}")
    execute_process(COMMAND ${CMAKE_COMMAND} -P "/home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build/jpeg/src/jpeg-stamp/download-jpeg.cmake")
  endif()
  math(EXPR attempt "${attempt} + 1")
endwhile()

if(${succeeded})
  message(STATUS "verifying file... done")
else()
  message(FATAL_ERROR "error: SHA256 hash of
  ${file}
does not match expected value
  expected: ${expect_value}
    actual: ${actual_value}
")
endif()
