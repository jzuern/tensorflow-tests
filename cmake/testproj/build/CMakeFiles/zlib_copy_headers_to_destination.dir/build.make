# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build

# Utility rule file for zlib_copy_headers_to_destination.

# Include the progress variables for this target.
include CMakeFiles/zlib_copy_headers_to_destination.dir/progress.make

CMakeFiles/zlib_copy_headers_to_destination:


zlib_copy_headers_to_destination: CMakeFiles/zlib_copy_headers_to_destination
zlib_copy_headers_to_destination: CMakeFiles/zlib_copy_headers_to_destination.dir/build.make
	/usr/bin/cmake -E copy_if_different /home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build/zlib/install/include/zconf.h /home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build/external/zlib_archive
	/usr/bin/cmake -E copy_if_different /home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build/zlib/install/include/zlib.h /home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build/external/zlib_archive
.PHONY : zlib_copy_headers_to_destination

# Rule to build all files generated by this target.
CMakeFiles/zlib_copy_headers_to_destination.dir/build: zlib_copy_headers_to_destination

.PHONY : CMakeFiles/zlib_copy_headers_to_destination.dir/build

CMakeFiles/zlib_copy_headers_to_destination.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/zlib_copy_headers_to_destination.dir/cmake_clean.cmake
.PHONY : CMakeFiles/zlib_copy_headers_to_destination.dir/clean

CMakeFiles/zlib_copy_headers_to_destination.dir/depend:
	cd /home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj /home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj /home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build /home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build /home/jzuern/Dropbox/develop/hiwi_mrt/cmake/testproj/build/CMakeFiles/zlib_copy_headers_to_destination.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/zlib_copy_headers_to_destination.dir/depend

