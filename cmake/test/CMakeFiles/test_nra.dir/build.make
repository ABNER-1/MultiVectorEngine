# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

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
CMAKE_COMMAND = /root/cmake-3.17.5-Linux-x86_64/bin/cmake

# The command to remove a file.
RM = /root/cmake-3.17.5-Linux-x86_64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/workspace/MultiVectorEngine

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/workspace/MultiVectorEngine/cmake

# Include any dependencies generated for this target.
include test/CMakeFiles/test_nra.dir/depend.make

# Include the progress variables for this target.
include test/CMakeFiles/test_nra.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/test_nra.dir/flags.make

test/CMakeFiles/test_nra.dir/test_nra.cpp.o: test/CMakeFiles/test_nra.dir/flags.make
test/CMakeFiles/test_nra.dir/test_nra.cpp.o: ../test/test_nra.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/workspace/MultiVectorEngine/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/test_nra.dir/test_nra.cpp.o"
	cd /root/workspace/MultiVectorEngine/cmake/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_nra.dir/test_nra.cpp.o -c /root/workspace/MultiVectorEngine/test/test_nra.cpp

test/CMakeFiles/test_nra.dir/test_nra.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_nra.dir/test_nra.cpp.i"
	cd /root/workspace/MultiVectorEngine/cmake/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/workspace/MultiVectorEngine/test/test_nra.cpp > CMakeFiles/test_nra.dir/test_nra.cpp.i

test/CMakeFiles/test_nra.dir/test_nra.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_nra.dir/test_nra.cpp.s"
	cd /root/workspace/MultiVectorEngine/cmake/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/workspace/MultiVectorEngine/test/test_nra.cpp -o CMakeFiles/test_nra.dir/test_nra.cpp.s

# Object files for target test_nra
test_nra_OBJECTS = \
"CMakeFiles/test_nra.dir/test_nra.cpp.o"

# External object files for target test_nra
test_nra_EXTERNAL_OBJECTS =

test/test_nra: test/CMakeFiles/test_nra.dir/test_nra.cpp.o
test/test_nra: test/CMakeFiles/test_nra.dir/build.make
test/test_nra: src/libmulti_vector.so
test/test_nra: test/CMakeFiles/test_nra.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/workspace/MultiVectorEngine/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_nra"
	cd /root/workspace/MultiVectorEngine/cmake/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_nra.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/test_nra.dir/build: test/test_nra

.PHONY : test/CMakeFiles/test_nra.dir/build

test/CMakeFiles/test_nra.dir/clean:
	cd /root/workspace/MultiVectorEngine/cmake/test && $(CMAKE_COMMAND) -P CMakeFiles/test_nra.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/test_nra.dir/clean

test/CMakeFiles/test_nra.dir/depend:
	cd /root/workspace/MultiVectorEngine/cmake && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/workspace/MultiVectorEngine /root/workspace/MultiVectorEngine/test /root/workspace/MultiVectorEngine/cmake /root/workspace/MultiVectorEngine/cmake/test /root/workspace/MultiVectorEngine/cmake/test/CMakeFiles/test_nra.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/test_nra.dir/depend

