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
include test/CMakeFiles/test_gist_l2.dir/depend.make

# Include the progress variables for this target.
include test/CMakeFiles/test_gist_l2.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/test_gist_l2.dir/flags.make

test/CMakeFiles/test_gist_l2.dir/test_gist_l2.cpp.o: test/CMakeFiles/test_gist_l2.dir/flags.make
test/CMakeFiles/test_gist_l2.dir/test_gist_l2.cpp.o: ../test/test_gist_l2.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/workspace/MultiVectorEngine/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/test_gist_l2.dir/test_gist_l2.cpp.o"
	cd /root/workspace/MultiVectorEngine/cmake/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_gist_l2.dir/test_gist_l2.cpp.o -c /root/workspace/MultiVectorEngine/test/test_gist_l2.cpp

test/CMakeFiles/test_gist_l2.dir/test_gist_l2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_gist_l2.dir/test_gist_l2.cpp.i"
	cd /root/workspace/MultiVectorEngine/cmake/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/workspace/MultiVectorEngine/test/test_gist_l2.cpp > CMakeFiles/test_gist_l2.dir/test_gist_l2.cpp.i

test/CMakeFiles/test_gist_l2.dir/test_gist_l2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_gist_l2.dir/test_gist_l2.cpp.s"
	cd /root/workspace/MultiVectorEngine/cmake/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/workspace/MultiVectorEngine/test/test_gist_l2.cpp -o CMakeFiles/test_gist_l2.dir/test_gist_l2.cpp.s

test/CMakeFiles/test_gist_l2.dir/utils.cpp.o: test/CMakeFiles/test_gist_l2.dir/flags.make
test/CMakeFiles/test_gist_l2.dir/utils.cpp.o: ../test/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/workspace/MultiVectorEngine/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object test/CMakeFiles/test_gist_l2.dir/utils.cpp.o"
	cd /root/workspace/MultiVectorEngine/cmake/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_gist_l2.dir/utils.cpp.o -c /root/workspace/MultiVectorEngine/test/utils.cpp

test/CMakeFiles/test_gist_l2.dir/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_gist_l2.dir/utils.cpp.i"
	cd /root/workspace/MultiVectorEngine/cmake/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/workspace/MultiVectorEngine/test/utils.cpp > CMakeFiles/test_gist_l2.dir/utils.cpp.i

test/CMakeFiles/test_gist_l2.dir/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_gist_l2.dir/utils.cpp.s"
	cd /root/workspace/MultiVectorEngine/cmake/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/workspace/MultiVectorEngine/test/utils.cpp -o CMakeFiles/test_gist_l2.dir/utils.cpp.s

# Object files for target test_gist_l2
test_gist_l2_OBJECTS = \
"CMakeFiles/test_gist_l2.dir/test_gist_l2.cpp.o" \
"CMakeFiles/test_gist_l2.dir/utils.cpp.o"

# External object files for target test_gist_l2
test_gist_l2_EXTERNAL_OBJECTS =

test/test_gist_l2: test/CMakeFiles/test_gist_l2.dir/test_gist_l2.cpp.o
test/test_gist_l2: test/CMakeFiles/test_gist_l2.dir/utils.cpp.o
test/test_gist_l2: test/CMakeFiles/test_gist_l2.dir/build.make
test/test_gist_l2: src/libmulti_vector.so
test/test_gist_l2: test/CMakeFiles/test_gist_l2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/workspace/MultiVectorEngine/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable test_gist_l2"
	cd /root/workspace/MultiVectorEngine/cmake/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_gist_l2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/test_gist_l2.dir/build: test/test_gist_l2

.PHONY : test/CMakeFiles/test_gist_l2.dir/build

test/CMakeFiles/test_gist_l2.dir/clean:
	cd /root/workspace/MultiVectorEngine/cmake/test && $(CMAKE_COMMAND) -P CMakeFiles/test_gist_l2.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/test_gist_l2.dir/clean

test/CMakeFiles/test_gist_l2.dir/depend:
	cd /root/workspace/MultiVectorEngine/cmake && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/workspace/MultiVectorEngine /root/workspace/MultiVectorEngine/test /root/workspace/MultiVectorEngine/cmake /root/workspace/MultiVectorEngine/cmake/test /root/workspace/MultiVectorEngine/cmake/test/CMakeFiles/test_gist_l2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/test_gist_l2.dir/depend

