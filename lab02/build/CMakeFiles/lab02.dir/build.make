# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_SOURCE_DIR = /home/sitindustry/Documents/CVLab/lab02

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sitindustry/Documents/CVLab/lab02/build

# Include any dependencies generated for this target.
include CMakeFiles/lab02.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/lab02.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/lab02.dir/flags.make

CMakeFiles/lab02.dir/rectify_homo.cpp.o: CMakeFiles/lab02.dir/flags.make
CMakeFiles/lab02.dir/rectify_homo.cpp.o: ../rectify_homo.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sitindustry/Documents/CVLab/lab02/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/lab02.dir/rectify_homo.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lab02.dir/rectify_homo.cpp.o -c /home/sitindustry/Documents/CVLab/lab02/rectify_homo.cpp

CMakeFiles/lab02.dir/rectify_homo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lab02.dir/rectify_homo.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sitindustry/Documents/CVLab/lab02/rectify_homo.cpp > CMakeFiles/lab02.dir/rectify_homo.cpp.i

CMakeFiles/lab02.dir/rectify_homo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lab02.dir/rectify_homo.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sitindustry/Documents/CVLab/lab02/rectify_homo.cpp -o CMakeFiles/lab02.dir/rectify_homo.cpp.s

# Object files for target lab02
lab02_OBJECTS = \
"CMakeFiles/lab02.dir/rectify_homo.cpp.o"

# External object files for target lab02
lab02_EXTERNAL_OBJECTS =

lab02: CMakeFiles/lab02.dir/rectify_homo.cpp.o
lab02: CMakeFiles/lab02.dir/build.make
lab02: /usr/local/opencv-4.5.2/lib/libopencv_gapi.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_stitching.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_alphamat.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_aruco.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_bgsegm.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_bioinspired.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_ccalib.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_cvv.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_dnn_objdetect.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_dnn_superres.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_dpm.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_face.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_freetype.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_fuzzy.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_hdf.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_hfs.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_img_hash.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_intensity_transform.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_line_descriptor.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_mcc.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_quality.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_rapid.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_reg.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_rgbd.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_saliency.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_stereo.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_structured_light.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_superres.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_surface_matching.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_tracking.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_videostab.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_viz.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_wechat_qrcode.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_xfeatures2d.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_xobjdetect.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_xphoto.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_shape.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_highgui.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_datasets.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_plot.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_text.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_ml.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_phase_unwrapping.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_optflow.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_ximgproc.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_video.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_videoio.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_dnn.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_imgcodecs.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_objdetect.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_calib3d.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_features2d.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_flann.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_photo.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_imgproc.so.4.5.2
lab02: /usr/local/opencv-4.5.2/lib/libopencv_core.so.4.5.2
lab02: CMakeFiles/lab02.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sitindustry/Documents/CVLab/lab02/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable lab02"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lab02.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/lab02.dir/build: lab02

.PHONY : CMakeFiles/lab02.dir/build

CMakeFiles/lab02.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/lab02.dir/cmake_clean.cmake
.PHONY : CMakeFiles/lab02.dir/clean

CMakeFiles/lab02.dir/depend:
	cd /home/sitindustry/Documents/CVLab/lab02/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sitindustry/Documents/CVLab/lab02 /home/sitindustry/Documents/CVLab/lab02 /home/sitindustry/Documents/CVLab/lab02/build /home/sitindustry/Documents/CVLab/lab02/build /home/sitindustry/Documents/CVLab/lab02/build/CMakeFiles/lab02.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/lab02.dir/depend

