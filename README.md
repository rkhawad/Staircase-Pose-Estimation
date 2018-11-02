# Staircase-Pose-Estimation
Detection of Staircases from the scene and estimating the pose of it for navigation of MAVs.

mkdir build
cd build
cmake ..
make

run the command:
./bin/lsd_opencv_example image bin_size name_for_saving_image

Here:
./bin/lsd_opencv_example -------> executable file
image ----------> test case image (data set)
bin_size ---------> size for creating the histogram of the detected horizontal staircase lines (varied according ti test case)
