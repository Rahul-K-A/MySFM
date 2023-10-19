# Visual Odometry
This project was created with the aim of implementing Monocular Visual Odometry and to test it using estabalished datasets.

## Setup instructions:

`sudo apt install libopencv-dev` in Ubuntu 20.04 or newer

`sudo apt install libeigen3-dev` in Ubuntu 20.04 or newer

## Dataset:
This project is based on the TU Munich Monocular Visual Odometry dataset and certain parts of the code are also adapted from their dataset

## Setting up the dataset
1. Download any of the sequences from the TUM Munich MVO dataset
2. Extract the .zip file
3. Within the extracted folder, extract the images zip file within the same folder
4. Note: the dataset source directory is considered as the folder which contains the calibration text files and images

## Running the project
1. Clone this project
2. Within the project folder `mkdir build`
3. `cd build`
4. `cmake ..`
5. `make`
6. Now you can run the project using `./VisualOdometry ./VisualOdometry <path to dataset source dir> <Camera FOV type>`
7. If the sequence contains images from wide angle cameras set Camera FOV type to 1
8. If the sequence contains images from narrow lens set Camera FOV type to 2






