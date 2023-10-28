# MySFM
This project was created with the aim of implementing a Structure-From-Motion (SFM) pipeline and to test it using estabalished datasets.

# Dependencies
This project requires OpenCV 4 and CVSBA. The version of cvsba used is adapted from [willdzeng's cvsba repository.](https://github.com/willdzeng/cvsba)

OpenCV 4 can be found in the OpenCV website. When installing OpenCV 4 from source, follow [this tutorial here](https://docs.opencv.org/4.x/db/d05/tutorial_config_reference.html) to build it with the contrib submodule and with non-free features enabled.

# Notes on CVSBA
CVSBA seems very sensitive to incorrect feature correspondences. While setting the error threshold for the KNN matcher start with low threshold and slowly increase from there. It's all about finding a balance between having enough points for performing PnP pose calculation while still maintaining good correspondence.
