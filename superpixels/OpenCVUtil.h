// This file contains utility functions for general purpose use and interaction with OpenCV.

#ifndef OPENCV_UTIL_H
#define	OPENCV_UTIL_H

#include <opencv2/opencv.hpp>
#include <unordered_map>

using namespace std;
using namespace cv;

class Coord;

// Convert a vector of 3 bytes into a signed 32bit integer.
// The values range for a 3 byte tag is 0 -> 0x00FFFFFF and
// the value -1 is still valid in this case.

static inline
int32_t Vec3BToUID(Vec3b tag) {
  // B = tagVec[0], G = tagVec[1], R = tagVec[2] -> BGRA
  uint32_t B = tag[0];
  uint32_t G = tag[1];
  uint32_t R = tag[2];
  uint32_t tagVal = (R << 16) | (G << 8) | B;
  int32_t tagValS = (int32_t) tagVal;
  return tagValS;
}

// Convert a 24bit signed int value to a Scalar

static inline
Scalar UIDToScalar(int32_t tag) {
  // BGR0
  uint32_t uTag = tag & 0x00FFFFFF;
  uint32_t B = uTag & 0xFF;
  uint32_t G = (uTag >> 8) & 0xFF;
  uint32_t R = (uTag >> 16) & 0xFF;
  return Scalar(B, G, R);
}

static inline
Vec3b PixelToVec3b(uint32_t pixel) {
  // BGR0
  uint32_t B = pixel & 0xFF;
  uint32_t G = (pixel >> 8) & 0xFF;
  uint32_t R = (pixel >> 16) & 0xFF;
  return Vec3b(B, G, R);
}

// Print SSIM for two images to cout

int printSSIM(Mat inImage1, Mat inImage2);

// Find a single "center" pixel in region of interest matrix. This logic
// accepts an input matrix that contains binary pixel values (0x0 or 0xFF)
// and computes a consistent center pixel. When this method returns the
// region binMat is unchanged. The orderMat is set to the size of the roi and
// it is filled with distance transformed gray values. Note that this method
// has to create a buffer zone of 1 pixel so that pixels on the edge have
// a very small distance.

Coord findRegionCenter(Mat &binMat, cv::Rect roi, Mat &outDistMat, int tag);

#endif // OPENCV_UTIL_H
