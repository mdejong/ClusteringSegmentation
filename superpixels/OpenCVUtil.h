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

static inline
uint32_t Vec4BToPixel(Vec4b tag) {
  // B = tagVec[0], G = tagVec[1], R = tagVec[2], A = tagVec[3]
  uint32_t B = tag[0];
  uint32_t G = tag[1];
  uint32_t R = tag[2];
  uint32_t A = tag[3];
  uint32_t pixel = (A << 24) | (R << 16) | (G << 8) | B;
  return pixel;
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

static inline
Vec3f xyzDeltaToUnitVec3f(int32_t &dR, int32_t &dG, int32_t &dB) {
  float scale = sqrt(float(dR*dR + dG*dG + dB*dB));
  
  if (scale == 0.0f) {
    dR = 0;
    dG = 0;
    dB = 0;
    return Vec3f(dB, dG, dR);
  } else {
    Vec3f vec(dB, dG, dR);
    vec = vec / scale;
    return vec;
  }
}

// Logical not operation for byte matrix. If the value is 0x0 then
// write 0xFF otherwise write 0x0.

static inline
void binMatInvert(Mat &binMat) {
#if defined(DEBUG)
  assert(binMat.channels() == 1);
#endif // DEBUG
  
  for (int y = 0; y < binMat.rows; y++) {
    for (int x = 0; x < binMat.cols; x++) {
      uint8_t bVal = binMat.at<uint8_t>(y, x);
      if (bVal == 0) {
        bVal = 0xFF;
      } else {
        bVal = 0;
      }
      binMat.at<uint8_t>(y, x) = bVal;
    }
  }
}

// This iterator loop over each byte in a constant Mat
// and invokes the function foreach byte. The original
// Mat is constant and cannot be changed.

template <class T>
static inline
void mat_byte_const_foreach (const Mat & mat, std::function<void(uint8_t)> f) noexcept
{
  auto it = mat.begin<uint8_t>();
  auto end = mat.end<uint8_t>();
  for ( ; it != end; ++it ) {
    uint8_t bVal = *it;
    f(bVal);
  }
  return;
}

// This iterator loop over each byte in a Mat and offers
// the ability to write a byte value back to the Mat.
// The function should return a byte value, in the case
// where the byte value returned is not the same as the
// original then that byte is written back to the Mat.

static inline
void mat_byte_foreach (Mat & mat, std::function<uint8_t(uint8_t)> f) noexcept
{
  auto it = mat.begin<uint8_t>();
  auto end = mat.end<uint8_t>();
  for ( ; it != end; ++it ) {
    uint8_t bVal = *it;
    uint8_t retBVal = f(bVal);
    if (retBVal != bVal) {
      *it = retBVal;
    }
  }
  return;
}

// Double iterator that loops over a pair of Mat objects. The function is invoked
// with the current value from mat1 and mat2 and if the result returned by the
// function is changed then that change is written back to mat1. Note that mat2
// is considered constant during the loop and will not be changed.

static inline
void mat_byte_foreach (Mat & mat1, const Mat & mat2, std::function<uint8_t(uint8_t, uint8_t)> f) noexcept
{
#if defined(DEBUG)
  assert(mat1.size() == mat2.size());
#endif // DEBUG
  
  auto it1 = mat1.begin<uint8_t>();
  auto end1 = mat1.end<uint8_t>();
  
  auto it2 = mat2.begin<uint8_t>();
  auto end2 = mat2.end<uint8_t>();
  
#if defined(DEBUG)
  assert(it1 != end1);
  assert(it2 != end2);
#endif // DEBUG
  
  for ( ; it1 != end1; it1++, it2++ ) {
#if defined(DEBUG)
    assert(it2 != end2);
#endif // DEBUG
    
    uint8_t bVal1 = *it1;
    uint8_t bVal2 = *it2;
    
    uint8_t retBVal = f(bVal1, bVal2);
    if (retBVal != bVal1) {
      *it1 = retBVal;
    }
  }
  return;
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

// Given an input binary Mat (0x0 or 0xFF) perform a dilate() operation that will expand
// the white region inside a black region. This makes use of a circular operator and
// an expansion size indicated by the caller.

Mat expandWhiteInRegion(const Mat &binMat, int expandNumPixelsSize, int tag);

// Given an input binary Mat (0x0 or 0xFF) perform a erode() operation that will decrease
// the white region inside a black region. This makes use of a circular operator and
// an expansion size indicated by the caller.

Mat decreaseWhiteInRegion(const Mat &binMat, int decreaseNumPixelsSize, int tag);

// Given a superpixel tag that indicates a region segmented into 4x4 squares
// map (X,Y) coordinates to a minimized Mat representation that can be
// quickly morphed with minimal CPU and memory usage.

Mat expandBlockRegion(int32_t tag,
                      const vector<Coord> &coords,
                      int expandNum,
                      int blockWidth, int blockHeight,
                      int superpixelDim);

// Given a Mat that contains quant pixels and a colortable, map the quant
// pixels to indexes in the colortable. If the asGreyscale) flag is true
// then each index is assumed to be a byte and is written as a greyscale pixel.

Mat mapQuantPixelsToColortableIndexes(const Mat & inQuantPixels, const vector<uint32_t> &colortable, bool asGreyscale);

// Given a Mat that contains pixels count each pixel and return a histogram
// of the number of times each pixel is found in the image.

void generatePixelHistogram(const Mat & inQuantPixels,
                            unordered_map<uint32_t, uint32_t> &pixelToCountTable);

// Return color cube with divided by 5 points along each axis.

vector<uint32_t> getSubdividedColors();

// Vote for pixels that have neighbors that are the exact same value, this method examines each
// pixel by getting the 8 connected neighbors and recoring a vote for a given pixel when it has
// a neighbor that is exactly the same.

void vote_for_identical_neighbors(unordered_map<uint32_t, uint32_t> &pixelToNumVotesMap,
                                  const Mat &inImage,
                                  const Mat &inMaskImage);

// Given a series of 3D points, generate a center of mass in (x,y,z) for the points.

Vec3b centerOfMass3d(const vector<Vec3b> &points);

// Given a series of 3D pixels, generate a center of mass in (B,G,R) for the points.

uint32_t centerOfMassPixels(const vector<uint32_t> & pixels);

// Generate a vector of pixels from one point to another

vector<uint32_t> generateVector(uint32_t fromPixel, uint32_t toPixel);

// Flood fill based on region of zero values. Input comes from inBinMask and the results
// are written to outBinMask. Black pixels are filled and white pixels are not filled.

int floodFillMask(Mat &inBinMask, Mat &outBinMask, Point2i startPoint, int connectivity);

#endif // OPENCV_UTIL_H
