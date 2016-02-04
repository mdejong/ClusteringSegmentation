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

void binMatInvert(Mat &binMat);

// Generate a skeleton based on simple morphological operations.

void skelReduce(Mat &binMat);

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

// Like cv::drawContours() except that this simplified method
// renders just one contour.

void drawOneContour(
                    Mat & mat,
                    const vector<Point2i> &contourPointsVec,
                    const Scalar& color,
                    int thickness,
                    int lineType );

// Invoke drawOneContour() with the results of looking up points in an existing
// points vector. This method assumes a vector of int as returned by convexHull().

void drawOneHull(
                 Mat & mat,
                 const vector<int> &hull,
                 const vector<Point2i> &points,
                 const Scalar& color,
                 int thickness,
                 int lineType );

void drawLine(
              Mat & mat,
              const vector<Point2i> &linePointsVec,
              const Scalar& color,
              int thickness,
              int lineType );

// Quickly convert vector of Coord to Point2i for passing to OpenCV functions

vector<Point2i>
convertCoordsToPoints(const vector<Coord> &coordsVec);

// Quickly convert vector of Coord to Point2i for passing to OpenCV functions

vector<Coord>
convertPointsToCoords(const vector<Point2i> &pointsVec);

#endif // OPENCV_UTIL_H
