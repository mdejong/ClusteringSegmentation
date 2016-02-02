// This module contains inline C++ template functions for OpenCV Mat iteration.
// The many details of iterating over a grayscale byte image or a 24BPP or
// 32BPP (with alpha channel) image can lead to a lot of duplicated code and
// non-optimal execution. This module makes use of templated functions that
// can accept functor/lambda blocks so that optimal boilerplate code need
// not appear over and over again in user code.

#ifndef OPENCV_MAT_ITER_H
#define	OPENCV_MAT_ITER_H

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// This iterator loops over all the byte values in a
// const Mat of grayscale bytes and invokes a user
// supplied function that takes a byte value.

// F = std::function<void(uint8_t bVal)>

template <typename F>
void for_each_const_byte (const Mat & binMat, F f) noexcept
{
#if defined(DEBUG)
  assert(binMat.channels() == 1);
  std::function<void(uint8_t)> funcPtr = f;
#endif // DEBUG
  
  int numRows = binMat.rows;
  int numCols = binMat.cols;
  
  if (binMat.isContinuous()) {
    numCols *= numRows;
    numRows = 1;
  }
  
  for (int y = 0; y < numRows; y++) {
    const uint8_t *rowPtr = binMat.ptr<uint8_t>(y);
    for (int x = 0; x < numCols; x++) {
      uint8_t bVal = rowPtr[x];
      f(bVal);
    }
  }
  
  return;
}

// This iterator loop over each byte in a Mat and offers
// the ability to write a byte value back to the Mat
// as the iteration progresses.

// F = std::function<void(uint8_t* bytePtr)>

template <typename F>
void for_each_byte (Mat & binMat, F f) noexcept
{
#if defined(DEBUG)
  assert(binMat.channels() == 1);
  std::function<void(uint8_t*)> funcPtr = f;
#endif // DEBUG
  
  int numRows = binMat.rows;
  int numCols = binMat.cols;
  
  if (binMat.isContinuous()) {
    numCols *= numRows;
    numRows = 1;
  }
  
  for (int y = 0; y < numRows; y++) {
    uint8_t *rowPtr = binMat.ptr<uint8_t>(y);
    for (int x = 0; x < numCols; x++) {
      uint8_t *bytePtr = rowPtr + x;
      f(bytePtr);
    }
  }
  
  return;
}

// Double iterator that loops over a pair of Mat objects. The function is invoked
// with a pointer to each value from binMat1 and binMat2. Note that binMat2
// is const so the second parameter passed to the function is a const pointer.

// F = std::function<void(uint8_t*, const uint8_t*)>

template <typename F>
void for_each_byte (Mat & binMat1, const Mat & binMat2, F f) noexcept
{
#if defined(DEBUG)
  assert(binMat1.size() == binMat2.size());
  assert(binMat1.channels() == 1);
  assert(binMat2.channels() == 1);
  assert(binMat1.isContinuous() == binMat2.isContinuous());
  std::function<void(uint8_t*, const uint8_t*)> funcPtr = f;
#endif // DEBUG
  
  int numRows = binMat1.rows;
  int numCols = binMat1.cols;
  
  if (binMat1.isContinuous()) {
    numCols *= numRows;
    numRows = 1;
  }

  for (int y = 0; y < numRows; y++) {
    uint8_t *rowPtr1 = binMat1.ptr<uint8_t>(y);
    const uint8_t *rowPtr2 = binMat2.ptr<uint8_t>(y);
    for (int x = 0; x < numCols; x++) {
      uint8_t *bytePtr1 = rowPtr1 + x;
      const uint8_t *bytePtr2 = rowPtr2 + x;
      f(bytePtr1, bytePtr2);
    }
  }

  return;
}

// This iterator reads all the 24BPP pixels out of an image Mat
// as (B G R) component triples. This iterator operates on
// a constant Mat, so no data is written back to Mat.

// F = std::function<void(uint8_t B, uint8_t G, uint8_t R)>

template <typename F>
void for_each_const_bgr (const Mat & mat, F f) noexcept
{
#if defined(DEBUG)
  assert(mat.channels() == 3);
  std::function<void(uint8_t,uint8_t,uint8_t)> funcPtr = f;
#endif // DEBUG
  
  int numRows = mat.rows;
  int numCols = mat.cols;
  
  if (mat.isContinuous()) {
    numCols *= numRows;
    numRows = 1;
  }
  
  for (int y = 0; y < numRows; y++) {
    const uint8_t *rowPtr = mat.ptr<uint8_t>(y);
    const uint8_t * const rowMaxPtr = rowPtr + (3 * numCols);
    for ( ; rowPtr < rowMaxPtr; rowPtr += 3) {
      uint8_t B = rowPtr[0];
      uint8_t G = rowPtr[1];
      uint8_t R = rowPtr[2];
      f(B, G, R);
    }
  }
  
  return;
}

// This iterator reads all the 24BPP pixels out of an image Mat
// as (B G R) component triples. The return Vec3b value returned
// from the method is written back to the non-constant Mat.

// F = std::function<Vec3b(uint8_t B, uint8_t G, uint8_t R)>

template <typename F>
void for_each_bgr (Mat & mat, F f) noexcept
{
#if defined(DEBUG)
  assert(mat.channels() == 3);
  std::function<Vec3b(uint8_t,uint8_t,uint8_t)> funcPtr = f;
#endif // DEBUG
  
  int numRows = mat.rows;
  int numCols = mat.cols;
  
  if (mat.isContinuous()) {
    numCols *= numRows;
    numRows = 1;
  }
  
  for (int y = 0; y < numRows; y++) {
    uint8_t *rowPtr = mat.ptr<uint8_t>(y);
    const uint8_t * const rowMaxPtr = rowPtr + (3 * numCols);
    for ( ; rowPtr < rowMaxPtr; rowPtr += 3) {
      uint8_t B = rowPtr[0];
      uint8_t G = rowPtr[1];
      uint8_t R = rowPtr[2];
      Vec3b result = f(B, G, R);
      rowPtr[0] = result[0]; // B
      rowPtr[1] = result[1]; // G
      rowPtr[2] = result[2]; // R
    }
  }
  
  return;
}

#endif // OPENCV_MAT_ITER_H
