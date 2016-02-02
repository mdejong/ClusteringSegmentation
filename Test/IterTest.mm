//
//  CoordTest.m
//
//  Test functionality defined in Coord.h, this is a 32bit value that stores
//  a pair of (x,y) coords in minimal memory space.

#include <opencv2/opencv.hpp> // Include OpenCV before any Foundation headers

#import <Foundation/Foundation.h>

#include "Util.h"
#include "OpenCVUtil.h"
#include "OpenCVIter.hpp"

#import <XCTest/XCTest.h>

@interface IterTest : XCTestCase

@end

@implementation IterTest

+ (void) fillImageWithPixels:(NSArray*)pixelNums img:(Mat&)img
{
  uint32_t offset = 0;
  
  for( int y = 0; y < img.rows; y++ ) {
    for( int x = 0; x < img.cols; x++ ) {
      uint32_t pixel = [[pixelNums objectAtIndex:offset] unsignedIntValue];
      offset += 1;
      
      Vec3b pixelVec;
      pixelVec[0] = pixel & 0xFF;
      pixelVec[1] = (pixel >> 8) & 0xFF;
      pixelVec[2] = (pixel >> 16) & 0xFF;
      
      img.at<Vec3b>(y, x) = pixelVec;
    }
  }
  
  return;
}

- (void)setUp {
    [super setUp];
    // Put setup code here. This method is called before the invocation of each test method in the class.
}

- (void)tearDown {
    // Put teardown code here. This method is called after the invocation of each test method in the class.
    [super tearDown];
}

- (void) testConstBGRIterator
{
  NSArray *pixelsArr = @[
                         @(0x00030201), @(0x00060504),
                         @(0x00090807), @(0x000C0B0A),
                         ];
  
  Mat tagsImg(2, 2, CV_MAKETYPE(CV_8U, 3));
  
  [self.class fillImageWithPixels:pixelsArr img:tagsImg];
  
  std::stringstream outStream;
  
  for_each_const_bgr(tagsImg, [&outStream](uint8_t R, uint8_t G, uint8_t B)->void {
    char buffer[1000];
    snprintf(buffer, sizeof(buffer), "%3d %3d %3d\n", B, G, R);
    outStream << buffer;
    snprintf(buffer, sizeof(buffer), "0x00%02X%02X%02X\n", R, G, B);
    outStream << buffer;
  });
  
  string result = outStream.str();

  string expectedOutput =
  "  1   2   3\n"
  "0x00030201\n"
  "  4   5   6\n"
  "0x00060504\n"
  "  7   8   9\n"
  "0x00090807\n"
  " 10  11  12\n"
  "0x000C0B0A\n";
  
  XCTAssert(result == expectedOutput, @"output");

  return;
}

// This test calls the for_each_bgr() iterator with a non-constant
// argument even though the returned result is always the same value.

- (void) testNonConstBGRIterator
{
  NSArray *pixelsArr = @[
                         @(0x00030201), @(0x00060504),
                         @(0x00090807), @(0x000C0B0A),
                         ];
  
  Mat tagsImg(2, 2, CV_MAKETYPE(CV_8U, 3));
  
  [self.class fillImageWithPixels:pixelsArr img:tagsImg];
  
  std::stringstream outStream;
  
  for_each_bgr(tagsImg, [&outStream](uint8_t R, uint8_t G, uint8_t B)->Vec3b {
    char buffer[1000];
    snprintf(buffer, sizeof(buffer), "%3d %3d %3d\n", B, G, R);
    outStream << buffer;
    snprintf(buffer, sizeof(buffer), "0x00%02X%02X%02X\n", R, G, B);
    outStream << buffer;
    return Vec3b(B, G, R);
  });
  
  string result = outStream.str();
  
  string expectedOutput =
  "  1   2   3\n"
  "0x00030201\n"
  "  4   5   6\n"
  "0x00060504\n"
  "  7   8   9\n"
  "0x00090807\n"
  " 10  11  12\n"
  "0x000C0B0A\n";
  
  XCTAssert(result == expectedOutput, @"output");
  
  return;
}

// Implement channel swap between the B and R channels using
// a non constant iterator and then a constant iterator

- (void) testChannelSwapBR
{
  NSArray *pixelsArr = @[
                         @(0x00010203), @(0x00040506),
                         @(0x00070809), @(0x000A0B0C),
                         ];
  
  Mat tagsImg(2, 2, CV_MAKETYPE(CV_8U, 3));
  
  [self.class fillImageWithPixels:pixelsArr img:tagsImg];
  
  for_each_bgr(tagsImg, [](uint8_t R, uint8_t G, uint8_t B)->Vec3b {
    return Vec3b(R, G, B); // Swap B and R channels
  });
  
  std::stringstream outStream;
  
  for_each_const_bgr(tagsImg, [&outStream](uint8_t R, uint8_t G, uint8_t B)->void {
    char buffer[1000];
    snprintf(buffer, sizeof(buffer), "%3d %3d %3d\n", B, G, R);
    outStream << buffer;
    snprintf(buffer, sizeof(buffer), "0x00%02X%02X%02X\n", R, G, B);
    outStream << buffer;
  });
  
  string result = outStream.str();
  
  string expectedOutput =
  "  1   2   3\n"
  "0x00030201\n"
  "  4   5   6\n"
  "0x00060504\n"
  "  7   8   9\n"
  "0x00090807\n"
  " 10  11  12\n"
  "0x000C0B0A\n";
  
  XCTAssert(result == expectedOutput, @"output");
  
  return;
}


// Number of Mat iteration loops inside the main test method

#define NUM_ITER_LOOPS 40

- (void) DISABLED_testPerformanceExampleBin1 {
  // This is an example of a performance test case.
  Mat binMat(1000, 1000, CV_8UC1);
  binMat = Scalar(0);
  
  int index = 0;
  for_each_byte (binMat, [&index](uint8_t *bytePtr)->void {
    uint8_t bVal;
    if ((index%2) == 0) {
      bVal = 0;
    } else {
      bVal = 1;
    }
    *bytePtr = bVal;
    index++;
  });
  
  Mat *ptrMat = &binMat;
  
  [self measureBlock:^{
    for (int i = 0; i < NUM_ITER_LOOPS; i++ ) {
      binMatInvert(*ptrMat);
      binMatInvert(*ptrMat);
    }
  }];
}

// Inline loop using X and Y values

- (void)testPerformanceExampleBGRConst1 {
  Mat mat(1000, 1000, CV_8UC3);
  
  uint32_t offset = 0;
  for_each_bgr (mat, [&offset](uint8_t R, uint8_t G, uint8_t B)->Vec3b {
    B = offset & 0xFF;
    G = (offset >> 8) & 0xFF;
    R = (offset >> 16) & 0xFF;
    offset++;
    return Vec3b(B, G, R);
  });
  
  [self measureBlock:^{
    int sum = 0;
    for (int i = 0; i < NUM_ITER_LOOPS; i++ ) {
      for(int y = 0; y < mat.rows; y++) {
        for(int x = 0; x < mat.cols; x++) {
          Vec3b vec = mat.at<Vec3b>(y, x);
          sum += vec[0] + vec[1] + vec[2];
        }
      }
    }
    
    printf("sum is %d\n", sum);
  }];
  
  return;
}

// This performance test makes use of for_each_const_bgr() which
// should be faster that repeated invocations of Mat.at<Vec3b>(y,x)

- (void)testPerformanceExampleBGRConst2 {
  Mat mat(1000, 1000, CV_8UC3);
  
  uint32_t offset = 0;
  for_each_bgr (mat, [&offset](uint8_t R, uint8_t G, uint8_t B)->Vec3b {
    B = offset & 0xFF;
    G = (offset >> 8) & 0xFF;
    R = (offset >> 16) & 0xFF;
    offset++;
    return Vec3b(B, G, R);
  });
  
  [self measureBlock:^{
    int sum = 0;
    for (int i = 0; i < NUM_ITER_LOOPS; i++ ) {
      for_each_const_bgr(mat, [&sum](uint8_t R, uint8_t G, uint8_t B)->void {
        // Do something with the values to avoid optimizing away loop
        sum += B + G + R;
      });
    }
    
    printf("sum is %d\n", sum);
  }];
  
  return;
}

// Non-constant loop that reads BGR components and writes them back to the Mat

- (void)testPerformanceExampleBGRNonConst1 {
  Mat mat(1000, 1000, CV_8UC3);
  
  uint32_t offset = 0;
  for_each_bgr (mat, [&offset](uint8_t R, uint8_t G, uint8_t B)->Vec3b {
    B = offset & 0xFF;
    G = (offset >> 8) & 0xFF;
    R = (offset >> 16) & 0xFF;
    offset++;
    return Vec3b(B, G, R);
  });
  
  Mat *matPtr = &mat;
  
  [self measureBlock:^{
    Mat &mat = *matPtr;
    int sum = 0;
    for (int i = 0; i < NUM_ITER_LOOPS; i++ ) {
      for(int y = 0; y < mat.rows; y++) {
        for(int x = 0; x < mat.cols; x++) {
          Vec3b vec = mat.at<Vec3b>(y, x);
          sum += vec[0] + vec[1] + vec[2];
          vec[0] += 1;
          mat.at<Vec3b>(y, x) = vec;
        }
      }
    }
    
    printf("sum is %d\n", sum);
  }];
  
  return;
}

// Non-constant loop that reads and write components with for_each_bgr()

- (void)testPerformanceExampleBGRNonConst2 {
  Mat mat(1000, 1000, CV_8UC3);
  
  uint32_t offset = 0;
  for_each_bgr (mat, [&offset](uint8_t R, uint8_t G, uint8_t B)->Vec3b {
    B = offset & 0xFF;
    G = (offset >> 8) & 0xFF;
    R = (offset >> 16) & 0xFF;
    offset++;
    return Vec3b(B, G, R);
  });
  
  Mat *matPtr = &mat;
  
  [self measureBlock:^{
    Mat &mat = *matPtr;
    int sum = 0;
    for (int i = 0; i < NUM_ITER_LOOPS; i++ ) {
      for_each_bgr(mat, [&sum](uint8_t R, uint8_t G, uint8_t B)->Vec3b {
        sum += B + G + R;
        B += 1;
        return Vec3b(B, G, R);
      });
    }
    
    printf("sum is %d\n", sum);
  }];
  
  return;
}

@end
