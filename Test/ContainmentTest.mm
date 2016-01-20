//
//  ContainmentTest.m
//
//  Test containment logic where a superpixel is scanned to determine which
//  superpixels are contained inside of other superpixels. The result is
//  basically a tree where each node can have N children.

#include <opencv2/opencv.hpp> // Include OpenCV before any Foundation headers

#import <Foundation/Foundation.h>

#include "Coord.h"
#include "Superpixel.h"
#include "SuperpixelEdge.h"
#include "SuperpixelImage.h"

#include "SuperpixelEdgeFuncs.h"
#include "MergeSuperpixelImage.h"

#include "OpenCVUtil.h"

#include "ClusteringSegmentation.hpp"

#import <XCTest/XCTest.h>

@interface ContainmentTest : XCTestCase

@end

@implementation ContainmentTest

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

+ (NSArray*) formatSuperpixelCoords:(Superpixel*)spPtr {
  NSMutableArray *mArr = [NSMutableArray array];
  
  for (auto it = spPtr->coords.begin(); it != spPtr->coords.end(); ++it) {
    int32_t X = it->x;
    int32_t Y = it->y;
    [mArr addObject:@[@(X),@(Y)]];
  }
  
  return mArr;
}

+ (NSArray*) formatCoords:(vector<Coord>&)coords {
  NSMutableArray *mArr = [NSMutableArray array];
  
  for (auto it = coords.begin(); it != coords.end(); ++it) {
    int32_t X = it->x;
    int32_t Y = it->y;
    [mArr addObject:@[@(X),@(Y)]];
  }
  
  return mArr;
}

// Read pixels at coordinates from input Mat

+ (NSArray*) getSuperpixelCoordsAsPixels:(Superpixel*)spPtr input:(Mat)input {
  NSMutableArray *mArr = [NSMutableArray array];
  
  for (auto it = spPtr->coords.begin(); it != spPtr->coords.end(); ++it) {
    int32_t X = it->x;
    int32_t Y = it->y;
    
    Vec3b pixelVec = input.at<Vec3b>(Y, X);
    
    uint32_t pixel = pixelVec[0];
    pixel |= (pixelVec[1] << 8);
    pixel |= (pixelVec[2] << 16);
    
    [mArr addObject:@(pixel)];
  }
  
  return mArr;
}

- (void)setUp {
    [super setUp];
    // Put setup code here. This method is called before the invocation of each test method in the class.
}

- (void)tearDown {
    // Put teardown code here. This method is called after the invocation of each test method in the class.
    [super tearDown];
}

- (void)testParse1x1Containment {
  
  NSArray *pixelsArr = @[
                         @(0)
                         ];
  
  Mat tagsImg(1, 1, CV_MAKETYPE(CV_8U, 3));
  
  [self.class fillImageWithPixels:pixelsArr img:tagsImg];
  
  SuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(tagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  vector<int32_t> superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 1, @"num sumperpixels");
  
  XCTAssert(superpixels[0] == 0+1, @"tag");
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[0])];
    
    NSArray *expected = @[
                          @[@(0), @(0)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  // Scan for containment
  
  unordered_map<int32_t, vector<int32_t> > containsTreeMap;
  
  vector<int32_t> rootTags = recurseSuperpixelContainment(spImage, tagsImg, containsTreeMap);
  
  // Lambda w capture
  auto lambdaFunc = [=](int32_t tag, const vector<int32_t> &children)->void {
    fprintf(stdout, "tag %5d has %5d children\n", tag, (int)children.size());

    XCTAssert(tag == 1, @"tag");
    XCTAssert(children.size() == 0, @"children");
  };
  
  recurseSuperpixelIterate(rootTags, containsTreeMap, lambdaFunc);

  XCTAssert(rootTags.size() == 1, @"tags");
  XCTAssert(rootTags[0] == 1, @"tags");
  
  XCTAssert(containsTreeMap.size() == 1, @"map");
}

- (void)testParse2x2Containment {
  
  NSArray *pixelsArr = @[
                         @(0), @(0),
                         @(1), @(1),
                         ];
  
  Mat tagsImg(2, 2, CV_MAKETYPE(CV_8U, 3));
  
  [self.class fillImageWithPixels:pixelsArr img:tagsImg];
  
  SuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(tagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  vector<int32_t> superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 2, @"num sumperpixels");
  
  XCTAssert(superpixels[0] == 0+1, @"tag");
  XCTAssert(superpixels[1] == 1+1, @"tag");
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[0])];
    
    NSArray *expected = @[
                          @[@(0), @(0)],
                          @[@(1), @(0)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[1])];
    
    NSArray *expected = @[
                          @[@(0), @(1)],
                          @[@(1), @(1)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  // Scan for containment
  
  unordered_map<int32_t, vector<int32_t> > containsTreeMap;
  
  vector<int32_t> rootTags = recurseSuperpixelContainment(spImage, tagsImg, containsTreeMap);
  
  // Lambda w capture
  auto lambdaFunc = [=](int32_t tag, const vector<int32_t> &children)->void {
    fprintf(stdout, "tag %5d has %5d children\n", tag, (int)children.size());
    
    XCTAssert(tag == 1 || tag == 2, @"tag");
    XCTAssert(children.size() == 0, @"children");
  };
  
  recurseSuperpixelIterate(rootTags, containsTreeMap, lambdaFunc);
  
  XCTAssert(rootTags.size() == 2, @"tags");
  XCTAssert(rootTags[0] == 1, @"tags");
  XCTAssert(rootTags[1] == 2, @"tags");
  
  XCTAssert(containsTreeMap.size() == 2, @"map");
}

@end
