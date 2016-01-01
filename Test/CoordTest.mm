//
//  CoordTest.m
//
//  Test functionality defined in Coord.h, this is a 32bit value that stores
//  a pair of (x,y) coords in minimal memory space.

#include <opencv2/opencv.hpp> // Include OpenCV before any Foundation headers

#import <Foundation/Foundation.h>

#include "Superpixel.h"
#include "SuperpixelEdge.h"
#include "SuperpixelImage.h"
#include "Coord.h"

#import <XCTest/XCTest.h>

@interface CoordTest : XCTestCase

@end

@implementation CoordTest

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

- (void)testParse1x1ZeroEdges {
  
  NSArray *pixelsArr = @[
                         @(0)
                         ];
  
  Mat tagsImg(1, 1, CV_MAKETYPE(CV_8U, 3));
  
  [self.class fillImageWithPixels:pixelsArr img:tagsImg];
  
  SuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(tagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  vector<int32_t> superpixels = spImage.superpixels;
  XCTAssert(superpixels.size() == 1, @"num sumperpixels");
  
  XCTAssert(superpixels[0] == 0+1, @"tag");
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[0])];
    
    NSArray *expected = @[
                          @[@(0), @(0)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 0, @"num edges");
}

- (void)setUp {
    [super setUp];
    // Put setup code here. This method is called before the invocation of each test method in the class.
}

- (void)tearDown {
    // Put teardown code here. This method is called after the invocation of each test method in the class.
    [super tearDown];
}

//- (void)testPerformanceExample {
//    // This is an example of a performance test case.
//    [self measureBlock:^{
//        // Put the code you want to measure the time of here.
//    }];
//}

@end
