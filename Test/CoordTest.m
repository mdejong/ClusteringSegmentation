//
//  CoordTest.m
//
//  Test functionality defined in Coord.h, this is a 32bit value that stores
//  a pair of (x,y) coords in minimal memory space.

#import <Foundation/Foundation.h>

#include <opencv2/opencv.hpp>

#include "Superpixel.h"
#include "SuperpixelEdge.h"
#include "SuperpixelImage.h"

#include "Coord.h"

#import <XCTest/XCTest.h>

@interface CoordTest : XCTestCase

@end

@implementation CoordTest

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

- (void)testExample {
    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
}

- (void)testPerformanceExample {
    // This is an example of a performance test case.
    [self measureBlock:^{
        // Put the code you want to measure the time of here.
    }];
}

@end
