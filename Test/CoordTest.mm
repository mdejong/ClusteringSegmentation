//
//  CoordTest.m
//
//  Test functionality defined in Coord.h, this is a 32bit value that stores
//  a pair of (x,y) coords in minimal memory space.

#include <opencv2/opencv.hpp> // Include OpenCV before any Foundation headers

#import <Foundation/Foundation.h>

#include "Coord.h"
#include "Superpixel.h"
#include "SuperpixelEdge.h"
#include "SuperpixelImage.h"

#include "SuperpixelEdgeFuncs.h"
#include "MergeSuperpixelImage.h"

#include "OpenCVUtil.h"

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

- (void)testParse1x1ZeroEdges {
  
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
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 0, @"num edges");
}

- (void)testParse2x2OneEdges {
  
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
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 1, @"num edges");
  
  SuperpixelEdge edge = edges[0];
  XCTAssert(edge.A == superpixels[0], @"edge.A");
  XCTAssert(edge.B == superpixels[1], @"edge.B");
}

- (void)testParse1and3V1 {
  
  NSArray *pixelsArr = @[
                         @(0), @(1),
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
                          @[@(0), @(0)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[1])];
    
    NSArray *expected = @[
                          @[@(1), @(0)],
                          @[@(0), @(1)],
                          @[@(1), @(1)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 1, @"num edges");
  
  SuperpixelEdge edge = edges[0];
  XCTAssert(edge.A == superpixels[0], @"edge.A");
  XCTAssert(edge.B == superpixels[1], @"edge.B");
}

- (void)testParse1and3V2 {
  
  NSArray *pixelsArr = @[
                         @(1), @(0),
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
                          @[@(1), @(0)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[1])];
    
    NSArray *expected = @[
                          @[@(0), @(0)],
                          @[@(0), @(1)],
                          @[@(1), @(1)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 1, @"num edges");
  
  SuperpixelEdge edge = edges[0];
  XCTAssert(edge.A == superpixels[0], @"edge.A");
  XCTAssert(edge.B == superpixels[1], @"edge.B");
}


- (void)testParse3x3TwoEdges {
  
  NSArray *pixelsArr = @[
                         @(0), @(0), @(0),
                         @(1), @(1), @(1),
                         @(2), @(2), @(2),
                         ];
  
  Mat tagsImg(3, 3, CV_MAKETYPE(CV_8U, 3));
  
  [self.class fillImageWithPixels:pixelsArr img:tagsImg];
  
  SuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(tagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  vector<int32_t> superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 3, @"num sumperpixels");
  
  XCTAssert(superpixels[0] == 0+1, @"tag");
  XCTAssert(superpixels[1] == 1+1, @"tag");
  XCTAssert(superpixels[2] == 2+1, @"tag");
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[0])];
    
    NSArray *expected = @[
                          @[@(0), @(0)],
                          @[@(1), @(0)],
                          @[@(2), @(0)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[1])];
    
    NSArray *expected = @[
                          @[@(0), @(1)],
                          @[@(1), @(1)],
                          @[@(2), @(1)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[2])];
    
    NSArray *expected = @[
                          @[@(0), @(2)],
                          @[@(1), @(2)],
                          @[@(2), @(2)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 2, @"num edges");
  
  {
    SuperpixelEdge &edge = edges[0];
    XCTAssert(edge.A == superpixels[0], @"edge.A");
    XCTAssert(edge.B == superpixels[1], @"edge.B");
  }
  
  {
    SuperpixelEdge &edge = edges[1];
    XCTAssert(edge.A == superpixels[1], @"edge.A");
    XCTAssert(edge.B == superpixels[2], @"edge.B");
  }
}

- (void)testParse3x3ThreeEdges {
  
  NSArray *pixelsArr = @[
                         @(0), @(0), @(0),
                         @(0), @(1), @(1),
                         @(2), @(2), @(2),
                         ];
  
  Mat tagsImg(3, 3, CV_MAKETYPE(CV_8U, 3));
  
  [self.class fillImageWithPixels:pixelsArr img:tagsImg];
  
  SuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(tagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  vector<int32_t> superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 3, @"num sumperpixels");
  
  XCTAssert(superpixels[0] == 0+1, @"tag");
  XCTAssert(superpixels[1] == 1+1, @"tag");
  XCTAssert(superpixels[2] == 2+1, @"tag");
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[0])];
    
    NSArray *expected = @[
                          @[@(0), @(0)],
                          @[@(1), @(0)],
                          @[@(2), @(0)],
                          @[@(0), @(1)],
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[1])];
    
    NSArray *expected = @[
                          @[@(1), @(1)],
                          @[@(2), @(1)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[2])];
    
    NSArray *expected = @[
                          @[@(0), @(2)],
                          @[@(1), @(2)],
                          @[@(2), @(2)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 3, @"num edges");
  
  {
    SuperpixelEdge &edge = edges[0];
    XCTAssert(edge.A == superpixels[0], @"edge.A");
    XCTAssert(edge.B == superpixels[1], @"edge.B");
  }
  
  {
    SuperpixelEdge &edge = edges[1];
    XCTAssert(edge.A == superpixels[0], @"edge.A");
    XCTAssert(edge.B == superpixels[2], @"edge.B");
  }
  
  {
    SuperpixelEdge &edge = edges[2];
    XCTAssert(edge.A == superpixels[1], @"edge.A");
    XCTAssert(edge.B == superpixels[2], @"edge.B");
  }
}

// Parse into superpixels and edges and then join a very
// simple pair of superpixels with only 1 coordinate
// into a merged superpixel.

- (void)testMerge2x1 {
  
  NSArray *pixelsArr = @[
                         @(0), @(1)
                         ];
  
  Mat tagsImg(1, 2, CV_MAKETYPE(CV_8U, 3));
  
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
                          @[@(0), @(0)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[1])];
    
    NSArray *expected = @[
                          @[@(1), @(0)],
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 1, @"num edges");
  
  {
    SuperpixelEdge &edge = edges[0];
    XCTAssert(edge.A == superpixels[0], @"edge.A");
    XCTAssert(edge.B == superpixels[1], @"edge.B");
  }
  
  // (0 1) are neighbors and vice versa
  
  vector<int32_t> neighbors;
  
  neighbors = spImage.edgeTable.getNeighbors(superpixels[0]);
  XCTAssert(neighbors.size() == 1, @"neighbors size");
  XCTAssert(neighbors[0] == superpixels[1], @"neighbors");
  
  neighbors = spImage.edgeTable.getNeighbors(superpixels[1]);
  XCTAssert(neighbors.size() == 1, @"neighbors size");
  XCTAssert(neighbors[0] == superpixels[0], @"neighbors");
  
  // Merge superpixels given an edge between them, note that
  // a merge operation always chooses the UID to use as the
  // result based on the number of pixels in the superpixel.
  // In this case, there is a tie and the smaller UID is used.
  
  XCTAssert(spImage.tagToSuperpixelMap.size() == 2, @"sumperpixel UID table");
  
  spImage.mergeEdge(edges[0]);
  
  XCTAssert(spImage.tagToSuperpixelMap.size() == 1, @"sumperpixel UID table");
  
  superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 1, @"num sumperpixels");
  
  XCTAssert(superpixels[0] == 0+1, @"tag");
  
  edges = spImage.getEdges();
  XCTAssert(edges.size() == 0, @"num edges");
  
  // The merge operation must have removed the neighbor 1 from 0
  
  neighbors = spImage.edgeTable.getNeighbors(superpixels[0]);
  XCTAssert(neighbors.size() == 0, @"neighbors size");
  
  // Verify that key for 1 was removed from neighbors table
  
  vector<int32_t> tags = spImage.edgeTable.getAllTagsInNeighborsTable();
  XCTAssert(tags.size() == 1, @"neighbors table size");
  XCTAssert(tags[0] == superpixels[0], @"neighbors");
}

// In this case 1 contains more pixels than 0 so the merge src will be 0

- (void)testMerge2x2 {
  
  NSArray *pixelsArr = @[
                         @(0), @(1),
                         @(1), @(1)
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
                          @[@(0), @(0)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[1])];
    
    NSArray *expected = @[
                          @[@(1), @(0)],
                          @[@(0), @(1)],
                          @[@(1), @(1)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 1, @"num edges");
  
  {
    SuperpixelEdge &edge = edges[0];
    XCTAssert(edge.A == superpixels[0], @"edge.A");
    XCTAssert(edge.B == superpixels[1], @"edge.B");
  }
  
  // Merge superpixels given an edge between them, note that
  // a merge operation always chooses the UID to use as the
  // result based on the number of pixels in the superpixel.
  // In this case, there is a tie and the smaller UID is used.
  
  XCTAssert(spImage.tagToSuperpixelMap.size() == 2, @"sumperpixel UID table");
  
  spImage.mergeEdge(edges[0]);
  
  XCTAssert(spImage.tagToSuperpixelMap.size() == 1, @"sumperpixel UID table");
  
  superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 1, @"num sumperpixels");
  
  XCTAssert(superpixels[0] == 1+1, @"tag");
  
  edges = spImage.getEdges();
  XCTAssert(edges.size() == 0, @"num edges");
  
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[0])];
    
    NSArray *expected = @[
                          @[@(1), @(0)],
                          @[@(0), @(1)],
                          @[@(1), @(1)],
                          // coords from A appended to B
                          @[@(0), @(0)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  // The merge operation must have removed the neighbor 0 from 1
  
  vector<int32_t> neighbors;
  
  neighbors = spImage.edgeTable.getNeighbors(superpixels[1]);
  XCTAssert(neighbors.size() == 0, @"neighbors size");
  
  // Verify that key for 0 was removed from neighbors table
  
  vector<int32_t> tags = spImage.edgeTable.getAllTagsInNeighborsTable();
  XCTAssert(tags.size() == 1, @"neighbors table size");
  XCTAssert(tags[0] == superpixels[0], @"neighbors");
}

// In this test case 2 of the superpixel are merged but one is not.
// The edges that are not merged need to be updated so that the
// merged edge UID is rewritten with the UID from the larger
// superpixel.

- (void)testMerge2x2Merge2of3 {
  
  NSArray *pixelsArr = @[
                         @(0), @(0),
                         @(1), @(2)
                         ];
  
  Mat tagsImg(2, 2, CV_MAKETYPE(CV_8U, 3));
  
  [self.class fillImageWithPixels:pixelsArr img:tagsImg];
  
  SuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(tagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  vector<int32_t> superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 3, @"num sumperpixels");
  
  XCTAssert(superpixels[0] == 0+1, @"tag");
  XCTAssert(superpixels[1] == 1+1, @"tag");
  XCTAssert(superpixels[2] == 2+1, @"tag");
  
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
                          @[@(0), @(1)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[2])];
    
    NSArray *expected = @[
                          @[@(1), @(1)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 3, @"num edges");
  
  {
    SuperpixelEdge &edge = edges[0];
    XCTAssert(edge.A == superpixels[0], @"edge.A");
    XCTAssert(edge.B == superpixels[1], @"edge.B");
  }
  
  {
    SuperpixelEdge &edge = edges[1];
    XCTAssert(edge.A == superpixels[0], @"edge.A");
    XCTAssert(edge.B == superpixels[2], @"edge.B");
  }
  
  {
    SuperpixelEdge &edge = edges[2];
    XCTAssert(edge.A == superpixels[1], @"edge.A");
    XCTAssert(edge.B == superpixels[2], @"edge.B");
  }
  
  // Merge superpixels given an edge between them, note that
  // a merge operation always chooses the UID to use as the
  // result based on the number of pixels in the superpixel.
  // In this case, there is a tie and the smaller UID is used.
  
  XCTAssert(spImage.tagToSuperpixelMap.size() == 3, @"sumperpixel UID table");
  
  // Merge superpixels (0 1) together
  
  spImage.mergeEdge(edges[0]);
  
  XCTAssert(spImage.tagToSuperpixelMap.size() == 2, @"sumperpixel UID table");
  
  superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 2, @"num sumperpixels");
  
  XCTAssert(superpixels[0] == 0+1, @"tag");
  XCTAssert(superpixels[1] == 2+1, @"tag");
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[0])];
    
    NSArray *expected = @[
                          @[@(0), @(0)],
                          @[@(1), @(0)],
                          @[@(0), @(1)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[1])];
    
    NSArray *expected = @[
                          @[@(1), @(1)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  // Superpixel 1 merged into 0, so check for update from (1 2) which should have
  // been updated to (0 2).
  
  // result:
  //
  // 0 0
  // 0 2
  
  edges = spImage.getEdges();
  XCTAssert(edges.size() == 1, @"num edges");
  
  SuperpixelEdge edge = edges[0];
  
  XCTAssert(edge.A == superpixels[0], @"edge");
  XCTAssert(edge.B == superpixels[1], @"edge");
  
  // The merge operation must have removed the neighbor 1 from 2
  // and 1 must have been removed from 0
  
  vector<int32_t> neighbors;
  
  neighbors = spImage.edgeTable.getNeighbors(superpixels[0]);
  XCTAssert(neighbors.size() == 1, @"neighbors size");
  XCTAssert(neighbors[0] == superpixels[1], @"neighbors");
  
  neighbors = spImage.edgeTable.getNeighbors(superpixels[1]);
  XCTAssert(neighbors.size() == 1, @"neighbors size");
  XCTAssert(neighbors[0] == superpixels[0], @"neighbors");
  
  // Verify that key for 1 was removed from neighbors table
  
  vector<int32_t> tags = spImage.edgeTable.getAllTagsInNeighborsTable();
  XCTAssert(tags.size() == 2, @"neighbors table size");
  XCTAssert(tags[0] == superpixels[0], @"neighbors");
  XCTAssert(tags[1] == superpixels[1], @"neighbors");
  
  return;
}

// In this 3x3 test grid a merge results in a new edge being created
// between the larger superpixel and a superpixel that was not a neighbor
// before the merge. In this case 2 is merged into 0 and that must
// create an edge between 0 and 3.

- (void)testMerge3x3MergeAdd {
  
  NSArray *pixelsArr = @[
                         @(0), @(0), @(0),
                         @(1), @(2), @(2),
                         @(3), @(3), @(3),
                         ];
  
  Mat tagsImg(3, 3, CV_MAKETYPE(CV_8U, 3));
  
  [self.class fillImageWithPixels:pixelsArr img:tagsImg];
  
  SuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(tagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  vector<int32_t> superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 4, @"num sumperpixels");
  
  XCTAssert(superpixels[0] == 0+1, @"tag");
  XCTAssert(superpixels[1] == 1+1, @"tag");
  XCTAssert(superpixels[2] == 2+1, @"tag");
  XCTAssert(superpixels[3] == 3+1, @"tag");
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[0])];
    
    NSArray *expected = @[
                          @[@(0), @(0)],
                          @[@(1), @(0)],
                          @[@(2), @(0)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[1])];
    
    NSArray *expected = @[
                          @[@(0), @(1)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[2])];
    
    NSArray *expected = @[
                          @[@(1), @(1)],
                          @[@(2), @(1)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[3])];
    
    NSArray *expected = @[
                          @[@(0), @(2)],
                          @[@(1), @(2)],
                          @[@(2), @(2)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 5, @"num edges");
  
  {
    SuperpixelEdge &edge = edges[0];
    XCTAssert(edge.A == superpixels[0], @"edge.A");
    XCTAssert(edge.B == superpixels[1], @"edge.B");
  }
  
  {
    SuperpixelEdge &edge = edges[1];
    XCTAssert(edge.A == superpixels[0], @"edge.A");
    XCTAssert(edge.B == superpixels[2], @"edge.B");
  }
  
  {
    SuperpixelEdge &edge = edges[2];
    XCTAssert(edge.A == superpixels[1], @"edge.A");
    XCTAssert(edge.B == superpixels[2], @"edge.B");
  }
  
  {
    SuperpixelEdge &edge = edges[3];
    XCTAssert(edge.A == superpixels[1], @"edge.A");
    XCTAssert(edge.B == superpixels[3], @"edge.B");
  }
  
  {
    SuperpixelEdge &edge = edges[4];
    XCTAssert(edge.A == superpixels[2], @"edge.A");
    XCTAssert(edge.B == superpixels[3], @"edge.B");
  }
  
  // Merge superpixels given an edge between them, note that
  // a merge operation always chooses the UID to use as the
  // result based on the number of pixels in the superpixel.
  // In this case, there is a tie and the smaller UID is used.
  
  XCTAssert(spImage.tagToSuperpixelMap.size() == superpixels.size(), @"sumperpixel UID table");
  
  // Merge superpixels (0 2) aka (1 3) together
  // Pre merged
  
  // 0 0 0
  // 1 2 2
  // 3 3 3
  
  spImage.mergeEdge(edges[1]);
  
  XCTAssert(spImage.tagToSuperpixelMap.size() == 3, @"sumperpixel UID table");
  
  superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 3, @"num sumperpixels");
  
  XCTAssert(superpixels[0] == 0+1, @"tag");
  XCTAssert(superpixels[1] == 1+1, @"tag");
  XCTAssert(superpixels[2] == 3+1, @"tag");
  
  // Merged result
  
  // 0 0 0
  // 1 0 0
  // 3 3 3
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[0])];
    
    NSArray *expected = @[
                          @[@(0), @(0)],
                          @[@(1), @(0)],
                          @[@(2), @(0)],
                          @[@(1), @(1)],
                          @[@(2), @(1)],
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[1])];
    
    NSArray *expected = @[
                          @[@(0), @(1)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[2])];
    
    NSArray *expected = @[
                          @[@(0), @(2)],
                          @[@(1), @(2)],
                          @[@(2), @(2)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  vector<int32_t> neighbors;
  
  // Post merge, sp 0 should have 2 neighbors
  
  neighbors = spImage.edgeTable.getNeighbors(superpixels[0]);
  XCTAssert(neighbors.size() == 2, @"neighbors size");
  XCTAssert(neighbors[0] == superpixels[1], @"neighbors");
  XCTAssert(neighbors[1] == superpixels[2], @"neighbors");
  
  // sp 1 should have 2 neighbors
  
  neighbors = spImage.edgeTable.getNeighbors(superpixels[1]);
  XCTAssert(neighbors.size() == 2, @"neighbors size");
  XCTAssert(neighbors[0] == superpixels[0], @"neighbors");
  XCTAssert(neighbors[1] == superpixels[2], @"neighbors");
  
  // sp 3 should have 2 neighbors
  
  neighbors = spImage.edgeTable.getNeighbors(superpixels[2]);
  XCTAssert(neighbors.size() == 2, @"neighbors size");
  XCTAssert(neighbors[0] == superpixels[0], @"neighbors");
  XCTAssert(neighbors[1] == superpixels[1], @"neighbors");
  
  vector<int32_t> tags = spImage.edgeTable.getAllTagsInNeighborsTable();
  XCTAssert(tags.size() == 3, @"neighbors table size");
  XCTAssert(tags[0] == superpixels[0], @"neighbors");
  XCTAssert(tags[1] == superpixels[1], @"neighbors");
  XCTAssert(tags[2] == superpixels[2], @"neighbors");
  
  // Superpixel 2 merged into 0 with new edge created between 0 and 3
  
  edges = spImage.getEdges();
  XCTAssert(edges.size() == 3, @"num edges");
  
  SuperpixelEdge edge = edges[0];
  
  XCTAssert(edge.A == superpixels[0], @"num edges");
  XCTAssert(edge.B == superpixels[1], @"num edges");
  
  edge = edges[1];
  
  XCTAssert(edge.A == superpixels[0], @"num edges");
  XCTAssert(edge.B == superpixels[2], @"num edges");
  
  edge = edges[2];
  
  XCTAssert(edge.A == superpixels[1], @"num edges");
  XCTAssert(edge.B == superpixels[2], @"num edges");
  
  return;
}

// This 3x3 matrix merges 1 into 2 which creates no
// new edges, the result has 2 fewer edges since
// (0 2) and (1 2) and no longer needed.

- (void)testMerge3x3MergeAdd2 {
  
  NSArray *pixelsArr = @[
                         @(0), @(0), @(1),
                         @(2), @(2), @(2),
                         @(3), @(3), @(4)
                         ];
  
  Mat tagsImg(3, 3, CV_MAKETYPE(CV_8U, 3));
  
  [self.class fillImageWithPixels:pixelsArr img:tagsImg];
  
  SuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(tagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  vector<int32_t> superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 5, @"num sumperpixels");
  
  XCTAssert(superpixels[0] == 0+1, @"tag");
  XCTAssert(superpixels[1] == 1+1, @"tag");
  XCTAssert(superpixels[2] == 2+1, @"tag");
  XCTAssert(superpixels[3] == 3+1, @"tag");
  XCTAssert(superpixels[4] == 4+1, @"tag");
  
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
                          @[@(2), @(0)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[2])];
    
    NSArray *expected = @[
                          @[@(0), @(1)],
                          @[@(1), @(1)],
                          @[@(2), @(1)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[3])];
    
    NSArray *expected = @[
                          @[@(0), @(2)],
                          @[@(1), @(2)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[4])];
    
    NSArray *expected = @[
                          @[@(2), @(2)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 6, @"num edges");
  
  {
    SuperpixelEdge &edge = edges[0];
    XCTAssert(edge.A == superpixels[0], @"edge.A");
    XCTAssert(edge.B == superpixels[1], @"edge.B");
  }
  
  {
    SuperpixelEdge &edge = edges[1];
    XCTAssert(edge.A == superpixels[0], @"edge.A");
    XCTAssert(edge.B == superpixels[2], @"edge.B");
  }
  
  {
    SuperpixelEdge &edge = edges[2];
    XCTAssert(edge.A == superpixels[1], @"edge.A");
    XCTAssert(edge.B == superpixels[2], @"edge.B");
  }
  
  {
    SuperpixelEdge &edge = edges[3];
    XCTAssert(edge.A == superpixels[2], @"edge.A");
    XCTAssert(edge.B == superpixels[3], @"edge.B");
  }
  
  {
    SuperpixelEdge &edge = edges[4];
    XCTAssert(edge.A == superpixels[2], @"edge.A");
    XCTAssert(edge.B == superpixels[4], @"edge.B");
  }
  
  {
    SuperpixelEdge &edge = edges[5];
    XCTAssert(edge.A == superpixels[3], @"edge.A");
    XCTAssert(edge.B == superpixels[4], @"edge.B");
  }
  
  // Merge superpixels given an edge between them, note that
  // a merge operation always chooses the UID to use as the
  // result based on the number of pixels in the superpixel.
  // In this case, there is a tie and the smaller UID is used.
  
  XCTAssert(spImage.tagToSuperpixelMap.size() == superpixels.size(), @"sumperpixel UID table");
  
  // Merge superpixels (1 2) together
  
  spImage.mergeEdge(edges[2]);
  
  XCTAssert(spImage.tagToSuperpixelMap.size() == 4, @"sumperpixel UID table");
  
  superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 4, @"num sumperpixels");
  
  XCTAssert(superpixels[0] == 0+1, @"tag");
  XCTAssert(superpixels[1] == 2+1, @"tag");
  XCTAssert(superpixels[2] == 3+1, @"tag");
  XCTAssert(superpixels[3] == 4+1, @"tag");
  
  // Merged result
  
  // 0 0 2
  // 2 2 2
  // 3 3 4
  
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
                          @[@(1), @(1)],
                          @[@(2), @(1)],
                          @[@(2), @(0)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[2])];
    
    NSArray *expected = @[
                          @[@(0), @(2)],
                          @[@(1), @(2)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[3])];
    
    NSArray *expected = @[
                          @[@(2), @(2)]
                          ];
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  // Superpixel 2 merged into 1 with no new edges and two removed
  
  edges = spImage.getEdges();
  XCTAssert(edges.size() == 4, @"num edges");
  
  SuperpixelEdge edge = edges[0];
  
  // sp[0] = 0
  // sp[1] = 2
  // sp[2] = 3
  // sp[3] = 4
  
  XCTAssert(edge.A == superpixels[0], @"num edges");
  XCTAssert(edge.B == superpixels[1], @"num edges"); // tag = 2
  
  edge = edges[1];
  
  XCTAssert(edge.A == superpixels[1], @"num edges"); // tag = 2
  XCTAssert(edge.B == superpixels[2], @"num edges"); // tag 3
  
  edge = edges[2];
  
  XCTAssert(edge.A == superpixels[1], @"num edges"); // tag = 2
  XCTAssert(edge.B == superpixels[3], @"num edges"); // tag = 4
  
  edge = edges[3];
  
  XCTAssert(edge.A == superpixels[2], @"num edges"); // tag = 3
  XCTAssert(edge.B == superpixels[3], @"num edges"); // tag = 4
  
  return;
}


// Test case for extracting superpixel pixels from larger image into a flat LUT
// and then extracting the data from the LUT back into an image of the original
// size.

- (void)testFillUtil {
  
  NSArray *pixelsArr = @[
                         @(10), @(11),
                         @(10), @(12)
                         ];
  
  Mat tagsImg(2, 2, CV_MAKETYPE(CV_8U, 3));
  
  [self.class fillImageWithPixels:pixelsArr img:tagsImg];
  
  SuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(tagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  vector<int32_t> superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 3, @"num sumperpixels");
  
  int32_t uid = superpixels[0];
  
  // Copy 2 pixels with the value 0 into a 2 element LUT
  
  Mat lutFromTags;
  
  spImage.fillMatrixFromCoords(tagsImg, uid, lutFromTags);
  
  XCTAssert(lutFromTags.rows == 1, @"rows");
  XCTAssert(lutFromTags.cols == 2, @"cols");
  
  Vec3b ten3B(uid, 0, 0);
  
  XCTAssert(lutFromTags.at<Vec3b>(0, 0) == ten3B, @"pixel");
  XCTAssert(lutFromTags.at<Vec3b>(0, 1) == ten3B, @"pixel");
  
  // Reverse the process and write the pixel values 0 and 0 back to a matrix
  // that has been initialized to 1.
  
  Vec3b one3B(1, 1, 1);
  
  Mat ones(2, 2, CV_MAKETYPE(CV_8U, 3), Scalar(1, 1, 1));
  
  XCTAssert(ones.at<Vec3b>(0, 0) == one3B, @"pixel");
  XCTAssert(ones.at<Vec3b>(0, 1) == one3B, @"pixel");
  XCTAssert(ones.at<Vec3b>(1, 0) == one3B, @"pixel");
  XCTAssert(ones.at<Vec3b>(1, 1) == one3B, @"pixel");
  
  spImage.reverseFillMatrixFromCoords(lutFromTags, false, uid, ones);
  
  // Result should be
  
  // 10 1,1,1
  // 10 1,1,1
  
  XCTAssert(ones.at<Vec3b>(0, 0) == ten3B, @"pixel");
  XCTAssert(ones.at<Vec3b>(0, 1) == one3B, @"pixel");
  XCTAssert(ones.at<Vec3b>(1, 0) == ten3B, @"pixel");
  XCTAssert(ones.at<Vec3b>(1, 1) == one3B, @"pixel");
}

// Test bbox util method on Superpixel class, this should return the X,Y,W,H of the bbox
// for the superpixel.

- (void)testSuperpixelBbox1 {
  
  NSArray *pixelsArr = @[
                         @(10), @(10),
                         @(11), @(12)
                         ];
  
  Mat tagsImg(2, 2, CV_MAKETYPE(CV_8U, 3));
  
  [self.class fillImageWithPixels:pixelsArr img:tagsImg];
  
  SuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(tagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  vector<int32_t> superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 3, @"num sumperpixels");
  
  // Got bbox for superpixel 10
  
  int32_t uid = superpixels[0];
  XCTAssert(uid == 10+1);
  
  Superpixel *spPtr = spImage.getSuperpixelPtr(uid);
  
  int32_t originX, originY, width, height;
  
  spPtr->bbox(originX, originY, width, height);
  
  XCTAssert(originX == 0, @"bbox");
  XCTAssert(originY == 0, @"bbox");
  XCTAssert(width == 2, @"bbox");
  XCTAssert(height == 1, @"bbox");
  
  // Got bbox for superpixel 11
  
  uid = superpixels[1];
  XCTAssert(uid == 11+1);
  
  spPtr = spImage.getSuperpixelPtr(uid);
  
  spPtr->bbox(originX, originY, width, height);
  
  XCTAssert(originX == 0, @"bbox");
  XCTAssert(originY == 1, @"bbox");
  XCTAssert(width == 1, @"bbox");
  XCTAssert(height == 1, @"bbox");
  
  // Got bbox for superpixel 12
  
  uid = superpixels[2];
  XCTAssert(uid == 12+1);
  
  spPtr = spImage.getSuperpixelPtr(uid);
  
  spPtr->bbox(originX, originY, width, height);
  
  XCTAssert(originX == 1, @"bbox");
  XCTAssert(originY == 1, @"bbox");
  XCTAssert(width == 1, @"bbox");
  XCTAssert(height == 1, @"bbox");
}

// Test bbox util method on Superpixel class, this should return the X,Y,W,H of the bbox
// for the superpixel.

- (void)testSuperpixelBbox2 {
  
  NSArray *pixelsArr = @[
                         @(10), @(10), @(10), @(10),
                         @(10), @(11), @(11), @(10),
                         @(10), @(11), @(11), @(11),
                         @(10), @(10), @(11), @(10)
                         ];
  
  Mat tagsImg(4, 4, CV_MAKETYPE(CV_8U, 3));
  
  [self.class fillImageWithPixels:pixelsArr img:tagsImg];
  
  SuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(tagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  vector<int32_t> superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 2, @"num sumperpixels");
  
  // Got bbox for superpixel 10
  
  int32_t uid = superpixels[0];
  XCTAssert(uid == 10+1);
  
  Superpixel *spPtr = spImage.getSuperpixelPtr(uid);
  
  int32_t originX, originY, width, height;
  
  spPtr->bbox(originX, originY, width, height);
  
  XCTAssert(originX == 0, @"bbox");
  XCTAssert(originY == 0, @"bbox");
  XCTAssert(width == 4, @"bbox");
  XCTAssert(height == 4, @"bbox");
  
  // Got bbox for superpixel 11
  
  uid = superpixels[1];
  XCTAssert(uid == 11+1);
  
  spPtr = spImage.getSuperpixelPtr(uid);
  
  spPtr->bbox(originX, originY, width, height);
  
  XCTAssert(originX == 1, @"bbox");
  XCTAssert(originY == 1, @"bbox");
  XCTAssert(width == 3, @"bbox");
  XCTAssert(height == 3, @"bbox");
}

- (void)testFilterEdgeCoords1 {
  
  NSArray *pixelsArr = @[
                         @(10), @(10), @(10), @(10),
                         @(10), @(11), @(11), @(11),
                         @(10), @(11), @(11), @(11),
                         @(10), @(10), @(10), @(10)
                         ];
  
  Mat tagsImg(4, 4, CV_MAKETYPE(CV_8U, 3));
  
  [self.class fillImageWithPixels:pixelsArr img:tagsImg];
  
  SuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(tagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  vector<int32_t> superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 2, @"num sumperpixels");
  
  // Get edge coords between these two superpixels
  
  XCTAssert(superpixels[0] == 10+1);
  XCTAssert(superpixels[1] == 11+1);
  
  // Verify coords for superpixel 10
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[0])];
    
    NSArray *expected = @[
                          @[@(0), @(0)],
                          @[@(1), @(0)],
                          @[@(2), @(0)],
                          @[@(3), @(0)],
                          @[@(0), @(1)],
                          @[@(0), @(2)],
                          @[@(0), @(3)],
                          @[@(1), @(3)],
                          @[@(2), @(3)],
                          @[@(3), @(3)]
                          ];
    
    XCTAssert(result.count == 10, @"coords");
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  // Verify coords for superpixel 11
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[1])];
    
    NSArray *expected = @[
                          @[@(1), @(1)],
                          @[@(2), @(1)],
                          @[@(3), @(1)],
                          @[@(1), @(2)],
                          @[@(2), @(2)],
                          @[@(3), @(2)]
                          ];
    
    XCTAssert(result.count == 6, @"coords");
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  Superpixel *spPtr1 = spImage.getSuperpixelPtr(superpixels[0]);
  Superpixel *spPtr2 = spImage.getSuperpixelPtr(superpixels[1]);
  
  XCTAssert(superpixels.size() == 2, @"num sumperpixels");
  
  vector<Coord> edgeCoords1;
  vector<Coord> edgeCoords2;
  
  Superpixel::filterEdgeCoords(spPtr1, edgeCoords1, spPtr2, edgeCoords2);
  
  XCTAssert((edgeCoords1.size() + edgeCoords2.size()) == 16, @"bbox");
  
  // Verify edge coords for superpixel 10
  
  {
    NSArray *result = [self.class formatCoords:edgeCoords1];
    
    NSArray *expected = @[
                          @[@(0), @(0)],
                          @[@(1), @(0)],
                          @[@(2), @(0)],
                          @[@(3), @(0)],
                          
                          @[@(0), @(1)],
                          
                          @[@(0), @(2)],
                          
                          @[@(0), @(3)],
                          @[@(1), @(3)],
                          @[@(2), @(3)],
                          @[@(3), @(3)]
                          
                          ];
    
    XCTAssert(result.count == 10, @"coords");
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  // Verify edge coords for superpixel 11
  
  {
    NSArray *result = [self.class formatCoords:edgeCoords2];
    
    NSArray *expected = @[
                          
                          @[@(1), @(1)],
                          @[@(2), @(1)],
                          @[@(3), @(1)],
                          
                          @[@(1), @(2)],
                          @[@(2), @(2)],
                          @[@(3), @(2)]
                          
                          ];
    
    XCTAssert(result.count == 6, @"coords");
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
}

// This smaller superpixel will be rooted at (1, 1) and it
// will contain 1 pixel that is not an edge pixel.

- (void)testFilterEdgeCoords2 {
  
  NSArray *pixelsArr = @[
                         @(10), @(10), @(10), @(10), @(10), @(10),
                         @(10), @(10), @(10), @(10), @(10), @(10),
                         @(10), @(10), @(11), @(11), @(11), @(10),
                         @(10), @(10), @(11), @(11), @(11), @(10),
                         @(10), @(10), @(11), @(11), @(11), @(10),
                         @(10), @(10), @(10), @(10), @(10), @(10)
                         ];
  
  Mat tagsImg(6, 6, CV_MAKETYPE(CV_8U, 3));
  
  [self.class fillImageWithPixels:pixelsArr img:tagsImg];
  
  SuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(tagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  vector<int32_t> superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 2, @"num sumperpixels");
  
  // Get edge coords between these two superpixels
  
  XCTAssert(superpixels[0] == 10+1);
  XCTAssert(superpixels[1] == 11+1);
  
  // Verify coords for superpixel 10
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[0])];
    
    NSArray *expected = @[
                          @[@(0), @(0)],
                          @[@(1), @(0)],
                          @[@(2), @(0)],
                          @[@(3), @(0)],
                          @[@(4), @(0)],
                          @[@(5), @(0)],
                          
                          @[@(0), @(1)],
                          @[@(1), @(1)],
                          @[@(2), @(1)],
                          @[@(3), @(1)],
                          @[@(4), @(1)],
                          @[@(5), @(1)],
                          
                          @[@(0), @(2)],
                          @[@(1), @(2)],
                          @[@(5), @(2)],
                          
                          @[@(0), @(3)],
                          @[@(1), @(3)],
                          @[@(5), @(3)],
                          
                          @[@(0), @(4)],
                          @[@(1), @(4)],
                          @[@(5), @(4)],
                          
                          @[@(0), @(5)],
                          @[@(1), @(5)],
                          @[@(2), @(5)],
                          @[@(3), @(5)],
                          @[@(4), @(5)],
                          @[@(5), @(5)]
                          
                          ];
    
    XCTAssert(result.count == 27, @"coords");
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  // Verify coords for superpixel 11
  
  {
    NSArray *result = [self.class formatSuperpixelCoords:spImage.getSuperpixelPtr(superpixels[1])];
    
    NSArray *expected = @[
                          @[@(2), @(2)],
                          @[@(3), @(2)],
                          @[@(4), @(2)],
                          
                          @[@(2), @(3)],
                          @[@(3), @(3)],
                          @[@(4), @(3)],
                          
                          @[@(2), @(4)],
                          @[@(3), @(4)],
                          @[@(4), @(4)]
                          
                          ];
    
    XCTAssert(result.count == 9, @"coords");
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  Superpixel *spPtr1 = spImage.getSuperpixelPtr(superpixels[0]);
  Superpixel *spPtr2 = spImage.getSuperpixelPtr(superpixels[1]);
  
  XCTAssert(superpixels.size() == 2, @"num sumperpixels");
  
  vector<Coord> edgeCoords1;
  vector<Coord> edgeCoords2;
  
  Superpixel::filterEdgeCoords(spPtr1, edgeCoords1, spPtr2, edgeCoords2);
  
  XCTAssert((edgeCoords1.size() + edgeCoords2.size()) == 16+8, @"bbox");
  
  // Verify edge coords for superpixel 10
  
  {
    NSArray *result = [self.class formatCoords:edgeCoords1];
    
    NSArray *expected = @[
                          
                          @[@(1), @(1)],
                          @[@(2), @(1)],
                          @[@(3), @(1)],
                          @[@(4), @(1)],
                          @[@(5), @(1)],
                          
                          @[@(1), @(2)],
                          @[@(5), @(2)],
                          
                          @[@(1), @(3)],
                          @[@(5), @(3)],
                          
                          @[@(1), @(4)],
                          @[@(5), @(4)],
                          
                          @[@(1), @(5)],
                          @[@(2), @(5)],
                          @[@(3), @(5)],
                          @[@(4), @(5)],
                          @[@(5), @(5)]
                          
                          ];
    
    XCTAssert(result.count == 16, @"coords");
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
  
  // Verify edge coords for superpixel 11
  
  {
    NSArray *result = [self.class formatCoords:edgeCoords2];
    
    NSArray *expected = @[
                          
                          @[@(2), @(2)],
                          @[@(3), @(2)],
                          @[@(4), @(2)],
                          
                          @[@(2), @(3)],
                          // Note that (3,3) is not an edge pixel
                          @[@(4), @(3)],
                          
                          @[@(2), @(4)],
                          @[@(3), @(4)],
                          @[@(4), @(4)]
                          
                          ];
    
    XCTAssert(result.count == 8, @"coords");
    
    XCTAssert([result isEqualToArray:expected], @"coords");
  }
}

// Compare edges between pixels in terms of LAB colorspace dist.

- (void)testCompareEdgesTwoSuperpixels {
  
  NSArray *pixelsArr = @[
                         @(0), @(0),
                         @(1), @(1),
                         ];
  
  Mat tagsImg(2, 2, CV_MAKETYPE(CV_8U, 3));
  
  [self.class fillImageWithPixels:pixelsArr img:tagsImg];
  
  MergeSuperpixelImage spImage;
  
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
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 1, @"num edges");
  
  // RGB image will be converted to LAB colorspace
  
  NSArray *rgbPixelsArr = @[
                            @(0x000000), @(0x000000),
                            @(0xFFFFFF), @(0xFFFFFF),
                            ];
  
  Mat rgbImg(2, 2, CV_MAKETYPE(CV_8U, 3));
  
  [self.class fillImageWithPixels:rgbPixelsArr img:rgbImg];
  
  vector<CompareNeighborTuple> results;
  
  SuperpixelEdgeFuncs::compareNeighborEdges(spImage, rgbImg, superpixels[0], results, NULL, 0, true);
  
  XCTAssert(results.size() == 1, @"results");
  
  CompareNeighborTuple tuple = results[0];
  double dist = get<0>(tuple);
  
  XCTAssert(dist == 1.0, @"dist");
  
  return;
}

- (void)testCompareEdgesTwoSuperpixels2 {
  
  NSArray *pixelsArr = @[
                         @(0), @(0),
                         @(1), @(1),
                         ];
  
  Mat tagsImg(2, 2, CV_MAKETYPE(CV_8U, 3));
  
  [self.class fillImageWithPixels:pixelsArr img:tagsImg];
  
  MergeSuperpixelImage spImage;
  
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
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 1, @"num edges");
  
  // RGB image will be converted to LAB colorspace
  
  NSArray *rgbPixelsArr = @[
                            @(0x000000), @(0x000000),
                            @(0xFF7777), @(0xFFFFFF),
                            ];
  
  Mat rgbImg(2, 2, CV_MAKETYPE(CV_8U, 3));
  
  [self.class fillImageWithPixels:rgbPixelsArr img:rgbImg];
  
  vector<CompareNeighborTuple> results;
  
  SuperpixelEdgeFuncs::compareNeighborEdges(spImage, rgbImg, superpixels[0], results, NULL, 0, true);
  
  XCTAssert(results.size() == 1, @"results");
  
  CompareNeighborTuple tuple = results[0];
  double dist = get<0>(tuple);
  
  // Note that this result is normalized
  
  XCTAssert(round(dist) == 1.0, @"dist");
  
  return;
}

- (void)testCompareEdgesThreeSuperpixels {
  
  NSArray *pixelsArr = @[
                         @(0), @(0), @(0), @(0),
                         @(0), @(0), @(1), @(1),
                         @(2), @(2), @(2), @(2),
                         @(2), @(2), @(2), @(2)
                         ];
  
  Mat tagsImg(4, 4, CV_MAKETYPE(CV_8U, 3));
  
  [self.class fillImageWithPixels:pixelsArr img:tagsImg];
  
  MergeSuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(tagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  vector<int32_t> superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 3, @"num sumperpixels");
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 3, @"num edges");
  
  // RGB image will be converted to LAB colorspace
  
  NSArray *rgbPixelsArr = @[
                            @(0x000000), @(0x000000), @(0x000000), @(0x000000),
                            @(0x000000), @(0x000000), @(0x777777), @(0x777777),
                            @(0x999999), @(0x999999), @(0x999999), @(0x999999),
                            @(0x999999), @(0x999999), @(0x999999), @(0x999999)
                            ];
  
  Mat rgbImg(4, 4, CV_MAKETYPE(CV_8U, 3));
  
  [self.class fillImageWithPixels:rgbPixelsArr img:rgbImg];
  
  vector<CompareNeighborTuple> results;
  
  // Compare 0 to neighbors 1 and 2
  
  SuperpixelEdgeFuncs::compareNeighborEdges(spImage, rgbImg, superpixels[0], results, NULL, 0, true);
  
  XCTAssert(results.size() == 2, @"results");
  
  CompareNeighborTuple tuple;
  double dist;
  
  // normalized 128.0 -> 0.795
  
  tuple = results[0];
  dist = get<0>(tuple);
  XCTAssert(round(dist * 100.0) == 80.0, @"dist");
  
  // Normalized to max of 161.0 -> 1.0
  
  tuple = results[1];
  dist = get<0>(tuple);
  XCTAssert(dist == 1.0, @"dist");
  
  return;
}

// Testing C++ details related to putting a SuperpixelEdge
// in an unordered map

- (void)testSuperpixelEdgeObject {
  unordered_map<SuperpixelEdge, float> edgeStrengthMap;
  
  SuperpixelEdge edge(1, 0);
  
  float val;
  
  XCTAssert(edge.A == 0, @"A");
  XCTAssert(edge.B == 1, @"B");
  
  edgeStrengthMap[edge] = 1.0;
  
  val = edgeStrengthMap[edge];
  
  XCTAssert(val == 1.0f, @"val");
  
  edgeStrengthMap[edge] = 2.0;
  
  val = edgeStrengthMap[edge];
  
  XCTAssert(val == 2.0f, @"val");
  
  SuperpixelEdge delEdge(0, 1);
  edgeStrengthMap.erase(delEdge);
  
  XCTAssert(edgeStrengthMap.count(edge) == 0, @"val");
}

// Testing C++ details related to Coord class and
// putting it into an unordered map.

- (void)testSuperpixelCoordObject {
  unordered_map<Coord, bool> visitedCoordsMap;
  
  Coord cDefault;
  Coord c0(0, 0);
  Coord c1(1, 0);
  
  XCTAssert(cDefault == c0, @"op =");
  XCTAssert(cDefault != c1, @"op !=");
  XCTAssert((c0 < c1) == true, @"op <");
  XCTAssert((c0 <= c1) == true, @"op <");
  XCTAssert((c0 > c1) == false, @"op >");
  XCTAssert((c0 >= c1) == false, @"op >=");
  XCTAssert((c1 > c0) == true, @"op >");
  XCTAssert((c0 == c1) == false, @"op ==");
  XCTAssert((c0 != c1) == true, @"op !=");
  
  visitedCoordsMap[c0] = true;
  
  bool visited;
  
  visited = visitedCoordsMap[c0];
  XCTAssert(visited == true, @"visited");
  
  visited = visitedCoordsMap[c1];
  XCTAssert(visited == false, @"visited");
  
  XCTAssert(visitedCoordsMap.size() == 2, @"size");
  
  Coord delCoord(0, 0);
  visitedCoordsMap.erase(delCoord);
  
  XCTAssert(visitedCoordsMap.size() == 1, @"size");
  
  int count = (int) visitedCoordsMap.count(c0);
  XCTAssert(count == 0, @"count");
  
  // Implicit create of key, return value is false
  
  visited = visitedCoordsMap[c0];
  XCTAssert(visited == false, @"visited");
  
  // Operator +
  
  Coord tmp;
  
  tmp = c0 + c1;
  XCTAssert(tmp.x == 1 && tmp.y == 0, @"coord");
  
  Coord cTwo(2, 4);
  Coord cThree(3, 6);

  tmp = cTwo;
  tmp += cThree;
  
  XCTAssert(tmp.x == 5 && tmp.y == 10, @"coord");
  
  // Operator -
  
  tmp = cThree - cTwo;
  XCTAssert(tmp.x == 1 && tmp.y == 2, @"coord");
  
  tmp = cThree;
  tmp -= cTwo;
  XCTAssert(tmp.x == 1 && tmp.y == 2, @"coord");
  
  return;
}

// Test case for color quant logic in OpenCVUtil.cpp that attempts
// to subdivide the color cube into even segments

- (void)testSegmentColorCube {
  vector<uint32_t> vec = getSubdividedColors();
  
  XCTAssert(vec.size() == 125, @"getSubdividedColors");
  
  vector<uint32_t> filtered;
  
  for ( uint32_t pixel : vec ) {
    uint32_t RG = pixel & 0x00FFFF00;
    if (RG == 0) {
      filtered.push_back(pixel & 0xFF);
    }
  }
  
  XCTAssert(filtered.size() == 5, @"filtered");
  
//   colortable[   0] = 0xFF000000
//   colortable[   1] = 0xFF00003F
//   colortable[   2] = 0xFF00007F
//   colortable[   3] = 0xFF0000BF
//   colortable[   4] = 0xFF0000FF
  
  XCTAssert(filtered[0] == 0, @"filtered");
  XCTAssert(filtered[1] == 63, @"filtered");
  XCTAssert(filtered[2] == 127, @"filtered");
  XCTAssert(filtered[3] == 191, @"filtered");
  XCTAssert(filtered[4] == 255, @"filtered");
  
  // Foreach number in the range 0 -> 255 check the bin that
  // would qualify as the closest match. Each bin should contain
  // the same number of matches.
  
  unordered_map<int, int> map;
  
  int delta[5];
  
  for ( int i = 0; i < 256; i++ ) {

    int min = 256;
    int mini;
    
    for ( int j = 0; j < 5; j++ ) {
      int f = filtered[j];
      int d = i - f;
      delta[j] = d;
      int absDelta = d < 0 ? -d : d;
      
      if (absDelta < min) {
        min = absDelta;
        mini = j;
      }
    }
    
    //fprintf(stdout, "for %3d : deltas (%d %d %d %d %d) : min d %d : min offset %d\n", i, delta[0], delta[1], delta[2], delta[3], delta[4], min, mini);
    
    map[mini] += 1;
  }
  
  int totalCount = 0;
  
  for ( auto &pair : map ) {
    fprintf(stdout, "%d -> %d\n", pair.first, pair.second);
    
    totalCount += pair.second;
  }

  //fprintf(stdout, "totalCount %d\n", totalCount);
  
  XCTAssert(totalCount == 256, @"filtered");
  
  return;
}

- (void)testSegmentColorCube2 {
  vector<uint32_t> filtered(4);

  // Divide so that coordinates in (0, 255) range get
  // evenly split into 4 buckets.
  
  filtered[0] = 0;
  filtered[1] = 127;
  filtered[2] = 128;
  filtered[3] = 255;
  
  // Foreach number in the range 0 -> 255 check the bin that
  // would qualify as the closest match. Each bin should contain
  // the same number of matches.
  
  unordered_map<int, int> map;
  
  int delta[filtered.size()];
  
  for ( int i = 0; i < 256; i++ ) {
    
    int min = 256;
    int mini;
    
    for ( int j = 0; j < filtered.size(); j++ ) {
      int f = filtered[j];
      int d = i - f;
      delta[j] = d;
      int absDelta = d < 0 ? -d : d;
      
      if (absDelta < min) {
        min = absDelta;
        mini = j;
      }
    }
    
    fprintf(stdout, "for %3d : deltas (%d %d %d %d) : min d %d : min offset %d\n", i, delta[0], delta[1], delta[2], delta[3], min, mini);
    
    map[mini] += 1;
  }
  
  int totalCount = 0;
  
  for ( auto &pair : map ) {
    fprintf(stdout, "%d -> %d\n", pair.first, pair.second);
    
    totalCount += pair.second;
  }
  
  fprintf(stdout, "totalCount %d\n", totalCount);
  
  XCTAssert(totalCount == 256, @"filtered");
  
  return;
}

// Test case for color quant logic in OpenCVUtil.cpp that attempts
// to subdivide the color cube into even segments

- (void)testIdenticalNeighbors1 {
  
  NSArray *pixelsArr = @[
                         @(0), @(0), @(0),
                         @(1), @(1), @(1),
                         @(2), @(2), @(3),
                         ];
  
  Mat tagsImg(3, 3, CV_8UC3);
  Mat maskImg(3, 3, CV_8UC1);
  
  maskImg = (Scalar) 0xFF;
  
  [self.class fillImageWithPixels:pixelsArr img:tagsImg];
  
  unordered_map<uint32_t, uint32_t> pixelToNumVotesMap;
  
  vote_for_identical_neighbors(pixelToNumVotesMap, tagsImg, maskImg);
  
  for ( auto &pair : pixelToNumVotesMap ) {
    fprintf(stdout, "%d -> %d\n", pair.first, pair.second);
  }

  XCTAssert(pixelToNumVotesMap.size() == 3, @"size");
  
  XCTAssert(pixelToNumVotesMap[0] == 4, @"votes");
  XCTAssert(pixelToNumVotesMap[1] == 4, @"votes");
  XCTAssert(pixelToNumVotesMap[2] == 2, @"votes");
  
  return;
}

@end
