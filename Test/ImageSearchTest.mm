//
//  ImageSearchTest.m
//
//  This test code checks logic related to iterating over image regions
//  based on how alike subregions are.

#include <opencv2/opencv.hpp>

#import <Foundation/Foundation.h>

#include "Superpixel.h"
#include "SuperpixelEdge.h"
#include "SuperpixelImage.h"
#include "MergeSuperpixelImage.h"

#import <XCTest/XCTest.h>

// imports

void generateStaticColortable(Mat &inputImg, SuperpixelImage &spImage);

void writeTagsWithStaticColortable(SuperpixelImage &spImage, Mat &resultImg);

void writeTagsWithGraytable(SuperpixelImage &spImage, Mat &origImg, Mat &resultImg);

@interface Test : XCTestCase

+ (void) fillImageWithPixels:(NSArray*)pixelNums img:(Mat&)img;

+ (NSArray*) formatSuperpixelCoords:(Superpixel*)spPtr;

+ (NSArray*) formatCoords:(vector<pair<int32_t,int32_t> >&)coords;

+ (NSArray*) getSuperpixelCoordsAsPixels:(Superpixel*)spPtr input:(Mat)input;

@end

// class

@interface ImageSearchTest : XCTestCase

+ (Mat) parseImageGrayscale:(string)str
                      width:(int)width
                     height:(int)height
                      scale:(float)scale;
@end

@implementation ImageSearchTest

+ (Mat) parseImageGrayscale:(string)str
                      width:(int)width
                     height:(int)height
                      scale:(float)scale
{
  Mat img(height, width, CV_8UC(3));
  
  int i = 0;
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      int c = str[i++];
      if (c == '\0') {
        // Parsed the end of string marker
        assert(0);
      }
      assert(c <= 'Z');
      int amt;
      if (c >= 'A') {
        // Treat A as 10
        amt = c - 'A' + 10;
      } else {
        // Treat 0 -> 9 as decimal
        assert(c >= '0' && c <= '9');
        amt = c - '0';
      }
      assert(amt >= 0);
      amt = (int) round(amt * scale);
      assert(amt >= 0);
      assert(amt <= 255);
      Vec3b pixelVec(amt, 0, 0);
      img.at<Vec3b>(row, col) = pixelVec;
    }
  }
  return img;
}

// Get a simple matrix where 2x2 superpixels define a region that can clearly be merged
// together because all the pixels are exactly identical.

+ (void) getSimple4by4BW1:(Mat&)pixelsImage
        tagsImage:(Mat&)tagsImage
{
  int width = 4;
  int height = 4;
  
  string pixels =
  "0011"
  "0011"
  "0000"
  "0000";
  
  string tags =
  "0011"
  "0011"
  "2233"
  "2233";
  
  pixelsImage = [self parseImageGrayscale:pixels width:width height:height scale:255.0f];
  
  tagsImage = [self parseImageGrayscale:tags width:width height:height scale:1.0f];
  
  return;
}

// Get maze1 image as a Mat. This image features a region (0+1) where the value is zero
// and the edge value to (6+1) or (5+1) is the same max value. The result is that a
// BFS will not expand over the edges since the backprojection sees zero alike pixels.

+ (void) getMaze1:(Mat&)pixelsImage
        tagsImage:(Mat&)tagsImage
{
  int width = 10;
  int height = 5;
  
  string pixels =
  "0000000101"
  "0000000101"
  "1111110101"
  "0010010001"
  "0010011111";
  
  string tags =
  "0000000605"
  "0000000605"
  "5555550605"
  "1152250005"
  "1152255555";

  pixelsImage = [self parseImageGrayscale:pixels width:width height:height scale:255.0f];
  
  tagsImage = [self parseImageGrayscale:tags width:width height:height scale:1.0f];
  
  return;
}

// Get maze2 image as a Mat. This image features a region (0+1) next to (1+1) that
// is exactly the same color so the two superpixels should be merged. The back projection
// finds 100% alike pixels in the histogram compare.

+ (void) getMaze2:(Mat&)pixelsImage
        tagsImage:(Mat&)tagsImage
{
  int width = 10;
  int height = 5;
  
  string pixels =
  "0000000101"
  "0000000101"
  "1111110101"
  "0010010001"
  "0010011111";
  
  string tags =
  "0000001615"
  "0000001615"
  "5555551615"
  "2253351115"
  "2253355555";
  
  pixelsImage = [self parseImageGrayscale:pixels width:width height:height scale:255.0f];
  
  tagsImage = [self parseImageGrayscale:tags width:width height:height scale:1.0f];
  
  return;
}

// This test case is like maze2 except that there are too few alike pixels for the
// back projection to consider the superpixels alike enough to merge. 5 / 9
// values for superpixel (1+1) have a zero histogram likeness.

+ (void) getMaze3:(Mat&)pixelsImage
        tagsImage:(Mat&)tagsImage
{
  int width = 10;
  int height = 5;
  
  string pixels =
  "0000000212"
  "0000000212"
  "2222220212"
  "0020020112"
  "0020022222";
  
  string tags =
  "0000001615"
  "0000001615"
  "5555551615"
  "2253351115"
  "2253355555";
  
  pixelsImage = [self parseImageGrayscale:pixels width:width height:height scale:255.0f/2];
  
  tagsImage = [self parseImageGrayscale:tags width:width height:height scale:1.0f];
  
  return;
}

// This test case features a block that can be easily merged and other blocks that cannot
// be merged because then extend right up to a hard edge. The identical regions tagged
// as 0 and 1 can be merged by an identical merge or by a 95% the same merge. The region
// tagged as 3 should not merge with 2 or 5. The region marked as 7 should not merge with 5.
// In this case contourRelaxedSuperpixels would generate roughly this segmentation with
// a 4x4 region size but not 5x5.

+ (void) getMaze4:(Mat&)pixelsImage
        tagsImage:(Mat&)tagsImage
{
  int width = 10;
  int height = 5;
  
  string pixels =
  "0000550000"
  "0000550000"
  "9999990000"
  "0000990000"
  "0000999999";
  
  string tags =
  "0011223333"
  "0011223333"
  "5555553333"
  "7777553333"
  "7777555555";
  
  pixelsImage = [self parseImageGrayscale:pixels width:width height:height scale:255.0f/9.0f];
  
  tagsImage = [self parseImageGrayscale:tags width:width height:height scale:1.0f];
  
  return;
}

// This image contains 5x5 blocks where the superpixel (1+1) contain a strong
// edge. This strong edge makes up 20 percent of the pixels in (1+1) so the top
// 95% test will not merge these two superpixels. This is the worse case since
// the hard edge was not detected and place along the edge of a superpixel.

+ (void) getMaze5:(Mat&)pixelsImage
        tagsImage:(Mat&)tagsImage
{
  int width = 10;
  int height = 5;
  
  string pixels =
  "0000000900"
  "0000000900"
  "0000000900"
  "0000000900"
  "0000000900";
  
  string tags =
  "0000011111"
  "0000011111"
  "0000011111"
  "0000011111"
  "0000011111";
  
  pixelsImage = [self parseImageGrayscale:pixels width:width height:height scale:255.0f/9.0f];
  
  tagsImage = [self parseImageGrayscale:tags width:width height:height scale:1.0f];
  
  return;
}

// This image contains four 5x5 blocks where one superpixel has a strong edge
// and the other has a very weak edge. The BFS should merge the identical
// superpixels first and then the larger identical superpixel is compared
// to the remaining two superpixels. Since there is no other information
// available, the edge strength of the two edges must be compared and the
// result should be that the weak edge gets merged while the strong edge
// does not.

+ (void) getMaze6:(Mat&)pixelsImage
        tagsImage:(Mat&)tagsImage
{
  int width = 10;
  int height = 10;
  
  string pixels =
  "0000090000"
  "0000090000"
  "0000090000"
  "0000090000"
  "0000090000"
  "0000010000"
  "0000010000"
  "0000010000"
  "0000010000"
  "0000010000"
  "0000010000";
  
  string tags =
  "0000011111"
  "0000011111"
  "0000011111"
  "0000011111"
  "0000011111"
  "2222233333"
  "2222233333"
  "2222233333"
  "2222233333"
  "2222233333";
  
  pixelsImage = [self parseImageGrayscale:pixels width:width height:height scale:255.0f/9.0f];
  
  tagsImage = [self parseImageGrayscale:tags width:width height:height scale:1.0f];
  
  return;
}

// This image contains 4x4 blocks in a maze pattern. The blocks with zero
// edge weights should merge while other blocks should be detected as
// hard edges and they should not merge. Note that because the

+ (void) getMaze7:(Mat&)pixelsImage
        tagsImage:(Mat&)tagsImage
{
  int width = 8;
  int height = 4;
  
  string pixels =
  "00990000"
  "00999999"
  "00000000"
  "00000000";
  
  string tags =
  "00112222"
  "00112222"
  "33333333"
  "33333333";
  
  pixelsImage = [self parseImageGrayscale:pixels width:width height:height scale:255.0f/9.0f];
  
  tagsImage = [self parseImageGrayscale:tags width:width height:height scale:1.0f];
  
  return;
}

// In this case, the edge weight between (2+1) and (1+1) is near the unmerged
// edge weight between (2+1) and (3+1) so a merge is not done even though the
// histogram values are alike enough to consider a merge.

+ (void) getMaze8:(Mat&)pixelsImage
        tagsImage:(Mat&)tagsImage
{
  int width = 8;
  int height = 4;
  
  string pixels =
  "00991999"
  "00991999"
  "00000000"
  "00000000";
  
  string tags =
  "00112222"
  "00112222"
  "33333333"
  "33333333";
  
  pixelsImage = [self parseImageGrayscale:pixels width:width height:height scale:255.0f/9.0f];
  
  tagsImage = [self parseImageGrayscale:tags width:width height:height scale:1.0f];
  
  return;
}

// This test case features a hard edge from the
// largest superpixel to the next two so as
// to prevent a merge of (0+1) and (1+1) or (2+1).
// Then (1+1) and (2+1) are merged since the
// weak edge between them is much less than the
// strong edge. Finally the non-zero but weak
// edges between the small superpixels are process
// and merged into (1+1).

+ (void) getMaze9:(Mat&)pixelsImage
        tagsImage:(Mat&)tagsImage
{
  int width  = 8;
  int height = 6;
  
  string pixels =
  "12121212"
  "12121212"
  "11001100"
  "11001100"
  "99999999"
  "99999999";
  
  string tags =
  "33335555"
  "44446666"
  "11112222"
  "11112222"
  "00000000"
  "00000000";
  
  pixelsImage = [self parseImageGrayscale:pixels width:width height:height scale:255.0f/9.0f];
  
  tagsImage = [self parseImageGrayscale:tags width:width height:height scale:1.0f];
  
  return;
}

// Multiple merges on image made up of 4x4 blocks. In this example there is
// no specific size advantage to any one block so initial iteration starts
// at the upper left corner and the BFS expands to the right and down.
// The range 0 -> 7 is over the entire grayscale range of 0 -> 255.
// Blocks that contains pixels 3,4,5,6,7 contain no overlapping pixels so
// they will not be merged. Blocks with the values 0,1,2 contain some
// overlap and so the merge logic finds blocks and has to determine the
// merge order based on percentages matched. Block (0+1) is merged with
// (1+1) first since they are exactly identical. Block (2+1) is then
// merged with 50% likeness. The tricky aspect of this second step is
// that (6+1) is scanned but it has a 0% likeness after one merge so
// it is not merged at that point. On the third step of the merge process
// the neighbors (3+1), (6+1), and (7+1) are considered. The superpixels
// (6+1) and (7+1) both have 100% matches so they should be merged in
// one iteration of the merge by percentage logic. At this point, a
// tricky case is found because now (B+1) is a 100% merge but (3+1)
// is only a 50% merge. The 100% merge should be done first since this
// is a BFS process followed by the 50% merge.

+ (void) getMaze10:(Mat&)pixelsImage
        tagsImage:(Mat&)tagsImage
{
  int width  = 8;
  int height = 6;
  
  string pixels =
  "00001100"
  "00000022"
  "33443300"
  "33443300"
  "55667700"
  "55667700";
  
  string tags =
  "00112233"
  "00112233"
  "44556677"
  "44556677"
  "8899AABB"
  "8899AABB";
  
  pixelsImage = [self parseImageGrayscale:pixels width:width height:height scale:255.0f/7.0f];
  
  tagsImage = [self parseImageGrayscale:tags width:width height:height scale:1.0f];
  
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

- (void)testSimple4by4BW1 {
  const bool debugDumpImage = true;

  Mat maze1Img;
  Mat maze1TagsImg;
  
  [self.class getSimple4by4BW1:maze1Img tagsImage:maze1TagsImg];
  
  //cout << maze1Img << endl;
  //cout << maze1TagsImg << endl;
  
  MergeSuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(maze1TagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  if (debugDumpImage) {
    // Needed if dumping is enabled in SuperpixelImage methods
    generateStaticColortable(maze1Img, spImage);
  }
  
  // There should be 4 superpixels of size 2x2
  
  vector<int32_t> superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 4, @"num sumperpixels");
  
  return;
}

- (void)testBFSMaze1NoMergeImage {
  const bool debugDumpImage = false;
  
  Mat maze1Img;
  Mat maze1TagsImg;
  
  [self.class getMaze1:maze1Img tagsImage:maze1TagsImg];
  
//  cout << maze1Img << endl;
//  cout << maze1TagsImg << endl;
  
  MergeSuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(maze1TagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  if (debugDumpImage) {
    // Needed if dumping is enabled in SuperpixelImage methods
    generateStaticColortable(maze1Img, spImage);
  }
  
  vector<int32_t> superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 5, @"num sumperpixels");
  
  XCTAssert(superpixels[0] == 0+1, @"tag");
  XCTAssert(superpixels[1] == 1+1, @"tag");
  XCTAssert(superpixels[2] == 2+1, @"tag");
  
  XCTAssert(superpixels[3] == 5+1, @"tag");
  XCTAssert(superpixels[4] == 6+1, @"tag");
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 4, @"num edges");

  {
    SuperpixelEdge vecEdge = edges[0];
    SuperpixelEdge edge = SuperpixelEdge(superpixels[0], superpixels[3]);
    XCTAssert((vecEdge.A == edge.A && vecEdge.B == edge.B), @"edge");
  }

  {
    SuperpixelEdge vecEdge = edges[1];
    SuperpixelEdge edge = SuperpixelEdge(superpixels[0], superpixels[4]);
    XCTAssert((vecEdge.A == edge.A && vecEdge.B == edge.B), @"edge");
  }

  {
    SuperpixelEdge vecEdge = edges[2];
    SuperpixelEdge edge = SuperpixelEdge(superpixels[1], superpixels[3]);
    XCTAssert((vecEdge.A == edge.A && vecEdge.B == edge.B), @"edge");
  }
  
  {
    SuperpixelEdge vecEdge = edges[3];
    SuperpixelEdge edge = SuperpixelEdge(superpixels[2], superpixels[3]);
    XCTAssert((vecEdge.A == edge.A && vecEdge.B == edge.B), @"edge");
  }
  
  // Largest superpixel is superpixels[0] so merge starts with this
  // superpixel and finds the most easily merged neighbor first.

  int mergeStep = 0;
  
  mergeStep = spImage.mergeBredthFirstRecursive(maze1Img, 0, mergeStep, NULL, 16);
  
  if (debugDumpImage) {
    Mat resultImg;
    writeTagsWithGraytable(spImage, maze1Img, resultImg);

    char *fname = (char*)"tags_after_merge.png";
    cout << "wrote " << fname << " in dir " << getwd(NULL) << endl;
    imwrite(fname, resultImg);
  }

  superpixels = spImage.getSuperpixelsVec();
  
//  cout << "num superpixels after merge: " << superpixels.size() << endl;
  
  // None of the superpixels could be merged since the histogram
  // back projection logic found no alike pixels between the
  // smooth areas and the edges.
  
  XCTAssert(superpixels.size() == 5, @"num sumperpixels");

  XCTAssert(mergeStep == 0, @"merges");
  
  return;
}

- (void)testBFSMaze2OneMergeImage {
  const bool debugDumpImage = false;
  
  Mat mazeImg;
  Mat mazeTagsImg;
  
  [self.class getMaze2:mazeImg tagsImage:mazeTagsImg];
  
//  cout << mazeImg << endl;
//  cout << mazeTagsImg << endl;
  
  MergeSuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(mazeTagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  if (debugDumpImage) {
    // Needed if dumping is enabled in SuperpixelImage methods
    generateStaticColortable(mazeImg, spImage);
  }
  
  vector<int32_t> superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 6, @"num sumperpixels");
  
  XCTAssert(superpixels[0] == 0+1, @"tag");
  XCTAssert(superpixels[1] == 1+1, @"tag");
  XCTAssert(superpixels[2] == 2+1, @"tag");
  XCTAssert(superpixels[3] == 3+1, @"tag");
  
  XCTAssert(superpixels[4] == 5+1, @"tag");
  XCTAssert(superpixels[5] == 6+1, @"tag");
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 6, @"num edges");
  
  {
    SuperpixelEdge vecEdge = edges[0];
    SuperpixelEdge edge = SuperpixelEdge(superpixels[0], superpixels[1]);
    XCTAssert((vecEdge.A == edge.A && vecEdge.B == edge.B), @"edge");
  }
  
  {
    SuperpixelEdge vecEdge = edges[1];
    SuperpixelEdge edge = SuperpixelEdge(superpixels[0], superpixels[4]);
    XCTAssert((vecEdge.A == edge.A && vecEdge.B == edge.B), @"edge");
  }
  
  {
    SuperpixelEdge vecEdge = edges[2];
    SuperpixelEdge edge = SuperpixelEdge(superpixels[1], superpixels[4]);
    XCTAssert((vecEdge.A == edge.A && vecEdge.B == edge.B), @"edge");
  }
  
  {
    SuperpixelEdge vecEdge = edges[3];
    SuperpixelEdge edge = SuperpixelEdge(superpixels[1], superpixels[5]);
    XCTAssert((vecEdge.A == edge.A && vecEdge.B == edge.B), @"edge");
  }

  {
    SuperpixelEdge vecEdge = edges[4];
    SuperpixelEdge edge = SuperpixelEdge(superpixels[2], superpixels[4]);
    XCTAssert((vecEdge.A == edge.A && vecEdge.B == edge.B), @"edge");
  }

  {
    SuperpixelEdge vecEdge = edges[5];
    SuperpixelEdge edge = SuperpixelEdge(superpixels[3], superpixels[4]);
    XCTAssert((vecEdge.A == edge.A && vecEdge.B == edge.B), @"edge");
  }
  
  // Largest superpixel is superpixels[0] so merge starts with this
  // superpixel and finds the most easily merged neighbor first.
  
  int mergeStep = 0;
  
  mergeStep = spImage.mergeBredthFirstRecursive(mazeImg, 0, mergeStep, NULL, 16);
  
  if (debugDumpImage) {
    Mat resultImg;
    writeTagsWithGraytable(spImage, mazeImg, resultImg);
    
    char *fname = (char*)"tags_after_merge.png";
    cout << "wrote " << fname << " in dir " << getwd(NULL) << endl;
    imwrite(fname, resultImg);
  }
  
  superpixels = spImage.getSuperpixelsVec();
  
  //cout << "num superpixels after merge: " << superpixels.size() << endl;
  
  // One superpixel should have been merged
  
  XCTAssert(superpixels.size() == 5, @"num sumperpixels");
  
  XCTAssert(mergeStep == 1, @"merges");
  
  return;
}

// Fail to merge die to histogram 50% non-zero test

- (void)testBFSMaze3NoMergeImage {
  const bool debugDumpImage = false;
  
  Mat mazeImg;
  Mat mazeTagsImg;
  
  [self.class getMaze3:mazeImg tagsImage:mazeTagsImg];
  
//  cout << mazeImg << endl;
//  cout << mazeTagsImg << endl;
  
  MergeSuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(mazeTagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  if (debugDumpImage) {
    // Needed if dumping is enabled in SuperpixelImage methods
    generateStaticColortable(mazeImg, spImage);
  }
  
  vector<int32_t> superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 6, @"num sumperpixels");
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 6, @"num edges");
  
  // Largest superpixel is superpixels[0] so merge starts with this
  // superpixel and finds the most easily merged neighbor first.
  
  int mergeStep = 0;
  
  mergeStep = spImage.mergeBredthFirstRecursive(mazeImg, 0, mergeStep, NULL, 16);
  
  if (debugDumpImage) {
    Mat resultImg;
    writeTagsWithGraytable(spImage, mazeImg, resultImg);
    
    char *fname = (char*)"tags_after_merge.png";
    cout << "wrote " << fname << " in dir " << getwd(NULL) << endl;
    imwrite(fname, resultImg);
  }
  
  superpixels = spImage.getSuperpixelsVec();
  
  //cout << "num superpixels after merge: " << superpixels.size() << endl;
  
  // No merge should have been done
  
  XCTAssert(superpixels.size() == 6, @"num sumperpixels");
  
  XCTAssert(mergeStep == 0, @"merges");
  
  return;
}

// Two of the tagged groups will be merged as identical.

- (void)testBFSMaze4OneMergeIdentical {
  const bool debugDumpImage = false;
  
  Mat mazeImg;
  Mat mazeTagsImg;
  
  [self.class getMaze4:mazeImg tagsImage:mazeTagsImg];

  if (debugDumpImage) {
    cout << mazeImg << endl;
    cout << mazeTagsImg << endl;
    
    imwrite("maze4_pixels.png", mazeImg);
    imwrite("maze4_tags.png", mazeTagsImg);
  }
  
  SuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(mazeTagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  if (debugDumpImage) {
    // Needed if dumping is enabled in SuperpixelImage methods
    generateStaticColortable(mazeImg, spImage);
  }
  
  vector<int32_t> superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 6, @"num sumperpixels");
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 8, @"num edges");
  
  // An identical pixels merge will find one successful merge.
  
  spImage.mergeIdenticalSuperpixels(mazeImg);
  
  if (debugDumpImage) {
    Mat resultImg;
    writeTagsWithGraytable(spImage, mazeImg, resultImg);
    
    char *fname = (char*)"tags_after_merge.png";
    cout << "wrote " << fname << " in dir " << getwd(NULL) << endl;
    imwrite(fname, resultImg);
  }
 
  superpixels = spImage.getSuperpixelsVec();
  
  //cout << "num superpixels after merge: " << superpixels.size() << endl;
  
  // No merge should have been done
  
  XCTAssert(superpixels.size() == 5, @"num sumperpixels");
  
  edges = spImage.getEdges();
  
  XCTAssert(edges.size() == 6, @"num sumperpixels");
  
  return;
}

// This 95% the same test will merge two identical superpixels.

- (void)testBFSMaze4OneMerge95Percent {
  const bool debugDumpImage = false;
  
  Mat mazeImg;
  Mat mazeTagsImg;
  
  [self.class getMaze4:mazeImg tagsImage:mazeTagsImg];
  
  if (debugDumpImage) {
    cout << mazeImg << endl;
    cout << mazeTagsImg << endl;
    
    imwrite("maze4_pixels.png", mazeImg);
    imwrite("maze4_tags.png", mazeTagsImg);
  }
  
  MergeSuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(mazeTagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  if (debugDumpImage) {
    // Needed if dumping is enabled in SuperpixelImage methods
    generateStaticColortable(mazeImg, spImage);
  }
  
  vector<int32_t> superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 6, @"num sumperpixels");
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 8, @"num edges");

  // Merge 95% alike superpixels based on a backprojection test. Note that
  // this logic starts from the largest superpixel (3+1) and then it
  // continues to merge until (0+1) and (1+1) are successfully merged.
  
  int mergeStep = 0;
  
  mergeStep = spImage.mergeBackprojectSuperpixels(mazeImg, 0, mergeStep, BACKPROJECT_HIGH_FIVE);
  
  if (debugDumpImage) {
    Mat resultImg;
    writeTagsWithGraytable(spImage, mazeImg, resultImg);
    
    char *fname = (char*)"tags_after_merge.png";
    cout << "wrote " << fname << " in dir " << getwd(NULL) << endl;
    imwrite(fname, resultImg);
  }
  
  superpixels = spImage.getSuperpixelsVec();
  
  //cout << "num superpixels after merge: " << superpixels.size() << endl;
  
  // No merge should have been done
  
  XCTAssert(superpixels.size() == 5, @"num sumperpixels");
  
  edges = spImage.getEdges();
  
  XCTAssert(edges.size() == 6, @"num sumperpixels");

  XCTAssert(mergeStep == 1, @"mergeStep");
  
  return;
}

// This example uses the recursive BFS merge logic to scan backprojection
// values and edge weights and does the same merge as the other steps.

- (void)testBFSMaze4OneMergeBFSRecursive {
  const bool debugDumpImage = false;
  
  Mat mazeImg;
  Mat mazeTagsImg;
  
  [self.class getMaze4:mazeImg tagsImage:mazeTagsImg];
  
  if (debugDumpImage) {
    cout << mazeImg << endl;
    cout << mazeTagsImg << endl;
    
    imwrite("maze4_pixels.png", mazeImg);
    imwrite("maze4_tags.png", mazeTagsImg);
  }
  
  MergeSuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(mazeTagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  if (debugDumpImage) {
    // Needed if dumping is enabled in SuperpixelImage methods
    generateStaticColortable(mazeImg, spImage);
  }
  
  vector<int32_t> superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 6, @"num sumperpixels");
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 8, @"num edges");
  
  // Merge 95% alike superpixels based on a backprojection test. Note that
  // this logic starts from the largest superpixel (3+1) and then it
  // continues to merge until (0+1) and (1+1) are successfully merged.
  
  int mergeStep = 0;
  
  vector<int32_t> veryLargeSuperpixels;
  
  mergeStep = spImage.mergeBredthFirstRecursive(mazeImg, 0, mergeStep, &veryLargeSuperpixels, 16);
  
  if (debugDumpImage) {
    Mat resultImg;
    writeTagsWithGraytable(spImage, mazeImg, resultImg);
    
    char *fname = (char*)"tags_after_merge.png";
    cout << "wrote " << fname << " in dir " << getwd(NULL) << endl;
    imwrite(fname, resultImg);
  }
  
  superpixels = spImage.getSuperpixelsVec();
  
  //cout << "num superpixels after merge: " << superpixels.size() << endl;
  
  // No merge should have been done
  
  XCTAssert(superpixels.size() == 5, @"num sumperpixels");
  
  edges = spImage.getEdges();
  
  XCTAssert(edges.size() == 6, @"num sumperpixels");
  
  XCTAssert(mergeStep == 1, @"mergeStep");
  
  return;
}

// This example checks the worse case of an edge in a 5x5 matrix where there is
// a strong edge in a superpixel. A 95% test will not merge this superpixel
// since the neighbor non-zero percentage is 80%.

- (void)testBFSMaze5NoMerge95PerHardEdge {
  const bool debugDumpImage = false;
  
  Mat mazeImg;
  Mat mazeTagsImg;
  
  [self.class getMaze5:mazeImg tagsImage:mazeTagsImg];
  
  if (debugDumpImage) {
    cout << mazeImg << endl;
    cout << mazeTagsImg << endl;
    
    imwrite("maze5_pixels.png", mazeImg);
    imwrite("maze5_tags.png", mazeTagsImg);
  }
  
  MergeSuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(mazeTagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  if (debugDumpImage) {
    // Needed if dumping is enabled in SuperpixelImage methods
    generateStaticColortable(mazeImg, spImage);
  }
  
  vector<int32_t> superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 2, @"num sumperpixels");
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 1, @"num edges");
  
  // Merge 95% alike superpixels based on a backprojection test. Note that
  // this logic starts from the largest superpixel (3+1) and then it
  // continues to merge until (0+1) and (1+1) are successfully merged.
  
  int mergeStep = 0;
  
  mergeStep = spImage.mergeBackprojectSuperpixels(mazeImg, 0, mergeStep, BACKPROJECT_HIGH_FIVE);
  
  if (debugDumpImage) {
    Mat resultImg;
    writeTagsWithGraytable(spImage, mazeImg, resultImg);
    
    char *fname = (char*)"tags_after_merge.png";
    cout << "wrote " << fname << " in dir " << getwd(NULL) << endl;
    imwrite(fname, resultImg);
  }
  
  superpixels = spImage.getSuperpixelsVec();
  
  //cout << "num superpixels after merge: " << superpixels.size() << endl;
  
  // No merge should have been done
  
  XCTAssert(superpixels.size() == 2, @"num sumperpixels");
  
  edges = spImage.getEdges();
  
  XCTAssert(edges.size() == 1, @"num sumperpixels");
  
  XCTAssert(mergeStep == 0, @"mergeStep");
  
  return;
}

// This example runs the 5x5 worst case where a 5x5 block contains
// a hard edge. While the 95% test will not merge these superpixels
// the BFS will because the edge weight is zero with an 80% hist
// match. The histogram logic depends on the block segmentation
// to correctly detect the hard edge and not present a case like this
// since it will improperly merge the hard edge.

- (void)testBFSMaze5BFSShouldNotMergeHardEdgeButDoes {
  const bool debugDumpImage = false;
  
  Mat mazeImg;
  Mat mazeTagsImg;
  
  [self.class getMaze5:mazeImg tagsImage:mazeTagsImg];
  
  if (debugDumpImage) {
    cout << mazeImg << endl;
    cout << mazeTagsImg << endl;
    
    imwrite("maze5_pixels.png", mazeImg);
    imwrite("maze5_tags.png", mazeTagsImg);
  }
  
  MergeSuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(mazeTagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  if (debugDumpImage) {
    // Needed if dumping is enabled in SuperpixelImage methods
    generateStaticColortable(mazeImg, spImage);
  }
  
  vector<int32_t> superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 2, @"num sumperpixels");
  
  XCTAssert(superpixels[0] == (0+1), @"sumperpixels");
  XCTAssert(superpixels[1] == (1+1), @"sumperpixels");
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 1, @"num edges");
  
  // Force cache insertion
  
  spImage.checkNeighborEdgeWeights(mazeImg, superpixels[0], NULL, spImage.edgeTable.edgeStrengthMap, 0);
  
  SuperpixelEdge cachedEdge(superpixels[0], superpixels[1]);
  
  XCTAssert(spImage.edgeTable.edgeStrengthMap.count(cachedEdge) >= 0, @"cached edge weight");
  
  // Merge 95% alike superpixels based on a backprojection test. Note that
  // this logic starts from the largest superpixel (3+1) and then it
  // continues to merge until (0+1) and (1+1) are successfully merged.
  
  int mergeStep = 0;
  
  mergeStep = spImage.mergeBredthFirstRecursive(mazeImg, 0, mergeStep, NULL, 16);
  
  if (debugDumpImage) {
    Mat resultImg;
    writeTagsWithGraytable(spImage, mazeImg, resultImg);
    
    char *fname = (char*)"tags_after_merge.png";
    cout << "wrote " << fname << " in dir " << getwd(NULL) << endl;
    imwrite(fname, resultImg);
  }
  
  superpixels = spImage.getSuperpixelsVec();
  
  //cout << "num superpixels after merge: " << superpixels.size() << endl;
  
  XCTAssert(superpixels.size() == 1, @"num sumperpixels");
  
  edges = spImage.getEdges();
  
  XCTAssert(edges.size() == 0, @"num sumperpixels");
  
  XCTAssert(mergeStep == 1, @"mergeStep");
  
  // Edge weight should not exist in cache
  
  XCTAssert(spImage.edgeTable.edgeStrengthMap.count(cachedEdge) == 0, @"cached edge weight");
  
  return;
}

// Merge one very weak edge but do not merge the strong edge

- (void)testBFSMaze6OneWeakEdgeMerge {
  const bool debugDumpImage = false;
  
  Mat mazeImg;
  Mat mazeTagsImg;
  
  [self.class getMaze6:mazeImg tagsImage:mazeTagsImg];
  
  if (debugDumpImage) {
    cout << mazeImg << endl;
    cout << mazeTagsImg << endl;
    
    imwrite("maze6_pixels.png", mazeImg);
    imwrite("maze6_tags.png", mazeTagsImg);
  }
  
  MergeSuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(mazeTagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  if (debugDumpImage) {
    // Needed if dumping is enabled in SuperpixelImage methods
    generateStaticColortable(mazeImg, spImage);
  }
  
  vector<int32_t> superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 4, @"num sumperpixels");
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 6, @"num edges");
  
  // tags (0+1) and (2+1) should be merged first as a 100% match.
  // After the first merge, the other two superpixels are the same
  // size and have the same back projection percentage so edge
  // strength comparison is needed in order to determine which
  // edges to merge.
  
  int mergeStep = 0;
  
  mergeStep = spImage.mergeBredthFirstRecursive(mazeImg, 0, mergeStep, NULL, 16);
  
  if (debugDumpImage) {
    Mat resultImg;
    writeTagsWithGraytable(spImage, mazeImg, resultImg);
    
    char *fname = (char*)"tags_after_merge.png";
    cout << "wrote " << fname << " in dir " << getwd(NULL) << endl;
    imwrite(fname, resultImg);
  }
  
  superpixels = spImage.getSuperpixelsVec();
  
  //cout << "num superpixels after merge: " << superpixels.size() << endl;
  
  XCTAssert(superpixels.size() == 2, @"num sumperpixels");
  
  edges = spImage.getEdges();
  
  XCTAssert(edges.size() == 1, @"num sumperpixels");
  
  XCTAssert(mergeStep == 2, @"mergeStep");
  
  return;
}

// This maze contains multiple edges that can be merged
// and the edge is exactly between the unmerged and
// can merge valeus so it is merged.

- (void)testBFSMaze7MergeMultipleEdges {
  const bool debugDumpImage = false;
  
  Mat mazeImg;
  Mat mazeTagsImg;
  
  [self.class getMaze7:mazeImg tagsImage:mazeTagsImg];
  
  if (debugDumpImage) {
    cout << mazeImg << endl;
    cout << mazeTagsImg << endl;
    
    imwrite("maze7_pixels.png", mazeImg);
    imwrite("maze7_tags.png", mazeTagsImg);
  }
  
  MergeSuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(mazeTagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  if (debugDumpImage) {
    // Needed if dumping is enabled in SuperpixelImage methods
    generateStaticColortable(mazeImg, spImage);
  }
  
  vector<int32_t> superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 4, @"num sumperpixels");
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 5, @"num edges");
  
  // tags (0+1) and (3+1) merge since the match is 100% and
  // the edge weight is zero. Then 2 is examined and the match
  // amount is 100% and the edge weight is exactly between
  // the two extremes and as a result the edges are merged.
  // This case is the exact edge that will successfully merge
  // if the edge weight were a little higher then only one
  // merge would be done.
  
  int mergeStep = 0;
  
  mergeStep = spImage.mergeBredthFirstRecursive(mazeImg, 0, mergeStep, NULL, 16);
  
  if (debugDumpImage) {
    Mat resultImg;
    writeTagsWithGraytable(spImage, mazeImg, resultImg);
    
    char *fname = (char*)"tags_after_merge.png";
    cout << "wrote " << fname << " in dir " << getwd(NULL) << endl;
    imwrite(fname, resultImg);
  }
  
  superpixels = spImage.getSuperpixelsVec();
  
  //cout << "num superpixels after merge: " << superpixels.size() << endl;
  
  XCTAssert(superpixels.size() == 2, @"num sumperpixels");
  
  edges = spImage.getEdges();
  
  XCTAssert(edges.size() == 1, @"num sumperpixels");
  
  XCTAssert(mergeStep == 2, @"mergeStep");
  
  return;
}

- (void) DISABLED_testBFSMaze8NoMergeSinceEdgeIsTooBig {
  const bool debugDumpImage = false;
  
  Mat mazeImg;
  Mat mazeTagsImg;
  
  [self.class getMaze8:mazeImg tagsImage:mazeTagsImg];
  
  if (debugDumpImage) {
    cout << mazeImg << endl;
    cout << mazeTagsImg << endl;
    
    imwrite("maze8_pixels.png", mazeImg);
    imwrite("maze8_tags.png", mazeTagsImg);
  }
  
  MergeSuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(mazeTagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  if (debugDumpImage) {
    // Needed if dumping is enabled in SuperpixelImage methods
    generateStaticColortable(mazeImg, spImage);
  }
  
  vector<int32_t> superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 4, @"num sumperpixels");
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 5, @"num edges");
  
  // tags (0+1) and (3+1) merge since the match is 100% and
  // the edge weight is zero. Then 2 is examined and a merge
  // is not done because the edge is a strong edge as determined
  // by comparing to the other edge for the superpixel.
  
  int mergeStep = 0;
  
  mergeStep = spImage.mergeBredthFirstRecursive(mazeImg, 0, mergeStep, NULL, 16);
  
  if (debugDumpImage) {
    Mat resultImg;
    writeTagsWithGraytable(spImage, mazeImg, resultImg);
    
    char *fname = (char*)"tags_after_merge.png";
    cout << "wrote " << fname << " in dir " << getwd(NULL) << endl;
    imwrite(fname, resultImg);
  }
  
  superpixels = spImage.getSuperpixelsVec();
  
  //cout << "num superpixels after merge: " << superpixels.size() << endl;
  
  XCTAssert(superpixels.size() == 3, @"num sumperpixels");
  
  edges = spImage.getEdges();
  
  XCTAssert(edges.size() == 3, @"num sumperpixels");
  
  XCTAssert(mergeStep == 1, @"mergeStep");
  
  // The unmerged edge weights must have been filled in by the BFS merge.
  // Note that the merge operation will sort by superpixel size.

  XCTAssert(superpixels[0] == (3+1), @"num sumperpixels");
  XCTAssert(superpixels[1] == (2+1), @"num sumperpixels");
  XCTAssert(superpixels[2] == (1+1), @"num sumperpixels");
  
  {
    // (1+1) was not merged so it contains 2 unmerged edge weights since
    // it was processed after (3+1) which did do a merge.
    
    Superpixel *spPtr = spImage.getSuperpixelPtr(superpixels[2]);
    XCTAssert(spPtr->mergedEdgeWeights.size() == 0, @"merged weights");
    XCTAssert(spPtr->unmergedEdgeWeights.size() == 2, @"unmerged weights");
  }
  
  {
    // (2+1) was not merged so it contains 2 unmerged edge weights
    
    Superpixel *spPtr = spImage.getSuperpixelPtr(superpixels[1]);
    XCTAssert(spPtr->mergedEdgeWeights.size() == 0, @"merged weights");
    XCTAssert(spPtr->unmergedEdgeWeights.size() == 2, @"unmerged weights");
  }
  
  {
    // (3+1) was merged with (0+1) so it contains 2 unmerged edge weights
    // and 1 merged weight. But, the merged edge weight was zero and
    // zero edge weights are not recorded.
    
    Superpixel *spPtr = spImage.getSuperpixelPtr(superpixels[0]);
    XCTAssert(spPtr->mergedEdgeWeights.size() == 1, @"merged weights");
    XCTAssert(spPtr->unmergedEdgeWeights.size() == 2, @"unmerged weights");
  }
  
  return;
}

// Do multiple merges and make sure edge weights are
// being copied as one superpixel is merged into another.

- (void)DISABLED_testBFSMaze9NoMergeHardThenMergeWeights {
  const bool debugDumpImage = false;
  
  Mat mazeImg;
  Mat mazeTagsImg;
  
  [self.class getMaze9:mazeImg tagsImage:mazeTagsImg];
  
  if (debugDumpImage) {
    cout << mazeImg << endl;
    cout << mazeTagsImg << endl;
    
    imwrite("maze9_pixels.png", mazeImg);
    imwrite("maze9_tags.png", mazeTagsImg);
  }
  
  MergeSuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(mazeTagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  if (debugDumpImage) {
    // Needed if dumping is enabled in SuperpixelImage methods
    generateStaticColortable(mazeImg, spImage);
  }
  
  vector<int32_t> superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 7, @"num sumperpixels");
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 13, @"num edges");
  
  // tags (0+1) and (3+1) merge since the match is 100% and
  // the edge weight is zero. Then 2 is examined and a merge
  // is not done because the edge is a strong edge as determined
  // by comparing to the other edge for the superpixel.
  
  int mergeStep = 0;
  
  mergeStep = spImage.mergeBredthFirstRecursive(mazeImg, 0, mergeStep, NULL, 16);
  
  if (debugDumpImage) {
    Mat resultImg;
    writeTagsWithGraytable(spImage, mazeImg, resultImg);
    
    char *fname = (char*)"tags_after_merge.png";
    cout << "wrote " << fname << " in dir " << getwd(NULL) << endl;
    imwrite(fname, resultImg);
  }
  
  superpixels = spImage.getSuperpixelsVec();
  
  //cout << "num superpixels after merge: " << superpixels.size() << endl;
  
  XCTAssert(superpixels.size() == 2, @"num sumperpixels");
  
  edges = spImage.getEdges();
  
  XCTAssert(edges.size() == 1, @"num sumperpixels");
  
  XCTAssert(mergeStep == 5, @"mergeStep");
  
  // The unmerged edge weights must have been filled in by the BFS merge
  
  XCTAssert(superpixels[0] == (0+1), @"num sumperpixels");
  XCTAssert(superpixels[1] == (1+1), @"num sumperpixels");
  
  {
    // (0+1) was not merged so it contains 2 unmerged edge weights
    
    Superpixel *spPtr = spImage.getSuperpixelPtr(superpixels[0]);
    XCTAssert(spPtr->mergedEdgeWeights.size() == 0, @"merged weights");
    XCTAssert(spPtr->unmergedEdgeWeights.size() == 2, @"unmerged weights");
  }
  
  {
    // (1+1) merged in 5 other superpixels. There are 2 unmerged edge
    // values that are exactly the same since (1+1) had one and the
    // other came from (2+1).
    
    Superpixel *spPtr = spImage.getSuperpixelPtr(superpixels[1]);
    XCTAssert(spPtr->mergedEdgeWeights.size() == 5, @"merged weights");
    XCTAssert(spPtr->unmergedEdgeWeights.size() == 2, @"unmerged weights");
  }
  
  return;
}

// This test case focuses on BFS merge logic and the order that multiple
// merges are done in. When processing chunks of percentage values the
// logic should merge until no more values of a specific percentage
// amount are left in a possible expansion.

- (void)testBFSMaze10MergeOrder {
  const bool debugDumpImage = false;
  
  Mat mazeImg;
  Mat mazeTagsImg;
  
  [self.class getMaze10:mazeImg tagsImage:mazeTagsImg];
  
  if (debugDumpImage) {
    cout << mazeImg << endl;
    cout << mazeTagsImg << endl;
    
    imwrite("maze10_pixels.png", mazeImg);
    imwrite("maze10_tags.png", mazeTagsImg);
  }
  
  MergeSuperpixelImage spImage;
  
  bool worked = SuperpixelImage::parse(mazeTagsImg, spImage);
  XCTAssert(worked, @"SuperpixelImage parse");
  
  if (debugDumpImage) {
    // Needed if dumping is enabled in SuperpixelImage methods
    generateStaticColortable(mazeImg, spImage);
  }
  
  vector<int32_t> superpixels = spImage.getSuperpixelsVec();
  XCTAssert(superpixels.size() == 12, @"num sumperpixels");
  
  vector<SuperpixelEdge> edges = spImage.getEdges();
  XCTAssert(edges.size() == 29, @"num edges");
  
  // Should start with tag (0+1) and then merge with (1+1)
  // since there is a zero edge and the hist match is 95%
  
  int mergeStep = 0;
  
  mergeStep = spImage.mergeBredthFirstRecursive(mazeImg, 0, mergeStep, NULL, 16);
  
  if (debugDumpImage) {
    Mat resultImg;
    writeTagsWithGraytable(spImage, mazeImg, resultImg);
    
    char *fname = (char*)"tags_after_merge.png";
    cout << "wrote " << fname << " in dir " << getwd(NULL) << endl;
    imwrite(fname, resultImg);
  }
  
  superpixels = spImage.getSuperpixelsVec();
  
  //cout << "num superpixels after merge: " << superpixels.size() << endl;
  
  XCTAssert(superpixels.size() == 7, @"num sumperpixels");
  
  edges = spImage.getEdges();
  
  XCTAssert(edges.size() == 15, @"num sumperpixels");
  
  XCTAssert(mergeStep == 5, @"mergeStep");
  
  // The unmerged edge weights must have been filled in by the BFS merge
  
  XCTAssert(superpixels[0] == (0+1), @"num sumperpixels");
  XCTAssert(superpixels[1] == (4+1), @"num sumperpixels");
  XCTAssert(superpixels[2] == (5+1), @"num sumperpixels");
  XCTAssert(superpixels[3] == (6+1), @"num sumperpixels");
  XCTAssert(superpixels[4] == (8+1), @"num sumperpixels");
  XCTAssert(superpixels[5] == (9+1), @"num sumperpixels");
  XCTAssert(superpixels[6] == (10+1), @"num sumperpixels");

  // Verify merge order
  
  vector<SuperpixelEdge> &merges = spImage.mergeOrder;

  XCTAssert(merges.size() == 5, @"num merges");
  
  XCTAssert(merges[0] == SuperpixelEdge(0+1, 1+1), @"edge");
  XCTAssert(merges[1] == SuperpixelEdge(0+1, 2+1), @"edge");

  XCTAssert(merges[2] == SuperpixelEdge(0+1, 7+1), @"edge");

  // When (7+1) is merged then the back projection loop must be
  // entered to enable finding (11+1) which was not a neighbor
  // previously but it has a higher percentage than (3+1)
  
  XCTAssert(merges[3] == SuperpixelEdge(0+1, 11+1), @"edge");
  XCTAssert(merges[4] == SuperpixelEdge(0+1, 3+1), @"edge");
  
  return;
}



// When adding unmerged edge weights and merged edge weights, should the logic "push"
// unmerged values up higher when merged values have been found that are near the
// unmerged values? For example, a hard edge with the value 10.0 might be found when
// histograms are not alike. But, then what if another edge weight is found that is
// also 10? Should a successfully merged edge weight remove a previous cannot merge
// weight or would it just add another value to the average?

@end
