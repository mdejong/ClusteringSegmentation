//
//  ClusteringSegmentation.cpp
//  ClusteringSegmentation
//
//  Created by Mo DeJong on 1/17/16.
//  Copyright © 2016 helpurock. All rights reserved.
//

#include "ClusteringSegmentation.hpp"

#include <opencv2/opencv.hpp>

#include "Util.h"
#include "OpenCVUtil.h"
#include "OpenCVIter.hpp"
#include "OpenCVHull.hpp"

#include "Superpixel.h"
#include "SuperpixelEdge.h"
#include "SuperpixelImage.h"

#include "quant_util.h"
#include "DivQuantHeader.h"

#include "MergeSuperpixelImage.h"

#include "srm.h"

#include "peakdetect.h"

#include <stack>

using namespace cv;
using namespace std;

void
captureVeryCloseRegion(SuperpixelImage &spImage,
                       const Mat & inputImg,
                       const Mat & srmTags,
                       int32_t tag,
                       int blockWidth,
                       int blockHeight,
                       int superpixelDim,
                       Mat &outBlockMask,
                       const vector<Coord> &regionCoords,
                       const vector<Coord> &srmRegionCoords,
                       int estNumColors);

void
captureNotCloseRegion(SuperpixelImage &spImage,
                      const Mat & inputImg,
                      const Mat & srmTags,
                      int32_t tag,
                      int blockWidth,
                      int blockHeight,
                      int superpixelDim,
                      Mat &mask,
                      const vector<Coord> &regionCoords,
                      const vector<Coord> &srmRegionCoords,
                      int estNumColors,
                      const Mat &blockBasedQuantMat);

void
captureRegion(SuperpixelImage &spImage,
              const Mat & inputImg,
              const Mat & srmTags,
              int32_t tag,
              int blockWidth,
              int blockHeight,
              int superpixelDim,
              Mat &mask,
              const vector<Coord> &regionCoords,
              const vector<Coord> &srmRegionCoords,
              const Mat &blockBasedQuantMat);

vector<SuperpixelEdge>
getEdgesInRegion(SuperpixelImage &spImage,
                 const Mat & tagsImg,
                 int32_t tag,
                 const vector<Coord> &coords);

void
clockwiseScanForShapeBounds(
                            const Mat & inputImg,
                            const Mat & tagsImg,
                            int32_t tag,
                            const vector<Coord> &regionCoords);

// Data and method for scanning ranges of tags around a shape.
// The total number of divisions (start, end) depends on the
// size of the bbox.

typedef struct {
  int start;
  int end;
  vector<int32_t> tags;
  vector<Coord> coords;
  bool flag;
} TagsAroundShape;

void
clockwiseScanForTagsAroundShape(
                                const Mat & tagsImg,
                                int32_t tag,
                                const vector<Coord> &coords,
                                vector<TagsAroundShape> &tagsAroundVec);

// Given a set of pixels, scan those pixels and determine the peaks
// in the histogram to find likely most common graph peak values.

vector<uint32_t> gatherPeakPixels(const vector<uint32_t> & pixels,
                                  unordered_map<uint32_t, uint32_t> & pixelToNumVotesMap);

// This method will contract or expand a region defined by coordinates by N pixel.
// In the case where the region cannot be expanded or contracted anymore this
// method returns false.

bool
contractOrExpandRegion(const Mat & inputImg,
                       int32_t tag,
                       const vector<Coord> &coords,
                       bool isExpand,
                       int numPixels,
                       vector<Coord> &outCoords);

vector<Coord> genRectangleOutline(int regionWidth, int regionHeight);

// Given an input image and a pixel buffer that is of the same dimensions
// write the buffer of pixels out as an image in a file.

Mat dumpQuantImage(string filename, const Mat &inputImg, uint32_t *pixels) {
  Mat quantOutputMat = inputImg.clone();
  quantOutputMat = (Scalar) 0;
  
  const bool debugOutput = false;
  
  int pi = 0;
  for (int y = 0; y < quantOutputMat.rows; y++) {
    for (int x = 0; x < quantOutputMat.cols; x++) {
      uint32_t pixel = pixels[pi++];
      
      if ((debugOutput)) {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "for (%4d,%4d) pixel is 0x%08X\n", x, y, pixel);
        cout << buffer;
      }
      
      Vec3b vec = PixelToVec3b(pixel);
      
      quantOutputMat.at<Vec3b>(y, x) = vec;
    }
  }
  
  imwrite(filename, quantOutputMat);
  cout << "wrote " << filename << endl;
  return quantOutputMat;
}

// Dump N x 1 image that contains pixels

void dumpQuantTableImage(string filename, const Mat &inputImg, uint32_t *colortable, uint32_t numColortableEntries)
{
  // Write image that contains one color in each row in a N x 1 image
  
  Mat qtableOutputMat = Mat(numColortableEntries, 1, CV_8UC3);
  qtableOutputMat = (Scalar) 0;
  
  vector<uint32_t> clusterCenterPixels;
  
  for ( int i = 0; i < numColortableEntries; i++) {
    uint32_t pixel = colortable[i];
    clusterCenterPixels.push_back(pixel);
  }
  
#if defined(DEBUG)
  if ((1)) {
    fprintf(stdout, "numClusters %5d\n", numColortableEntries);
    
    unordered_map<uint32_t, uint32_t> seen;
    
    for ( int i = 0; i < numColortableEntries; i++ ) {
      uint32_t pixel;
      pixel = colortable[i];
      
      if (seen.count(pixel) > 0) {
        fprintf(stdout, "cmap[%3d] = 0x%08X (DUP of %d)\n", i, pixel, seen[pixel]);
      } else {
        fprintf(stdout, "cmap[%3d] = 0x%08X\n", i, pixel);
        
        // Note that only the first seen index is retained, this means that a repeated
        // pixel value is treated as a dup.
        
        seen[pixel] = i;
      }
    }
    
    fprintf(stdout, "cmap contains %3d unique entries\n", (int)seen.size());
    
    int numQuantUnique = (int)seen.size();
    
    assert(numQuantUnique == numColortableEntries);
  }
#endif // DEBUG
  
  vector<uint32_t> sortedOffsets = generate_cluster_walk_on_center_dist(clusterCenterPixels);
  
  for (int i = 0; i < numColortableEntries; i++) {
    int si = (int) sortedOffsets[i];
    uint32_t pixel = colortable[si];
    Vec3b vec = PixelToVec3b(pixel);
    qtableOutputMat.at<Vec3b>(i, 0) = vec;
  }
  
  imwrite(filename, qtableOutputMat);
  cout << "wrote " << filename << endl;
  return;
}

// Generate a tags Mat from the original input pixels based on SRM algo.

Mat generateSRM(const Mat &inputImg, double Q)
{
  // SRM
  
  const bool debugOutput = false;
  const bool debugDumpImage = true;
  
  int numPixels = inputImg.rows * inputImg.cols;
  
  assert(inputImg.channels() == 3);
  
  const int channels = 3;
  
  uint8_t *in = new uint8_t[numPixels * channels]();
  uint8_t *out = new uint8_t[numPixels * channels]();
  
  int i = 0;
  for(int y = 0; y < inputImg.rows; y++) {
    for(int x = 0; x < inputImg.cols; x++) {
      Vec3b vec = inputImg.at<Vec3b>(y, x);
      
      uint8_t B = vec[0];
      uint8_t G = vec[1];
      uint8_t R = vec[2];
      
      if ((debugOutput)) {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "for IN (%4d,%4d) pixel is 0x00%02X%02X%02X -> offset %d\n", x, y, R, G, B, i);
        cout << buffer;
      }
      
      in[(i*3)+0] = B;
      in[(i*3)+1] = G;
      in[(i*3)+2] = R;
      i += 1;
    }
  }
  
  //double Q = 512.0;
  //double Q = 255.0;
  
  SRM(Q, inputImg.cols, inputImg.rows, channels, in, out, 0);
  
  Mat outImg = inputImg.clone();
  outImg = (Scalar) 0;
  
  bool foundWhitePixel = false;
  uint32_t largestNonWhitePixel = 0x0;
  
  i = 0;
  for(int y = 0; y < outImg.rows; y++) {
    for(int x = 0; x < outImg.cols; x++) {
      uint32_t B = out[(i*3)+0];
      uint32_t G = out[(i*3)+1];
      uint32_t R = out[(i*3)+2];
      
      i += 1;
      
      if ((debugOutput)) {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "for OUT (%4d,%4d) pixel is 0x00%02X%02X%02X -> offset %d\n", x, y, R, G, B, i);
        cout << buffer;
      }
      
      Vec3b vec;
      
      vec[0] = B;
      vec[1] = G;
      vec[2] = R;
      
      outImg.at<Vec3b>(y, x) = vec;
      
      if (B == 0xFF && G == 0xFF && R == 0xFF) {
        foundWhitePixel = true;
      } else {
        uint32_t pixel = (R << 16) | (G << 8) | (B);
        if (pixel > largestNonWhitePixel) {
          largestNonWhitePixel = pixel;
        }
      }
    }
  }
  
  if (foundWhitePixel) {
    // SRM output must not include the special case of color 0xFFFFFFFF since the
    // implicit +1 during paring would overflow the int value. Simply find an unused
    // near white color and use that instead.
    
    uint32_t nonWhitePixel = 0x00FFFFFF;
    
    nonWhitePixel -= 1;
    
    while (1) {
      if (nonWhitePixel != largestNonWhitePixel) {
        break;
      }
    }
    
    // nonWhitePixel now contains an unused pixel value
    
    Vec3b nonWhitePixelVec = PixelToVec3b(nonWhitePixel);
    
    if ((debugOutput)) {
      char buffer[1024];
      snprintf(buffer, sizeof(buffer), "rewrite white pixel 0x%08X as 0x%08X\n", 0x00FFFFFF, nonWhitePixel);
      cout << buffer;
    }
    
    for(int y = 0; y < outImg.rows; y++) {
      for(int x = 0; x < outImg.cols; x++) {
        Vec3b vec = outImg.at<Vec3b>(y, x);
        uint32_t pixel = Vec3BToUID(vec);
        if (pixel == 0x00FFFFFF) {
          vec = nonWhitePixelVec;
          outImg.at<Vec3b>(y, x) = vec;
        }
      }
    }
  }
  
  if (debugDumpImage) {
      std::stringstream fnameStream;
      fnameStream << "srm" << int(Q) << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, outImg);
      cout << "wrote " << fname << endl;
  }
  
  delete [] in;
  delete [] out;
  
  return outImg;
}

// Generate a histogram for each block of 4x4 pixels in the input image.
// This logic maps input pixels to an even quant division of the color cube
// so that comparison based on the pixel frequency is easy on a region
// by region basis.

Mat genHistogramsForBlocks(const Mat &inputImg,
                           unordered_map<Coord, HistogramForBlock> &blockMap,
                           int blockWidth,
                           int blockHeight,
                           int superpixelDim)
{
  const bool debugOutput = false;
  const bool dumpOutputImages = true;
  
  uint32_t width = inputImg.cols;
  uint32_t height = inputImg.rows;
  
  uint32_t numPixels = width * height;
  uint32_t *inPixels = new uint32_t[numPixels];
  uint32_t *outPixels = new uint32_t[numPixels]();
  
  int pi = 0;
  for(int y = 0; y < height; y++) {
    for(int x = 0; x < width; x++) {
      Vec3b vec = inputImg.at<Vec3b>(y, x);
      uint32_t pixel = Vec3BToUID(vec);
      
      if ((debugOutput)) {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "for (%4d,%4d) pixel is 0x%08X\n", x, y, pixel);
        cout << buffer;
      }
      
      inPixels[pi++] = pixel;
    }
  }
  
  vector<uint32_t> quantColors = getSubdividedColors();
  uint32_t numColors = (uint32_t) quantColors.size();
  uint32_t *colortable = new uint32_t[numColors];
  
  {
    int i = 0;
    for ( uint32_t color : quantColors ) {
      colortable[i++] = color;
    }
  }
  
  map_colors_mps(inPixels, numPixels, outPixels, colortable, numColors);
  
  if (dumpOutputImages) {
    Mat quantMat = dumpQuantImage("block_quant_full_output.png", inputImg, outPixels);
  }
  
  // Allocate Mat where a single quant value is selected for each block. Iterate over
  // each block and query the coordinates associated with a specific block.
  
  Mat blockMat = Mat(blockHeight, blockWidth, CV_8UC3);
  blockMat = (Scalar) 0;
  
  pi = 0;
  for(int by = 0; by < blockMat.rows; by++) {
    for(int bx = 0; bx < blockMat.cols; bx++) {
      Coord blockC(bx, by);
      
      if ((debugOutput)) {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "block (%4d,%4d)", bx, by);
        cout << buffer << endl;
      }
      
      int actualX = blockC.x * superpixelDim;
      int actualY = blockC.y * superpixelDim;
      
      Coord min(actualX, actualY);
      Coord max(actualX+superpixelDim-1, actualY+superpixelDim-1);
      
      if ((debugOutput)) {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "block min (%4d,%4d) max (%4d,%4d)", min.x, min.y, max.x, max.y);
        cout << buffer << endl;
      }
      
      vector<uint32_t> pixelsThisBlock;
      
      bool isAllSamePixel = true;
      bool isFirstPixelSet = false;
      uint32_t firstPixel = 0x0;
      
      for (int y = actualY; y <= max.y; y++) {
        for (int x = actualX; x <= max.x; x++) {
          if ((debugOutput) && false) {
            char buffer[1024];
            snprintf(buffer, sizeof(buffer), "(%4d,%4d)", x, y);
            cout << buffer << endl;
          }
          
          if (x > width-1) {
            continue;
          }
          if (y > height-1) {
            continue;
          }
          
          Coord c(x, y);
          uint32_t pi = (y * width) + x;
          uint32_t quantPixel = outPixels[pi];
          
          if (!isFirstPixelSet) {
            // First pixel in block
            isFirstPixelSet = true;
            firstPixel = quantPixel;
            
            if (debugOutput) {
              cout << "detected first pixel in block at " << x << "," << y << endl;
            }
          }
          
          if ((debugOutput)) {
            char buffer[1024];
            snprintf(buffer, sizeof(buffer), "for (%4d,%4d) offset is %d pixel is 0x%08X\n", x, y, pi, quantPixel);
            cout << buffer;
          }
          
          pixelsThisBlock.push_back(quantPixel);
          
          if (isAllSamePixel) {
            if (quantPixel != firstPixel) {
              isAllSamePixel = false;
            }
          }
        }
      }
      
      if (debugOutput) {
        cout << "isAllSamePixel " << isAllSamePixel << " isFirstPixelSet " << isFirstPixelSet << " num pixelsThisBlock " << pixelsThisBlock.size() << endl;
      }
      
      assert(isFirstPixelSet && pixelsThisBlock.size() > 0);
      
      // Examine each quant pixel value in pixelsThisBlock and determine which quant pixel best
      // represents this block. Note that coord is the upper left coord in the block.
      
      HistogramForBlock &hfb = blockMap[blockC];
      
      unordered_map<uint32_t, uint32_t> &pixelToCountTable = hfb.pixelToCountTable;
      
      uint32_t maxPixel = 0x0;
      
      if (isAllSamePixel) {
        // Optimized common case where all pixels are exactly the same value
        
        maxPixel = pixelsThisBlock[0];
        
        pixelToCountTable[maxPixel] = (uint32_t)pixelsThisBlock.size();
        
        if ((debugOutput)) {
          char buffer[1024];
          snprintf(buffer, sizeof(buffer), "all pixels in block optimized case for pixel 0x%08X\n", maxPixel);
          cout << buffer;
        }
      } else {
        
        for ( uint32_t qp : pixelsThisBlock ) {
          if ((debugOutput)) {
            char buffer[1024];
            snprintf(buffer, sizeof(buffer), "histogram pixel 0x%08X\n", qp);
            cout << buffer;
          }
          
          pixelToCountTable[qp] += 1;
        }
        
        int maxCount = 0;
        
        for ( auto it = begin(pixelToCountTable); it != end(pixelToCountTable); ++it) {
          uint32_t pixel = it->first;
          uint32_t count = it->second;
          
          if ((debugOutput)) {
            printf("count table[0x%08X] = %6d\n", pixel, count);
          }
          
          if (count > maxCount) {
            maxCount = count;
            maxPixel = pixel;
          }
        }
        
        // FIXME: if these are anywhere close, then do a stddev and choose one that is way
        // larger than the others. But if really close then choose no specific pixel.
        
        if ((debugOutput)) {
          printf("maxCount %5d : maxPixel 0x%08X\n", maxCount, maxPixel);
          printf("done\n");
        }
      }
      
      hfb.regionQuantPixel = maxPixel;
      
      Vec3b vec = PixelToVec3b(maxPixel);
      blockMat.at<Vec3b>(by, bx) = vec;
    }
  }
  
  if (dumpOutputImages) {
    char *filename = (char*) "block_quant_output.png";
    imwrite(filename, blockMat);
    cout << "wrote " << filename << endl;
  }
  
  delete [] colortable;
  delete [] inPixels;
  delete [] outPixels;
  
  return blockMat;
}

// Given input pixels and a range of coordinates (gathered from a region mask), determine
// a quant table that gives good quant results. This estimation determines the number of
// pixels and the actual cluster centers for different cases.

bool
estimateClusterCenters(const Mat & inputImg,
                       int32_t tag,
                       const vector<Coord> &regionCoords,
                       vector<uint32_t> &clusterCenters)
{
  const bool debug = true;
  const bool debugDumpImages = true;
  
  bool isVeryClose = false;
  
  if (debug) {
    cout << "estimateClusterCenters" << endl;
  }
  
  // Quant to evenly spaced grid to get estimate for number of clusters N
  
  vector<uint32_t> subdividedColors = getSubdividedColors();
  
  uint32_t numColors = (uint32_t) subdividedColors.size();
  uint32_t *colortable = new uint32_t[numColors];
  
  {
    int i = 0;
    for ( uint32_t color : subdividedColors ) {
      colortable[i++] = color;
    }
  }
  
  // Copy input pixels into array that can be passed to map_colors_mps()
  
  uint32_t numPixels = (uint32_t)regionCoords.size();
  uint32_t *inPixels = new uint32_t[numPixels];
  uint32_t *outPixels = new uint32_t[numPixels];
  
  for ( int i = 0; i < numPixels; i++ ) {
    Coord c = regionCoords[i];
    Vec3b vec = inputImg.at<Vec3b>(c.y, c.x);
    uint32_t pixel = Vec3BToUID(vec);
    inPixels[i] = pixel;
  }
  
  map_colors_mps(inPixels, numPixels, outPixels, colortable, numColors);
  
  // Count each quant pixel in outPixels
  
  Mat countMat(1, numPixels, CV_8UC3);
  
  for (int i = 0; i < numPixels; i++) {
    uint32_t pixel = outPixels[i];
    Vec3b vec = PixelToVec3b(pixel);
    countMat.at<Vec3b>(0, i) = vec;
  }
  
  unordered_map<uint32_t, uint32_t> outPixelToCountTable;
  
  generatePixelHistogram(countMat, outPixelToCountTable);
  
  if (debug) {
    for ( auto it = begin(outPixelToCountTable); it != end(outPixelToCountTable); ++it) {
      uint32_t pixel = it->first;
      uint32_t count = it->second;
      
      printf("count table[0x%08X] = %6d\n", pixel, count);
    }
  }
  
  // Dump quant output, each pixel is replaced by color in colortable
  
  if (debugDumpImages) {
    
    Mat tmpResultImg = inputImg.clone();
    tmpResultImg = Scalar(0,0,0xFF);
    
    for ( int i = 0; i < numPixels; i++ ) {
      Coord c = regionCoords[i];
      uint32_t pixel = outPixels[i];
      Vec3b vec = PixelToVec3b(pixel);
      tmpResultImg.at<Vec3b>(c.y, c.x) = vec;
    }
    
    {
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_quant_est_output" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, tmpResultImg);
      cout << "wrote " << fname << endl;
    }

    {
      vector<uint32_t> colortableVec;
      
      for (int i = 0; i < numColors; i++) {
        uint32_t pixel = colortable[i];
        colortableVec.push_back(pixel);
      }
      
      // Add phony entry for Red (the mask color)
      colortableVec.push_back(0x00FF0000);
      
      Mat quantOffsetsMat = mapQuantPixelsToColortableIndexes(tmpResultImg, colortableVec, true);
      
      {
        std::stringstream fnameStream;
        fnameStream << "srm" << "_tag_" << tag << "_quant_est_offsets" << ".png";
        string fname = fnameStream.str();
        
        imwrite(fname, quantOffsetsMat);
        cout << "wrote " << fname << endl;
      }
    }
  }
  
  // Check for the trivial case where the total number of pixels
  // is smallish and the pixels are quite a distance apart. This
  // can be detected by detecting the case when there are few
  // total pixels and the total number of quant pixels is the
  // same as the number of unique original original pixels.
  
  unordered_map<uint32_t, uint32_t> inUniqueTable;
  unordered_map<uint32_t, uint32_t> outUniqueTable;
  
  for ( int i = 0; i < numPixels; i++ ) {
    uint32_t pixel = inPixels[i];
    inUniqueTable[pixel] += 1;
  }
  
  for ( int i = 0; i < numPixels; i++ ) {
    uint32_t pixel = outPixels[i];
    outUniqueTable[pixel] += 1;
  }
  
  bool doClustering = true;
  
  if (inUniqueTable.size() < 32 && outUniqueTable.size() < 32) {
    if (debug) {
      int i = 0;
      for ( auto it = begin(inUniqueTable); it != end(inUniqueTable); ++it) {
        uint32_t pixel = it->first;
        uint32_t count = it->second;
        
        printf(" inUniqueTable[%5d] : 0x%08X -> %d\n", i, pixel, count);
      }
      for ( auto it = begin(outUniqueTable); it != end(outUniqueTable); ++it) {
        uint32_t pixel = it->first;
        uint32_t count = it->second;
        
        printf("outUniqueTable[%5d] : 0x%08X -> %d\n", i, pixel, count);
      }
    }
    
    if (inUniqueTable.size() == outUniqueTable.size()) {
      if (debug) {
        cout << "estimateClusterCenters return since small num in pixels is exact maping for size " << inUniqueTable.size() << endl;
      }

      for ( auto &pair : inUniqueTable ) {
        uint32_t pixel = pair.first;
        clusterCenters.push_back(pixel);
      }
      
      doClustering = false;
      isVeryClose = true;
    }
  }
  
  if (doClustering) {
    
    // Generate a clustering using the estimated number of clusters found by doing
    // the quant to an evenly spaced grid.
    
    uint32_t numActualClusters = numColors;
    
    int allPixelsUnique = 0;
    
    quant_recurse(numPixels, inPixels, outPixels, &numActualClusters, colortable, allPixelsUnique );
    
    if (debugDumpImages) {
      Mat tmpResultImg = inputImg.clone();
      tmpResultImg = Scalar(0,0,0xFF);
      
      for ( int i = 0; i < numPixels; i++ ) {
        Coord c = regionCoords[i];
        uint32_t pixel = outPixels[i];
        Vec3b vec = PixelToVec3b(pixel);
        tmpResultImg.at<Vec3b>(c.y, c.x) = vec;
      }
      
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_quant_est2_output" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, tmpResultImg);
      cout << "wrote " << fname << endl;
      cout << "";
    }
    
    // table
    
    {
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_quant_est2_table" << ".png";
      string fname = fnameStream.str();
      
      dumpQuantTableImage(fname, inputImg, colortable, numActualClusters);
    }
    
    unordered_map<uint32_t, uint32_t> inToOutTable;
    
    for ( int i = 0; i < numPixels; i++ ) {
      uint32_t inPixel = inPixels[i];
      uint32_t outPixel = outPixels[i];
      inToOutTable[inPixel] = outPixel;
    }
    
    // Process each unique pixel in outUniqueTable and lookup the difference
    // from the input pixel to the output pixel.
    
    uint32_t totalAbsComponentDeltas = 0;
    
    for ( auto &pair : inToOutTable ) {
      uint32_t inPixel = pair.first;
      uint32_t outPixel = pair.second;
      
      // Determine the delta for each component
      
      uint32_t deltaPixel = predict_trivial_component_sub(inPixel, outPixel);
      
      if (debug) {
        printf("unique pixel delta 0x%08X -> 0x%08X = 0x%08X\n", inPixel, outPixel, deltaPixel);
      }
      
      uint32_t absDeltaPixel = absPixel(deltaPixel);
      
      if (debug) {
        printf("abs    pixel delta 0x%08X\n", absDeltaPixel);
      }
      
      totalAbsComponentDeltas += absDeltaPixel;
    }
    
    if (totalAbsComponentDeltas == 0) {
      isVeryClose = true;
      
      // Gather cluster centers and return to caller
      
      for ( auto &pair : inToOutTable ) {
        //uint32_t inPixel = pair.first;
        uint32_t outPixel = pair.second;
        
        clusterCenters.push_back(outPixel);
      }
    }
    
  } // end doClustering if block
  
  delete [] colortable;
  delete [] inPixels;
  delete [] outPixels;
  
  return isVeryClose;
}

// Morph the "region mask", this is basically a way to expand the 2D region around the shape
// in a way that should capture pixels around the superpixel.

void
morphRegionMask(const Mat & inputImg,
                int32_t tag,
                const vector<Coord> &coords,
                int blockWidth,
                int blockHeight,
                int superpixelDim,
                vector<Coord> &regionCoords)
{
  const bool debug = true;
  const bool debugDumpImages = true;
  
  if (debug) {
    cout << "morphRegionMask" << endl;
  }
  
  Mat expandedBlockMat = expandBlockRegion(tag, coords, 2, blockWidth, blockHeight, superpixelDim);
  
  // Map morph blocks back to rectangular ROI in original image and extract ROI
  
  vector<Point> locations;
  findNonZero(expandedBlockMat, locations);
  
  vector<Coord> minMaxCoords;
  
  for ( Point p : locations ) {
    int actualX = p.x * superpixelDim;
    int actualY = p.y * superpixelDim;
    
    Coord min(actualX, actualY);
    minMaxCoords.push_back(min);
    
    Coord max(actualX+superpixelDim-1, actualY+superpixelDim-1);
    
    if (max.x > inputImg.cols-1) {
      max.x = inputImg.cols-1;
    }
    if (max.y > inputImg.rows-1) {
      max.y = inputImg.rows-1;
    }
    
    minMaxCoords.push_back(max);
  }
  
  int32_t originX, originY, width, height;
  bbox(originX, originY, width, height, minMaxCoords);
  Rect expandedRoi(originX, originY, width, height);
  
  if (debugDumpImages) {
    Mat roiInputMat = inputImg(expandedRoi);
    
    std::stringstream fnameStream;
    fnameStream << "srm" << "_tag_" << tag << "_morph_block_input" << ".png";
    string fname = fnameStream.str();
    
    imwrite(fname, roiInputMat);
    cout << "wrote " << fname << endl;
  }
  
  if (debugDumpImages) {
    int width = inputImg.cols;
    int height = inputImg.rows;
    
    Mat tmpExpandedBlockMat(height, width, CV_8UC1);
    
    tmpExpandedBlockMat = (Scalar) 0;
    
    for ( Point p : locations ) {
      int actualX = p.x * superpixelDim;
      int actualY = p.y * superpixelDim;
      
      Coord min(actualX, actualY);
      Coord max(actualX+superpixelDim-1, actualY+superpixelDim-1);
      
      for ( int y = min.y; y <= max.y; y++ ) {
        for ( int x = min.x; x <= max.x; x++ ) {
          
          if (x > width-1) {
            continue;
          }
          if (y > height-1) {
            continue;
          }
          
          tmpExpandedBlockMat.at<uint8_t>(y, x) = 0xFF;
        }
      }
    }
    
    std::stringstream fnameStream;
    fnameStream << "srm" << "_tag_" << tag << "_morph_block_bw" << ".png";
    string fname = fnameStream.str();
    
    imwrite(fname, tmpExpandedBlockMat);
    cout << "wrote " << fname << endl;
  }
  
  // Generate a collection of pixels from the blocks included in the
  // expanded mask.
  
  int inputWidthMinus1 = inputImg.cols - 1;
  int inputHeightMinus1 = inputImg.rows - 1;
  
  regionCoords.clear();
  regionCoords.reserve(locations.size() * (superpixelDim * superpixelDim));
  
  for ( Point p : locations ) {
    int actualX = p.x * superpixelDim;
    int actualY = p.y * superpixelDim;
    
    Coord min(actualX, actualY);
    Coord max(actualX+superpixelDim-1, actualY+superpixelDim-1);
    
    for ( int y = min.y; y <= max.y; y++ ) {
      for ( int x = min.x; x <= max.x; x++ ) {
        Coord c(x, y);
        
        if (x > inputWidthMinus1) {
          continue;
        }
        if (y > inputHeightMinus1) {
          continue;
        }
        
        regionCoords.push_back(c);
      }
    }
  }
    
  int numPixels = (int) regionCoords.size();
  
  if (debugDumpImages) {
    Mat tmpResultImg(inputImg.rows, inputImg.cols, CV_8UC3);
    tmpResultImg = Scalar(0,0,0xFF);
    
    for ( int i = 0; i < numPixels; i++ ) {
      Coord c = regionCoords[i];
      Vec3b vec = inputImg.at<Vec3b>(c.y, c.x);
      tmpResultImg.at<Vec3b>(c.y, c.x) = vec;
    }
    
    {
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_morph_masked_input" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, tmpResultImg);
      cout << "wrote " << fname << endl;
    }
  }
  
  if (debugDumpImages) {
    Mat tmpResultImg(inputImg.rows, inputImg.cols, CV_8UC4);
    tmpResultImg = Scalar(0,0,0,0);
    
    for ( int i = 0; i < numPixels; i++ ) {
      Coord c = regionCoords[i];
      Vec3b vec = inputImg.at<Vec3b>(c.y, c.x);
      Vec4b vec4;
      vec4[0] = vec[0];
      vec4[1] = vec[1];
      vec4[2] = vec[2];
      vec4[3] = 0xFF;
      tmpResultImg.at<Vec4b>(c.y, c.x) = vec4;
    }
    
    {
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_morph_alpha_input" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, tmpResultImg);
      cout << "wrote " << fname << endl;
    }
  }
  
  if (debug) {
    cout << "return morphRegionMask" << endl;
  }
  
  return;
}

// Given a tag indicating a superpixel generate a mask that captures the region in terms of
// exact pixels. On input, the mask contains either 0x0 or 0xFF to indicate if a given
// pixel was already consumed by a previous merge process. On return, the mask should contain
// 0xFF for pixels that are known to be inside the region.

bool
captureRegionMask(SuperpixelImage &spImage,
                  const Mat & inputImg,
                  const Mat & srmTags,
                  int32_t tag,
                  int blockWidth,
                  int blockHeight,
                  int superpixelDim,
                  Mat &mask,
                  const Mat &blockBasedQuantMat)
{
  const bool debug = true;
  const bool debugDumpImages = true;
  
  if (debug) {
    cout << "captureRegionMask" << endl;
  }
  
  assert(mask.rows == inputImg.rows);
  assert(mask.cols == inputImg.cols);
  assert(mask.channels() == 1);
  
  auto &coords = spImage.getSuperpixelPtr(tag)->coords;
  
  if (coords.size() <= ((superpixelDim*superpixelDim) >> 1)) {
    // A region contained in only a single block, don't process by itself
    
    if (debug) {
      cout << "captureRegionMask : region indicated by tag " << tag << " is too small to process with N coords " << coords.size() << endl;
    }
    
    return false;
  }
  
  vector<Coord> regionCoords;
  
  morphRegionMask(inputImg, tag, coords, blockWidth, blockHeight, superpixelDim, regionCoords);
  
  // Remove pixels from regionCoords that are known to be on in the mask. This limits the pixels
  // found with the region mask so that known regions that have already been processed will not
  // be included in the regionCoords.
  
  if ((1)) {
    unordered_map<Coord, bool> regionCoordsMap;
    
    for ( Coord c : regionCoords ) {
      regionCoordsMap[c] = true;
    }
    
    for ( int y = 0; y < mask.rows; y++ ) {
      for ( int x = 0; x < mask.cols; x++ ) {
        uint8_t bval = mask.at<uint8_t>(y, x);
        if (bval != 0) {
          // Do not consider this coord
          Coord c(x, y);
          regionCoordsMap.erase(c);
        }
      }
    }
    
    vector<Coord> trimRegionCoords;
    
    for ( Coord c : regionCoords ) {
      if (regionCoordsMap.count(c) > 0) {
        trimRegionCoords.push_back(c);
      }
    }
    
    regionCoords = trimRegionCoords;
    
    if (debugDumpImages) {
      Mat tmpResultImg(inputImg.rows, inputImg.cols, CV_8UC4);
      tmpResultImg = Scalar(0,0,0,0);
      
      int numPixels = (int)regionCoords.size();
      
      for ( int i = 0; i < numPixels; i++ ) {
        Coord c = regionCoords[i];
        Vec3b vec = inputImg.at<Vec3b>(c.y, c.x);
        Vec4b vec4;
        vec4[0] = vec[0];
        vec4[1] = vec[1];
        vec4[2] = vec[2];
        vec4[3] = 0xFF;
        tmpResultImg.at<Vec4b>(c.y, c.x) = vec4;
      }
      
      {
        std::stringstream fnameStream;
        fnameStream << "srm" << "_tag_" << tag << "_morph_minus_mask_alpha_input" << ".png";
        string fname = fnameStream.str();
        
        imwrite(fname, tmpResultImg);
        cout << "wrote " << fname << endl;
        cout << "";
      }
    }
  }
  
  // Init mask after possible early return
  
  mask = (Scalar) 0;
  
//  vector<uint32_t> estClusterCenters;
//  
//  bool isVeryClose = estimateClusterCenters(inputImg, tag, regionCoords, estClusterCenters);
//  
//  if (isVeryClose) {
//    captureVeryCloseRegion(spImage, inputImg, srmTags, tag, blockWidth, blockHeight, superpixelDim, mask, regionCoords, coords, (int)estClusterCenters.size());
//  } else {
//    captureNotCloseRegion(spImage, inputImg, srmTags, tag, blockWidth, blockHeight, superpixelDim, mask, regionCoords, coords, (int)estClusterCenters.size(), blockBasedQuantMat);
//  }
  
  captureRegion(spImage, inputImg, srmTags, tag, blockWidth, blockHeight, superpixelDim, mask, regionCoords, coords, blockBasedQuantMat);
  
  // Capture mask output as alpha pixels
  
  if (debugDumpImages) {
    Mat tmpResultImg(inputImg.rows, inputImg.cols, CV_8UC4);
    tmpResultImg = Scalar(0,0,0,0);
    
    for ( int y = 0; y < tmpResultImg.rows; y++ ) {
      for ( int x = 0; x < tmpResultImg.cols; x++ ) {
        uint8_t isOn = mask.at<uint8_t>(y, x);
        
        if (isOn) {
          Vec3b vec = inputImg.at<Vec3b>(y, x);
          Vec4b vec4;
          vec4[0] = vec[0];
          vec4[1] = vec[1];
          vec4[2] = vec[2];
          vec4[3] = 0xFF;
          tmpResultImg.at<Vec4b>(y, x) = vec4;
        }
      }
    }
    
    {
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_morph_minus_mask_alpha_output" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, tmpResultImg);
      cout << "wrote " << fname << endl;
      cout << "";
    }
  }
  
  if (debug) {
    cout << "return captureRegionMask" << endl;
  }
  
  return true;
}

// This implementation will examine the bounds of a region after collapsing and then expanding the region back
// out to discover where the true edges of regions are located.

void
captureRegion(SuperpixelImage &spImage,
              const Mat & inputImg,
              const Mat & srmTags,
              int32_t tag,
              int blockWidth,
              int blockHeight,
              int superpixelDim,
              Mat &mask,
              const vector<Coord> &regionCoords,
              const vector<Coord> &srmRegionCoords,
              const Mat &blockBasedQuantMat)
{
  const bool debug = true;
  const bool debugDumpImages = true;
  
  if (debug) {
    cout << "captureRegion " << tag << endl;
  }
  
  // Gather the tags associated with all the regions
  // indicated by regionCoords.
  
  vector<int32_t> allRegionTags;
  
  unordered_map<int32_t, int32_t> allRegionTagsMap;
  
  for ( Coord c : regionCoords ) {
    Vec3b vec = srmTags.at<Vec3b>(c.y, c.x);
    int32_t tag = Vec3BToUID(vec);
    allRegionTagsMap[tag] = tag;
  }
  for ( auto &pair : allRegionTagsMap ) {
    allRegionTags.push_back(pair.first);
  }
  
  if (debug) {
    cout << "allRegionTags:" << endl;
    for ( int32_t tag : allRegionTags ) {
      cout << tag << endl;
    }
    cout << "";
  }
  
  // How many edge would there be in the expanded regionCoords ?
  
  vector<SuperpixelEdge> extendedRegionEdges;
  
  {
    vector<SuperpixelEdge> edges = getEdgesInRegion(spImage, srmTags, tag, regionCoords);
    
    cout << "getEdgesInRegion returned " << edges.size() << " edges" << endl;
    
    for ( SuperpixelEdge edge : edges ) {
      cout << "edge " << edge << endl;
    }
    
    cout << "";
    
    extendedRegionEdges = edges;
  }
  
  if (extendedRegionEdges.size() == 0) {
    // No edges between regions, this would typically happen when a parent that contains
    // interior regions has already consumed all the interior region pixels.
    
    if (debug) {
      cout << "captureRegion returned region as mask since zero edges detected for tag " << tag << endl;
    }
    
    for ( Coord c : regionCoords ) {
      mask.at<uint8_t>(c.y, c.x) = 0xFF;
    }

    return;
  }
  
  // At the start of a region identification operation, a rough estimate of the region bounds is available
  // but the contracted or expanded bounds are not known. Scan clockwise to determine likely bounds based
  // on the initial region shape.
  
  clockwiseScanForShapeBounds(inputImg, srmTags, tag, srmRegionCoords);
  
  vector<Coord> outCoords;
  
  vector<vector<Coord> > contractStack;
  vector<vector<Coord> > expandStack;
  
  // Contract the mask area starting from the region pixels
  
  int contractStep = 1;
  
  Vec3b contractingCenterOfMass(0,0,0);
  
  for ( ; 1 ; contractStep++) {
    bool worked = contractOrExpandRegion(inputImg, tag, srmRegionCoords, false, contractStep, outCoords);
    if (!worked) {
      // Deduct 1 so that the step number is the value just before the iteration stopped
      // due to a COM being the same or no more pixels in the mask.
      
      contractStep -= 1;
      
      break;
    }
    
    // If outCoords step does not reduce the colors more then stop at this point
    
    vector<Vec3b> vecOfPixels;
    
    for ( Coord c : outCoords ) {
      Vec3b vec = inputImg.at<Vec3b>(c.y,c.x);
      vecOfPixels.push_back(vec);
    }
    
    Vec3b centerOfMass = centerOfMass3d(vecOfPixels);
    
    if (debug) {
      uint32_t p1 = Vec3BToUID(contractingCenterOfMass);
      uint32_t p2 = Vec3BToUID(centerOfMass);
      
      printf("contract region returned %5d coords, contractingCenterOfMass 0x%08X vs COM 0x%08X\n", (int)outCoords.size(), p1, p2);
    }
    
    if (centerOfMass == contractingCenterOfMass) {
      if (debug) {
        printf("stopping after %d contraction steps since internal center of mass color is consistent\n", contractStep);
      }
      
      // Deduct 1 so that the step number is the value just before the iteration stopped
      // due to a COM being the same or no more pixels in the mask.
      
      contractStep -= 1;
      
      break;
    } else {
      contractingCenterOfMass = centerOfMass;
      
      contractStack.push_back(outCoords);
    }
    
    // If zero edges inside the region is found, then stop contracting
    
    if (debug) {
      vector<SuperpixelEdge> edges = getEdgesInRegion(spImage, srmTags, tag, outCoords);
      
      cout << "getEdgesInRegion returned " << edges.size() << " edges" << endl;
      
      for ( SuperpixelEdge edge : edges ) {
        cout << "edge " << edge << endl;
      }
      
      cout << "";
    }
    
    if (1) {
      vector<SuperpixelEdge> edges = getEdgesInRegion(spImage, srmTags, tag, outCoords);
      
      if (edges.size() == 0) {
        // No edges means that the interior region contains only pixels associated
        // with the region identified by tag. Note that the contract step is
        // left alone here so that the coords for this region step are appened
        // to contractStack.
        
        if (debug) {
          printf("stopping after %d contraction steps since no edges found inside region\n", contractStep);
        }
        
        break;
      }
    }
  }
  
  // Expand region out so that more pixels are included from 1 to N neighbors.
  
  int expandStep = 1;
  bool oneMoreStep = false;
  
  //  Vec3b expandingCenterOfMass(0,0,0);
  
  for ( ; 1 ; expandStep++) {
    bool worked = contractOrExpandRegion(inputImg, tag, srmRegionCoords, true, expandStep, outCoords);
    if (!worked) {
      // Deduct 1 so that the step number is the value just before the iteration stopped
      // due to a COM being the same or no more pixels in the mask.
      
      expandStep -= 1;
      
      break;
    }
    
    // If outCoords step does not reduce the colors more then stop at this point
    
    vector<Vec3b> vecOfPixels;
    
    for ( Coord c : outCoords ) {
      Vec3b vec = inputImg.at<Vec3b>(c.y,c.x);
      vecOfPixels.push_back(vec);
    }
    
//    Vec3b centerOfMass = centerOfMass3d(vecOfPixels);
    
//    if (debug) {
//      uint32_t p1 = Vec3BToUID(expandingCenterOfMass);
//      uint32_t p2 = Vec3BToUID(centerOfMass);
//      
//      printf("expand region returned %5d coords, expandingCenterOfMass 0x%08X vs COM 0x%08X\n", (int)outCoords.size(), p1, p2);
//    }
    
    if (debug) {
      vector<SuperpixelEdge> edges = getEdgesInRegion(spImage, srmTags, tag, outCoords);
      
      cout << "getEdgesInRegion returned " << edges.size() << " edges" << endl;
      
      for ( SuperpixelEdge edge : edges ) {
        cout << "edge " << edge << endl;
      }
      
      cout << "";
    }
    
    if (oneMoreStep) {
      //      if (debug) {
      //        printf("stopping after %d expansion steps since internal center of mass color is consistent\n", expandStep);
      //      }
      
      // Deduct 1 so that the step number is the value just before the iteration stopped
      // due to a COM being the same or no more pixels in the mask.
      
      //expandStep -= 1;
      
      expandStack.push_back(outCoords);
      
      break;
    } else {
      //expandingCenterOfMass = centerOfMass;
      
      expandStack.push_back(outCoords);
    }
    
    if (1) {
      vector<SuperpixelEdge> edges = getEdgesInRegion(spImage, srmTags, tag, outCoords);
      
      if (edges.size() == extendedRegionEdges.size()) {
        if (debug) {
          printf("stopping after %d expansion steps since the number of edges %d matches the extended region num edges\n", expandStep, (int)edges.size());
        }
        
        oneMoreStep = true;
        
        //break;
      }
    }
  }
  
  // Use best expanded range of coords
  
  vector<Coord> bestRegionCoords = expandStack[expandStack.size() - 1];
  
  // Copy input pixels
  
  int numPixels = (int)bestRegionCoords.size();
  
  uint32_t *inPixels = new uint32_t[numPixels];
  uint32_t *outPixels = new uint32_t[numPixels];
  
//  for ( int i = 0; i < numPixels; i++ ) {
//    Coord c = bestRegionCoords[i];
//    
//    Vec3b vec = inputImg.at<Vec3b>(c.y, c.x);
//    uint32_t pixel = Vec3BToUID(vec);
//    inPixels[i] = pixel;
//  }
  
  // Generate mask Mat that is the same dimensions as blockMat but contains just one
  // byte for each pixel and acts as a mask. The white pixels indicate the blocks
  // that are included in the mask.
  
  Mat blockMaskMat(blockBasedQuantMat.size(), CV_8UC1);
  blockMaskMat = (Scalar) 0;
  
  for ( Coord c : bestRegionCoords ) {
    // Convert (X,Y) to block (X,Y)
    
    int blockX = c.x / superpixelDim;
    int blockY = c.y / superpixelDim;
    
    blockMaskMat.at<uint8_t>(blockY, blockX) = 0xFF;
  }
  
  if (debugDumpImages) {
    std::stringstream fnameStream;
    fnameStream << "srm" << "_tag_" << tag << "_best_region_mask_blocks" << ".png";
    string fname = fnameStream.str();
    
    imwrite(fname, blockMaskMat);
    cout << "wrote " << fname << endl;
  }
  
  if (debugDumpImages) {
    Mat tmpMat(inputImg.size(), CV_8UC4);
    tmpMat = Scalar(0, 0, 0, 0);
    
    for ( Coord c : bestRegionCoords ) {
      Vec3b vec = inputImg.at<Vec3b>(c.y, c.x);
      Vec4b vec4;
      vec4[0] = vec[0];
      vec4[1] = vec[1];
      vec4[2] = vec[2];
      vec4[3] = 0xFF;
      tmpMat.at<Vec4b>(c.y, c.x) = vec4;
    }
    
    std::stringstream fnameStream;
    fnameStream << "srm" << "_tag_" << tag << "_best_region_alpha" << ".png";
    string fname = fnameStream.str();
    
    imwrite(fname, tmpMat);
    cout << "wrote " << fname << endl;
    cout << "";
  }
  
  // Do a clockwise scan of the pixels identified in the best region mask and
  // gather all the tags in that direction. This is basically a line that is
  // rotated around via a Matrix rotation.
  
  vector<TagsAroundShape> tagsAround;
  
  clockwiseScanForTagsAroundShape(srmTags, tag, bestRegionCoords, tagsAround);
  
  cout << "num TAS ranges N = " << tagsAround.size() << endl;
  
  for ( TagsAroundShape tas : tagsAround ) {
    cout << "for TAS range " << tas.start << " to " << tas.end << " contains tags (";
    for ( int32_t tag : tas.tags ) {
      cout << tag << " ";
    }
    cout << ") and coords N = " << tas.coords.size();
    cout << endl;
    cout << "";
    
    // Display coords that correspond to this range
    
    if (debugDumpImages) {
      Mat tmpMat(inputImg.size(), CV_8UC4);
      tmpMat = Scalar(0, 0, 0, 0);
      
      for ( Coord c : tas.coords ) {
        Vec4b vec4;
        
        Vec3b vec = inputImg.at<Vec3b>(c.y, c.x);
        
        vec4[0] = vec[0];
        vec4[1] = vec[1];
        vec4[2] = vec[2];
        vec4[3] = 0xFF;
        tmpMat.at<Vec4b>(c.y, c.x) = vec4;
      }
      
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_tas_range" << tas.start << "_" << tas.end << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, tmpMat);
      cout << "wrote " << fname << endl;
      cout << "";
    }
  }
  
  // Iteration must be in terms of the scan regions, since breaking the input up into
  // known "secants" is required in order to process regions in terms of one
  // gradient to the next. Note the special case where there is only 1 tag
  // (likely this is a parent tag). In that case, do 1 merge from the interior
  // region all the way out to the containing region.
  
  typedef struct {
    uint32_t count;
    TagsAroundShape *tasPtr;
  } WeightedEdgeValue;
  
  unordered_map<uint32_t, WeightedEdgeValue> weightedEdges;

  // If any coords are not used as "most common" then append them here
  
  unordered_map<uint32_t, vector<Coord>> notMostCommonCoords;
  
  for ( TagsAroundShape &tas : tagsAround ) {
    if (tas.coords.size() == 0) {
      // Skip region that contains no coords
      
      if (debug) {
        cout << "skip TAS range " << tas.start << " to " << tas.end << " since it contains zero coords" << endl;
      }
      
      continue;
      
    } else if (tas.start == tas.end) {
      // Skip region that is only a single step.
      
      if (debug) {
          cout << "skip TAS range " << tas.start << " to " << tas.end << " since it is a single step" << endl;
      }
      
      for ( Coord c : tas.coords ) {
        Vec3b tagVec = srmTags.at<Vec3b>(c.y, c.x);
        int32_t tag = Vec3BToUID(tagVec);
        notMostCommonCoords[tag].push_back(c);
      }
      
      // FIXME: in this tas loop, if a small region is the only ref to a neighbor then
      // that neighbor has to be found in one of the searches. Save tas to another vec
      // and then do another loop looking for elements that were not found.
      
      continue;
    }
    
    cout << "for TAS range " << tas.start << " to " << tas.end << endl;
    
    // Find path from the interior region to a tag identified in the region.
    
    unordered_map<uint32_t, uint32_t> tagHistogram;
    
    for ( Coord c : tas.coords ) {
      Vec3b tagVec = srmTags.at<Vec3b>(c.y, c.x);
      int32_t tag = Vec3BToUID(tagVec);
      tagHistogram[tag] += 1;
    }
    
    vector<uint32_t> sortedKeys = sort_keys_by_count(tagHistogram, true);
    
    cout << "tags sorted by count:" << endl;
    
    for ( uint32_t key : sortedKeys ) {
      uint32_t key_count = tagHistogram[key];
      printf("%5d = %5d\n", key, key_count);
    }
    
    assert(sortedKeys.size() > 0);
    int32_t mostCommonOtherTag = sortedKeys[0];
    int32_t mostCommonOtherTagCount = tagHistogram[mostCommonOtherTag];
    
    // Edge (tag, mostCommonTag)
    
    SuperpixelEdge edge(tag, mostCommonOtherTag);
    
    cout << "edge " << edge << endl;

    int currentLargest = 0;
    if (weightedEdges.count(mostCommonOtherTag) > 0) {
      currentLargest = weightedEdges[mostCommonOtherTag].count;
    }
    
    if (mostCommonOtherTagCount > currentLargest) {
      WeightedEdgeValue wev;
      wev.count = mostCommonOtherTagCount;
      wev.tasPtr = &tas;
      weightedEdges[mostCommonOtherTag] = wev;
    }
    
    // FIXME: how to track coords not used in this logic ? aka notMostCommonCoords
  }
  
  for ( auto & pair : weightedEdges ) {
    int32_t mostCommonOtherTag = pair.first;
    int32_t mostCommonOtherTagCount = pair.second.count;
    cout << "weightedEdges[" << mostCommonOtherTag << "] = " << mostCommonOtherTagCount << endl;
  }
  
  // The edge table indicates how to get from one common region to the interior
  // region. Each edge indicates a subset of points to be iterated.
  
  for ( auto & pair : weightedEdges ) {
    int32_t mostCommonOtherTag = pair.first;

    WeightedEdgeValue &wev = pair.second;
    
    // If this other tag is largely a flat color region then it it easy to
    // create a vector of one color to the other since anything that is
    // not exactly the flat pixel is in the gradient part.
    
    vector<Coord> outsideCoords;
    
    for ( Coord c : wev.tasPtr->coords ) {
      outsideCoords.push_back(c);
    }
    
    if (notMostCommonCoords.count(mostCommonOtherTag) > 0) {
      for ( Coord c : notMostCommonCoords[mostCommonOtherTag] ) {
        outsideCoords.push_back(c);
      }
    }
    
//    vector<int32_t> tags = wev.tasPtr->tags;
//    for (int i = 0; i < tags.size(); i++) {
//      int32_t currentTag = tags[i];
//      if (currentTag == tag) {
//        continue;
//      }
//      if (notMostCommonCoords.count(currentTag) > 0) {
//        for ( Coord c : notMostCommonCoords[currentTag] ) {
//          combinedCoords.push_back(c);
//        }
//      }
//    }
    
//    Superpixel *otherSpPtr = spImage.getSuperpixelPtr(mostCommonOtherTag);
//    
//    for ( Coord c : otherSpPtr->coords ) {
//      combinedCoords.push_back(c);
//    }
    
    // FIXME: might want to only select the pixels near the edge coords.
    
    Superpixel *spPtr = spImage.getSuperpixelPtr(tag);
    
    vector<Coord> &insideCoords = spPtr->coords;
    
    vector<Coord> combinedCoords;
    combinedCoords.reserve(outsideCoords.size() + insideCoords.size());
    
    append_to_vector(combinedCoords, outsideCoords);
    append_to_vector(combinedCoords, insideCoords);
    
    if (debugDumpImages) {
      Mat tmpMat(inputImg.size(), CV_8UC4);
      tmpMat = Scalar(0, 0, 0, 0);
      
      for ( Coord c : combinedCoords ) {
        Vec4b vec4;
        
        Vec3b vec = inputImg.at<Vec3b>(c.y, c.x);
        
        vec4[0] = vec[0];
        vec4[1] = vec[1];
        vec4[2] = vec[2];
        vec4[3] = 0xFF;
        tmpMat.at<Vec4b>(c.y, c.x) = vec4;
      }
      
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_to_" << mostCommonOtherTag << "_combined" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, tmpMat);
      cout << "wrote " << fname << endl;
      cout << "";
    }
    
    // --------------------- QUANT ---------------------------
    
    // Run quant logic on this subset of pixels to determine if there is a gradient
    // between the pixels identified in this region.
    
    // With 2 colors, detect spread between the input pixels N = 4
    
    int N = 4;
    
    const int numClusters = N;
    
    uint32_t *colortable = new uint32_t[numClusters];
    
    uint32_t numActualClusters = numClusters;
    
    int allPixelsUnique = 0;
    
    // FIXME: only want to include the inPixels that are defined for this region!
    
    assert(combinedCoords.size() <= bestRegionCoords.size());
    
    int numPixels = (int)combinedCoords.size();
    
    for ( int i = 0; i < numPixels; i++ ) {
      Coord c = combinedCoords[i];
      
      Vec3b vec = inputImg.at<Vec3b>(c.y, c.x);
      uint32_t pixel = Vec3BToUID(vec);
      inPixels[i] = pixel;
    }
    
    quant_recurse(numPixels, inPixels, outPixels, &numActualClusters, colortable, allPixelsUnique );
    
    // Write quant output where each original pixel is replaced with the closest
    // colortable entry.
    
    if (debugDumpImages) {
      Mat tmpResultImg(inputImg.size(), CV_8UC4);
      
      tmpResultImg = Scalar(0,0,0,0);
      
      for ( int i = 0; i < numPixels; i++ ) {
        Coord c = combinedCoords[i];
        uint32_t pixel = inPixels[i];
        Vec3b vec = PixelToVec3b(pixel);
        Vec4b vec4;
        vec4[0] = vec[0];
        vec4[1] = vec[1];
        vec4[2] = vec[2];
        vec4[3] = 0xFF;
        tmpResultImg.at<Vec4b>(c.y, c.x) = vec4;
      }
      
      {
        std::stringstream fnameStream;
        fnameStream << "srm" << "_tag_" << tag << "_to_" << mostCommonOtherTag << "_quant3_input" << ".png";
        string fname = fnameStream.str();
        
        imwrite(fname, tmpResultImg);
        cout << "wrote " << fname << endl;
      }
      
      tmpResultImg = Scalar(0,0,0,0);
      
      for ( int i = 0; i < numPixels; i++ ) {
        Coord c = combinedCoords[i];
        uint32_t pixel = outPixels[i];
        Vec3b vec = PixelToVec3b(pixel);
        Vec4b vec4;
        vec4[0] = vec[0];
        vec4[1] = vec[1];
        vec4[2] = vec[2];
        vec4[3] = 0xFF;
        tmpResultImg.at<Vec4b>(c.y, c.x) = vec4;
      }
      
      {
        std::stringstream fnameStream;
        fnameStream << "srm" << "_tag_" << tag << "_to_" << mostCommonOtherTag << "_quant3_output" << ".png";
        string fname = fnameStream.str();
        
        imwrite(fname, tmpResultImg);
        cout << "wrote " << fname << endl;
      }
      
      // table
      
      {
        std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_to_" << mostCommonOtherTag << "_quant3_table" << ".png";
        string fname = fnameStream.str();
        
        dumpQuantTableImage(fname, inputImg, colortable, numActualClusters);
      }
    }
    
    // Generate color sorted clusters
    
    vector<uint32_t> clusterCenterPixels;
    
    for ( int i = 0; i < numActualClusters; i++) {
      uint32_t pixel = colortable[i];
      clusterCenterPixels.push_back(pixel);
    }
    
#if defined(DEBUG)
    if ((1)) {
      unordered_map<uint32_t, uint32_t> seen;
      
      for ( int i = 0; i < numActualClusters; i++ ) {
        uint32_t pixel;
        pixel = colortable[i];
        
        if (seen.count(pixel) > 0) {
        } else {
          // Note that only the first seen index is retained, this means that a repeated
          // pixel value is treated as a dup.
          
          seen[pixel] = i;
        }
      }
      
      int numQuantUnique = (int)seen.size();
      assert(numQuantUnique == numActualClusters);
    }
#endif // DEBUG
    
    if (debug) {
      fprintf(stdout, "numClusters %5d : numActualClusters %5d \n", numClusters, numActualClusters);
      
      unordered_map<uint32_t, uint32_t> seen;
      
      for ( int i = 0; i < numActualClusters; i++ ) {
        uint32_t pixel;
        pixel = colortable[i];
        
        if (seen.count(pixel) > 0) {
          fprintf(stdout, "cmap[%3d] = 0x%08X (DUP of %d)\n", i, pixel, seen[pixel]);
        } else {
          fprintf(stdout, "cmap[%3d] = 0x%08X\n", i, pixel);
          
          // Note that only the first seen index is retained, this means that a repeated
          // pixel value is treated as a dup.
          
          seen[pixel] = i;
        }
      }
      
      fprintf(stdout, "cmap contains %3d unique entries\n", (int)seen.size());
      
      int numQuantUnique = (int)seen.size();
      
      assert(numQuantUnique == numActualClusters);
    }
    
    vector<uint32_t> sortedOffsets = generate_cluster_walk_on_center_dist(clusterCenterPixels);
    
    // Once cluster centers have been sorted by 3D color cube distance, emit as PNG
    
    Mat sortedQtableOutputMat = Mat(numActualClusters, 1, CV_8UC3);
    sortedQtableOutputMat = (Scalar) 0;
    
    vector<uint32_t> sortedColortable;
    
    for (int i = 0; i < numActualClusters; i++) {
      int si = (int) sortedOffsets[i];
      uint32_t pixel = colortable[si];
      Vec3b vec = PixelToVec3b(pixel);
      sortedQtableOutputMat.at<Vec3b>(i, 0) = vec;
      
      sortedColortable.push_back(pixel);
    }
    
    if (debugDumpImages)
    {
      std::stringstream fnameStream;
      
      fnameStream << "srm" << "_tag_" << tag << "_to_" << mostCommonOtherTag << "_quant3_table_sorted" << ".png";
      
      string filename = fnameStream.str();
      
      char *outQuantTableFilename = (char*) filename.c_str();
      imwrite(outQuantTableFilename, sortedQtableOutputMat);
      cout << "wrote " << outQuantTableFilename << endl;
    }
    
    // Map pixels to sorted colortable offset
    
    unordered_map<uint32_t, uint32_t> pixel_to_sorted_offset;
    
    assert(numActualClusters <= 256);
    
    for (int i = 0; i < numActualClusters; i++) {
      int si = (int) sortedOffsets[i];
      uint32_t pixel = colortable[si];
      pixel_to_sorted_offset[pixel] = si;
    }
    
    if (debugDumpImages) {
      Mat sortedQuantOutputMat = inputImg.clone();
      sortedQuantOutputMat = Scalar(0,0,0xFF);
      
      for ( int i = 0; i < numPixels; i++ ) {
        Coord c = combinedCoords[i];
        uint32_t pixel = outPixels[i];
        
        assert(pixel_to_sorted_offset.count(pixel) > 0);
        uint32_t offset = pixel_to_sorted_offset[pixel];
        
        if ((debug) && false) {
          char buffer[1024];
          snprintf(buffer, sizeof(buffer), "for (%4d,%4d) pixel is %d -> offset %d\n", c.x, c.y, pixel, offset);
          cout << buffer;
        }
        
        assert(offset <= 256);
        uint32_t grayscalePixel = (offset << 16) | (offset << 8) | offset;
        
        Vec3b vec = PixelToVec3b(grayscalePixel);
        sortedQuantOutputMat.at<Vec3b>(c.y, c.x) = vec;
      }
      
      {
        std::stringstream fnameStream;
        
        fnameStream << "srm" << "_tag_" << tag << "_to_" << mostCommonOtherTag << "_quant3_table_offsets" << ".png";
        
        string filename = fnameStream.str();
        
        char *outQuantFilename = (char*)filename.c_str();
        imwrite(outQuantFilename, sortedQuantOutputMat);
        cout << "wrote " << outQuantFilename << endl;
      }
    }
    
    delete [] colortable;
    
    // For the coords that define the inside region, gather all the out quant pixels
    // and record the colortable offsets.
    
    unordered_map<Coord, uint32_t> coordToQuantPixelMap;
    
    for ( int i = 0; i < numPixels; i++ ) {
      Coord c = combinedCoords[i];
      uint32_t pixel = outPixels[i];
      coordToQuantPixelMap[c] = pixel;
      
#if defined(DEBUG)
      assert(pixel_to_sorted_offset.count(pixel) > 0);
#endif // DEBUG
    }
    
    // Map colortable offset to count "inside"
    unordered_map<uint32_t, uint32_t> insideOffsetHistogram;
    // Map colortable offset to count "outside"
    unordered_map<uint32_t, uint32_t> outsideOffsetHistogram;
    
    for ( Coord c : insideCoords ) {
      uint32_t pixel = coordToQuantPixelMap[c];
      uint32_t offset = pixel_to_sorted_offset[pixel];
      insideOffsetHistogram[offset] += 1;
    }
    for ( Coord c : outsideCoords ) {
      uint32_t pixel = coordToQuantPixelMap[c];
      uint32_t offset = pixel_to_sorted_offset[pixel];
      outsideOffsetHistogram[offset] += 1;
    }
    
    vector<uint32_t> sortedInsideOffsetKeys = sort_keys_by_count(insideOffsetHistogram, true);
    
    if (debug) {
      fprintf(stdout, "sortedInsideOffsetKeys\n");
      for ( uint32_t offset : sortedInsideOffsetKeys ) {
        uint32_t count = insideOffsetHistogram[offset];
        fprintf(stdout, "%8d -> %5d = 0x%08X\n", offset, count, sortedColortable[offset]);
      }
      fprintf(stdout, "done\n");
    }

    vector<uint32_t> sortedOutsideOffsetKeys = sort_keys_by_count(outsideOffsetHistogram, true);
    
    if (debug) {
      fprintf(stdout, "sortedOutsideOffsetKeys\n");
      for ( uint32_t offset : sortedOutsideOffsetKeys ) {
        uint32_t count = outsideOffsetHistogram[offset];
        fprintf(stdout, "%8d -> %5d = 0x%08X\n", offset, count, sortedColortable[offset]);
      }
      fprintf(stdout, "done\n");
    }
    
    // Choose most likely offset for vector end point based on histogram counts
    
    uint32_t insideColortableOffset = sortedInsideOffsetKeys[0];
    uint32_t outsideColortableOffset = sortedOutsideOffsetKeys[0];
    
    uint32_t insideQuantPixel = sortedColortable[insideColortableOffset];
    uint32_t outsideQuantPixel = sortedColortable[outsideColortableOffset];
    
    if (debug) {
      fprintf(stdout, "inside  %5d -> 0x%08X\n", insideColortableOffset, insideQuantPixel);
      fprintf(stdout, "outside %5d -> 0x%08X\n", outsideColortableOffset, outsideQuantPixel);
    }
    
    // Filter out insideQuantPixel and outsideQuantPixel
    
    vector<uint32_t> filteredQuantVector;
    
    for ( uint32_t pixel : sortedColortable ) {
      if (pixel == insideQuantPixel) {
        continue;
      }
      if (pixel == outsideQuantPixel) {
        continue;
      }
      filteredQuantVector.push_back(pixel);
    }
    
    if (debug) {
      for ( uint32_t pixel : filteredQuantVector ) {
        fprintf(stdout, "filtered quant vector  0x%08X\n", pixel);
      }
    }
    
    // Iterate over the pixels in the outside region trimming away the least likely pixels
    
    unordered_map<uint32_t, uint32_t> outsideInputHistogram;
    
    for ( Coord c : outsideCoords ) {
      Vec3b vec = inputImg.at<Vec3b>(c.y, c.x);
      uint32_t pixel = Vec3BToUID(vec);
      outsideInputHistogram[pixel] += 1;
    }
    
    vector<uint32_t> sortedOutsideInputKeys = sort_keys_by_count(outsideInputHistogram, true);
    
    uint32_t outsideExactPixel = sortedOutsideInputKeys[0];
    
    if (debug) {
      fprintf(stdout, "sortedOutsideInputKeys\n");
      for ( uint32_t pixel : sortedOutsideInputKeys ) {
        uint32_t count = outsideInputHistogram[pixel];
        fprintf(stdout, "0x%08X -> %5d\n", pixel, count);
      }
      fprintf(stdout, "done\n");
    }
    
    if (debug) {
      fprintf(stdout, "outside exact 0x%08X\n", outsideExactPixel);
    }
    
    // Generate histogram for each pixel in the outside the region
    // area and then calculate slope for histogram counts.
    
    bool isOutsideSmoothRegion = false;
    
    if (1) {
      float mean, stddev;
      
      vector<float> floatSizes;
      
      unordered_map<uint32_t, uint32_t> countMap;
      
      for ( uint32_t pixel : sortedOutsideInputKeys ) {
        uint32_t count = outsideInputHistogram[pixel];
        countMap[count] += 1;
        
        if (debug) {
          char buffer[1000];
          snprintf(buffer, sizeof(buffer), "outside pixel 0x%08X = count %5d", pixel, count);
          cout << (char*)buffer << endl;
        }
      }
      
      // Sort countMap keys by descending int value
      
      vector<int32_t> countKeys;
      for ( auto &pair : countMap ) {
        int32_t count = pair.first;
        countKeys.push_back(count);
      }
      
      sort(begin(countKeys), end(countKeys), greater<int32_t>());
      
      vector<int32_t> countDeltas = deltas(countKeys);
      
      if (debug) {
        fprintf(stdout, "countDeltas\n");
        int max = (int) countKeys.size();
        assert(max == countDeltas.size());
        for ( int i = 0; i < max; i++) {
          int32_t count = countKeys[i];
          int32_t delta = countDeltas[i];
          fprintf(stdout, "%5d -> %5d\n", count, delta);
        }
        fprintf(stdout, "done countDeltas\n");
      }

      if (countDeltas.size() > 0) {
        countDeltas.erase(begin(countDeltas)); // delete first slope
      }
      
      for ( int32_t count : countDeltas ) {
        // Treat each negative slope value as a positive number
        assert(count <= 0);
        floatSizes.push_back((float)(count * -1));
      }
      
      sample_mean(floatSizes, &mean);
      sample_mean_delta_squared_div(floatSizes, mean, &stddev);
      
      if (1) {
        char buffer[1024];
        
        snprintf(buffer, sizeof(buffer), "mean %0.4f stddev %0.4f", mean, stddev);
        cout << (char*)buffer << endl;
        
        snprintf(buffer, sizeof(buffer), "1 stddev %0.4f", (mean + (stddev * 0.5f * 1.0f)));
        cout << (char*)buffer << endl;
        
        snprintf(buffer, sizeof(buffer), "2 stddev %0.4f", (mean + (stddev * 0.5f * 2.0f)));
        cout << (char*)buffer << endl;
        
        snprintf(buffer, sizeof(buffer), "3 stddev %0.4f", (mean + (stddev * 0.5f * 3.0f)));
        cout << (char*)buffer << endl;
        
        snprintf(buffer, sizeof(buffer), "-1 stddev %0.4f", (mean - (stddev * 0.5f * 1.0f)));
        cout << (char*)buffer << endl;
        
        snprintf(buffer, sizeof(buffer), "-2 stddev %0.4f", (mean - (stddev * 0.5f * 2.0f)));
        cout << (char*)buffer << endl;
      }
      
      vector<float> largerSizes;
      
      const float minusOneStddev = (mean - (stddev * 0.5f * 1.0f));
      
      for ( float f : floatSizes ) {
        if (f > minusOneStddev) {
          largerSizes.push_back(f);
        }
      }
      if (largerSizes.size() == 0) {
        // Include just the first delta when all are the same
        assert(floatSizes.size() > 0);
        largerSizes.push_back(floatSizes[0]);
      }
      
      if (debug) {
        cout << "num larger " << largerSizes.size() << endl;
        for ( float f : largerSizes ) {
          cout << "larger " << f << endl;
        }
      }
      
      // Generate an average slope that considers the first N
      // counts in the histogram.
      
      sample_mean(largerSizes, &mean);
      mean *= -1.0f;
      
      if (debug) {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "resulting ave slope %0.4f for N = %d elements", mean, (int)largerSizes.size());
        cout << (char*)buffer << endl;
      }
      
      // Detect if this histogram indicates a significant negative slope or if it looks like
      // a gradient into a flat region.
      
      if (mean < -30.0f) {
        isOutsideSmoothRegion = true;
        
        if (debug) {
          cout << "isOutsideSmoothRegion = true" << endl;
        }
      }
    }
    
    // Generate vector from quant centers of mass
    
    if ((1)) {
      vector<uint32_t> insideToOutsideVec = generateVector(insideQuantPixel, outsideQuantPixel);
      
      if (debugDumpImages) {
        // Dump points generated on the line that make up the vector
        
        int numPoints = (int) insideToOutsideVec.size();
        
        Mat qtableOutputMat = Mat(numPoints, 1, CV_8UC3);
        qtableOutputMat = (Scalar) 0;
        
        for (int i = 0; i < numPoints; i++) {
          uint32_t pixel = insideToOutsideVec[i];
          Vec3b vec = PixelToVec3b(pixel);
          qtableOutputMat.at<Vec3b>(i, 0) = vec;
        }
        
        std::stringstream fnameStream;
        
        fnameStream << "srm" << "_tag_" << tag << "_to_" << mostCommonOtherTag << "_quant3_inside_outside_quant_vec" << ".png";
        
        string fname = fnameStream.str();
        
        imwrite(fname, qtableOutputMat);
        cout << "wrote " << fname << endl;
        cout << "";
      }
    }
    
    // Determine where to place the book end pixel. In the smooth outside region case, place
    // the book end at close to the flat region as possible. In the case of a outside that
    // is not smooth, choose a distance 1/2 the way from the 3rd vector coordinate.

    uint32_t bookEndPixel;
    
    if (isOutsideSmoothRegion) {
      vector<uint32_t> insideToOutsideVec = generateVector(insideQuantPixel, outsideExactPixel);
      
      if (debugDumpImages) {
        // Dump points generated on the line that make up the vector
        
        int numPoints = (int) insideToOutsideVec.size();
        
        Mat qtableOutputMat = Mat(numPoints, 1, CV_8UC3);
        qtableOutputMat = (Scalar) 0;
        
        for (int i = 0; i < numPoints; i++) {
          uint32_t pixel = insideToOutsideVec[i];
          Vec3b vec = PixelToVec3b(pixel);
          qtableOutputMat.at<Vec3b>(i, 0) = vec;
        }
        
        std::stringstream fnameStream;
        
        fnameStream << "srm" << "_tag_" << tag << "_to_" << mostCommonOtherTag << "_quant3_inside_outside_exact_vec" << ".png";
        
        string fname = fnameStream.str();
        
        imwrite(fname, qtableOutputMat);
        cout << "wrote " << fname << endl;
        cout << "";
      }
      
      int secondToLastOffset = 0;
      if (insideToOutsideVec.size() > 1) {
        secondToLastOffset = (int)insideToOutsideVec.size() - 2;
      }
      
      bookEndPixel = insideToOutsideVec[secondToLastOffset];
      assert(outsideExactPixel != bookEndPixel);
    } else {
      // Choose the 3rd quant pixel point as the bookend
      
      outsideExactPixel = outsideQuantPixel;
      
      bookEndPixel = filteredQuantVector[filteredQuantVector.size() - 1]; // last pixel in quant vector
    }
    
    // Generate a colortable that contains the exact pixel, the next pixel in the vector
    // and then the 2 midpoints in the quant vector along with the quant center of mass.
    
    vector<uint32_t> generatedQuantVector;
    
    generatedQuantVector.push_back(outsideExactPixel);
    generatedQuantVector.push_back(bookEndPixel);
    
    if (debug) {
      fprintf(stdout, "outsideExactPixel = 0x%08X\n", outsideExactPixel);
      fprintf(stdout, "bookEndPixel = 0x%08X\n", bookEndPixel);
    }
    
    for ( uint32_t pixel : filteredQuantVector ) {
      if (pixel == outsideExactPixel) {
        continue;
      }
      if (pixel == bookEndPixel) {
        continue;
      }
      generatedQuantVector.push_back(pixel);
      
      if (debug) {
        fprintf(stdout, "colortable pixel = 0x%08X\n", pixel);
      }
    }
    
    generatedQuantVector.push_back(insideQuantPixel);
    
    if (debug) {
      fprintf(stdout, "insideQuantPixel = 0x%08X\n", insideQuantPixel);
    }
    
    // Print the generated colortable
    
    if (debug) {
      int i = 0;
      for ( uint32_t pixel : generatedQuantVector ) {
        fprintf(stdout, "generated[%5d] = 0x%08X\n", i, pixel);
        i += 1;
      }
    }
    
    // Sort the generated colortable in terms of the outsideExactPixel
    // as the starting point of the sort iteration.
    
    {
      vector<uint32_t> sortedOffsets = generate_cluster_walk_on_center_dist(generatedQuantVector, outsideExactPixel);

      vector<uint32_t> sortedColortable;
      
      for (int i = 0; i < numActualClusters; i++) {
        int si = (int) sortedOffsets[i];
        uint32_t pixel = generatedQuantVector[si];
        sortedColortable.push_back(pixel);
      }
      
      generatedQuantVector = sortedColortable;
    }
    
    // Print the generated colortable
    
    if (debug) {
      for ( uint32_t pixel : generatedQuantVector ) {
        fprintf(stdout, "sorted generated 0x%08X\n", pixel);
      }
    }
    
    if (debugDumpImages) {
      // Dump points generated on the line that make up the vector
      
      int numPoints = (int) generatedQuantVector.size();
      
      Mat qtableOutputMat = Mat(numPoints, 1, CV_8UC3);
      qtableOutputMat = (Scalar) 0;
      
      for (int i = 0; i < numPoints; i++) {
        uint32_t pixel = generatedQuantVector[i];
        Vec3b vec = PixelToVec3b(pixel);
        qtableOutputMat.at<Vec3b>(i, 0) = vec;
      }
      
      std::stringstream fnameStream;
      
      fnameStream << "srm" << "_tag_" << tag << "_to_" << mostCommonOtherTag << "_quant3_inside_outside_generated_vec" << ".png";
      
      string fname = fnameStream.str();
      
      imwrite(fname, qtableOutputMat);
      cout << "wrote " << fname << endl;
      cout << "";
    }
    
    // Rerun quant logic with the generated table
    
    numActualClusters = (int) generatedQuantVector.size();
    
    colortable = new uint32_t[numActualClusters];
    
    for (int i = 0; i < numActualClusters; i++) {
      uint32_t pixel = generatedQuantVector[i];
      colortable[i] = pixel;
    }
    
    map_colors_mps(inPixels, numPixels, outPixels, colortable, numActualClusters);
    
    delete [] colortable;
    
    // Dump output, which is the input colors run through the color table
    
    if (debugDumpImages) {
      Mat tmpResultImg = inputImg.clone();
      tmpResultImg = Scalar(0,0,0xFF);
      
      for ( int i = 0; i < numPixels; i++ ) {
        Coord c = combinedCoords[i];
        uint32_t pixel = outPixels[i];
        Vec3b vec = PixelToVec3b(pixel);
        tmpResultImg.at<Vec3b>(c.y, c.x) = vec;
      }
      
      {
        std::stringstream fnameStream;
        fnameStream << "srm" << "_tag_" << tag << "_to_" << mostCommonOtherTag << "_quant3_generated_output" << ".png";
        string fname = fnameStream.str();
        
        imwrite(fname, tmpResultImg);
        cout << "wrote " << fname << endl;
      }
    }

    // Set inside/outside flagsfor the whole region based on the gradient vector for
    // this specific pair of regions.
    
    unordered_map<uint32_t, InsideOutsideRecord> pixelToInside;
    
    for ( uint32_t pixel : generatedQuantVector ) {
      InsideOutsideRecord &inOutRef = pixelToInside[pixel];
      
      if (pixel == outsideExactPixel) {
        inOutRef.isInside = false;
      } else {
        inOutRef.isInside = true;
      }
    }
    
    if (debugDumpImages)
    {
      std::stringstream fnameStream;
      
      fnameStream << "srm" << "_tag_" << tag << "_to_" << mostCommonOtherTag << "_quant3_table_votes" << ".png";
      string fname = fnameStream.str();
      
      // Write image that contains one color in each row in a N x 2 image
      
      int numColortableEntries = (int) generatedQuantVector.size();
      
      Mat qtableOutputMat = Mat(numColortableEntries, 2, CV_8UC3);
      qtableOutputMat = (Scalar) 0;
      
      for (int i = 0; i < numColortableEntries; i++) {
        uint32_t pixel = generatedQuantVector[i];
        Vec3b vec = PixelToVec3b(pixel);
        qtableOutputMat.at<Vec3b>(i, 0) = vec;
        
        bool isInside = pixelToInside[pixel].isInside;
        
        if (isInside) {
          vec = Vec3b(0xFF, 0xFF, 0xFF);
        } else {
          vec = Vec3b(0, 0, 0);
        }
        
        qtableOutputMat.at<Vec3b>(i, 1) = vec;
      }
      
      imwrite(fname, qtableOutputMat);
      cout << "wrote " << fname << endl;
      cout << "";
    }
    
    const bool debugOnOff = false;
    
    for ( int i = 0; i < numPixels; i++ ) {
      Coord c = combinedCoords[i];
      uint32_t quantPixel = outPixels[i];
      
#if defined(DEBUG)
      assert(pixelToInside.count(quantPixel));
#endif // DEBUG
      bool isInside = pixelToInside[quantPixel].isInside;
      
      if (isInside) {
        mask.at<uint8_t>(c.y, c.x) = 0xFF;
        
        if (debug && debugOnOff) {
          printf("pixel 0x%08X at (%5d,%5d) is marked on (inside)\n", quantPixel, c.x, c.y);
        }
      } else {
        if (debug && debugOnOff) {
          printf("pixel 0x%08X at (%5d,%5d) is marked off (outside)\n", quantPixel, c.x, c.y);
        }
      }
    }

    // continue to next region to region pair
  }
  
  // After each region segment has been processed, it is possible that the quant processing logic
  // would have selected certain pixels to active that are actually not 8 connected to the rest
  // of the pixels.
  
  if ((1)) {
    vector<Point> locations;
    findNonZero(mask, locations);
    
    vector<Coord> coords;
    
    for ( Point p : locations ) {
      Coord c(p.x, p.y);
      coords.push_back(c);
    }
    
    int32_t originX, originY, regionWidth, regionHeight;
    bbox(originX, originY, regionWidth, regionHeight, coords);
    Rect roiRect(originX, originY, regionWidth, regionHeight);

    Mat outDistMat;
    outDistMat = Scalar(0);
    
    Coord roiCenter = findRegionCenter(mask, roiRect, outDistMat, tag);

    Coord regionCenter(originX + roiCenter.x, originY + roiCenter.y);
    
    Point2i center2i(regionCenter.x, regionCenter.y);
    
    if (debugDumpImages) {
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_pre_flood_region_mask" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, mask);
      cout << "wrote " << fname << endl;
      cout << "";
    }
    
    Mat invMaskMat = mask.clone();
    binMatInvert(invMaskMat);

    if (debugDumpImages) {
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_pre_flood_inv_region_mask" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, invMaskMat);
      cout << "wrote " << fname << endl;
      cout << "";
    }
    
    invMaskMat.at<uint8_t>(center2i.y, center2i.x) = 0xFF;
    
    Mat outFloodMat = mask.clone();
    outFloodMat = Scalar(0);
    
    int numPixelsFilled = floodFillMask(invMaskMat, outFloodMat, center2i, 8);
    assert(numPixelsFilled > 0);
    
    if (debugDumpImages && false) {
      Mat tmpResultImg = mask.clone();
      
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_post_flood_region_mask" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, tmpResultImg);
      cout << "wrote " << fname << endl;
      cout << "";
    }
    
    if (debugDumpImages) {
      Mat tmpResultImg = outFloodMat.clone();
      
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_post_flood_out_mask" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, tmpResultImg);
      cout << "wrote " << fname << endl;
      cout << "";
    }
    
    // Any pixel that is on in mask but off in outFloodMat should be turned
    // off in mask since this pixel was not included in the flood fill.
    
    assert(mask.size() == outFloodMat.size());
    
    // Do dump that shows any pixels that should avtually be off because
    // they were not included in the flood mask.
    
    if (debugDumpImages) {
      Mat tmpResultImg = outFloodMat.clone();
      int numRemoved = 0;
      
      // Loop over each value in outFloodMat and set the output
      // to 0xFF only in the case where the mask is on and the
      // flood mask is off.
      
      for_each_byte(tmpResultImg, mask,
                    [&numRemoved](uint8_t *floodBPtr, const uint8_t *maskBPtr)->void {
                      uint8_t floodB = *floodBPtr;
                      const uint8_t maskB = *maskBPtr;
                      if (maskB && !floodB) {
                        *floodBPtr = 0xFF;
                        numRemoved++;
                        //fprintf(stdout, "maskB %3d, floodB %3d -> %3d\n", maskB, floodB, *floodBPtr);
                      } else {
                        //fprintf(stdout, "maskB %3d, floodB %3d -> %3d\n", maskB, floodB, floodB);
                      }
                    });
      
      {
        std::stringstream fnameStream;
        fnameStream << "srm" << "_tag_" << tag << "_post_flood_out_mask_removed" << ".png";
        string fname = fnameStream.str();
        
        imwrite(fname, tmpResultImg);
        cout << "wrote " << fname << endl;
        cout << "";
      }
      
      if (debug) {
        cout << "numRemoved " << numRemoved << endl;
      }
    }
    
    // Optimal impl that iterates over each Mat result and calls lambda with pointers
    
    for_each_byte(mask, outFloodMat,
                     [](uint8_t *maskBPtr, const uint8_t *floodBPtr)->void {
                       uint8_t maskB = *maskBPtr;
                       uint8_t floodB = *floodBPtr;
                       if (maskB && !floodB) {
                         *maskBPtr = 0;
                         //fprintf(stdout, "maskB %3d, floodB %3d -> %3d\n", maskB, floodB, *maskBPtr);
                       } else {
                         //fprintf(stdout, "maskB %3d, floodB %3d -> %3d\n", maskB, floodB, floodB);
                       }
                       return;
                     });
  }
  
  if (debug) {
    cout << "return captureRegion" << endl;
  }
  
  return;
}

// In this case the pixels are from a very small colortable or all the entries
// are so close together that one can assume that the colors are very simple
// and can be represented by quant that uses the original colors as a colortable.

// FIXME: why both regionCoords (region including expand around) and the
// srm region as coords ?

void
captureVeryCloseRegion(SuperpixelImage &spImage,
                  const Mat & inputImg,
                  const Mat & srmTags,
                  int32_t tag,
                  int blockWidth,
                  int blockHeight,
                  int superpixelDim,
                  Mat &mask,
                  const vector<Coord> &regionCoords,
                  const vector<Coord> &srmRegionCoords,
                  int estNumColors)
{
  const bool debug = true;
  const bool debugDumpImages = true;
  
  if (debug) {
    cout << "captureVeryCloseRegion" << endl;
  }
  
  int numPixels = (int)regionCoords.size();
  
  assert(estNumColors > 0);
  uint32_t numActualClusters = estNumColors;
  
  uint32_t *colortable = new uint32_t[numActualClusters];
  uint32_t *inPixels = new uint32_t[numPixels];
  uint32_t *outPixels = new uint32_t[numPixels];
  
  for ( int i = 0; i < numPixels; i++ ) {
    Coord c = regionCoords[i];
    
    Vec3b vec = inputImg.at<Vec3b>(c.y, c.x);
    uint32_t pixel = Vec3BToUID(vec);
    inPixels[i] = pixel;
  }
  
  // In this case the pixels are from a very small colortable or all the entries
  // are so close together that one can assume that the colors are very simple
  // and can be represented by quant that uses the original colors as a colortable.
  
  // Vote inside/outside for each pixel after we know what colortable entry a specific
  // pixel is associated with.
  
  unordered_map<uint32_t, uint32_t> mapSrcPixelToSRMTag;
  
  // Iterate over the coords and gather up srmTags that correspond
  // to the area indicated by tag
  
  for ( Coord c : srmRegionCoords ) {
    Vec3b srmVec = srmTags.at<Vec3b>(c.y, c.x);
    uint32_t srmTag = Vec3BToUID(srmVec);
    
    Vec3b vec = inputImg.at<Vec3b>(c.y, c.x);
    uint32_t pixel = Vec3BToUID(vec);
    
    mapSrcPixelToSRMTag[pixel] = srmTag;
  }
  
  for ( auto &pair : mapSrcPixelToSRMTag ) {
    uint32_t pixel = pair.first;
    uint32_t srmTag = pair.second;
    printf("pixel->srmTag table[0x%08X] = 0x%08X\n", pixel, srmTag);
  }
  
  if (debugDumpImages) {
    Mat tmpResultImg = inputImg.clone();
    tmpResultImg = Scalar(0,0,0);
    
    for ( Coord c : srmRegionCoords ) {
      Vec3b srmVec = srmTags.at<Vec3b>(c.y, c.x);
      //uint32_t srmTag = Vec3BToUID(srmVec);
      
      //Vec3b vec = inputImg.at<Vec3b>(c.y, c.x);
      //uint32_t pixel = Vec3BToUID(vec);
      
      tmpResultImg.at<Vec3b>(c.y, c.x) = srmVec;
    }
    
    {
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_srm_region_tags" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, tmpResultImg);
      cout << "wrote " << fname << endl;
      cout << "";
    }
  }
  
  // Generate cluster centers based on the indicated number of clusters N
  
  int allPixelsUnique = 0;
  
  quant_recurse(numPixels, inPixels, outPixels, &numActualClusters, colortable, allPixelsUnique );
  
  // Write quant output where each original pixel is replaced with the closest
  // colortable entry.
  
  if (debugDumpImages) {
    Mat tmpResultImg = inputImg.clone();
    tmpResultImg = Scalar(0,0,0xFF);
    
    for ( int i = 0; i < numPixels; i++ ) {
      Coord c = regionCoords[i];
      uint32_t pixel = outPixels[i];
      Vec3b vec = PixelToVec3b(pixel);
      tmpResultImg.at<Vec3b>(c.y, c.x) = vec;
    }
    
    {
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_quant_inside_output" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, tmpResultImg);
      cout << "wrote " << fname << endl;
    }
    
    {
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_quant_inside_table" << ".png";
      string fname = fnameStream.str();
      
      dumpQuantTableImage(fname, inputImg, colortable, numActualClusters);
    }
    
    vector<uint32_t> colortableVec;
    
    for (int i = 0; i < numActualClusters; i++) {
      uint32_t pixel = colortable[i];
      colortableVec.push_back(pixel);
    }
    
    // Add phony entry for Red (the mask color)
    colortableVec.push_back(0x00FF0000);
    
    Mat quantOffsetsMat = mapQuantPixelsToColortableIndexes(tmpResultImg, colortableVec, true);
    
    {
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_quant_inside_offsets" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, quantOffsetsMat);
      cout << "wrote " << fname << endl;
    }
  }
  
  // FIXME: make sure that colortableVec passed to insideOutsideTest() is actually sorted
  
  vector<uint32_t> sortedColortable;
  
  // Copy cluster colors into colortable after resorting
  {
    for (int i = 0; i < numActualClusters; i++) {
      uint32_t pixel = colortable[i];
      sortedColortable.push_back(pixel);
    }
    
    vector<uint32_t> sortedOffsets = generate_cluster_walk_on_center_dist(sortedColortable);
    
    sortedColortable.clear();
    
    for (int i = 0; i < numActualClusters; i++) {
      int si = (int) sortedOffsets[i];
      uint32_t pixel = colortable[si];
      sortedColortable.push_back(pixel);
    }
  }
  
  unordered_map<uint32_t, InsideOutsideRecord> pixelToInside;
  
  insideOutsideTest(inputImg.rows, inputImg.cols, srmRegionCoords, tag, regionCoords, outPixels, sortedColortable, pixelToInside);
  
  // Each pixel in the input is now mapped to a boolean condition that
  // indicates if that pixel is inside or outside the shape.
  
  const bool debugOnOff = false;
  
  for ( int i = 0; i < numPixels; i++ ) {
    Coord c = regionCoords[i];
    uint32_t quantPixel = outPixels[i];
    
#if defined(DEBUG)
    assert(pixelToInside.count(quantPixel));
#endif // DEBUG
    bool isInside = pixelToInside[quantPixel].isInside;
    
    if (isInside) {
      mask.at<uint8_t>(c.y, c.x) = 0xFF;
      
      if (debug && debugOnOff) {
        printf("pixel 0x%08X at (%5d,%5d) is marked on (inside)\n", quantPixel, c.x, c.y);
      }
    } else {
      if (debug && debugOnOff) {
        printf("pixel 0x%08X at (%5d,%5d) is marked off (outside)\n", quantPixel, c.x, c.y);
      }
    }
  }
  
  delete [] colortable;
  delete [] inPixels;
  delete [] outPixels;
  
  if (debug) {
    cout << "return captureVeryCloseRegion" << endl;
  }
}

// This implementation is invoked when there is significant pixel spread between
// entries in a colortable. The logic uses custer quant to determine a best
// inside vs outside testing by looking for known simple vectors between major
// colors.

void
captureNotCloseRegion(SuperpixelImage &spImage,
                       const Mat & inputImg,
                       const Mat & srmTags,
                       int32_t tag,
                       int blockWidth,
                       int blockHeight,
                       int superpixelDim,
                       Mat &mask,
                       const vector<Coord> &regionCoords,
                       const vector<Coord> &srmRegionCoords,
                       int estNumColors,
                      const Mat &blockBasedQuantMat)
{
  const bool debug = true;
  const bool debugDumpImages = true;
  
  if (debug) {
    cout << "captureNotCloseRegion " << tag << endl;
  }
  
  int numPixels = (int)regionCoords.size();
  
  uint32_t *inPixels = new uint32_t[numPixels];
  uint32_t *outPixels = new uint32_t[numPixels];
  
  for ( int i = 0; i < numPixels; i++ ) {
    Coord c = regionCoords[i];
    
    Vec3b vec = inputImg.at<Vec3b>(c.y, c.x);
    uint32_t pixel = Vec3BToUID(vec);
    inPixels[i] = pixel;
  }
  
//  unordered_map<Coord, HistogramForBlock> blockMap;
//  
//  Mat blockMat =
//  genHistogramsForBlocks(inputImg, blockMap, blockWidth, blockHeight, superpixelDim);
  
  // Generate mask Mat that is the same dimensions as blockMat but contains just one
  // byte for each pixel and acts as a mask. The white pixels indicate the blocks
  // that are included in the mask.
  
  Mat blockMaskMat(blockBasedQuantMat.size(), CV_8UC1);
  blockMaskMat = (Scalar) 0;
  
  for ( Coord c : regionCoords ) {
    // Convert (X,Y) to block (X,Y)
    
    int blockX = c.x / superpixelDim;
    int blockY = c.y / superpixelDim;
    
    blockMaskMat.at<uint8_t>(blockY, blockX) = 0xFF;
  }
  
  if (debugDumpImages) {
    std::stringstream fnameStream;
    fnameStream << "srm" << "_tag_" << tag << "_block_mask" << ".png";
    string fname = fnameStream.str();
    
    imwrite(fname, blockMaskMat);
    cout << "wrote " << fname << endl;
  }
  
  // Examine the region mask and check for the case of splay pixels
  // that extend away from known solid color regions but are actually
  // gradients.
  
  // Count neighbors that share a quant pixel value after conversion to blocks
  
  unordered_map<uint32_t, uint32_t> pixelToNumVotesMap;
  
  vote_for_identical_neighbors(pixelToNumVotesMap, blockBasedQuantMat, blockMaskMat);
  
  vector<uint32_t> sortedPixelKeys = sort_keys_by_count(pixelToNumVotesMap, true);
  
  if (debug) {
    for ( uint32_t pixel : sortedPixelKeys ) {
      uint32_t count = pixelToNumVotesMap[pixel];
      fprintf(stdout, "0x%08X (%8d) -> %5d\n", pixel, pixel, count);
    }
    fprintf(stdout, "done\n");
  }
  
  // Instead of a stddev type of approach, use peak logic to examine the counts
  // and select the peaks in the distrobution.
  
  vector<uint32_t> sortedOffsets = generate_cluster_walk_on_center_dist(sortedPixelKeys);
  
  int numPoints = (int) sortedOffsets.size();
  
  Mat sortedQtableOutputMat = Mat(numPoints, 1, CV_8UC3);
  sortedQtableOutputMat = (Scalar) 0;
  
  vector<uint32_t> sortedColortable;
  
  for (int i = 0; i < numPoints; i++) {
    int si = (int) sortedOffsets[i];
    uint32_t pixel = sortedPixelKeys[si];
    Vec3b vec = PixelToVec3b(pixel);
    sortedQtableOutputMat.at<Vec3b>(i, 0) = vec;
    
    sortedColortable.push_back(pixel);
  }
  
  if (debug) {
    for ( uint32_t pixel : sortedColortable ) {
      uint32_t count = pixelToNumVotesMap[pixel];
      fprintf(stdout, "0x%08X (%8d) -> %5d\n", pixel, pixel, count);
    }
    fprintf(stdout, "done\n");
  }
  
  // Dump sorted pixel data as a CSV file, with int value and hex rep of int value for readability
  
  if (debugDumpImages) {
    std::stringstream fnameStream;
    fnameStream << "srm" << "_tag_" << tag << "_quant_table_sorted" << ".csv";
    string fname = fnameStream.str();
    
    FILE *fout = fopen(fname.c_str(), "w+");
    
    for ( uint32_t pixel : sortedColortable ) {
      uint32_t count = pixelToNumVotesMap[pixel];
      uint32_t pixelNoAlpha = pixel & 0x00FFFFFF;
      fprintf(fout, "%d,0x%08X,%d\n", pixelNoAlpha, pixelNoAlpha, count);
    }
    
    fclose(fout);
    cout << "wrote " << fname << endl;
  }
  
  if (debugDumpImages) {
    std::stringstream fnameStream;
    fnameStream << "srm" << "_tag_" << tag << "_block_mask_sorted" << ".png";
    string filename = fnameStream.str();
    
    char *outQuantTableFilename = (char*) filename.c_str();
    imwrite(outQuantTableFilename, sortedQtableOutputMat);
    cout << "wrote " << outQuantTableFilename << endl;
  }
  
  // Use peak detection logic to examine the 1D histogram in sorted order so as to find the
  // peaks in the distribution.
  
  vector<uint32_t> peakPixels = gatherPeakPixels(sortedColortable, pixelToNumVotesMap);
  
  int N = (int) peakPixels.size();
  
  // Min N must be at least 1 at this point
  
  if (N < 2) {
    N = 2;
  }
  
  N = N * 4;
  
  /*
   
   // Estimate N
   
   // Choice of N for splitting the masked area. Need 1 for the surrounding area, possibly
   // more if background is more than 1 color. But, need to select the other "target" color
   // to split from the background by looking at the density of the colors in (X,Y) terms.
   // For example, a dense patch of green should be seen as +1 over a surrounding gradient
   // even if there are more colors in the gradient but they are spread out.
   
   float mean, stddev;
   
   vector<float> floatSizes;
   
   for ( uint32_t pixel : sortedPixelKeys ) {
   uint32_t count = pixelToNumVotesMap[pixel];
   
   floatSizes.push_back(count);
   }
   
   sample_mean(floatSizes, &mean);
   sample_mean_delta_squared_div(floatSizes, mean, &stddev);
   
   if (1) {
   char buffer[1024];
   
   snprintf(buffer, sizeof(buffer), "mean %0.4f stddev %0.4f", mean, stddev);
   cout << (char*)buffer << endl;
   
   snprintf(buffer, sizeof(buffer), "1 stddev %0.4f", (mean + (stddev * 0.5f * 1.0f)));
   cout << (char*)buffer << endl;
   
   snprintf(buffer, sizeof(buffer), "2 stddev %0.4f", (mean + (stddev * 0.5f * 2.0f)));
   cout << (char*)buffer << endl;
   
   snprintf(buffer, sizeof(buffer), "3 stddev %0.4f", (mean + (stddev * 0.5f * 3.0f)));
   cout << (char*)buffer << endl;
   
   snprintf(buffer, sizeof(buffer), "-1 stddev %0.4f", (mean - (stddev * 0.5f * 1.0f)));
   cout << (char*)buffer << endl;
   
   snprintf(buffer, sizeof(buffer), "-2 stddev %0.4f", (mean - (stddev * 0.5f * 2.0f)));
   cout << (char*)buffer << endl;
   }
   
   // 1 for the background
   // 1 for the most common color group
   
   int N = 1;
   
   // Anything larger than 1 standard dev is likely to be a cluster of alike pixels
   
   float upOneStddev = (mean + (stddev * 0.5f * 1.0f));
   
   int countAbove = 0;
   
   for ( float floatSize : floatSizes ) {
   if (floatSize >= upOneStddev) {
   countAbove += 1;
   }
   }
   
   N += countAbove;
   //N = N * 2;
   N = N * 4;
   
   //          fprintf(stdout, "N = %5d\n", N);
   
   */
  
  // Generate quant based on the input
  
  const int numClusters = N;
  
  if (debug) {
    cout << "numClusters detected as " << numClusters << endl;
  }
  
  uint32_t *colortable = new uint32_t[numClusters];
  
  uint32_t numActualClusters = numClusters;
  
  int allPixelsUnique = 0;
  
  quant_recurse(numPixels, inPixels, outPixels, &numActualClusters, colortable, allPixelsUnique );
  
  // Write quant output where each original pixel is replaced with the closest
  // colortable entry.
  
  if (debugDumpImages) {
    Mat tmpResultImg = inputImg.clone();
    tmpResultImg = Scalar(0,0,0xFF);
    
    for ( int i = 0; i < numPixels; i++ ) {
      Coord c = regionCoords[i];
      uint32_t pixel = outPixels[i];
      Vec3b vec = PixelToVec3b(pixel);
      tmpResultImg.at<Vec3b>(c.y, c.x) = vec;
    }
    
    {
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_quant_output" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, tmpResultImg);
      cout << "wrote " << fname << endl;
    }
    
    // table
    
    {
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_quant_table" << ".png";
      string fname = fnameStream.str();
      
      dumpQuantTableImage(fname, inputImg, colortable, numActualClusters);
    }
  }
  
  // Generate color sorted clusters
  
  {
    vector<uint32_t> clusterCenterPixels;
    
    for ( int i = 0; i < numActualClusters; i++) {
      uint32_t pixel = colortable[i];
      clusterCenterPixels.push_back(pixel);
    }
    
#if defined(DEBUG)
    if ((1)) {
      unordered_map<uint32_t, uint32_t> seen;
      
      for ( int i = 0; i < numActualClusters; i++ ) {
        uint32_t pixel;
        pixel = colortable[i];
        
        if (seen.count(pixel) > 0) {
        } else {
          // Note that only the first seen index is retained, this means that a repeated
          // pixel value is treated as a dup.
          
          seen[pixel] = i;
        }
      }
      
      int numQuantUnique = (int)seen.size();
      assert(numQuantUnique == numActualClusters);
    }
#endif // DEBUG
    
    if (debug) {
      fprintf(stdout, "numClusters %5d : numActualClusters %5d \n", numClusters, numActualClusters);
      
      unordered_map<uint32_t, uint32_t> seen;
      
      for ( int i = 0; i < numActualClusters; i++ ) {
        uint32_t pixel;
        pixel = colortable[i];
        
        if (seen.count(pixel) > 0) {
          fprintf(stdout, "cmap[%3d] = 0x%08X (DUP of %d)\n", i, pixel, seen[pixel]);
        } else {
          fprintf(stdout, "cmap[%3d] = 0x%08X\n", i, pixel);
          
          // Note that only the first seen index is retained, this means that a repeated
          // pixel value is treated as a dup.
          
          seen[pixel] = i;
        }
      }
      
      fprintf(stdout, "cmap contains %3d unique entries\n", (int)seen.size());
      
      int numQuantUnique = (int)seen.size();
      
      assert(numQuantUnique == numActualClusters);
    }
    
    vector<uint32_t> sortedOffsets = generate_cluster_walk_on_center_dist(clusterCenterPixels);
    
    // Once cluster centers have been sorted by 3D color cube distance, emit as PNG
    
    Mat sortedQtableOutputMat = Mat(numActualClusters, 1, CV_8UC3);
    sortedQtableOutputMat = (Scalar) 0;
    
    vector<uint32_t> sortedColortable;
    
    for (int i = 0; i < numActualClusters; i++) {
      int si = (int) sortedOffsets[i];
      uint32_t pixel = colortable[si];
      Vec3b vec = PixelToVec3b(pixel);
      sortedQtableOutputMat.at<Vec3b>(i, 0) = vec;
      
      sortedColortable.push_back(pixel);
    }
    
    if (debugDumpImages)
    {
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_quant_table_sorted" << ".png";
      string filename = fnameStream.str();
      
      char *outQuantTableFilename = (char*) filename.c_str();
      imwrite(outQuantTableFilename, sortedQtableOutputMat);
      cout << "wrote " << outQuantTableFilename << endl;
    }
    
    // Map pixels to sorted colortable offset
    
    unordered_map<uint32_t, uint32_t> pixel_to_sorted_offset;
    
    assert(numActualClusters <= 256);
    
    for (int i = 0; i < numActualClusters; i++) {
      int si = (int) sortedOffsets[i];
      uint32_t pixel = colortable[si];
      pixel_to_sorted_offset[pixel] = si;
    }
    
    Mat sortedQuantOutputMat = inputImg.clone();
    sortedQuantOutputMat = Scalar(0,0,0xFF);
    
    for ( int i = 0; i < numPixels; i++ ) {
      Coord c = regionCoords[i];
      uint32_t pixel = outPixels[i];
      
      assert(pixel_to_sorted_offset.count(pixel) > 0);
      uint32_t offset = pixel_to_sorted_offset[pixel];
      
      if ((debug) && false) {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "for (%4d,%4d) pixel is %d -> offset %d\n", c.x, c.y, pixel, offset);
        cout << buffer;
      }
      
      assert(offset <= 256);
      uint32_t grayscalePixel = (offset << 16) | (offset << 8) | offset;
      
      Vec3b vec = PixelToVec3b(grayscalePixel);
      sortedQuantOutputMat.at<Vec3b>(c.y, c.x) = vec;
    }
    
    if (debugDumpImages)
    {
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_quant_table_offsets" << ".png";
      string filename = fnameStream.str();
      
      char *outQuantFilename = (char*)filename.c_str();
      imwrite(outQuantFilename, sortedQuantOutputMat);
      cout << "wrote " << outQuantFilename << endl;
    }
  }
  
  // Examine points in 3D space by gathering center of mass coordinates
  // for (B,G,R) values and emit as an image that contains the peak
  // points and the center of mass point.
  
  if ((1)) {
    vector<Vec3b> clusterCenterPoints;
    
    for (int i = 0; i < numActualClusters; i++) {
      uint32_t pixel = colortable[i];
      Vec3b vec = PixelToVec3b(pixel);
      clusterCenterPoints.push_back(vec);
    }
    
    Vec3b centerOfMass = centerOfMass3d(clusterCenterPoints);
    
    if (debugDumpImages) {
      
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_peak_center_of_mass" << ".png";
      string fname = fnameStream.str();
      
      // Write image that contains one color in each row in a N x 2 image
      
      int numPoints = (int) peakPixels.size() + 1;
      
      Mat outputMat = Mat(numPoints, 1, CV_8UC3);
      outputMat = (Scalar) 0;
      
      int i = 0;
      
      for ( uint32_t pixel : peakPixels ) {
        pixel = pixel & 0x00FFFFFF;
        if (debug) {
          printf("peak[%5d] = 0x%08X\n", i, pixel);
        }
        Vec3b vec = PixelToVec3b(pixel);
        outputMat.at<Vec3b>(i, 0) = vec;
        i += 1;
      }
      
      // Add center of mass pixel
      
      outputMat.at<Vec3b>(i, 0) = centerOfMass;
      
      if (debug) {
        printf("peak com = 0x%02X%02X%02X\n", centerOfMass[0], centerOfMass[1], centerOfMass[2]);
      }
      
      imwrite(fname, outputMat);
      cout << "wrote " << fname << endl;
    }
  }
  
  // Pass all input points to the fitLine() method in attempt to get a best
  // fit line.
  
  if ((1)) {
    vector<Vec3b> allRegionPoints;
    
    for (int i = 0; i < numPixels; i++) {
      uint32_t pixel = inPixels[i];
      Vec3b vec = PixelToVec3b(pixel);
      allRegionPoints.push_back(vec);
    }
    
    vector<float> lineVec;
    vector<Vec3b> linePoints;
    
    int distType = CV_DIST_L12;
    //int distType = CV_DIST_FAIR;
    
    fitLine(allRegionPoints, lineVec, distType, 0, 0.1, 0.1);
    
    // In case of 3D fitting (vx, vy, vz, x0, y0, z0)
    // where (vx, vy, vz) is a normalized vector collinear to the line
    // and (x0, y0, z0) is a point on the line.
    
    Vec3b centerOfMass(round(lineVec[3]), round(lineVec[4]), round(lineVec[5]));
    
    for (int i = 0; i < 300; i++) {
      linePoints.push_back(centerOfMass);
      linePoints.push_back(centerOfMass); // Dup COM
    }
    
    Vec3f colinearF(lineVec[0], lineVec[1], lineVec[2]);
    Vec3f centerOfMassF(centerOfMass[0], centerOfMass[1], centerOfMass[2]);
    
    // Add another point away from the com
    
    int scalar = 30; // num points in (x,y,z) space
    
    Vec3f comPlusUnitF = centerOfMassF + (colinearF * scalar);
    Vec3b comPlusUnit(round(comPlusUnitF[0]), round(comPlusUnitF[1]), round(comPlusUnitF[2]));
    
    Vec3f comMinusUnitF = centerOfMassF - (colinearF * scalar);
    Vec3b comMinusUnit(round(comMinusUnitF[0]), round(comMinusUnitF[1]), round(comMinusUnitF[2]));
    
    if (debug) {
      cout << "centerOfMassF " << centerOfMassF << endl;
      cout << "colinearF " << colinearF << endl;
      cout << "comPlusUnitF " << comPlusUnitF << endl;
      cout << "comMinusUnitF " << comMinusUnitF << endl;
      cout << "comPlusUnit " << comPlusUnit << endl;
      cout << "comMinusUnit " << comMinusUnit << endl;
    }
    
    for (int i = 0; i < 300; i++) {
      linePoints.push_back(comPlusUnit);
      linePoints.push_back(comMinusUnit);
    }
    
    if (debugDumpImages) {
      
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_all_center_of_mass" << ".png";
      string fname = fnameStream.str();
      
      // Write image that contains one color in each row in a N x 2 image
      
      int numPoints = (int) allRegionPoints.size() + (int) linePoints.size();
      
      Mat outputMat = Mat(numPoints, 1, CV_8UC3);
      outputMat = (Scalar) 0;
      
      int i = 0;
      
      for ( Vec3b vec : allRegionPoints ) {
        uint32_t pixel = Vec3BToUID(vec);
        pixel = pixel & 0x00FFFFFF;
        if (debug) {
          printf("peak[%5d] = 0x%08X\n", i, pixel);
        }
        //Vec3b vec = PixelToVec3b(pixel);
        outputMat.at<Vec3b>(i, 0) = vec;
        i += 1;
      }
      
      // Add each line point
      
      for ( Vec3b vec : linePoints ) {
        outputMat.at<Vec3b>(i, 0) = vec;
        
        if (debug) {
          printf("line point %5d = 0x%02X%02X%02X\n", i, vec[0], vec[1], vec[2]);
        }
        
        i += 1;
      }
      
      imwrite(fname, outputMat);
      cout << "wrote " << fname << endl;
    }
  }
  
  // Generate a best fit vector from the region colors that have been passed
  // through and initial even region quant. This should get rid of outliers
  // and generate a clean best fit line.
  
  if ((1)) {
    vector<Vec3b> quantPoints;
    
    const bool onlyLineOutput = true;
    
    for (int i = 0; i < numPixels; i++) {
      uint32_t pixel = outPixels[i];
      Vec3b vec = PixelToVec3b(pixel);
      quantPoints.push_back(vec);
    }
    
    vector<float> lineVec;
    vector<Vec3b> linePoints;
    
    int distType = CV_DIST_L12;
    //int distType = CV_DIST_FAIR;
    
    fitLine(quantPoints, lineVec, distType, 0, 0.1, 0.1);
    
    // In case of 3D fitting (vx, vy, vz, x0, y0, z0)
    // where (vx, vy, vz) is a normalized vector collinear to the line
    // and (x0, y0, z0) is a point on the line.
    
    Vec3b centerOfMass(round(lineVec[3]), round(lineVec[4]), round(lineVec[5]));
    
    if (onlyLineOutput == false) {
      for (int i = 0; i < 300; i++) {
        linePoints.push_back(centerOfMass);
        linePoints.push_back(centerOfMass); // Dup COM
      }
    } else {
      linePoints.push_back(centerOfMass);
    }
    
    Vec3f colinearF(lineVec[0], lineVec[1], lineVec[2]);
    Vec3f centerOfMassF(centerOfMass[0], centerOfMass[1], centerOfMass[2]);
    
    // Add another point away from the com
    
    int scalar = 30; // num points in (x,y,z) space
    
    Vec3f comPlusUnitF = centerOfMassF + (colinearF * scalar);
    Vec3b comPlusUnit(round(comPlusUnitF[0]), round(comPlusUnitF[1]), round(comPlusUnitF[2]));
    
    Vec3f comMinusUnitF = centerOfMassF - (colinearF * scalar);
    Vec3b comMinusUnit(round(comMinusUnitF[0]), round(comMinusUnitF[1]), round(comMinusUnitF[2]));
    
    if (debug) {
      cout << "centerOfMassF " << centerOfMassF << endl;
      cout << "colinearF " << colinearF << endl;
      cout << "comPlusUnitF " << comPlusUnitF << endl;
      cout << "comMinusUnitF " << comMinusUnitF << endl;
      cout << "comPlusUnit " << comPlusUnit << endl;
      cout << "comMinusUnit " << comMinusUnit << endl;
    }
    
    if (onlyLineOutput == false) {
      for (int i = 0; i < 300; i++) {
        linePoints.push_back(comPlusUnit);
        linePoints.push_back(comMinusUnit);
      }
    } else {
      // Generate line points from -1 -> 0 -> +1 assuming that the
      // scale is the total colorspace size.
      
      for (int scalar = 1; scalar < 255; scalar++ ) {
        Vec3f comPlusUnitF = centerOfMassF + (colinearF * scalar);
        Vec3b comPlusUnit(round(comPlusUnitF[0]), round(comPlusUnitF[1]), round(comPlusUnitF[2]));
        
        Vec3f comMinusUnitF = centerOfMassF - (colinearF * scalar);
        Vec3b comMinusUnit(round(comMinusUnitF[0]), round(comMinusUnitF[1]), round(comMinusUnitF[2]));
        
        if (comPlusUnitF[0] <= 255 && comPlusUnitF[1] <= 255 && comPlusUnitF[0] <= 255) {
          linePoints.push_back(comPlusUnit);
          
          if (debug) {
            cout << "comPlusUnit " << comPlusUnit << endl;
          }
        }
        
        if (comMinusUnitF[0] >= 0 && comMinusUnitF[1] >= 0 && comMinusUnitF[0] >= 0) {
          linePoints.push_back(comMinusUnit);
          
          if (debug) {
            cout << "comMinusUnit " << comMinusUnit << endl;
          }
        }
      }
      
    }
    
    if (debugDumpImages) {
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_quant_center_of_mass" << ".png";
      string fname = fnameStream.str();
      
      // Write image that contains one color in each row in a N x 2 image
      
      int numPoints;
      
      if (onlyLineOutput == false) {
        numPoints = (int) quantPoints.size() + (int) linePoints.size();
      } else {
        numPoints = (int) linePoints.size();
      }
      
      Mat outputMat = Mat(numPoints, 1, CV_8UC3);
      outputMat = (Scalar) 0;
      
      int i = 0;
      
      if (onlyLineOutput == false) {
        
        for ( Vec3b vec : quantPoints ) {
          uint32_t pixel = Vec3BToUID(vec);
          pixel = pixel & 0x00FFFFFF;
          if (debug) {
            printf("quant[%5d] = 0x%08X\n", i, pixel);
          }
          //Vec3b vec = PixelToVec3b(pixel);
          outputMat.at<Vec3b>(i, 0) = vec;
          i += 1;
        }
        
      }
      
      // Add each line point
      
      for ( Vec3b vec : linePoints ) {
        outputMat.at<Vec3b>(i, 0) = vec;
        
        if (debug) {
          printf("line point %5d = 0x%02X%02X%02X\n", i, vec[0], vec[1], vec[2]);
        }
        
        i += 1;
      }
      
      imwrite(fname, outputMat);
      cout << "wrote " << fname << endl;
    }
  }
  
  // Determine which cluster center is nearest to the peak pixels and use
  // that info to generate new cluster centers that are exactly at the
  // peak values. This means that the peak pixels will quant exactly and the
  // nearby cluster value will get the nearby but not exactly on pixels.
  // This should clearly separate the flat pixels from the gradient pixels.
  
    unordered_map<uint32_t, uint32_t> pixelToQuantCountTable;
    
    for (int i = 0; i < numActualClusters; i++) {
      uint32_t pixel = colortable[i];
      pixel = pixel & 0x00FFFFFF;
      pixelToQuantCountTable[pixel] = i;
    }
    
    {
      int i = 0;
      
      for ( uint32_t pixel : peakPixels ) {
        pixel = pixel & 0x00FFFFFF;
        if (debug) {
          printf("peak[%5d] = 0x%08X\n", i, pixel);
        }
        i += 1;
      }
    }
    
    {
      uint32_t prevPeak = 0x0;
      
      for ( int i = 0; i < peakPixels.size(); i++ ) {
        uint32_t pixel = peakPixels[i];
        pixel = pixel & 0x00FFFFFF;
        
        if (pixelToQuantCountTable.count(pixel) == 0) {
          pixelToQuantCountTable[pixel] = 0;
          
          if (debug) {
            printf("added peak pixel 0x%08X\n", pixel);
          }
        } else {
          if (debug) {
            printf("colortable already contains peak pixel 0x%08X\n", pixel);
          }
        }
        
        int32_t sR, sG, sB;
        
        xyzDelta(prevPeak, pixel, sR, sG, sB);
        
        if (debug) {
          printf("peakToPeakDelta 0x%08X -> 0x%08X = (%d %d %d)\n", prevPeak, pixel, sR, sG, sB);
        }
        
        xyzDeltaToUnitVector(sR, sG, sB);
        
        if (debug) {
          printf("unit vector (%5d %5d %5d)\n", sR, sG, sB);
        }
        
        if (1) {
          // Add book end in direction of next peak, but very close
          
          // FIXME: in additon to adding the peak, make sure that
          // another value is very near the peak to act as a book
          // end in the direction of the other peak(s). Might need
          // one bookend for each other major peak.
          
          if ((i == 0) && (peakPixels.size() > 1)) {
            // Add bookend by finding the delta to the next peak, since
            // this is the first element and it is likely to be 0x0.
            
            uint32_t nextPeak = peakPixels[1];
            
            // Note that this delta is not quite correct since the value
            // 0xFF must be treated as the maximum delta, not -1 to flip
            // from 0 to 255.
            
            xyzDelta(pixel, nextPeak, sR, sG, sB);
            
            if (debug) {
              printf("peakToPeakDelta 0x%08X -> 0x%08X = (%d %d %d)\n", pixel, nextPeak, sR, sG, sB);
            }
            
            xyzDeltaToUnitVector(sR, sG, sB);
            
            if (debug) {
              printf("unit vector (%5d %5d %5d)\n", sR, sG, sB);
            }
            
            // Add vector to current pixel
            
            Vec3f curVec(pixel & 0xFF, (pixel >> 8) & 0xFF, (pixel >> 16) & 0xFF);
            Vec3f deltaVec(sB, sG, sR);
            Vec3f sumVec = curVec + deltaVec;
            
            if (debug) {
              cout << "cur + delta = booekend : " << curVec << " + " << deltaVec << " = " << sumVec << endl;
            }
            
            uint32_t B = round(sumVec[0]);
            uint32_t G = round(sumVec[1]);
            uint32_t R = round(sumVec[2]);
            
            uint32_t bePixel = (R << 16) | (G << 8) | B;
            
            if (pixelToQuantCountTable.count(bePixel) == 0) {
              pixelToQuantCountTable[bePixel] = 0;
              
              if (debug) {
                printf("added bookend pixel 0x%08X\n", bePixel);
              }
            } else {
              if (debug) {
                printf("colortable already contains bookend pixel 0x%08X\n", bePixel);
              }
            }
            
          }
          
        }
        
        prevPeak = pixel;
      }
    }
    
    int numColors = (int)pixelToQuantCountTable.size();
  delete [] colortable;
    colortable = new uint32_t[numColors];
    
    {
      int i = 0;
      for ( auto &pair : pixelToQuantCountTable ) {
        uint32_t key = pair.first;
        assert(key == (key & 0x00FFFFFF)); // verify alpha is zero
        colortable[i] = key;
        i++;
      }
    }
    
    if (debug) {
      cout << "numActualClusters was " << numActualClusters << " while output numColors is " << numColors << endl;
    }
    
    // Resort cluster centers
    
    vector<uint32_t> resortedColortable;
    
    {
      for (int i = 0; i < numColors; i++) {
        resortedColortable.push_back(colortable[i]);
      }
      
      vector<uint32_t> sortedOffsets = generate_cluster_walk_on_center_dist(resortedColortable);
      
      resortedColortable.clear();
      
      for (int i = 0; i < numColors; i++) {
        int si = (int) sortedOffsets[i];
        uint32_t pixel = colortable[si];
        resortedColortable.push_back(pixel);
      }
      
      // Copy pixel values in sorted order back into colortable
      
      for (int i = 0; i < numColors; i++) {
        uint32_t pixel = resortedColortable[i];
        colortable[i] = pixel;
      }
    }
    
    if (debugDumpImages)
    {
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_quant_table2" << ".png";
      string fname = fnameStream.str();
      
      dumpQuantTableImage(fname, inputImg, colortable, numColors);
    }
  
  if (debug) {
    for ( int i = 0; i < numColors; i++) {
      uint32_t pixel = colortable[i];
      fprintf(stdout, "colortable[%5d] = 0x%08X\n", i, pixel);
    }
  }
    
    // Run input pixels through closest color quant logic using the
    // generated colortable. Note that the colortable should be
    // split such that one range of the colortable should be seen
    // as "inside" while the other range is "outside".
    
    map_colors_mps(inPixels, numPixels, outPixels, colortable, numColors);
    
    // Dump output, which is the input colors run through the color table
    
    if (debugDumpImages) {
      Mat tmpResultImg = inputImg.clone();
      tmpResultImg = Scalar(0,0,0xFF);
      
      for ( int i = 0; i < numPixels; i++ ) {
        Coord c = regionCoords[i];
        uint32_t pixel = outPixels[i];
        Vec3b vec = PixelToVec3b(pixel);
        tmpResultImg.at<Vec3b>(c.y, c.x) = vec;
      }
      
      {
        std::stringstream fnameStream;
        fnameStream << "srm" << "_tag_" << tag << "_quant_output2" << ".png";
        string fname = fnameStream.str();
        
        imwrite(fname, tmpResultImg);
        cout << "wrote " << fname << endl;
      }
    }
    
    // Vote inside/outside
    
    unordered_map<uint32_t, InsideOutsideRecord> pixelToInside;
    
    insideOutsideTest(inputImg.rows, inputImg.cols, srmRegionCoords, tag, regionCoords, outPixels, resortedColortable, pixelToInside);
    
    // Emit vote result table which basically shows the sorted pixel and the vote boolean as black or white
    
    if (debugDumpImages)
    {
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_quant_table3_votes" << ".png";
      string fname = fnameStream.str();
      
      // Write image that contains one color in each row in a N x 2 image
      
      int numColortableEntries = (int) resortedColortable.size();
      
      Mat qtableOutputMat = Mat(numColortableEntries, 2, CV_8UC3);
      qtableOutputMat = (Scalar) 0;
      
      for (int i = 0; i < numColortableEntries; i++) {
        uint32_t pixel = resortedColortable[i];
        Vec3b vec = PixelToVec3b(pixel);
        qtableOutputMat.at<Vec3b>(i, 0) = vec;
        
        bool isInside = pixelToInside[pixel].isInside;
        
        if (isInside) {
          vec = Vec3b(0xFF, 0xFF, 0xFF);
        } else {
          vec = Vec3b(0, 0, 0);
        }
        
        qtableOutputMat.at<Vec3b>(i, 1) = vec;
      }
      
      imwrite(fname, qtableOutputMat);
      cout << "wrote " << fname << endl;
    }
  
  // Determine the dominate vector(s) as we iterate through the colortable
  // so that vectors add as long as they are actually in the same vector
  // line.

  vector<vector<uint32_t> > vecOfPixels;
  vector<vector<Vec3f> > vecOfVectors;
  
  vecOfPixels.push_back(vector<uint32_t>());
  vector<uint32_t> *currentVecPtr = &vecOfPixels[0];
  
  vecOfVectors.push_back(vector<Vec3f>());
  vector<Vec3f> *currentVecOfVec3bPtr = &vecOfVectors[0];

  uint32_t prevPixel = 0x0;
  
  for ( int i = 0; i < numColors; i++) {
    uint32_t pixel = resortedColortable[i];
    fprintf(stdout, "resorted colortable[%5d] = 0x%08X\n", i, pixel);
    
    int prevSize = (int) currentVecPtr->size();
    currentVecPtr->push_back(pixel);
    
    if (prevSize == 0) {
      // currentVecPtr is currently empty, so append the first pixel
      // and continue on with the next pixel.
    } else {
      // At least 2 pixels, initial vector between pixels can now
      // be determined for the first time.
      
      int32_t sR, sG, sB;
      
      xyzDelta(prevPixel, pixel, sR, sG, sB);
      
      if (debug) {
        printf("peakToPeakDelta 0x%08X -> 0x%08X = (%d %d %d)\n", prevPixel, pixel, sR, sG, sB);
      }
      
      Vec3f unitVec = xyzDeltaToUnitVec3f(sR, sG, sB);
      
      if (debug) {
        cout << "unit vector " << unitVec << endl;
      }

      currentVecOfVec3bPtr->push_back(unitVec);
    }
    
    prevPixel = pixel;
  }

  for ( uint32_t pixel : *currentVecPtr ) {
    printf("0x%08X\n", pixel);
  }
  
  for ( Vec3f vec : *currentVecOfVec3bPtr ) {
    cout << "vec " << vec << endl;
  }
  
//  Should see this set of vectors as shifting from one
//  general slope to a new slope at offset 5
//  the fact that earlier slopes are not exactly
//  the same is not critical. Last 5 are basically
//  the same
  
//  vec [0.0794929, 0.0794929, 0.993661]
//  vec [0.111784, 0.111784, 0.987425]
//  vec [-0.131796, -0.131796, 0.982476]
//  vec [0.535899, 0.535899, 0.652399]
//  vec [0.730889, 0.426352, -0.53294]
//  vec [0.732098, 0.413795, -0.541116]
//  vec [0.732728, 0.429867, -0.527564]
//  vec [0.733142, 0.4339, -0.523673]

  
  // Determine the color inside the region, this could be rooted on a flat region
  // of all the same pixel value or it could be a region of alike colors that
  // define a cluster center point. The clustering logic above is useful in
  // that it identifies a best clustering, but that clustering needs to be
  // iterated such that a set of N vectors from one core color to the next is
  // defined.
  
  if (debugDumpImages) {
    Mat tmpResultImg(inputImg.rows, inputImg.cols, CV_8UC1);
    tmpResultImg = Scalar(0);

    Vec3b prevSrmVec;
    int i = 0;
    
    for ( Coord c : srmRegionCoords ) {
      Vec3b srmVec = srmTags.at<Vec3b>(c.y, c.x);

      if (i == 0) {
      } else {
        assert(srmVec == prevSrmVec);
      }
      prevSrmVec = srmVec;
      
      tmpResultImg.at<uint8_t>(c.y, c.x) = 0xFF;
      
      i += 1;
    }
    
    {
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_srm_region_decrease_mask" << "1" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, tmpResultImg);
      cout << "wrote " << fname << endl;
      cout << "";
    }
    
    {
      Mat alphaMaskResultImg(inputImg.rows, inputImg.cols, CV_8UC4);
      alphaMaskResultImg = Scalar(0, 0, 0, 0);
      
      for ( Coord c : srmRegionCoords ) {
        Vec3b vec = inputImg.at<Vec3b>(c.y, c.x);
        Vec4b vec4;
        vec4[0] = vec[0];
        vec4[1] = vec[1];
        vec4[2] = vec[2];
        vec4[3] = 0xFF;
        alphaMaskResultImg.at<Vec4b>(c.y, c.x) = vec4;
      }
      
      {
        std::stringstream fnameStream;
        fnameStream << "srm" << "_tag_" << tag << "_srm_region_decrease_alpha_mask" << "1" << ".png";
        string fname = fnameStream.str();
        
        imwrite(fname, alphaMaskResultImg);
        cout << "wrote " << fname << endl;
        cout << "";
      }
    }
    
    Mat savedTmpResultImg = tmpResultImg;
    
    // Call decrease white logic over and over until no more white area is left.
    
    vector<vector<Coord> > decreasingCoordVecStack;
    
    // Define max iter num in terms of sqrt(width^2, height^2)
    
    for ( int i = 2; i < 100; i++ ) {
      tmpResultImg = decreaseWhiteInRegion(tmpResultImg, 1, tag);
      
      vector<Point> locations;
      findNonZero(tmpResultImg, locations);
      
      vector<Coord> coords;
      
      for ( Point p : locations ) {
        Coord c(p.x, p.y);
        coords.push_back(c);
      }
      
      int numNonZero = (int) coords.size();
      
      if (numNonZero == 0) {
        break;
      }
      
      decreasingCoordVecStack.insert(begin(decreasingCoordVecStack), coords);
      
      {
        std::stringstream fnameStream;
        fnameStream << "srm" << "_tag_" << tag << "_srm_region_decrease_mask" << i << ".png";
        string fname = fnameStream.str();
        
        imwrite(fname, tmpResultImg);
        cout << "wrote " << fname << endl;
        cout << "";
      }
      
      // Dump alpha masked version of the original input.
      
      {
        Mat alphaMaskResultImg(inputImg.rows, inputImg.cols, CV_8UC4);
        alphaMaskResultImg = Scalar(0, 0, 0, 0);
        
        for ( Coord c : coords ) {
          Vec3b vec = inputImg.at<Vec3b>(c.y, c.x);
          Vec4b vec4;
          vec4[0] = vec[0];
          vec4[1] = vec[1];
          vec4[2] = vec[2];
          vec4[3] = 0xFF;
          alphaMaskResultImg.at<Vec4b>(c.y, c.x) = vec4;
        }
        
        {
          std::stringstream fnameStream;
          fnameStream << "srm" << "_tag_" << tag << "_srm_region_decrease_alpha_mask" << i << ".png";
          string fname = fnameStream.str();
          
          imwrite(fname, alphaMaskResultImg);
          cout << "wrote " << fname << endl;
          cout << "";
        }
      }

    }
    
    cout << "done" << endl;
    
    
    // Call decrease white logic over and over until no more white area is left.
    
    vector<vector<Coord> > expandingCoordVecStack;
    
    // Define max iter num in terms of sqrt(width^2, height^2)
    
    tmpResultImg = savedTmpResultImg;
    
    for ( int i = 2; i < 15; i++ ) {
      tmpResultImg = expandWhiteInRegion(tmpResultImg, 1, tag);
      
      vector<Point> locations;
      findNonZero(tmpResultImg, locations);
      
      vector<Coord> coords;
      
      for ( Point p : locations ) {
        Coord c(p.x, p.y);
        coords.push_back(c);
      }
      
      int numNonZero = (int) coords.size();
      
      if (numNonZero == 0) {
        break;
      }
      
      expandingCoordVecStack.insert(begin(expandingCoordVecStack), coords);
      
      {
        std::stringstream fnameStream;
        fnameStream << "srm" << "_tag_" << tag << "_srm_region_increase_mask" << i << ".png";
        string fname = fnameStream.str();
        
        imwrite(fname, tmpResultImg);
        cout << "wrote " << fname << endl;
        cout << "";
      }
      
      // Dump alpha masked version of the original input.
      
      {
        Mat alphaMaskResultImg(inputImg.rows, inputImg.cols, CV_8UC4);
        alphaMaskResultImg = Scalar(0, 0, 0, 0);
        
        for ( Coord c : coords ) {
          Vec3b vec = inputImg.at<Vec3b>(c.y, c.x);
          Vec4b vec4;
          vec4[0] = vec[0];
          vec4[1] = vec[1];
          vec4[2] = vec[2];
          vec4[3] = 0xFF;
          alphaMaskResultImg.at<Vec4b>(c.y, c.x) = vec4;
        }
        
        {
          std::stringstream fnameStream;
          fnameStream << "srm" << "_tag_" << tag << "_srm_region_increase_alpha_mask" << i << ".png";
          string fname = fnameStream.str();
          
          imwrite(fname, alphaMaskResultImg);
          cout << "wrote " << fname << endl;
          cout << "";
        }
      }
      
    }
    
    cout << "done" << endl;

  }
  
  // FIXME: the better approach would be to decrease until the region seems to stabilize around
  // on single color or one quant color and then expand until there is a jump up in the number
  // of colors or quant num. This has the effect of trimming off any "splay" pixels that are
  // improperly identified in the region and then expanding the region to account for any "under".
  // Expanding only according to the original SRM shape is not going to be correct in the case
  // of splay pixels.
  
  // An upper bound on the region bounds could be the containing region bounds.
  
  
    // If after the voting, it becomes clear that one of the regions
    // is always outside the in/out region defined by the tags, then
    // pixels in that range can be left out of the vector calculation.
    // For example, the black outside the red in OneCircleInsideAnother.png.
    // In this case, the pixels that quant to the black to red vector
    // can all be ignored and the blue to red vector can be extracted
    // cleanly. This is useful to turn off the in/out vote of certain
    // pixels if on with zero votes and to "capture" all the possible
    // pixels along the vector line up to the bookend and identify those
    // as in/out. The extends of the inner region should fully capture
    // gradient pixels at the bound, so that if the outer region is
    // otherwise uniform then it will not need to know about the inner
    // region color use at all and can be free to predict as if the
    // inner region shift does not happen.
    
    // Each pixel in the input is now mapped to a boolean condition that
    // indicates if that pixel is inside or outside the shape.
    
    const bool debugOnOff = false;
    
    for ( int i = 0; i < numPixels; i++ ) {
      Coord c = regionCoords[i];
      uint32_t quantPixel = outPixels[i];
      
#if defined(DEBUG)
      assert(pixelToInside.count(quantPixel));
#endif // DEBUG
      bool isInside = pixelToInside[quantPixel].isInside;
      
      if (isInside) {
        mask.at<uint8_t>(c.y, c.x) = 0xFF;
        
        if (debug && debugOnOff) {
          printf("pixel 0x%08X at (%5d,%5d) is marked on (inside)\n", quantPixel, c.x, c.y);
        }
      } else {
        if (debug && debugOnOff) {
          printf("pixel 0x%08X at (%5d,%5d) is marked off (outside)\n", quantPixel, c.x, c.y);
        }
      }
    }
    
    if (debug) {
      cout << "return captureNotCloseRegion" << endl;
    }
  
  delete [] colortable;
  
    return;
}

// Loop over each pixel passed through the quant logic and count up how
// often a pixel is "inside" the known region vs how often it is "outside".

// Foreach pixel in a colortable determine the "inside/outside" status of that
// pixel based on a stats test as compared to the current known region.

void insideOutsideTest(int32_t width,
                       int32_t height,
                       const vector<Coord> &coords,
                       int32_t tag,
                       const vector<Coord> &regionCoords,
                       const uint32_t *outPixels,
                       const vector<uint32_t> &sortedColortable,
                       unordered_map<uint32_t, InsideOutsideRecord> &pixelToInsideMap)
{
  const bool debug = true;
  const bool debugDumpImages = true;
  
  // Create region mask as byte mask
  
  Mat isInsideMask(height, width, CV_8UC1);
  isInsideMask = Scalar(0);
  
  for ( Coord c : coords ) {
    isInsideMask.at<uint8_t>(c.y, c.x) = 0xFF;
  }
  
  if (debugDumpImages) {
    std::stringstream fnameStream;
    fnameStream << "srm" << "_tag_" << tag << "_srm_region_mask" << ".png";
    string fname = fnameStream.str();
    
    imwrite(fname, isInsideMask);
    cout << "wrote " << fname << endl;
    cout << "";
  }
  
  //unordered_map<uint32_t, InsideOutside> srmTagInsideCount;
  
  int numPixels = (int) regionCoords.size();
  
  for ( int i = 0; i < numPixels; i++ ) {
    Coord c = regionCoords[i];
    uint32_t quantPixel = outPixels[i];
    
    if (debug && 0) {
      printf("quantPixel 0x%08X\n", quantPixel);
    }
    
    InsideOutsideRecord &inOut = pixelToInsideMap[quantPixel];
    uint8_t isInside = isInsideMask.at<uint8_t>(c.y, c.x);
    if (isInside) {
      inOut.inside += 1;
    } else {
      inOut.outside += 1;
    }
  }
  
  for ( int i = 0; i < sortedColortable.size(); i++ ) {
    uint32_t pixel = sortedColortable[i];
    
    if (pixelToInsideMap.count(pixel) == 0) {
      InsideOutsideRecord &inOut = pixelToInsideMap[pixel];
      // FIXME: assume it is inside somewhere in a gradient ?
      // May not be important since no output pixels match it.
      inOut.inside += 1;
      inOut.outside += 0;
      inOut.confidence = 0.0f;
    }
  }
  
  // If any colortable pixels did not appear in outPixels then add inside = 0
  // votes for that pixel ?
  
  // Vote for inside/outside status for each unique pixel based on a GT 50% chance
  
  for ( int i = 0; i < sortedColortable.size(); i++ ) {
    uint32_t pixel = sortedColortable[i];
    InsideOutsideRecord &inOut = pixelToInsideMap[pixel];
    
    if (debug) {
      printf("inout table[0x%08X] = (in out) (%5d %5d)\n", pixel, inOut.inside, inOut.outside);
    }
    
    float percentOn = (float)inOut.inside / (inOut.inside + inOut.outside);
    
    inOut.confidence = percentOn;
    
    if (debug) {
      printf("percent on [0x%08X] = %0.3f\n", pixel, percentOn);
    }
    
    if (percentOn > 0.5f) {
      inOut.isInside = true;
    } else {
      inOut.isInside = false;
    }
    
    if (debug) {
      printf("pixelToInsideMap[0x%08X].isInside = %d\n", pixel, inOut.isInside);
    }
  }
  
  if (debug) {
    printf("done voting\n");
  }
  
  // Dump the colortable in sorted order

  uint32_t prevPixel = sortedColortable[0];
  
  for ( int i = 0; i < sortedColortable.size(); i++ ) {
    uint32_t pixel = sortedColortable[i];
    
    if (debug) {
      printf("colortable[%5d] = 0x%08X\n", i, pixel);
    }
    
    uint32_t deltaPixel = predict_trivial_component_sub(prevPixel, pixel);
    
    if (debug) {
      printf("pixel delta 0x%08X -> 0x%08X = 0x%08X\n", prevPixel, pixel, deltaPixel);
    }
    
    uint32_t absDeltaPixel = absPixel(deltaPixel);
    
    if (debug) {
      printf("abs    pixel delta 0x%08X : %d %d %d\n", absDeltaPixel, (int)((absDeltaPixel >> 16)&0xFF), (int)((absDeltaPixel >> 8)&0xFF), (int)((absDeltaPixel >> 0)&0xFF));
    }
    
    prevPixel = pixel;
  }
  
  if (debug) {
    printf("done measure table diff\n");
  }
  
  // Print in/out state for each ordered table pixel
  
  for ( int i = 0; i < sortedColortable.size(); i++ ) {
    uint32_t pixel = sortedColortable[i];
    
#if defined(DEBUG)
    assert(pixelToInsideMap.count(pixel) > 0);
#endif // DEBUG
    
    bool isInside = pixelToInsideMap[pixel].isInside;
    
    if (debug) {
      printf("colortable[%5d] = 0x%08X : isInside %d\n", i, pixel, isInside);
    }
  }
  
  // FIXME: issue with in/out is that the src region does not expand all the way
  // out and as a result the nearest quant table value does not have many in
  // votes even though it is far from the black pixel. The better approach here
  // would be to place a pixel very near black so that even though it is black
  // with just a little green that pixel is seen as "inside". This kind of psudo
  // pixel that is known to not be outside as compared to a black background
  // (since it is a little bit green) should then be seen by this method as
  // inside. Basically need to have a way to indicate that a specific colortable
  // entry is "known to be inside" when it is automatically placed near the
  // background color cluster center.
  
//  inout table[0x00000000] = (in out) (    0  1267)
//  percent on [0x00000000] = 0.000
  
//  inout table[0x000C4400] = (in out) (    7    21)
//  percent on [0x000C4400] = 0.250

// w bookend (known to be inside, leading right up to known solid color)
  
//  inout table[0x00000000] = (in out) (    0  1215)
//  percent on [0x00000000] = 0.000
//  pixelToInsideMap[0x00000000] = 0
//  inout table[0x00010300] = (in out) (    0    52)
//  percent on [0x00010300] = 0.000
//  pixelToInsideMap[0x00010300] = 0
//  inout table[0x000C4400] = (in out) (    7    21)
//  percent on [0x000C4400] = 0.250
//  pixelToInsideMap[0x000C4400] = 0
//  inout table[0x00178000] = (in out) (  174     0)
//  percent on [0x00178000] = 1.000
  
  if (debug) {
    printf("done in out boolean\n");
  }
  
  return;
}

// Given a set of pixels, scan those pixels and determine the peaks
// in the histogram to find likely most common graph peak values.

vector<uint32_t> gatherPeakPixels(const vector<uint32_t> & pixels,
                                  unordered_map<uint32_t, uint32_t> & pixelToNumVotesMap)
{
  const bool debug = true;
  
  vector<uint32_t> peakPixels;
  
  // FIXME: dynamically allocate buffers to fit input size ?
  
  double*     data[2];
  //  double      row[2];
  
#define MAX_PEAK    256
  
  int         emi_peaks[MAX_PEAK];
  int         absorp_peaks[MAX_PEAK];
  
  int         emi_count = 0;
  int         absorp_count = 0;
  
  double      delta = 1e-6;
  int         emission_first = 0;
  
  int numDataPoints = (int) pixels.size();
  
  assert(numDataPoints <= 256);
  
  data[0] = new double[MAX_PEAK]();
  data[1] = new double[MAX_PEAK]();
  
  int i = 0;
  
  // Insert zero slow with zero count so that a peak can
  // be detected in the first position.
  i += 1;
  
  for ( uint32_t pixel : pixels ) {
    uint32_t count = pixelToNumVotesMap[pixel];
    uint32_t pixelNoAlpha = pixel & 0x00FFFFFF;
    
    data[0][i] = pixelNoAlpha;
//    data[0][i] = i;
    data[1][i] = count;
    
    if ((0)) {
      fprintf(stderr, "data[%05d] = 0x%08X -> count %d\n", i, pixelNoAlpha, count);
    }
    
    i += 1;
  }
  
  // +1 at the end of the samples
  i += 1;
  
  // Print the input data with zeros at the front and the back
  
  for ( int j = 0; j < i; j++ ) {
    uint32_t pixelNoAlpha = data[0][j];
    uint32_t count = data[1][j];
    
    if (debug) {
      fprintf(stderr, "pixel %05d : 0x%08X = %d\n", j, pixelNoAlpha, count);
    }
  }
  
  if(detect_peak(data[1], i,
                 emi_peaks, &emi_count, MAX_PEAK,
                 absorp_peaks, &absorp_count, MAX_PEAK,
                 delta, emission_first))
  {
    fprintf(stderr, "There are too many peaks.\n");
    exit(1);
  }
  
  if (debug) {
    fprintf(stdout, "num emi_peaks %d\n", emi_count);
  }
  
  for(i = 0; i < emi_count; ++i) {
    int offset = emi_peaks[i];
    if (debug) {
      fprintf(stdout, "%5d : %5d,%5d\n", offset, (int)data[0][offset], (int)data[1][offset]);
    }
    
    uint32_t pixel = (uint32_t) round(data[0][offset]);
    peakPixels.push_back(pixel);
  }
  
  if (debug) {
    fprintf(stdout, "num absorp_peaks %d\n", absorp_count);
  }
  
  for(i = 0; i < absorp_count; ++i) {
    int offset = absorp_peaks[i];
    if (debug) {
      fprintf(stdout, "%5d : %5d,%5d\n", offset, (int)data[0][offset],(int)data[1][offset]);
    }
  }
  
  delete [] data[0];
  delete [] data[1];
  
  return peakPixels;
}

// This method accepts a region defined by coords and returns the edges between
// superpixels in the region.

void
clockwiseScanForTagsAroundShape(
                                const Mat & tagsImg,
                                int32_t tag,
                                const vector<Coord> &regionCoords,
                                vector<TagsAroundShape> &tagsAroundVec)
{
  const bool debug = true;
  const bool debugDumpImages = true;
  const bool debugDumpStepImages = false;
  
  if (debug) {
    cout << "clockwiseScanForTagsAroundShape " << tag << endl;
  }
  
  // The tagsImg mat contains tags, so generate lines around the 360 degrees
  // of rotation and determine which tags the line pixels hit.
  
  int32_t originX, originY, regionWidth, regionHeight;
  bbox(originX, originY, regionWidth, regionHeight, regionCoords);
  Rect roiRect(originX, originY, regionWidth, regionHeight);
  
  Coord originCoord(originX, originY);
  
  Mat renderMat(roiRect.size(), CV_8UC1);
  
  renderMat = Scalar(0);
  
  // Generate coords that iterate around the region bbox starting from up which
  // is taken to be degree zero.
  
  vector<Coord> outlineCoords = genRectangleOutline(regionWidth, regionHeight);
  
  // Render points in outlineCoords to binary mat and debug dump
  
  if (debugDumpImages) {
    renderMat = Scalar(0);
    
    for ( Coord c : outlineCoords ) {
      renderMat.at<uint8_t>(c.y, c.x) = 0xFF;
    }
    
    std::stringstream fnameStream;
    fnameStream << "srm" << "_tag_" << tag << "_region_outline_coords" << ".png";
    string fname = fnameStream.str();
    
    imwrite(fname, renderMat);
    cout << "wrote " << fname << endl;
    cout << "";
  }
  
  // Map from global coordinates to the specific tag at that coordinate
  // but ignore the current shape tag since most coords will be for the
  // interior of the shape.
  
  unordered_map<Coord, int32_t> tagMap;
  
  for ( Coord c : regionCoords ) {
    Vec3b vec = tagsImg.at<Vec3b>(c.y, c.x);
    int32_t inRegionTag = Vec3BToUID(vec);
    if (inRegionTag == tag) {
      continue;
    }
    tagMap[c] = inRegionTag;
    if (debug) {
      cout << "add mapping for " << c << " -> " << vec << endl;
    }
  }
  
  // Render from region center to each coordinate around the outside edge of the bbox
  // and store the vector of tags found along the line.
  
  // Determine region center point
  
  // Find a single "center" pixel in region of interest matrix. This logic
  // accepts an input matrix that contains binary pixel values (0x0 or 0xFF)
  // and computes a consistent center pixel. When this method returns the
  // region binMat is unchanged. The orderMat is set to the size of the roi and
  // it is filled with distance transformed gray values. Note that this method
  // has to create a buffer zone of 1 pixel so that pixels on the edge have
  // a very small distance.
  
  renderMat = Scalar(0);
  
  // FIXME: could pass this in, but just query for now
  
  vector<Coord> currentTagCoords;
  
  for ( Coord c : regionCoords ) {
    Vec3b vec = tagsImg.at<Vec3b>(c.y, c.x);
    int32_t regionTag = Vec3BToUID(vec);
    if (tag == regionTag) {
      currentTagCoords.push_back(c);
    }
  }
  
  for ( Coord c : currentTagCoords ) {
    c = c - originCoord;
    renderMat.at<uint8_t>(c.y, c.x) = 0xFF;
  }
  
  Mat outDistMat;
  Coord regionCenter = findRegionCenter(renderMat, Rect2d(0,0,regionWidth,regionHeight), outDistMat, tag);
  
  Point2i center(regionCenter.x, regionCenter.y);
  
  // Iterate over each vector of (center, edgePoint) along the bbox bounds
  
  vector<set<int32_t> > allTagSetsForVectors;
  
  // Store coords found for each vector
  
  vector<vector<Coord> > allCoordForVectors;
  
  int stepi = 0;
  int stepMax = (int) outlineCoords.size();
  
  for ( ; stepi < stepMax; stepi++ ) {
    
    set<int32_t> tagsForVector;
    vector<Coord> coordsForVector;
    
    Coord edgeCoord = outlineCoords[stepi];
    
    Point2i edgePoint(edgeCoord.x, edgeCoord.y);
    
    renderMat = Scalar(0);
    line(renderMat, center, edgePoint, Scalar(0xFF));
    
    if (debug) {
      cout << "render center line from " << center << " to " << edgePoint << endl;
    }
    
    if (debugDumpImages && debugDumpStepImages) {
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_step" << stepi << "_region_vec" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, renderMat);
      cout << "wrote " << fname << endl;
      cout << "";
    }
    
    vector<Point> locations;
    findNonZero(renderMat, locations);
    
    for ( Point p : locations ) {
      Coord c(p.x, p.y);
      c = originCoord + c;
      if (tagMap.count(c) > 0) {
        int32_t regionTag = tagMap[c];
        tagsForVector.insert(regionTag);
        coordsForVector.push_back(c);
      }
    }
    
    allTagSetsForVectors.push_back(tagsForVector);
    allCoordForVectors.push_back(coordsForVector);
    
    if (debugDumpImages && debugDumpStepImages) {
      Mat regionRoiMat = tagsImg(roiRect);
      
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_step" << stepi << "_region_input_tags_roi" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, regionRoiMat);
      cout << "wrote " << fname << endl;
      cout << "";
    }
    
    if (debugDumpImages && debugDumpStepImages) {
      // Dump tags that are defined as on in regionCoords
      
      Mat allTagsOn = tagsImg.clone();
      allTagsOn = Scalar(0,0,0);
      
      for ( Coord c : regionCoords ) {
        allTagsOn.at<Vec3b>(c.y, c.x) = tagsImg.at<Vec3b>(c.y, c.x);
      }
      
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_step" << stepi << "_region_input_tags" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, allTagsOn);
      cout << "wrote " << fname << endl;
      cout << "";
    }
    
    if (debugDumpImages && debugDumpStepImages) {
      renderMat = Scalar(0);
      
      for ( Point p : locations ) {
        Coord c(p.x, p.y);
        c = originCoord + c;
        if (tagMap.count(c) > 0) {
          c = c - originCoord;
          renderMat.at<uint8_t>(c.y, c.x) = 0xFF;
        }
      }
      
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_step" << stepi << "_hits_for_tag_vec" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, renderMat);
      cout << "wrote " << fname << endl;
      cout << "";
    }
    
    if (debugDumpImages && debugDumpStepImages) {
      // Dump tags that are defined as on in regionCoords
      
      Mat allTagsHit = tagsImg.clone();
      allTagsHit = Scalar(0,0,0);
      
      for ( Point p : locations ) {
        Coord c(p.x, p.y);
        c = originCoord + c;
        if (tagMap.count(c) > 0) {
          allTagsHit.at<Vec3b>(c.y, c.x) = tagsImg.at<Vec3b>(c.y, c.x);
        }
      }
      
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_step" << stepi << "_hit_tags_for_tag_vec" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, allTagsHit);
      cout << "wrote " << fname << endl;
      cout << "";
    }
    
  }
  
  // Identify all the tags found in the region and with a tag other than this one
  
  unordered_map<int32_t, bool> allTagsCombined;
  
  for ( set<int32_t> & tagSet : allTagSetsForVectors ) {
    for ( int32_t regionTag : tagSet ) {
      allTagsCombined[regionTag] = true;
    }
  }
  
  if (debug) {
    cout << "all tags found around region" << endl;
    
    for ( auto & pair : allTagsCombined ) {
      int32_t regionTag = pair.first;
      printf("tag = 0x%08X aka %d\n", regionTag, regionTag);
    }
  }
  
  if (debugDumpImages) {
    // Dump tags that are defined as on in regionCoords
    
    Mat allTagsHit = tagsImg.clone();
    allTagsHit = Scalar(0,0,0);
    
    for ( Coord c : regionCoords ) {
      Vec3b vec = tagsImg.at<Vec3b>(c.y, c.x);
      int32_t regionTag = Vec3BToUID(vec);
      
      if (allTagsCombined.count(regionTag) > 0) {
        allTagsHit.at<Vec3b>(c.y, c.x) = vec;
        
        if (debug) {
          printf("found region tag %9d at coord (%5d, %5d)\n", regionTag, c.x, c.y);
        }
      }
    }
    
    std::stringstream fnameStream;
    fnameStream << "srm" << "_tag_" << tag << "_hit_tags_in_scan_region" << ".png";
    string fname = fnameStream.str();
    
    imwrite(fname, allTagsHit);
    cout << "wrote " << fname << endl;
    cout << "";
  }
  
  // Condense tags in regions starting from the top. A region range is condensed as long
  // as the tags in the range are the same or if there are no tags.
  
  unordered_map<Coord, bool> uniqueCoords;
  
  for ( stepi = 0 ; stepi < stepMax; ) {
    if (debug) {
      cout << "consider stepi " << stepi << endl;
    }
    
    set<int32_t> &currentSet = allTagSetsForVectors[stepi];
    
    int nextStepi = stepi + 1;
    
    for ( ; nextStepi < stepMax; nextStepi++ ) {
      set<int32_t> &nextSet = allTagSetsForVectors[nextStepi];
      
      // If the sets are identical then merge the regions.
      
      if (currentSet == nextSet) {
        if (debug && true) {
          cout << "same set for step " << nextStepi << endl;
        }
        if (debug && false) {
          cout << "set 1" << endl;
          for ( int32_t tag : currentSet ) {
            cout << tag << endl;
          }
          cout << "set 2" << endl;
          for ( int32_t tag : nextSet ) {
            cout << tag << endl;
          }
        }
      } else {
        nextStepi -= 1;
        break;
      }
    }
    
    // Range is (stepi, nextStepi)
    
    if (debug) {
      cout << "step same range (" << stepi << "," << nextStepi << ")" << endl;
    }
    
    tagsAroundVec.push_back(TagsAroundShape());
    TagsAroundShape &tas = tagsAroundVec[tagsAroundVec.size() - 1];
    
    tas.start = stepi;
    tas.end = nextStepi;
    
    vector<int32_t> vecOfTags;
    
    for ( int32_t tag : currentSet ) {
      vecOfTags.push_back(tag);
    }
    
    tas.tags = vecOfTags;
    
    // Gather all unique coords from combined range
    
#if defined(DEBUG)
    for ( auto &pair : uniqueCoords ) {
      bool uniqueThisLoop = pair.second;
      assert(uniqueThisLoop == false);
    }
#endif // DEBUG
    
    int maxStepi = mini((nextStepi + 1), stepMax);
    
#if defined(DEBUG)
    assert(allTagSetsForVectors.size() == allCoordForVectors.size());
    assert(maxStepi <= allCoordForVectors.size());
#endif // DEBUG
    
    for ( int i = stepi ; i < maxStepi; i++ ) {
#if defined(DEBUG)
      assert(i < allCoordForVectors.size());
#endif // DEBUG
      
      if (debug) {
        cout << "allCoordForVectors[" << i << "] num coords " << allCoordForVectors[i].size() << endl;
      }
      
      for ( Coord c : allCoordForVectors[i] ) {
        if (uniqueCoords.count(c) == 0) {
          uniqueCoords[c] = true;
        }
      }
    }
    
    vector<Coord> &uniqueCoordsVec = tas.coords;
    for ( auto &pair : uniqueCoords ) {
      Coord c = pair.first;
      bool uniqueThisLoop = pair.second;
      if (uniqueThisLoop) {
        pair.second = false;
        uniqueCoordsVec.push_back(c);
      }
    }
    
#if defined(DEBUG)
    assert((nextStepi + 1) > stepi);
#endif // DEBUG
    stepi = nextStepi + 1;
  }
  
  // In the special case where the final range is larger than 1 element
  // and the range extends to 12 oclock and the sets match, then combine
  // the last range with the first one.
  
  if (tagsAroundVec.size() > 1) {
    
    if (allTagSetsForVectors.size() > 2) {
      // Check for the special case of the first and second sets being exactly equal,
      // in this case iterate backwards from 12 oclock so that and initial same range
      // at the front of the vector is moved to the start of the vector.
      
      set<int32_t> &firstSet = allTagSetsForVectors[0];
      
      set<int32_t> &lastSet = allTagSetsForVectors[stepMax - 1];
      
      if (firstSet == lastSet) {
        if (debug) {
          cout << "first and last range sets are the same" << endl;
        }
        
        assert(tagsAroundVec.size() > 0);
        TagsAroundShape &firstTas = tagsAroundVec[0];
        TagsAroundShape &lastTas = tagsAroundVec[tagsAroundVec.size() - 1];
        
        firstTas.start = lastTas.start;
        //firstTas.end = nextStepi;
        
        for ( Coord c : lastTas.coords ) {
          firstTas.coords.push_back(c);
        }
        
        int numBefore = (int) tagsAroundVec.size();
        tagsAroundVec.erase(end(tagsAroundVec) - 1);
        int numAfter = (int) tagsAroundVec.size();
        assert(numBefore == (numAfter + 1));
      }
    }
    
    // Mark entries that are simple clusters of N tags
    
    for ( TagsAroundShape &tas : tagsAroundVec ) {
      if (tas.start == tas.end) {
        tas.flag = true;
      } else {
        tas.flag = false;
      }
    }
    
    // Do a second scan of the resulting TagsAroundShape elements and combine ranges that consist of just
    // one single step
    
    stepMax = (int) tagsAroundVec.size() - 1;
    
    bool mergeNext = false;
    
    for ( stepi = 0; stepi < stepMax; stepi += 1) {
      TagsAroundShape &oneTas = tagsAroundVec[stepi];
      TagsAroundShape &nextTas = tagsAroundVec[stepi+1];
      
      if (oneTas.flag && nextTas.flag) {
        // Merge 2 in a row that differ in set contents
        
        oneTas.end = nextTas.end;
        
        set<int32_t> uniqueTags;
        
        for ( int32_t tag : oneTas.tags ) {
          uniqueTags.insert(tag);
        }
        
        for ( int32_t tag : nextTas.tags ) {
          uniqueTags.insert(tag);
        }
        
        oneTas.tags.clear();
        
        for ( int32_t tag : uniqueTags ) {
          oneTas.tags.push_back(tag);
        }
        
        for ( Coord c : nextTas.coords ) {
          oneTas.coords.push_back(c);
        }
        
        tagsAroundVec.erase(begin(tagsAroundVec) + stepi+1);
        stepMax = (int) tagsAroundVec.size() - 1;
        mergeNext = true;
      } else {
        mergeNext = false;
      }
    }
    
  } // end if more than 1 segment block
  
  if (debug) {
    cout << "return clockwiseScanForTagsAroundShape " << tag << " with N = " << tagsAroundVec.size() << " ranges" << endl;
  }
  
  return;
}

// Generate rectangle coordinates given region width and height

vector<Coord> genRectangleOutline(int regionWidth, int regionHeight)
{
  vector<Coord> outlineCoords;
  
  // top right half
  {
    int y = 0;
    int xMax = regionWidth - 1;
    
    for ( int x = regionWidth / 2; x < xMax; x++ ) {
      Coord c(x, y);
      outlineCoords.push_back(c);
    }
  }
  
  // right side
  {
    int x = regionWidth - 1;
    int yMax = regionHeight - 1;
    
    for ( int y = 0; y < yMax; y++ ) {
      Coord c(x, y);
      outlineCoords.push_back(c);
    }
  }
  
  // bottom side
  {
    int y = regionHeight - 1;
    
    for ( int x = regionWidth - 1; x > 0; x-- ) {
      Coord c(x, y);
      outlineCoords.push_back(c);
    }
  }
  
  // left side
  {
    int x = 0;
    
    for ( int y = regionHeight - 1; y > 0; y-- ) {
      Coord c(x, y);
      outlineCoords.push_back(c);
    }
  }
  
  // top left half
  {
    int y = 0;
    int xMax = regionWidth / 2;
    
    for ( int x = 0; x < xMax; x++ ) {
      Coord c(x, y);
      outlineCoords.push_back(c);
    }
  }
  
#if defined(DEBUG)
  for ( int i = 1; i < outlineCoords.size(); i++ ) {
    Coord prevCoord = outlineCoords[i-1];
    Coord currentCoord = outlineCoords[i];
    if (currentCoord == prevCoord) {
      assert(0); // must not repeat
    }
  }
#endif // DEBUG
  
  return outlineCoords;
}

// Scan region given likely bounds and determine where most accurate region bounds are likely to be

void
clockwiseScanForShapeBounds(const Mat & inputImg,
                            const Mat & tagsImg,
                            int32_t tag,
                            const vector<Coord> &regionCoords)
{
  const bool debug = true;
  const bool debugDumpImages = true;
  
  if (debug) {
    cout << "clockwiseScanForShapeBounds " << tag << endl;
  }
  
  // If the shape is convex then wrap it in a convex hull and simplify the shape with
  // smallish straight lines so that perpendicular lines can be computed as compared
  // to each line segment in order to find the shape normals.

  vector<TypedHullCoords> hullCoordsVec = clockwiseScanOfHullCoords(tagsImg, tag, regionCoords);
  
  // The hull lines should have already been simplified when possible, so determine the
  // hulls by taking the first and last point in the coords vec and then find the midpoint.
  
  if (1) {
    // Iterate over endpoints for hull lines and gather a set of coords
    // where the nearest coord on the contour will be near at the hull line
    // end.
    
    vector<Coord> nearPoints;
    unordered_map<Coord, int> isMidpointMap;
    
    Coord firstPoint;
    Coord lastPoint;
    
    assert(hullCoordsVec.size() > 0);
    firstPoint = hullCoordsVec[0].coords[0];
    
    const float minStartEndDist = 2.0f;
    
    int typedHullOffset = 0;
    for ( TypedHullCoords &typedHullCoords : hullCoordsVec ) {
      vector<Coord> &coordVec = typedHullCoords.coords;
      
      if (debug) {
        cout << "coordVec points:" << endl;
        for ( Coord c : coordVec ) {
          cout << c << endl;
        }
        cout << "";
      }
      
      int numCoords = (int) coordVec.size();
      
      assert(numCoords > 0);
      if (numCoords == 1) {
        // Ignore single hull sets
        
        if (debug) {
          cout << "skip single coord hull entry" << endl;
        }
        
        lastPoint = coordVec[0];
        typedHullOffset++;
        continue;
      }
      
      Coord cStart = coordVec[0];
      Coord cEnd = coordVec[coordVec.size() - 1];
      
      float d = deltaDistance(cStart, cEnd);
      if (d >= minStartEndDist) {
        nearPoints.push_back(cStart);
        
        if (coordVec.size() > 2) {
          // Determine midpoint of hull line
          
          Point2i p1 = coordToPoint(cStart);
          Point2i p2 = coordToPoint(cEnd);
          Point2i d = (p2 - p1)  / 2;
          Point2i midP = p2 - d;
          Coord midC = pointToCoord(midP);
          
          nearPoints.push_back(midC);
          
          isMidpointMap[midC] = typedHullOffset;
          
          if (debug) {
            cout << "append midpoint " << midC << " in between " << cStart << " to " << cEnd << endl;
          }
        }
      }
      
      lastPoint = cEnd;
      typedHullOffset++;
    }
    if (hullCoordsVec.size() > 1) {
      float d = deltaDistance(firstPoint, lastPoint);
      
      if (d >= minStartEndDist) {
        nearPoints.push_back(lastPoint);
        
        if (debug) {
          cout << "append last point " << lastPoint << endl;
        }
      }
    }

    if (debug) {
      cout << "near Points:" << endl;
      for ( Coord c : nearPoints ) {
        cout << c << endl;
      }
      cout << "";
    }
    
    // Render near points as bin Mat
    
    if (debugDumpImages) {
      Mat binMat(tagsImg.size(), CV_8UC1, Scalar(0));
      
      for ( Coord c : nearPoints ) {
        binMat.at<uint8_t>(c.y, c.x) = 0xFF;
      }
    
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_hull_near_points" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, binMat);
      cout << "wrote " << fname << endl;
      cout << "" << endl;
    }
    
    // Calculate the hull midpoint to nearest contour point map.
    // In the case of a concave hull, circle the interior point
    // otherwise circle the midpoint of the hull line.
    
    if (debugDumpImages) {
      Mat colorMat(tagsImg.size(), CV_8UC3, Scalar(0, 0, 0));
      
      for ( TypedHullCoords &typedHullCoords : hullCoordsVec ) {
        uint32_t pixel = 0;
        pixel |= (rand() % 256);
        pixel |= ((rand() % 256) << 8);
        pixel |= ((rand() % 256) << 16);
        pixel |= (0xFF << 24);
        
        Vec3b vec = PixelToVec3b(pixel);
        
        for ( Coord c : typedHullCoords.coords ) {
          colorMat.at<Vec3b>(c.y, c.x) = vec;
        }
      }
      
      int typedHullOffset = 0;
      for ( TypedHullCoords &typedHullCoords : hullCoordsVec ) {
        if (typedHullCoords.isConcave) {
          Point2i defectP = coordToPoint(typedHullCoords.defectPoint);
          circle(colorMat, defectP, 4, Scalar(0,0,0xFF), 2);
        } else {
          for ( auto &pair : isMidpointMap ) {
            if (pair.second == typedHullOffset) {
              // Found midpoint that corresponds to this segment.
              Point2i midP = coordToPoint(pair.first);
              circle(colorMat, midP, 4, Scalar(0,0,0xFF), 2);
              break;
            }
          }
        }
        
        typedHullOffset++;
      }
      
      if (debug) {
        int numConcave = 0;
        int numConvex = 0;
        
        for ( TypedHullCoords &typedHullCoords : hullCoordsVec ) {
          if (typedHullCoords.isConcave) {
            numConcave++;
          } else {
            numConvex++;
          }
        }
        
        cout << "numConcave " << numConcave << " and numConvex " << numConvex << endl;
      }
      
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_hull_mid_defect_segments" << ".png";
      string fname = fnameStream.str();
      
      writeWroteImg(fname, colorMat);
      cout << "" << endl;
    }
    
    // Generate an iteration order for each hull region. Regions are different sizes
    // so these iteration orders are different lengths.
    
    {
      int consumedLeft = 0;
      int consumedRight = 0;
      int typedHullOffset = 0;
      
      vector<stack<int32_t> > coordsIterStackVec;
      
      typedHullOffset = 0;
      for ( TypedHullCoords &typedHullCoords : hullCoordsVec ) {
        auto &coordVec = typedHullCoords.coords;
        
        vector<int32_t> coordsIterOrder;

        if (typedHullCoords.isConcave) {
          Coord searchPoint = typedHullCoords.defectPoint;
          // Find the offsets of c in coordVec
          int offset = 0;
          int foundOffset = -1;
          for ( Coord c : coordVec ) {
            if (c == searchPoint) {
              foundOffset = offset;
              break;
            }
            offset++;
          }
          assert(foundOffset != -1);
          consumedLeft = foundOffset;
          consumedRight = foundOffset;
        } else {
          // Choose the mid index in the convex list of points.
          
          int foundOffset = (int) coordVec.size() / 2;
          consumedLeft = foundOffset;
          consumedRight = foundOffset;
        }
        
        // Process all coordinates in coordVec
        
        while (1) {
          if (debug) {
            cout << "consumedLeft  " << consumedLeft << endl;
            cout << "consumedRight " << consumedRight << endl;
          }
          
          if (consumedLeft == consumedRight) {
            // Starting point where 1 element is consumed
            
            coordsIterOrder.push_back(consumedLeft);
            consumedLeft--;
            consumedRight++;
          } else {
            if (consumedLeft > -1) {
              // Still coordinates to consume on left side of contour segment
              coordsIterOrder.push_back(consumedLeft);
              consumedLeft--;
            }
            
            if (consumedRight < coordVec.size()) {
              // Still coordinates to consume on right side of contour segment
              coordsIterOrder.push_back(consumedRight);
              consumedRight++;
            }
            
            if (consumedLeft == -1 && consumedRight == coordVec.size()) {
              // out of while(1) loop
              break;
            }
          }
        }
        
        assert(coordsIterOrder.size() == coordVec.size());
        
        if (debug) {
          cout << "consumedLeft  " << consumedLeft << endl;
          cout << "consumedRight " << consumedRight << endl;
          
          for ( int32_t offset : coordsIterOrder ) {
            cout << "iter offset " << offset << endl;
          }
        }
        
        // Create stack and insert each iter offset into it
        
        stack<int32_t> iterStack;
        
        for ( auto it = coordsIterOrder.rbegin(); it != coordsIterOrder.rend(); it++) {
          int32_t offset = *it;
          iterStack.push(offset);
        }
        
        coordsIterStackVec.push_back(iterStack);
        
        typedHullOffset++;
      }
      
      // Iterate over all sets of coords at the same time and determine
      // the order that coordinates on the contour would be consumed.

      vector<Coord> contourCoords;
      
      vector<int32_t> startOffsets;
      
      vector<int32_t> contourIterOrder;
      
      for ( TypedHullCoords &typedHullCoords : hullCoordsVec ) {
        auto &coordVec = typedHullCoords.coords;
        startOffsets.push_back((int32_t)contourCoords.size());
        append_to_vector(contourCoords, coordVec);
      }
      
      for ( int contourNumCoords = (int) contourCoords.size(); contourNumCoords > 0; contourNumCoords-- ) {
        // Consume 1 coordinate from each regions if possible
        
        if (debug) {
          cout << "contourNumCoords  " << contourNumCoords << endl;
        }
        
        for ( typedHullOffset = 0; typedHullOffset < hullCoordsVec.size(); typedHullOffset++ ) {
          stack<int32_t> &iterStack = coordsIterStackVec[typedHullOffset];
          
          if (iterStack.size() > 0) {
            int32_t offset = iterStack.top();
            iterStack.pop();
            
            // Make offset into contourCoords and push onto contourIterOrder
            int32_t contourOffset = startOffsets[typedHullOffset] + offset;
            contourIterOrder.push_back(contourOffset);
            
            if (debug) {
              cout << "popped  " << offset << " from typedHullOffset " << typedHullOffset << " which maps to contour offset " << contourOffset << endl;
            }
          }
        }
      }
      
      // contourIterOrder now contains contour iteration order

      if (debug) {
        for ( int32_t offset : contourIterOrder ) {
          cout << "iter offset " << offset << endl;
        }
        cout << "";
      }
      
#if defined(DEBUG)
      {
        set<int32_t> seen;
        for ( int32_t offset : contourIterOrder ) {
          if (seen.count(offset) > 0) {
            assert(0);
          }
          seen.insert(offset);
        }
      }
#endif // DEBUG
      
      // Dump grayscale pixels in iteration order
      
      if (debugDumpImages) {
        Mat binMat(tagsImg.size(), CV_8UC1, Scalar(0));
        
        int bVal = 0xFF;
        
        for ( int32_t offset : contourIterOrder ) {
          Coord c = contourCoords[offset];
          binMat.at<uint8_t>(c.y, c.x) = bVal;
          bVal -= 3;
          if (bVal < 127) {
            bVal = 0xFF;
          }
        }
        
        std::stringstream fnameStream;
        fnameStream << "srm" << "_tag_" << tag << "_hull_iter_order" << ".png";
        string fname = fnameStream.str();
        
        writeWroteImg(fname, binMat);
        cout << "" << endl;
      }
      
      // Return normal vector at indicated point
      
      auto normalAtOffset = [](const vector<Coord> &contourCoords, int32_t offset)->Point2f {
        int32_t o1 = offset - 2;
        int32_t o2 = offset + 2;
        
        int32_t act1 = vecOffsetAround((int32_t)contourCoords.size(), o1);
        int32_t act2 = vecOffsetAround((int32_t)contourCoords.size(), o2);
        
#if defined(DEBUG)
        assert(act1 >= 0 && act1 < contourCoords.size());
        assert(act2 >= 0 && act2 < contourCoords.size());
#endif // DEBUG
        
        Coord c1 = contourCoords[act1];
        Coord c2 = contourCoords[act2];
        
        Point2f pF1(c1.x, c1.y);
        Point2f pF2(c2.x, c2.y);
        
        // Get unit normalized vector from pF1 -> pF2
        // pointing away from the contour center.
        
        Point2f deltaN = pF1 - pF2;
        
        printf("delta vector %0.3f %0.3f\n", deltaN.x, deltaN.y);
        
        normalUnitVector(deltaN);
        
        printf("normal unit vector %0.3f %0.3f\n", deltaN.x, deltaN.y);
        
        return deltaN;
      };
      
      // Define N steps where a normal to the point in question passes through
      // the point and defines the vector away from the shape.

      if (debugDumpImages) {
        Mat binMat(tagsImg.size(), CV_8UC1, Scalar(0));
        
        // Dump all normal vectors as bin Mat, note that this is a lot of images
        
        if ((0)) {
        
        for ( int32_t offset : contourIterOrder ) {
          binMat = Scalar(0);
          
          for ( int32_t offset : contourIterOrder ) {
            Coord c = contourCoords[offset];
            binMat.at<uint8_t>(c.y, c.x) = 0x7F;
          }
          
          Coord c = contourCoords[offset];
          Point2f cF(c.x, c.y);
          
          Point2f normF = normalAtOffset(contourCoords, offset);
          
          Point2f normPoint = cF + (normF * 3);
          
          // Draw vector from cF to normPoint
          
          line(binMat, cF, normPoint, Scalar(0xFF), 1);
          
          std::stringstream fnameStream;
          fnameStream << "srm" << "_tag_" << tag << "_hull_iter_normal_" << offset << ".png";
          string fname = fnameStream.str();
          
          writeWroteImg(fname, binMat);
          cout << "" << endl;
        }
          
        }
        
        Rect roi(0,0,inputImg.size().width, inputImg.size().height);
        
        vector<HullLineOrCurveSegment> vecOfSeg = splitContourIntoLinesSegments(tag, inputImg.size(), roi, contourCoords, 1.4);
        
        // Note that iteration order of coordinates in vecOfSeg may not start on contourCoords[0] so
        // create a map from known line points to the common line slope.

        vector<Point2f> normalUnitVecTable;
        
        for ( HullLineOrCurveSegment & locSeg : vecOfSeg ) {
          if (locSeg.isLine) {
            Point2f slopeVec = locSeg.slope;
            Point2f normF;
            
            // Calculate normal vector
            float tmp = slopeVec.x;
            normF.x = slopeVec.y * -1;
            normF.y = tmp;
            normF *= -1; // invert
            
            normalUnitVecTable.push_back(normF);
          }
        }
        
        unordered_map<Coord, Point2f> lineCoordToNormalMap;
        
        // Util lambda that will determine the slope for a position by ave of L and R slopes
        
        auto aveSlope = [&normalUnitVecTable, &contourCoords, &lineCoordToNormalMap](int offset)->Point2f {
          const bool debug = true;
          
          if (debug) {
            cout << "aveSlope starting at offset " << offset << endl;
          }
          
          // Walk backwards until a normal is found
          
          int32_t offsetL = offset - 1;
          int32_t offsetR = offset + 1;
          
          int32_t actualOffsetL;
          int32_t actualOffsetR;
          
          Coord cL;
          Coord cR;
          
          while (1) {
            actualOffsetL = vecOffsetAround((int32_t)contourCoords.size(), offsetL);
            
            if (actualOffsetL == offset) {
              break;
            }
            
            cL = contourCoords[actualOffsetL];
            
            if (lineCoordToNormalMap.count(cL) > 0) {
              break;
            }
            
            offsetL--;
          }
          
          while (1) {
            actualOffsetR = vecOffsetAround((int32_t)contourCoords.size(), offsetR);
            
            if (actualOffsetR == offset) {
              break;
            }
            
            cR = contourCoords[actualOffsetR];
            
            if (lineCoordToNormalMap.count(cR) > 0) {
              break;
            }
            
            offsetR++;
          }
          
          // Smooth out the difference between the two normal vectors
          
#if defined(DEBUG)
          assert(lineCoordToNormalMap.count(cL) > 0);
          assert(lineCoordToNormalMap.count(cR) > 0);
#endif // DEBUG
          
          Point2f pF1 = lineCoordToNormalMap[cL];
          Point2f pF2 = lineCoordToNormalMap[cR];
          
          if (debug) {
            printf("Coord on Left  (%d,%d) from offset %d\n", cL.x, cL.y, actualOffsetL);
            printf("Coord on Right (%d,%d) from offset %d\n", cR.x, cR.y, actualOffsetR);
            
            printf("Norm on Left  (%0.3f,%0.3f)\n", pF1.x, pF1.y);
            printf("Norm on Right (%0.3f,%0.3f)\n", pF2.x, pF2.y);
          }
          
          Point2f sumF = pF1 + pF2;
          
          if (debug) {
            printf("Sum of directional vectors (%0.3f,%0.3f)\n", sumF.x, sumF.y);
          }
          
          makeUnitVector(sumF);
          
          if (debug) {
            printf("normal unit vector (%0.3f,%0.3f)\n", sumF.x, sumF.y);
          }
          
          return sumF;
        };
        
        vector<Coord> pendingLineEdges;
        
        int locOffset;
        
        locOffset = 0;
        for ( HullLineOrCurveSegment & locSeg : vecOfSeg ) {
          if (locSeg.isLine) {
            auto &pointsVec = locSeg.points;
            int startEndN;
            
            auto insideOutVec = iterInsideOut(pointsVec);

            if (pointsVec.size() <= 3) {
              startEndN = 2;
            } else if (pointsVec.size() <= 5) {
              startEndN = 2;
            } else {
              startEndN = 2;
//              startEndN = 1;
            }
            
            int maxOffset = (int) insideOutVec.size();
            
            for ( int i = 0; i < maxOffset; i++ ) {
              Point2i p = insideOutVec[i];
              
              if (i >= (maxOffset - startEndN)) {
                // Append rest of values
                Coord c = pointToCoord(p);
                pendingLineEdges.push_back(c);
              } else {
                Point2f normal = normalUnitVecTable[locOffset];
                Coord c = pointToCoord(p);
                lineCoordToNormalMap[c] = normal;
              }
            }
          } else {
            // All points on curves treated as average between line slopes.
            
            // FIXME: should curve points be added inside out, so that ave at
            // edges is done before other points?
            
            vector<Coord> vec = convertPointsToCoords(locSeg.points);
            append_to_vector(pendingLineEdges, vec);
          }
          
          locOffset++;
        }
        
        // Iterate over original contour points and determine if any points
        // still need to have normals calculated.
        
        unordered_map<Coord, int> contourCoordsToFirstOffsetMap;
        
        int contourOffset = 0;
        
        for ( Coord c : contourCoords ) {
          if (contourCoordsToFirstOffsetMap.count(c) == 0) {
            contourCoordsToFirstOffsetMap[c] = contourOffset;
          }
          contourOffset++;
        }
        
        // Average pending points in backward order so that the points
        // farthest from the line center are processed first.
        
        for ( auto it = pendingLineEdges.begin(); it != pendingLineEdges.end(); it++ ) {
          Coord c = *it;
          cout << c << endl;
          
#if defined(DEBUG)
          assert(contourCoordsToFirstOffsetMap.count(c) > 0);
#endif // DEBUG
          
          int contourOffset = contourCoordsToFirstOffsetMap[c];
          
          Point2f normF = aveSlope(contourOffset);
          
          lineCoordToNormalMap[c] = normF;
        }
        
        // Calculate all normals
        
        vector<vector<Point2f> > allNormalVectors;
        
        for ( Coord c : contourCoords ) {
          Point2f normF;
          
#if defined(DEBUG)
          assert(lineCoordToNormalMap.count(c) > 0);
#endif // DEBUG
          
          if (lineCoordToNormalMap.count(c) == 0) {
            assert(0);
          } else {
            normF = lineCoordToNormalMap[c];
          }
          
          Point2f cF(c.x, c.y);
          
          Point2f normOutside = cF + (normF * 1);
          Point2f normInside = cF + (normF * -1);
          
          vector<Point2f> vecPoints;
          
          const bool roundToPixels = false;
          
          if (roundToPixels) {
            round(normInside);
            round(normOutside);
          }
          
          vecPoints.push_back(normInside);
          vecPoints.push_back(cF);
          vecPoints.push_back(normOutside);
          
          allNormalVectors.push_back(vecPoints);
        }
        
        // Dump the pixels contained in allNormalVectors as a massive image where the pixels
        // from the original image are copied into rows of output.
        
        if ((1)) {
          binMat = Scalar(0);
          
          for ( auto &vec : allNormalVectors ) {
            for ( Point2f p : vec ) {
              round(p);
              Point2i pi = p;
              binMat.at<uint8_t>(pi.y, pi.x) = 0xFF;
            }
          }
          
          for ( int32_t offset : contourIterOrder ) {
            Coord c = contourCoords[offset];
            binMat.at<uint8_t>(c.y, c.x) = 0x7F;
          }
          
          
          std::stringstream fnameStream;
          fnameStream << "srm" << "_tag_" << tag << "_hull_iter_normal_over" << ".png";
          string fname = fnameStream.str();
            
          writeWroteImg(fname, binMat);
          cout << "" << endl;
        }
        
        // Emit an image where each vector of pixels is a row
        
        if (1) {
          int maxWidth = 0;
          
          for ( auto &vec : allNormalVectors ) {
            int N = (int) vec.size();
            if (N > maxWidth) {
              maxWidth = N;
            }
          }
          
          Mat colorMat((int)allNormalVectors.size(), maxWidth, CV_8UC3, Scalar(0,0,0));
          
          for ( int y = 0; y < colorMat.rows; y++) {
            auto &vec = allNormalVectors[y];
            int numCols = (int) vec.size();
            
            for ( int x = 0; x < numCols; x++) {
              Point2f p = vec[x];
              round(p);
              Point2i c = p;
              Vec3b vec = inputImg.at<Vec3b>(c.y, c.x);
              colorMat.at<Vec3b>(y, x) = vec;
            }
          }
          
          std::stringstream fnameStream;
          fnameStream << "srm" << "_tag_" << tag << "_hull_iter_vec_pixels" << ".png";
          string fname = fnameStream.str();
          
          writeWroteImg(fname, colorMat);
          cout << "" << endl;
        }
        
        
        // This dump image will enlarge the original image multiple times so that vectors
        // with arrow directions can be seen clearly over the enlarged images.
        
        if (1) {
          CvSize origSize = inputImg.size();
          CvSize largerSize = origSize;
          int multBy = 1;
          
          Vec3b grayVec(0x7F, 0x7F, 0x7F);
          
          while (largerSize.width < 1000 && largerSize.width < 1000) {
            largerSize.width *= 2;
            largerSize.height *= 2;
            multBy *= 2;
          }
          
          Mat smallColorMat(origSize, CV_8UC3, Scalar(0,0,0));
          
          Mat colorMat(largerSize, CV_8UC3, Scalar(0,0,0));
          
          for ( int32_t offset : contourIterOrder ) {
            Coord c = contourCoords[offset];
            smallColorMat.at<Vec3b>(c.y, c.x) = grayVec;
          }
          
          resize(smallColorMat, colorMat, colorMat.size(), 0, 0, INTER_CUBIC);
          
          // Render each normal vector as line with arrow at end
          
          for ( int y = 0; y < allNormalVectors.size(); y++) {
            auto &vec = allNormalVectors[y];

            Point2f p1 = vec[0];
            Point2f p2 = vec[vec.size() - 1];
            
            if ((1)) {
              // Add a little more to the second vector
              Point2f delta = p2 - p1;
              p2 += delta;
            }
            
            p1 *= multBy;
            p2 *= multBy;
            
            round(p1);
            round(p2);
            
            Point2i rp1 = p1;
            Point2i rp2 = p2;
            
            double tipLength = 0.2;
            
            arrowedLine(colorMat, rp1, rp2, Scalar(0, 0, 0xFF), 1, 8, 0, tipLength);
          }
          
          std::stringstream fnameStream;
          fnameStream << "srm" << "_tag_" << tag << "_hull_vecs_larger" << ".png";
          string fname = fnameStream.str();
          
          writeWroteImg(fname, colorMat);
          cout << "" << endl;
        }


      }
    }

  }
  
  // Dump skel generated from region bin Mat
  
  if ((1)) {
    Mat binMat(tagsImg.size(), CV_8UC1, Scalar(0));
    
    for ( Coord c : regionCoords ) {
      binMat.at<uint8_t>(c.y, c.x) = 0xFF;
    }
    
    skelReduce(binMat);
    
    if (debugDumpImages) {
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_region_skel" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, binMat);
      cout << "wrote " << fname << endl;
      cout << "" << endl;
    }
  }
  
  // The tagsImg mat contains tags, so generate lines around the 360 degrees
  // of rotation and determine which tags the line pixels hit.
  
  int32_t originX, originY, regionWidth, regionHeight;
  bbox(originX, originY, regionWidth, regionHeight, regionCoords);
  Rect roiRect(originX, originY, regionWidth, regionHeight);
  
  Coord originCoord(originX, originY);
  
  Mat renderMat(roiRect.size(), CV_8UC1);
  
  renderMat = Scalar(0);
  
  // Generate coords that iterate around the region bbox starting from up which
  // is taken to be degree zero.
  
  vector<Coord> outlineCoords = genRectangleOutline(regionWidth, regionHeight);
  
  // Render points in outlineCoords to binary mat and debug dump
  
  if (debugDumpImages) {
    renderMat = Scalar(0);
    
    for ( Coord c : outlineCoords ) {
      renderMat.at<uint8_t>(c.y, c.x) = 0xFF;
    }
    
    std::stringstream fnameStream;
    fnameStream << "srm" << "_tag_" << tag << "_region_outline_coords" << ".png";
    string fname = fnameStream.str();
    
    imwrite(fname, renderMat);
    cout << "wrote " << fname << endl;
    cout << "";
  }
  
  // Map from global coordinates to the specific tag at that coordinate
  // but ignore the current shape tag since most coords will be for the
  // interior of the shape.
  
  unordered_map<Coord, int32_t> tagMap;
  
  for ( Coord c : regionCoords ) {
    Vec3b vec = tagsImg.at<Vec3b>(c.y, c.x);
    int32_t inRegionTag = Vec3BToUID(vec);
    if (inRegionTag == tag) {
      continue;
    }
    tagMap[c] = inRegionTag;
    if (debug) {
      cout << "add mapping for " << c << " -> " << vec << endl;
    }
  }
  
  // Determine region center point and distance transform for approx region.
  
  renderMat = Scalar(0);
  
  // FIXME: could pass this in, but just query for now
  
  vector<Coord> currentTagCoords;
  
  for ( Coord c : regionCoords ) {
    Vec3b vec = tagsImg.at<Vec3b>(c.y, c.x);
    int32_t regionTag = Vec3BToUID(vec);
    if (tag == regionTag) {
      currentTagCoords.push_back(c);
    }
  }
  
  for ( Coord c : currentTagCoords ) {
    c = c - originCoord;
    renderMat.at<uint8_t>(c.y, c.x) = 0xFF;
  }
  
  Mat outDistMat;
  Coord regionCenter = findRegionCenter(renderMat, Rect2d(0,0,regionWidth,regionHeight), outDistMat, tag);
  
  Point2i center(regionCenter.x, regionCenter.y);
  
  if (debugDumpImages) {
    std::stringstream fnameStream;
    fnameStream << "srm" << "_tag_" << tag << "_region_center_dist" << ".png";
    string fname = fnameStream.str();
    
    imwrite(fname, outDistMat);
    cout << "wrote " << fname << endl;
    cout << "";
  }
  
  // Use the dist mat to determine the "most alike" area inside the region and then
  // start from the common or most alike bound and then move outward as the gradient
  // increases. So, this should segment into "layers" where each layer is how alike
  // a certain range is.
  
  /*
  
  // Iterate over each vector of (center, edgePoint) along the bbox bounds
  
  vector<set<int32_t> > allTagSetsForVectors;
  
  // Store coords found for each vector
  
  vector<vector<Coord> > allCoordForVectors;
  
  int stepi = 0;
  int stepMax = (int) outlineCoords.size();
  
  for ( ; stepi < stepMax; stepi++ ) {
    
    set<int32_t> tagsForVector;
    vector<Coord> coordsForVector;
    
    Coord edgeCoord = outlineCoords[stepi];
    
    Point2i edgePoint(edgeCoord.x, edgeCoord.y);
    
    renderMat = Scalar(0);
    line(renderMat, center, edgePoint, Scalar(0xFF));
    
    if (debug) {
      cout << "render center line from " << center << " to " << edgePoint << endl;
    }
    
    if (debugDumpImages && debugDumpStepImages) {
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_step" << stepi << "_region_vec" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, renderMat);
      cout << "wrote " << fname << endl;
      cout << "";
    }
    
    vector<Point> locations;
    findNonZero(renderMat, locations);
    
    for ( Point p : locations ) {
      Coord c(p.x, p.y);
      c = originCoord + c;
      if (tagMap.count(c) > 0) {
        int32_t regionTag = tagMap[c];
        tagsForVector.insert(regionTag);
        coordsForVector.push_back(c);
      }
    }
    
    allTagSetsForVectors.push_back(tagsForVector);
    allCoordForVectors.push_back(coordsForVector);
    
    if (debugDumpImages && debugDumpStepImages) {
      Mat regionRoiMat = tagsImg(roiRect);
      
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_step" << stepi << "_region_input_tags_roi" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, regionRoiMat);
      cout << "wrote " << fname << endl;
      cout << "";
    }
    
    if (debugDumpImages && debugDumpStepImages) {
      // Dump tags that are defined as on in regionCoords
      
      Mat allTagsOn = tagsImg.clone();
      allTagsOn = Scalar(0,0,0);
      
      for ( Coord c : regionCoords ) {
        allTagsOn.at<Vec3b>(c.y, c.x) = tagsImg.at<Vec3b>(c.y, c.x);
      }
      
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_step" << stepi << "_region_input_tags" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, allTagsOn);
      cout << "wrote " << fname << endl;
      cout << "";
    }
    
    if (debugDumpImages && debugDumpStepImages) {
      renderMat = Scalar(0);
      
      for ( Point p : locations ) {
        Coord c(p.x, p.y);
        c = originCoord + c;
        if (tagMap.count(c) > 0) {
          c = c - originCoord;
          renderMat.at<uint8_t>(c.y, c.x) = 0xFF;
        }
      }
      
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_step" << stepi << "_hits_for_tag_vec" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, renderMat);
      cout << "wrote " << fname << endl;
      cout << "";
    }
    
    if (debugDumpImages && debugDumpStepImages) {
      // Dump tags that are defined as on in regionCoords
      
      Mat allTagsHit = tagsImg.clone();
      allTagsHit = Scalar(0,0,0);
      
      for ( Point p : locations ) {
        Coord c(p.x, p.y);
        c = originCoord + c;
        if (tagMap.count(c) > 0) {
          allTagsHit.at<Vec3b>(c.y, c.x) = tagsImg.at<Vec3b>(c.y, c.x);
        }
      }
      
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_step" << stepi << "_hit_tags_for_tag_vec" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, allTagsHit);
      cout << "wrote " << fname << endl;
      cout << "";
    }
    
  }
  
  // Identify all the tags found in the region and with a tag other than this one
  
  unordered_map<int32_t, bool> allTagsCombined;
  
  for ( set<int32_t> & tagSet : allTagSetsForVectors ) {
    for ( int32_t regionTag : tagSet ) {
      allTagsCombined[regionTag] = true;
    }
  }
  
  if (debug) {
    cout << "all tags found around region" << endl;
    
    for ( auto & pair : allTagsCombined ) {
      int32_t regionTag = pair.first;
      printf("tag = 0x%08X aka %d\n", regionTag, regionTag);
    }
  }
  
  if (debugDumpImages) {
    // Dump tags that are defined as on in regionCoords
    
    Mat allTagsHit = tagsImg.clone();
    allTagsHit = Scalar(0,0,0);
    
    for ( Coord c : regionCoords ) {
      Vec3b vec = tagsImg.at<Vec3b>(c.y, c.x);
      int32_t regionTag = Vec3BToUID(vec);
      
      if (allTagsCombined.count(regionTag) > 0) {
        allTagsHit.at<Vec3b>(c.y, c.x) = vec;
        
        if (debug) {
          printf("found region tag %9d at coord (%5d, %5d)\n", regionTag, c.x, c.y);
        }
      }
    }
    
    std::stringstream fnameStream;
    fnameStream << "srm" << "_tag_" << tag << "_hit_tags_in_scan_region" << ".png";
    string fname = fnameStream.str();
    
    imwrite(fname, allTagsHit);
    cout << "wrote " << fname << endl;
    cout << "";
  }
  
  // Condense tags in regions starting from the top. A region range is condensed as long
  // as the tags in the range are the same or if there are no tags.
  
  unordered_map<Coord, bool> uniqueCoords;
  
  for ( stepi = 0 ; stepi < stepMax; ) {
    if (debug) {
      cout << "consider stepi " << stepi << endl;
    }
    
    set<int32_t> &currentSet = allTagSetsForVectors[stepi];
    
    int nextStepi = stepi + 1;
    
    for ( ; nextStepi < stepMax; nextStepi++ ) {
      set<int32_t> &nextSet = allTagSetsForVectors[nextStepi];
      
      // If the sets are identical then merge the regions.
      
      if (currentSet == nextSet) {
        if (debug && true) {
          cout << "same set for step " << nextStepi << endl;
        }
        if (debug && false) {
          cout << "set 1" << endl;
          for ( int32_t tag : currentSet ) {
            cout << tag << endl;
          }
          cout << "set 2" << endl;
          for ( int32_t tag : nextSet ) {
            cout << tag << endl;
          }
        }
      } else {
        nextStepi -= 1;
        break;
      }
    }
    
    // Range is (stepi, nextStepi)
    
    if (debug) {
      cout << "step same range (" << stepi << "," << nextStepi << ")" << endl;
    }
    
    tagsAroundVec.push_back(TagsAroundShape());
    TagsAroundShape &tas = tagsAroundVec[tagsAroundVec.size() - 1];
    
    tas.start = stepi;
    tas.end = nextStepi;
    
    vector<int32_t> vecOfTags;
    
    for ( int32_t tag : currentSet ) {
      vecOfTags.push_back(tag);
    }
    
    tas.tags = vecOfTags;
    
    // Gather all unique coords from combined range
    
#if defined(DEBUG)
    for ( auto &pair : uniqueCoords ) {
      bool uniqueThisLoop = pair.second;
      assert(uniqueThisLoop == false);
    }
#endif // DEBUG
    
    int maxStepi = mini((nextStepi + 1), stepMax);
    
#if defined(DEBUG)
    assert(allTagSetsForVectors.size() == allCoordForVectors.size());
    assert(maxStepi <= allCoordForVectors.size());
#endif // DEBUG
    
    for ( int i = stepi ; i < maxStepi; i++ ) {
#if defined(DEBUG)
      assert(i < allCoordForVectors.size());
#endif // DEBUG
      
      if (debug) {
        cout << "allCoordForVectors[" << i << "] num coords " << allCoordForVectors[i].size() << endl;
      }
      
      for ( Coord c : allCoordForVectors[i] ) {
        if (uniqueCoords.count(c) == 0) {
          uniqueCoords[c] = true;
        }
      }
    }
    
    vector<Coord> &uniqueCoordsVec = tas.coords;
    for ( auto &pair : uniqueCoords ) {
      Coord c = pair.first;
      bool uniqueThisLoop = pair.second;
      if (uniqueThisLoop) {
        pair.second = false;
        uniqueCoordsVec.push_back(c);
      }
    }
    
#if defined(DEBUG)
    assert((nextStepi + 1) > stepi);
#endif // DEBUG
    stepi = nextStepi + 1;
  }
  
  // In the special case where the final range is larger than 1 element
  // and the range extends to 12 oclock and the sets match, then combine
  // the last range with the first one.
  
  if (tagsAroundVec.size() > 1) {
    
    if (allTagSetsForVectors.size() > 2) {
      // Check for the special case of the first and second sets being exactly equal,
      // in this case iterate backwards from 12 oclock so that and initial same range
      // at the front of the vector is moved to the start of the vector.
      
      set<int32_t> &firstSet = allTagSetsForVectors[0];
      
      set<int32_t> &lastSet = allTagSetsForVectors[stepMax - 1];
      
      if (firstSet == lastSet) {
        if (debug) {
          cout << "first and last range sets are the same" << endl;
        }
        
        assert(tagsAroundVec.size() > 0);
        TagsAroundShape &firstTas = tagsAroundVec[0];
        TagsAroundShape &lastTas = tagsAroundVec[tagsAroundVec.size() - 1];
        
        firstTas.start = lastTas.start;
        //firstTas.end = nextStepi;
        
        for ( Coord c : lastTas.coords ) {
          firstTas.coords.push_back(c);
        }
        
        int numBefore = (int) tagsAroundVec.size();
        tagsAroundVec.erase(end(tagsAroundVec) - 1);
        int numAfter = (int) tagsAroundVec.size();
        assert(numBefore == (numAfter + 1));
      }
    }
    
    // Mark entries that are simple clusters of N tags
    
    for ( TagsAroundShape &tas : tagsAroundVec ) {
      if (tas.start == tas.end) {
        tas.flag = true;
      } else {
        tas.flag = false;
      }
    }
    
    // Do a second scan of the resulting TagsAroundShape elements and combine ranges that consist of just
    // one single step
    
    stepMax = (int) tagsAroundVec.size() - 1;
    
    bool mergeNext = false;
    
    for ( stepi = 0; stepi < stepMax; stepi += 1) {
      TagsAroundShape &oneTas = tagsAroundVec[stepi];
      TagsAroundShape &nextTas = tagsAroundVec[stepi+1];
      
      if (oneTas.flag && nextTas.flag) {
        // Merge 2 in a row that differ in set contents
        
        oneTas.end = nextTas.end;
        
        set<int32_t> uniqueTags;
        
        for ( int32_t tag : oneTas.tags ) {
          uniqueTags.insert(tag);
        }
        
        for ( int32_t tag : nextTas.tags ) {
          uniqueTags.insert(tag);
        }
        
        oneTas.tags.clear();
        
        for ( int32_t tag : uniqueTags ) {
          oneTas.tags.push_back(tag);
        }
        
        for ( Coord c : nextTas.coords ) {
          oneTas.coords.push_back(c);
        }
        
        tagsAroundVec.erase(begin(tagsAroundVec) + stepi+1);
        stepMax = (int) tagsAroundVec.size() - 1;
        mergeNext = true;
      } else {
        mergeNext = false;
      }
    }
    
  } // end if more than 1 segment block
   
   */
  
  if (debug) {
    cout << "return clockwiseScanForShapeBounds " << tag << " with N = " << 0 << " ranges" << endl;
  }
  
  return;
}

// This method accepts a region defined by coords and returns the edges between
// superpixels in the region.

vector<SuperpixelEdge>
getEdgesInRegion(SuperpixelImage &spImage,
                 const Mat & tagsImg,
                 int32_t tag,
                 const vector<Coord> &coords)
{
  const bool debug = true;
  
  // Generate vectors that determine how different colors that are nearby each other
  // in 2D space map to other nearby colors. It is only possible to determine that
  // regions are "neighbors" and that those regions then have gradient vectors from
  // one to the other. Colors that are near each other in 3D space may not be near
  // each other in 2D space, so only a pair of dominate colors at a time can be
  // considered.
  
  // Gather the tags associated with all the regions
  // indicated by regionCoords.
  
  unordered_map<int32_t, int32_t> allRegionTagsMap;
  
  for ( Coord c : coords ) {
#if defined(DEBUG)
    int maxX = tagsImg.cols - 1;
    int maxY = tagsImg.rows - 1;
    assert(c.x <= maxX);
    assert(c.y <= maxY);
#endif // DEBUG
    Vec3b vec = tagsImg.at<Vec3b>(c.y, c.x);
    int32_t tag = Vec3BToUID(vec);
    assert(tag != 0);
    allRegionTagsMap[tag] = tag;
  }
  
  if (allRegionTagsMap.size() < 2) {
    // Cannot possibly find an edge if there are not at least 2 tags
    
    if (debug) {
      cout << "did not find at least 2 tags, so no edges are inside region" << endl;
    }
    
    return vector<SuperpixelEdge>();
  }
  
  vector<int32_t> allUniqueRegionTags;
  
  for ( auto &pair : allRegionTagsMap ) {
    allUniqueRegionTags.push_back(pair.first);
  }
  
  if (debug) {
    cout << "allUniqueRegionTags:" << endl;
    for ( int32_t tag : allUniqueRegionTags ) {
      cout << tag << endl;
    }
    cout << "";
  }
  
  // Gather all tags in the extended region (masked as blocks)
  
  set<int32_t> neighborsSet;
  
  unordered_map<SuperpixelEdge, bool> allNeighborsPairsMap;
  
  for ( int32_t tag : allUniqueRegionTags ) {
    neighborsSet.insert(tag);
  }
  
  // Save all pairs where both sides appear in neighborsSet.
  
  for ( int32_t neighborTag : neighborsSet ) {
    auto &neighborsOfNeighborSet = spImage.edgeTable.getNeighborsSet(neighborTag);
    
    for ( int32_t neighborOfNeighborTag : neighborsOfNeighborSet ) {
      if (neighborsSet.count(neighborOfNeighborTag) > 0) {
        // Both neighborTag and neighborOfNeighborTag are in neighborsSet
        SuperpixelEdge edge(neighborTag, neighborOfNeighborTag);
        allNeighborsPairsMap[edge] = true;
      }
    }
  }
  
  if (debug) {
    cout << "all neighbor pairs: " << endl;
    
    for ( auto & pair : allNeighborsPairsMap ) {
      cout << pair.first << " -> " << pair.second << endl;
    }
  }
  
  vector<SuperpixelEdge> neighborsVecOfPairs;
  
  for ( auto & pair : allNeighborsPairsMap ) {
    neighborsVecOfPairs.push_back(pair.first);
  }
  
  if (debug) {
    cout << "neighborsVecOfPairs: " << endl;
    
    for ( auto & edge : neighborsVecOfPairs ) {
      cout << edge << endl;
    }
    
    cout << "done" << endl;
  }
  
  return neighborsVecOfPairs;
}

// This method will contract or expand a region defined by coordinates by N pixel.
// In the case where the region cannot be expanded or contracted anymore this
// method returns false.

bool
contractOrExpandRegion(const Mat & inputImg,
                       int32_t tag,
                       const vector<Coord> &coords,
                       bool isExpand,
                       int numPixels,
                       vector<Coord> &outCoords)
{
  const bool debug = false;
  const bool debugDumpImages = true;
  const bool debugDumpInputStateImages = true;
  
  if (debug) {
    cout << "contractOrExpandRegion " << tag << " with N = " << coords.size() << " and isExpand " << isExpand << endl;
  }
  
  outCoords.clear();
  
  int32_t originX, originY, regionWidth, regionHeight;
  bbox(originX, originY, regionWidth, regionHeight, coords);
  
  if (debug) {
    cout << "bbox " << originX << "," << originY << " with " << regionWidth << " x " << regionHeight << endl;
  }
  
  if (isExpand) {
    originX -= numPixels;
    if (originX < 0) {
      originX = 0;
    }
    originY -= numPixels;
    if (originY < 0) {
      originY = 0;
    }
    regionWidth += (numPixels * 2);
    if (regionWidth > inputImg.cols) {
      regionWidth = inputImg.cols;
    }
    regionHeight += (numPixels * 2);
    if (regionHeight > inputImg.rows) {
      regionHeight = inputImg.rows;
    }
    
    if (debug) {
      cout << "expanded bbox " << originX << "," << originY << " with " << regionWidth << " x " << regionHeight << endl;
    }
  }
  
  Rect expandedRoi(originX, originY, regionWidth, regionHeight);
  
  Mat inBoolMat(expandedRoi.size(), CV_8UC1);
  inBoolMat = Scalar(0);
  
  Mat outBoolMat;
  
  Coord origin(originX, originY);
  
  for ( Coord c : coords ) {
    Coord bC = c - origin;
    inBoolMat.at<uint8_t>(bC.y, bC.x) = 0xFF;
  }
  
  if (isExpand) {
    outBoolMat = expandWhiteInRegion(inBoolMat, numPixels, tag);
  } else {
    outBoolMat = decreaseWhiteInRegion(inBoolMat, numPixels, tag);
  }
  
  if (debugDumpInputStateImages) {
    string expandStr = isExpand ? "expand" : "contract";
    
    std::stringstream fnameStream;
    fnameStream << "srm" << "_tag_" << tag << "_srm_region_" << expandStr << "_input" << numPixels << ".png";
    string fname = fnameStream.str();
    
    imwrite(fname, inBoolMat);
    cout << "wrote " << fname << endl;
    cout << "";
  }
  
  vector<Point> locations;
  findNonZero(outBoolMat, locations);

  int maxX = inputImg.cols;
  int maxY = inputImg.rows;
  
  for ( Point p : locations ) {
    Coord c(p.x, p.y);
    c = origin + c;
    
    if ((c.x < maxX) && (c.y < maxY)) {
      // Inside the original image bounds
      outCoords.push_back(c);
      
      if (debug) {
        cout << "keep coord " << c << endl;
      }
    } else {
      if (debug) {
        cout << "skip coord " << c << endl;
      }
    }
  }
  
  int numNonZero = (int) outCoords.size();
  
  if (isExpand) {
    // Stop when expanded out to the entire region filled
    
    if (numNonZero == (inputImg.rows * inputImg.cols)) {
      return false;
    }
  } else {
    // Stop when contracting down to zero pixels
    
    if (numNonZero == 0) {
      return false;
    }
  }
  
  if (debugDumpImages)
  {
    string expandStr = isExpand ? "expand" : "contract";
    
    std::stringstream fnameStream;
    fnameStream << "srm" << "_tag_" << tag << "_srm_region_" << expandStr << "_mask" << numPixels << ".png";
    string fname = fnameStream.str();
    
    Mat tmpMat(inputImg.size(), CV_8UC1);
    tmpMat = Scalar(0);
    
    for ( Coord c : outCoords ) {
      tmpMat.at<uint8_t>(c.y, c.x) = 0xFF;
    }
    
    imwrite(fname, tmpMat);
    cout << "wrote " << fname << endl;
    cout << "";
  }
  
  // Dump alpha masked version of the original input.
  
  if (debugDumpImages)
  {
    string expandStr = isExpand ? "expand" : "contract";
    
    Mat alphaMaskResultImg(inputImg.size(), CV_8UC4);
    alphaMaskResultImg = Scalar(0, 0, 0, 0);
    
    for ( Coord c : outCoords ) {
      Vec3b vec = inputImg.at<Vec3b>(c.y, c.x);
      Vec4b vec4;
      vec4[0] = vec[0];
      vec4[1] = vec[1];
      vec4[2] = vec[2];
      vec4[3] = 0xFF;
      alphaMaskResultImg.at<Vec4b>(c.y, c.x) = vec4;
    }
    
    {
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_srm_region_" << expandStr << "_alpha_mask" << numPixels << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, alphaMaskResultImg);
      cout << "wrote " << fname << endl;
      cout << "";
    }
  }
  
#if defined(DEBUG)
  {
    int maxX = inputImg.cols;
    int maxY = inputImg.rows;
    for ( Coord c : outCoords ) {
      assert(c.x < maxX);
      assert(c.y < maxY);
    }
  }
#endif // DEBUG
  
  if (debug) {
    cout << "return contractOrExpandRegion" << endl;
  }
  
  return true;
}

// Invoked for each child of a container, returns the tags that are direct children of tag

void
recurseSuperpixelContainmentImpl(SuperpixelImage &spImage,
                                 unordered_map<int32_t, std::vector<int32_t> > &map,
                                 int32_t tag,
                                unordered_map<int32_t, int32_t> &superpixelTagToOffsetMap)
{
  const bool debug = true;
  
  Superpixel *spPtr = spImage.getSuperpixelPtr(tag);
  
  if (debug) {
    cout << "recurseSuperpixelContainmentImpl for tag " << tag << " with N = " << spPtr->coords.size() << endl;
  }
  
#if defined(DEBUG)
  if (map.count(tag) > 0) {
    cout << " already processed tag " << tag << endl;
    assert(0);
  }
#endif // DEBUG
  
  // Before processing neightbors, mark this superpixel as processed
  
  vector<int32_t> &children = map[tag];
  
  auto &neighborsSet = spImage.edgeTable.getNeighborsSet(tag);

  if (debug) {
    cout << "neighbors: " << endl;
    
    for ( int32_t neighborTag : neighborsSet ) {
      cout << neighborTag << endl;
    }
  }
  
  // Order neighborsSet by the offset in superpixelTagToOffsetMap
  
  // Filter neighborsSet into a set of siblings by removing
  // the neighbors that are known to already be processed.
  
  vector<int32_t> siblings;
  
  for ( int32_t neighborTag : neighborsSet ) {
    if (debug) {
      cout << "check for unprocessed neighbor " << neighborTag << endl;
    }
    
    if (map.count(neighborTag) > 0) {
      if (debug) {
        cout << "already processed neighbor tag " << neighborTag << endl;
      }
    } else {
      siblings.push_back(neighborTag);
    }
  }
  
  // order siblings by superpixelTagToOffsetMap
  sort(begin(siblings), end(siblings),
       [&](int tag1, int tag2) {
         int32_t offset1 = superpixelTagToOffsetMap[tag1];
         int32_t offset2 = superpixelTagToOffsetMap[tag2];
         return offset1 < offset2;
       });
  
  if (debug) {
    int offset = 0;
    
    for ( int32_t siblingTag : siblings ) {
      Superpixel *spPtr = spImage.getSuperpixelPtr(siblingTag);
      
      printf("sibling[%5d] : %d contains N = %d coords\n", offset, siblingTag, (int)spPtr->coords.size());
      
      offset += 1;
    }
  }
  
  // FIXME: might be better to impl siblings as a stack instead of a vector if N is large
  
  // Iterate over neighbors and recursively process, note that any
  // other siblings are inserted into the processed map in order
  // to properly handle recursion WRT siblings.

  while ( 1 ) {
    int32_t neighborTag;
    
    {
      // Scope ensures that iterators are released before loop body
      auto it = begin(siblings);
      auto itEnd = end(siblings);
      
      if (it == itEnd) {
        break;
      }
      
      neighborTag = *it;
      
      siblings.erase(it);
    }
    
    if (debug) {
      cout << "process neighbor " << neighborTag << endl;
    }
    
    children.push_back(neighborTag);
    
    for ( int32_t sibling : siblings ) {
#if defined(DEBUG)
      assert(sibling != neighborTag);
      assert(map.count(sibling) == 0);
#endif // DEBUG
      map[sibling] = vector<int32_t>();
    }
    
    // Recurse into neighbors of neighbor at this point
    
    recurseSuperpixelContainmentImpl(spImage, map, neighborTag, superpixelTagToOffsetMap);
    
    for ( int32_t sibling : siblings ) {
#if defined(DEBUG)
      assert(sibling != neighborTag);
      assert(map.count(sibling) > 0);
      assert(map[sibling].size() == 0);
#endif // DEBUG
      map.erase(sibling);
    }
  }

#if defined(DEBUG)
  assert(siblings.size() == 0);
#endif // DEBUG

  if (debug) {
    cout << "recurseSuperpixelContainmentImpl for " << tag << " finished with children: " << endl;
    
    for ( int32_t childTag : children ) {
      cout << childTag << endl;
    }
  }
  
  return;
}

// Recurse into each superpixel and determine the children of each superpixel.

std::vector<int32_t>
recurseSuperpixelContainment(SuperpixelImage &spImage,
                             const Mat &tagsImg,
                             unordered_map<int32_t, std::vector<int32_t> > &map)
{
  const bool debug = false;

  set<int32_t> rootSet;
  vector<int32_t> rootTags;
  
  // Determine the outermost set of tags by gathering all the tags along the edges of the image. In the
  // tricky case where more than 1 superpixel is a sibling at the toplevel this logic figures out where
  // to being with the recursion.
  
  int width = tagsImg.cols;
  int height = tagsImg.rows;
  
  int32_t lastTag = 0;
  
  for ( int y = 0; y < height; y++ ) {
    bool isFirstRow = (y == 0);
    bool isLastRow = (y == (height-1));
    
    for ( int x = 0; x < width; x++ ) {
      bool isFirstCol = (x == 0);
      bool isLastCol = (x == (width-1));
      
      if (isFirstRow || isLastRow) {
        // All pixels in first and last row processed
        
        if (debug) {
          cout << "check " << x << "," << y << endl;
        }
      } else if (isFirstCol || isLastCol) {
        // All pixels in first and last col processed
        
        if (debug) {
          cout << "check " << x << "," << y << endl;
        }
      } else {
        // Not on the edges
        
        if (debug) {
          cout << "skip " << x << "," << y << endl;
        }
        
        continue;
      }
      
      Vec3b vec = tagsImg.at<Vec3b>(y, x);
      int32_t tag = Vec3BToUID(vec);
      
      if (tag != lastTag) {
        if (debug) {
          cout << "check " << x << "," << y << " with tag " << tag << endl;
        }
        
        rootSet.insert(tag);
      }
      lastTag = tag;
    }
  }
  
  // Sort by superpixel size and then determine order by decreasing size
  
  vector<int32_t> sortedSuperpixelTags = spImage.sortSuperpixelsBySize();

  // Map superpixel UID to the offset in the sorted list, smaller offset
  // means that the superpixel is larger.
  
  unordered_map<int32_t, int32_t> superpixelTagToOffsetMap;
  
//  unordered_map<int32_t,int32_t> rootTagToSize;
  
  {
    int offset = 0;
    
    for ( int32_t tag : rootSet ) {
      superpixelTagToOffsetMap[tag] = offset;
      offset += 1;
    }
  }
  
  for ( int32_t tag : sortedSuperpixelTags ) {
    if (superpixelTagToOffsetMap.count(tag) > 0) {
      // This superpixel is in rootSet
      rootTags.push_back(tag);
    }
  }
  
  sortedSuperpixelTags = vector<int32_t>(); // Dealloc possibly large list
  
  assert(rootTags.size() == rootSet.size());

  set<int32_t> siblings;
  for ( int32_t tag : rootTags ) {
    siblings.insert(tag);
  }
  
  for ( int32_t tag : rootTags ) {
    siblings.erase(tag);
    
    for ( int32_t sibling : siblings ) {
      if (debug) {
        cout << "sibling " << sibling << endl;
      }
#if defined(DEBUG)
      assert(map.count(sibling) == 0);
#endif
      map[sibling] = vector<int32_t>();
    }
    
    recurseSuperpixelContainmentImpl(spImage, map, tag, superpixelTagToOffsetMap);
    
    for ( int32_t sibling : siblings ) {
#if defined(DEBUG)
      assert(map.count(sibling) > 0);
      assert(map[sibling].size() == 0);
#endif

      map.erase(sibling);
    }

  }
  
  return rootTags;
}

// Segment an input image with multiple passes of SRM approach and place the
// result tags in tagsMat.

bool srmMultiSegment(const Mat & inputImg, Mat & tagsMat) {
  // Run SRM logic to generate initial segmentation based on statistical "alikeness".
  // Very large regions are likely to be very alike or even contain many pixels that
  // are identical.
  
  const bool debugWriteIntermediateFiles = false;
  
  //    double Q = 16.0;
  //    double Q = 32.0;
  //    double Q = 64.0; // Not too small
  double Q = 128.0; // keeps small circles together
  //    double Q = 256.0;
  //    double Q = 512.0;
  
  //double Qmore = Q + 128.0; // break up into more regions
  double Qmore = 512.0; // break up into more regions
  
  Mat srmTags1 = generateSRM(inputImg, Q);
  
  Mat srmTags2 = generateSRM(inputImg, Qmore);
  
  // Collect the more precise segmentations into groups
  
  // Alloc object on stack
  SuperpixelImage spImage2;
  
  bool worked = SuperpixelImage::parse(srmTags2, spImage2);
  
  if (!worked) {
    return false;
  }
  
  if ((1)) {
    // Bypass second stage SRM
    
    tagsMat = srmTags1;
    
    return true;
  }
  
  // Scan each grouping to determine when pixels identified as being in
  // the same group in srmTags1 are not included in the group in srmTags2.
  
  vector<int32_t> sortedSuperpixelTags = spImage2.sortSuperpixelsBySize();
  
  vector<vector<Coord> > allMergeCoordsVec;
  
  for ( int32_t tag : sortedSuperpixelTags ) {
    
    // Find bbox for this superpixel region
    
    Superpixel *spPtr = spImage2.getSuperpixelPtr(tag);
    
    int32_t originX, originY, regionWidth, regionHeight;
    bbox(originX, originY, regionWidth, regionHeight, spPtr->coords);
    Rect roiRect(originX, originY, regionWidth, regionHeight);
    
    if (false && (originX == 0 && originY == 0 && regionWidth == inputImg.cols && regionHeight == inputImg.rows)) {
      if (1) {
        cout << "skip roi region that is the size of the whole image" << endl;
      }
      
      continue;
    }
    
    Mat roiInputMat = inputImg(roiRect);
    
    if (debugWriteIntermediateFiles) {
      std::stringstream fnameStream;
      fnameStream << "srm_multi" << "_tag_" << tag << "_check_region_bbox" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, roiInputMat);
      cout << "wrote " << fname << endl;
      cout << "";
    }

    // Mask the superpixel coordinates as alpha, so that only the in region
    // pixels appear in the dump output.
    
    if (debugWriteIntermediateFiles) {
      Mat alphaMaskedRegionPixels(inputImg.size(), CV_8UC4);
      alphaMaskedRegionPixels = Scalar(0, 0, 0, 0);
      
      for ( Coord c : spPtr->coords ) {
        Vec3b vec = inputImg.at<Vec3b>(c.y, c.x);
        Vec4b vec4;
        vec4[0] = vec[0];
        vec4[1] = vec[1];
        vec4[2] = vec[2];
        vec4[3] = 0xFF;
        alphaMaskedRegionPixels.at<Vec4b>(c.y, c.x) = vec4;
      }
      
      std::stringstream fnameStream;
      fnameStream << "srm_multi" << "_tag_" << tag << "_check_region_alpha_input" << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, alphaMaskedRegionPixels);
      cout << "wrote " << fname << endl;
      cout << "";
    }
    
    // Find all the UID values in srmTags1 that correspond to this region.
    // The regions defined by spImage2 should be more precise, so if there
    // are pixels with with the same tag then these pixels were split out
    // of the group.
    
    unordered_map<int32_t, int32_t> srm1TagsInRegion;
    
    for ( Coord c : spPtr->coords ) {
      Vec3b vec = srmTags1.at<Vec3b>(c.y, c.x);
      uint32_t tag = Vec3BToUID(vec);
      srm1TagsInRegion[tag] = 0;
    }
    
    for ( auto & pair : srm1TagsInRegion ) {
      uint32_t regionTag = pair.first;
      fprintf(stdout, "region %d unique tag %d\n", tag, regionTag);
    }

    fprintf(stdout, "done\n");
    
    // Copy UID for this region by appending the coords to a vector of all
    // merge coords for a specific group.
  
    // If multiple, chhose the largest set of coords and call that a region
    
    vector<Coord> coordsInRegion;
    
    assert(srm1TagsInRegion.size() != 0);

    if (srm1TagsInRegion.size() == 1) {
      for ( Coord c : spPtr->coords ) {
        coordsInRegion.push_back(c);
      }

    } else {
      for (auto it = begin(srm1TagsInRegion); it != end(srm1TagsInRegion); ++it) {
        uint32_t regionTag = it->first;
        
        for ( Coord c : spPtr->coords ) {
          Vec3b vec = srmTags1.at<Vec3b>(c.y, c.x);
          uint32_t foundRegionTag = Vec3BToUID(vec);
          if (regionTag == foundRegionTag) {
            coordsInRegion.push_back(c);
          }
        }
      }
    }
    
    allMergeCoordsVec.push_back(coordsInRegion);

  } // foreach tag in sortedSuperpixelTags
  
#if defined(DEBUG)
  int numCoordsExpected = inputImg.rows * inputImg.cols;
  int numCoordsTotal = 0;
  
  for ( auto vec : allMergeCoordsVec ) {
    for ( Coord c : vec ) {
      c = c;
      numCoordsTotal += 1;
    }
  }
  
  assert(numCoordsTotal == numCoordsExpected);
#endif // DEBUG

  tagsMat = inputImg.clone();
  
  uint32_t mergeUID = 0;
  
  for ( auto vec : allMergeCoordsVec ) {
    Vec3b mergeVec = PixelToVec3b(mergeUID);
    
    for ( Coord c : vec ) {
      tagsMat.at<Vec3b>(c.y, c.x) = mergeVec;
    }

    mergeUID += 1;
  }
  
#if defined(DEBUG)
  // Verify that each tags value is larger than zero
  
  for ( int y = 0; y < tagsMat.rows; y++ ) {
    for ( int x = 0; x < tagsMat.cols; x++ ) {
      Vec3b vec = tagsMat.at<Vec3b>(y, x);
      uint32_t pixel = Vec3BToUID(vec);
      assert(pixel > 0);
    }
  }
#endif // DEBUG

  return true;
}
