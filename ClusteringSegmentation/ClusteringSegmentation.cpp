//
//  ClusteringSegmentation.cpp
//  ClusteringSegmentation
//
//  Created by Mo DeJong on 1/17/16.
//  Copyright © 2016 helpurock. All rights reserved.
//

#include "ClusteringSegmentation.hpp"

#include <opencv2/opencv.hpp>

#include "Superpixel.h"
#include "SuperpixelEdge.h"
#include "SuperpixelImage.h"

#include "OpenCVUtil.h"
#include "Util.h"

#include "quant_util.h"
#include "DivQuantHeader.h"

#include "MergeSuperpixelImage.h"

#include "srm.h"

#include "peakdetect.h"

#include "Util.h"

using namespace cv;
using namespace std;

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
  const bool debugDumpImage = false;
  
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
  
  //uint32_t *outPixels = new uint32_t[numPixels]();
  
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
      
      //uint32_t pixel = (R << 16) | (G << 8) | B;
      //outPixels[i] = pixel;
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
    string filename = "srm.png";
    imwrite(filename, outImg);
    cout << "wrote " << filename << endl;
  }
  
  //  if (debugDumpImage) {
  //    dumpQuantImage("srm.png", inputImg, outPixels);
  //  }
  
  //  delete [] outPixels;
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
  const bool dumpOutputImages = false;
  
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
    
    for ( auto &pair : inToOutTable ) {
      uint32_t inPixel = pair.first;
      uint32_t outPixel = pair.second;
      
      // Determine the delta for each component
      
      uint32_t deltaPixel = predict_trivial_component_sub(inPixel, outPixel);
      
      if (debug) {
        printf("unique pixel delta 0x%08X -> 0x%08X = 0x%08X\n", inPixel, outPixel, deltaPixel);
      }
    }
    
    
    //  clusterCenters.push_back();
    
  } // end doClustering if block
  
  delete [] colortable;
  delete [] inPixels;
  delete [] outPixels;
  
  return isVeryClose;
}


// Given a tag indicating a superpixel generate a mask that captures the region in terms of
// exact pixels. This method returns a Mat that indicate a boolean region mask where 0xFF
// means that the pixel is inside the indicated region.

bool
captureRegionMask(SuperpixelImage &spImage,
                  const Mat & inputImg,
                  const Mat & srmTags,
                  int32_t tag,
                  int blockWidth,
                  int blockHeight,
                  int superpixelDim,
                  Mat &outBlockMask)
{
  const bool debug = true;
  const bool debugDumpImages = true;
  
  if (debug) {
    cout << "captureRegionMask" << endl;
  }
  
  assert(outBlockMask.rows == inputImg.rows);
  assert(outBlockMask.cols == inputImg.cols);
  assert(outBlockMask.channels() == 1);
  
  auto &coords = spImage.getSuperpixelPtr(tag)->coords;
  
  if (coords.size() <= (superpixelDim*superpixelDim)) {
    // A region contained in only a single block, don't process by itself
    
    if (debug) {
      cout << "captureRegionMask : region indicated by tag " << tag << " is too small to process" << endl;
    }
    
    return false;
  }

  // Init mask after possible early return
  
  outBlockMask = (Scalar) 0;
  
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
  Superpixel::bbox(originX, originY, width, height, minMaxCoords);
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
    
    Mat tmpExpandedBlockMat(height, width, CV_8U);
    
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
  
  if ((1)) {
    int width = inputImg.cols;
    int height = inputImg.rows;
    
    vector<Coord> regionCoords;
    regionCoords.reserve(locations.size() * (superpixelDim * superpixelDim));
    
    for ( Point p : locations ) {
      int actualX = p.x * superpixelDim;
      int actualY = p.y * superpixelDim;
      
      Coord min(actualX, actualY);
      Coord max(actualX+superpixelDim-1, actualY+superpixelDim-1);
      
      for ( int y = min.y; y <= max.y; y++ ) {
        for ( int x = min.x; x <= max.x; x++ ) {
          Coord c(x, y);
          
          if (x > width-1) {
            continue;
          }
          if (y > height-1) {
            continue;
          }
          
          regionCoords.push_back(c);
        }
      }
    }
    
    int numPixels = (int) regionCoords.size();
    
    uint32_t *inPixels = new uint32_t[numPixels];
    uint32_t *outPixels = new uint32_t[numPixels];
    
    for ( int i = 0; i < numPixels; i++ ) {
      Coord c = regionCoords[i];
      
      Vec3b vec = inputImg.at<Vec3b>(c.y, c.x);
      uint32_t pixel = Vec3BToUID(vec);
      inPixels[i] = pixel;
    }
    
    if (debugDumpImages) {
      Mat tmpResultImg = inputImg.clone();
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
    
    // Quant to evenly spaced grid to get estimate for number of clusters N
    
    vector<uint32_t> colors = getSubdividedColors();
    
    uint32_t numColors = (uint32_t) colors.size();
    uint32_t *colortable = new uint32_t[numColors];
    
    {
      int i = 0;
      for ( uint32_t color : colors ) {
        colortable[i++] = color;
      }
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
    
    for ( auto it = begin(outPixelToCountTable); it != end(outPixelToCountTable); ++it) {
      uint32_t pixel = it->first;
      uint32_t count = it->second;
      
      printf("count table[0x%08X] = %6d\n", pixel, count);
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
        fnameStream << "srm" << "_tag_" << tag << "_quant_output" << ".png";
        string fname = fnameStream.str();
        
        imwrite(fname, tmpResultImg);
        cout << "wrote " << fname << endl;
      }
    
      // Map quant pixels to colortable offsets
    
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
        fnameStream << "srm" << "_tag_" << tag << "_quant_offsets" << ".png";
        string fname = fnameStream.str();
        
        imwrite(fname, quantOffsetsMat);
        cout << "wrote " << fname << endl;
      }
    }
    
    delete [] colortable;
    
    
    // Invoke util method
    
    vector<uint32_t> estClusterCenters;
    
    bool isVeryClose = estimateClusterCenters(inputImg, tag, regionCoords, estClusterCenters);
    
    if (isVeryClose) {
      // In this case the pixels are from a very small colortable or all the entries
      // are so close together that one can assume that the colors are very simple
      // and can be represented by quant that uses the original colors as a colortable.
      
      // Vote inside/outside for each pixel after we know what colortable entry a specific
      // pixel is associated with.
      
      unordered_map<uint32_t, uint32_t> mapSrcPixelToSRMTag;
      
      // Iterate over the coords and gather up srmTags that correspond
      // to the area indicated by tags
      
      for ( Coord c : coords ) {
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
        
        for ( Coord c : coords ) {
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
      
      // Create region mask as byte mask
      
      Mat srmRegionMask(inputImg.rows, inputImg.cols, CV_8UC1);
      srmRegionMask = Scalar(0);
      
      for ( Coord c : coords ) {
        srmRegionMask.at<uint8_t>(c.y, c.x) = 0xFF;
      }
      
      if (debugDumpImages) {
        std::stringstream fnameStream;
        fnameStream << "srm" << "_tag_" << tag << "_srm_region_mask" << ".png";
        string fname = fnameStream.str();
        
        imwrite(fname, srmRegionMask);
        cout << "wrote " << fname << endl;
        cout << "";
      }
      
      // Quant the region pixels to the provided cluster centers
      
//      for ( uint32_t pixel : estClusterCenters ) {}
      
      // Generate quant based on the input
      
      int numColors = (uint32_t) estClusterCenters.size();
      uint32_t numActualClusters = numColors;

      uint32_t *colortable = new uint32_t[numActualClusters];
      
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
          
          dumpQuantTableImage(fname, inputImg, colortable, numColors);
        }
        
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
          fnameStream << "srm" << "_tag_" << tag << "_quant_inside_offsets" << ".png";
          string fname = fnameStream.str();
          
          imwrite(fname, quantOffsetsMat);
          cout << "wrote " << fname << endl;
        }
        
        // Loop over each pixel passed through the quant logic and count up how
        // often a pixel is "inside" the known region vs how often it is "outside".
        
        typedef struct {
          int inside;
          int outside;
        } InsideOutside;
        
        unordered_map<uint32_t, InsideOutside> srmTagInsideCount;
        
        for ( int i = 0; i < numPixels; i++ ) {
          Coord c = regionCoords[i];
          uint32_t quantPixel = outPixels[i];
          InsideOutside &inOut = srmTagInsideCount[quantPixel];
          uint8_t isInside = srmRegionMask.at<uint8_t>(c.y, c.x);
          if (isInside) {
            inOut.inside += 1;
          } else {
            inOut.outside += 1;
          }
        }
        
        // Vote for inside/outside status for each unique pixel based on a GT 50% chance
        
        unordered_map<uint32_t, bool> pixelToInside;
        
        for ( auto &pair : srmTagInsideCount ) {
          uint32_t pixel = pair.first;
          InsideOutside &inOut = pair.second;
          
          if (debug) {
            printf("inout table[0x%08X] = (in out) (%5d %5d)\n", pixel, inOut.inside, inOut.outside);
          }
          
          float percentOn = (float)inOut.inside / (inOut.inside + inOut.outside);
          
          if (debug) {
            printf("percent on [0x%08X] = %0.3f\n", pixel, percentOn);
          }
          
          if (percentOn > 0.5f) {
            pixelToInside[pixel] = true;
          } else {
            pixelToInside[pixel] = false;
          }
          
          if (debug) {
            printf("pixelToInside[0x%08X] = %d\n", pixel, pixelToInside[pixel]);
          }
        }
        
        if (debug) {
          printf("done\n");
        }
        
        // Each pixel in the input is now mapped to a boolean condition that
        // indicates if that pixel is inside or outside the shape.
        
        for ( int i = 0; i < numPixels; i++ ) {
          Coord c = regionCoords[i];
          uint32_t quantPixel = outPixels[i];
          
#if defined(DEBUG)
          assert(pixelToInside.count(quantPixel));
#endif // DEBUG
          bool isInside = pixelToInside[quantPixel];
          
          if (isInside) {
            outBlockMask.at<uint8_t>(c.y, c.x) = 0xFF;
            
            if (debug) {
              printf("pixel 0x%08X at (%5d,%5d) is marked on (inside)\n", quantPixel, c.x, c.y);
            }
          } else {
            if (debug) {
              printf("pixel 0x%08X at (%5d,%5d) is marked off (outside)\n", quantPixel, c.x, c.y);
            }
          }
        }
      }
      
      delete [] colortable;
    }
    
    // Estimate the number of clusters to use in a quant operation by
    // mapping the input pixels through an even quant table and then
    // convert to blocks that represent the quant regions. This logic
    // counts quant pixels that are next to other quant pixels such
    // that dense areas that quant to the same pixel are promoted to
    // a high count.
    
    if (!isVeryClose) {
      
      unordered_map<Coord, HistogramForBlock> blockMap;
      
      Mat blockMat =
      genHistogramsForBlocks(inputImg, blockMap, blockWidth, blockHeight, superpixelDim);
      
      // Generate mask Mat that is the same dimensions as blockMat but contains just one
      // byte for each pixel and acts as a mask. The white pixels indicate the blocks
      // that are included in the mask.
      
      Mat blockMaskMat(blockMat.rows, blockMat.cols, CV_8U);
      blockMaskMat= (Scalar) 0;
      
      for ( Point p : locations ) {
        int blockX = p.x;
        int blockY = p.y;
        blockMaskMat.at<uint8_t>(blockY, blockX) = 0xFF;
      }
      
      if (debugDumpImages) {
        std::stringstream fnameStream;
        fnameStream << "srm" << "_tag_" << tag << "_block_mask" << ".png";
        string fname = fnameStream.str();
        
        imwrite(fname, blockMaskMat);
        cout << "wrote " << fname << endl;
      }
      
      // Count neighbors that share a quant pixel value after conversion to blocks
      
      unordered_map<uint32_t, uint32_t> pixelToNumVotesMap;
      
      vote_for_identical_neighbors(pixelToNumVotesMap, blockMat, blockMaskMat);
      
      vector<uint32_t> sortedPixelKeys = sort_keys_by_count(pixelToNumVotesMap, true);
      
      for ( uint32_t pixel : sortedPixelKeys ) {
        uint32_t count = pixelToNumVotesMap[pixel];
        fprintf(stdout, "0x%08X (%8d) -> %5d\n", pixel, pixel, count);
      }
      
      fprintf(stdout, "done\n");
      
      // Instead of a stddev type of approach, use grap peak logic to examine the counts
      // and select the peaks in the distrobution.
      
      vector<uint32_t> sortedOffsets = generate_cluster_walk_on_center_dist(sortedPixelKeys);
      
      // Once cluster centers have been sorted by 3D color cube distance, emit "centers.png"
      
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
      
      for ( uint32_t pixel : sortedColortable ) {
        uint32_t count = pixelToNumVotesMap[pixel];
        fprintf(stdout, "0x%08X (%8d) -> %5d\n", pixel, pixel, count);
      }
      
      fprintf(stdout, "done\n");
      
      // Dump sorted pixel data as a CSV file, with int value and hex rep of int value for readability
      
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
      
      {
        std::stringstream fnameStream;
        fnameStream << "srm" << "_tag_" << tag << "_block_mask_sorted" << ".png";
        string filename = fnameStream.str();
        
        char *outQuantTableFilename = (char*) filename.c_str();
        imwrite(outQuantTableFilename, sortedQtableOutputMat);
        cout << "wrote " << outQuantTableFilename << endl;
      }
      
      // Use peak detection logic to examine the 1D histogram in sorted order so as to find the
      // peaks in the distribution.
      
      int N = 0;
      vector<uint32_t> peakPixels;
      
      {
        // FIXME: dynamically allocate buffers to fit input size ?
        
        double*     data[2];
        //              double      row[2];
        
#define MAX_PEAK    256
        
        int         emi_peaks[MAX_PEAK];
        int         absorp_peaks[MAX_PEAK];
        
        int         emi_count = 0;
        int         absorp_count = 0;
        
        double      delta = 1e-6;
        int         emission_first = 0;
        
        int numDataPoints = (int) sortedColortable.size();
        
        assert(numDataPoints <= 256);
        
        data[0] = (double*) malloc(sizeof(double) * MAX_PEAK);
        data[1] = (double*) malloc(sizeof(double) * MAX_PEAK);
        
        memset(data[0], 0, sizeof(double) * MAX_PEAK);
        memset(data[1], 0, sizeof(double) * MAX_PEAK);
        
        int i = 0;
        
        i += 1;
        
        for ( uint32_t pixel : sortedColortable ) {
          uint32_t count = pixelToNumVotesMap[pixel];
          uint32_t pixelNoAlpha = pixel & 0x00FFFFFF;
          
          data[0][i] = pixelNoAlpha;
          //data[0][i] = i;
          data[1][i] = count;
          
          if ((0)) {
            fprintf(stderr, "pixel %05d : 0x%08X = %d\n", i, pixelNoAlpha, count);
          }
          
          i += 1;
        }
        
        // +1 at the end of the samples
        i += 1;
        
        // Print the input data with zeros at the front and the back
        
        for ( int j = 0; j < i; j++ ) {
          uint32_t pixelNoAlpha = data[0][j];
          uint32_t count = data[1][j];
          
          if ((1)) {
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
        
        fprintf(stdout, "num emi_peaks %d\n", emi_count);
        fprintf(stdout, "num absorp_peaks %d\n", absorp_count);
        
        for(i = 0; i < emi_count; ++i) {
          int offset = emi_peaks[i];
          fprintf(stdout, "%5d : %5d,%5d\n", offset, (int)data[0][offset], (int)data[1][offset]);
          
          uint32_t pixel = (uint32_t) round(data[0][offset]);
          peakPixels.push_back(pixel);
        }
        
        puts("");
        
        for(i = 0; i < absorp_count; ++i) {
          int offset = absorp_peaks[i];
          fprintf(stdout, "%5d : %5d,%5d\n", offset, (int)data[0][offset],(int)data[1][offset]);
        }
        
        free(data[0]);
        free(data[1]);
        
        // FIXME: if there seems to be just 1 peak, then it is likely that the other
        // points are another color range. Just assume N = 2 in that case ?
        
        N = (int) peakPixels.size();
        
        // Min N must be at least 1 at this point
        
        if (N < 2) {
          N = 2;
        }
        
        N = N * 4;
      }
      
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
      
      cout << "numClusters detected as " << numClusters << endl;
      
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
        
        if ((1)) {
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
        
        // Once cluster centers have been sorted by 3D color cube distance, emit "centers.png"
        
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
        
        // Generate histogram based on the sorted quant pixels
        
//        {
//          unordered_map<uint32_t, uint32_t> pixelToQuantCountTable;
//          
//          generatePixelHistogram(tmpResultImg, pixelToQuantCountTable);
//          
//          for ( uint32_t pixel : sortedColortable ) {
//            uint32_t count = pixelToQuantCountTable[pixel];
//            uint32_t pixelNoAlpha = pixel & 0x00FFFFFF;
//            fprintf(stdout, "0x%08X (%8d) -> %5d\n", pixelNoAlpha, pixelNoAlpha, count);
//          }
//          fprintf(stdout, "done\n");
//        }
        
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
          
          if ((debug)) {
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
      
      // Determine which cluster center is nearest to the peak pixels and use
      // that info to generate new cluster centers that are exactly at the
      // peak value. This means that the peak pixels will quant exactly and the
      // nearby cluster value will get the nearby but not exactly on pixels.
      // This should clearly separate the flat pixels from the gradient pixels.
      
      {
        unordered_map<uint32_t, uint32_t> pixelToQuantCountTable;
        
        for (int i = 0; i < numActualClusters; i++) {
          uint32_t pixel = colortable[i];
          pixel = pixel & 0x00FFFFFF;
          pixelToQuantCountTable[pixel] = i;
        }
        
        for ( uint32_t pixel : peakPixels ) {
          pixel = pixel & 0x00FFFFFF;
          pixelToQuantCountTable[pixel] = 0;
        }
        
        int numColors = (int)pixelToQuantCountTable.size();
        uint32_t *colortable = new uint32_t[numColors];
        
        {
          int i = 0;
          for ( auto &pair : pixelToQuantCountTable ) {
            uint32_t key = pair.first;
            assert(key == (key & 0x00FFFFFF)); // verify alpha is zero
            colortable[i] = key;
            i++;
          }
        }
        
        if (debugDumpImages)
        {
          std::stringstream fnameStream;
          fnameStream << "srm" << "_tag_" << tag << "_quant_table2" << ".png";
          string fname = fnameStream.str();
          
          dumpQuantTableImage(fname, inputImg, colortable, numColors);
        }
        
        for ( int i = 0; i < numColors; i++) {
          uint32_t pixel = colortable[i];
          fprintf(stdout, "colortable[%5d] = 0x%08X\n", i, pixel);
        }
        
        // Run input pixels through closest color quant logic using the
        // generated colortable. Note that the colortable should be
        // split such that one range of the colortable should be seen
        // as "inside" while the other range is "outside".
        
        map_colors_mps(inPixels, numPixels, outPixels, colortable, numColors);
        
        // Dump quant output, each pixel is replaced by color in colortable
        
        if (debugDumpImages) {
          Mat tmpResultImg = inputImg.clone();
          tmpResultImg = Scalar(0,0,0xFF);
          
          for ( int i = 0; i < numPixels; i++ ) {
            Coord c = regionCoords[i];
            
            // FIXME: need to identify the "inside" color and then deselect all others.
            // The "inside" one is typically the largest cound inside the "in" region.
            
            uint32_t pixel = outPixels[i];
            Vec3b vec;
            // vec = PixelToVec3b(pixel);
            if (pixel == 0x0) {
              vec = PixelToVec3b(pixel);
            } else {
              vec = PixelToVec3b(0xFFFFFFFF);
            }
            tmpResultImg.at<Vec3b>(c.y, c.x) = vec;
            
            if (pixel == 0x0) {
              // No-op when pixel is not on
            } else {
              outBlockMask.at<uint8_t>(c.y, c.x) = 0xFF;
            }
          }
          
          if (debugDumpImages)
          {
            std::stringstream fnameStream;
            fnameStream << "srm" << "_tag_" << tag << "_quant_output2" << ".png";
            string fname = fnameStream.str();
            
            imwrite(fname, tmpResultImg);
            cout << "wrote " << fname << endl;
          }          
        }
        
      }
      
      // FIXME: leaking memory currently
      
      // dealloc
      
      delete [] inPixels;
      delete [] outPixels;
      delete [] colortable;
      
    }
    
  }
  
  if (debug) {
    cout << "return captureRegionMask" << endl;
  }
  
  return true;
}

// Invoked for each child of a container, returns the tags that are direct children of tag

void
recurseSuperpixelContainmentImpl(SuperpixelImage &spImage,
                                 unordered_map<int32_t, std::vector<int32_t> > &map,
                                 int32_t tag)
{
  const bool debug = true;
  
  Superpixel *spPtr = spImage.getSuperpixelPtr(tag);
  
  if (debug) {
    cout << "recurseSuperpixelContainmentImpl for tag " << tag << " with N = " << spPtr->coords.size() << endl;
  }
  
  // Before processing neightbors, mark this superpixel as processed
  
  vector<int32_t> &children = map[tag];
  
  auto &neighbors = spImage.edgeTable.getNeighborsSet(tag);

  if (debug) {
    cout << "neighbors: " << endl;
    
    for ( int32_t neighborTag : neighbors ) {
      cout << neighborTag << endl;
    }
  }

  // When all the neighbors are iterated over and only tag is found to
  // be a common neighbor then we know that all the neighbors are
  // contained inside tag.
  
  set<int32_t> neighborsOfNeighborsSet;
  
  for ( int32_t neighborTag : neighbors ) {
    if (debug) {
      cout << "process neighbor " << neighborTag << endl;
    }
    
    if (map.count(neighborTag) > 0) {
      if (debug) {
        cout << "already processed neighbor tag " << neighborTag << endl;
      }
      continue;
    }
    
    neighborsOfNeighborsSet.insert(neighborTag);
    
    // Recurse into neighbors of neighbor at this point
    
    recurseSuperpixelContainmentImpl(spImage, map, neighborTag);
  }
  
  if (debug) {
    cout << "neighborsOfNeighborsSet: " << endl;
    
    for ( int32_t neighborTag : neighborsOfNeighborsSet ) {
      cout << neighborTag << endl;
    }
  }

  if (neighborsOfNeighborsSet.size() == 0) {
    // nop when no unprocecssed neighbors
  } else if (neighborsOfNeighborsSet.size() == 1) {
    // Special case of just 1 neighbor, neighbor must be inside tag
    
    for ( int32_t neighborTag : neighborsOfNeighborsSet ) {
      children.push_back(neighborTag);
    }
  } else {
    assert(0);
  }

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
                             unordered_map<int32_t, std::vector<int32_t> > &map)
{
  const bool debug = true;
  
  vector<int32_t> rootTags;
  
  // Sort the superpixel by size, if the background is the most common element then this is typically the largest
  // element.

  uint32_t firstTag = 1;
  
  Superpixel *spPtr = spImage.getSuperpixelPtr(firstTag);
  
  Coord firstCoord = spPtr->coords[0];
  
  assert(firstCoord.x == 0 && firstCoord.y == 0);
  
  vector<int32_t> sortedSuperpixelTags = spImage.sortSuperpixelsBySize();
  assert(sortedSuperpixelTags.size() > 0);

  for ( int32_t tag : sortedSuperpixelTags ) {
    // Recurse into children of tag
    
    if (map.count(tag) > 0) {
      if (debug) {
        cout << " already processed children for tag " << tag << endl;
      }
      continue;
    }
    
    // Find all neighbors of tag
    
    recurseSuperpixelContainmentImpl(spImage, map, tag);
  }
  
  rootTags.push_back(firstTag);
  
  return rootTags;
}
