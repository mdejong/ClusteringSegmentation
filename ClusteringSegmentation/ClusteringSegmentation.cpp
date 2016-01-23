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
                      Mat &outBlockMask,
                      const vector<Coord> &regionCoords,
                      const vector<Coord> &srmRegionCoords,
                      int estNumColors);

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
// in a way that should capture pixels around the outlines of the

Mat
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
  
  return expandedBlockMat;
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
  
  if (coords.size() <= ((superpixelDim*superpixelDim) >> 1)) {
    // A region contained in only a single block, don't process by itself
    
    if (debug) {
      cout << "captureRegionMask : region indicated by tag " << tag << " is too small to process with N coords " << coords.size() << endl;
    }
    
    return false;
  }
  
  // Init mask after possible early return
  
  outBlockMask = (Scalar) 0;
  
  vector<Coord> regionCoords;
  
  Mat expandedBlockMat = morphRegionMask(inputImg, tag, coords, blockWidth, blockHeight, superpixelDim, regionCoords);
  
  // Invoke util method
  
  vector<uint32_t> estClusterCenters;
  
  bool isVeryClose = estimateClusterCenters(inputImg, tag, regionCoords, estClusterCenters);
  
  if (isVeryClose) {
    captureVeryCloseRegion(spImage, inputImg, srmTags, tag, blockWidth, blockHeight, superpixelDim, outBlockMask, regionCoords, coords, (int)estClusterCenters.size());
  } else {
    captureNotCloseRegion(spImage, inputImg, srmTags, tag, blockWidth, blockHeight, superpixelDim, outBlockMask, regionCoords, coords, (int)estClusterCenters.size());
  }
  
  if (debug) {
    cout << "return captureRegionMask" << endl;
  }
  
  return true;
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
                  Mat &outBlockMask,
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
      outBlockMask.at<uint8_t>(c.y, c.x) = 0xFF;
      
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
                       Mat &outBlockMask,
                       const vector<Coord> &regionCoords,
                       const vector<Coord> &srmRegionCoords,
                       int estNumColors)
{
  const bool debug = true;
  const bool debugDumpImages = true;
  
  if (debug) {
    cout << "captureNotCloseRegion" << endl;
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
  
  unordered_map<Coord, HistogramForBlock> blockMap;
  
  Mat blockMat =
  genHistogramsForBlocks(inputImg, blockMap, blockWidth, blockHeight, superpixelDim);
  
  // Generate mask Mat that is the same dimensions as blockMat but contains just one
  // byte for each pixel and acts as a mask. The white pixels indicate the blocks
  // that are included in the mask.
  
  Mat blockMaskMat(blockMat.rows, blockMat.cols, CV_8U);
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
    
    // Insert zero slow with zero count so that a peak can
    // be detected in the first position.
    i += 1;
    
    for ( uint32_t pixel : sortedColortable ) {
      uint32_t count = pixelToNumVotesMap[pixel];
      uint32_t pixelNoAlpha = pixel & 0x00FFFFFF;
      
      data[0][i] = pixelNoAlpha;
      //data[0][i] = i;
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
    
    for(i = 0; i < emi_count; ++i) {
      int offset = emi_peaks[i];
      fprintf(stdout, "%5d : %5d,%5d\n", offset, (int)data[0][offset], (int)data[1][offset]);
      
      uint32_t pixel = (uint32_t) round(data[0][offset]);
      peakPixels.push_back(pixel);
    }
    
    fprintf(stdout, "num absorp_peaks %d\n", absorp_count);
    
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
      
      // Once cluster centers have been sorted by 3D color cube distance, emit as PNG
      
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
    
    for ( int i = 0; i < numColors; i++) {
      uint32_t pixel = colortable[i];
      fprintf(stdout, "colortable[%5d] = 0x%08X\n", i, pixel);
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
        outBlockMask.at<uint8_t>(c.y, c.x) = 0xFF;
        
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
  
  // Filter neighborsSet into a set of siblings by removing
  // the neighbors that are known to already be processed.
  
  set<int32_t> siblings;
  
  for ( int32_t neighborTag : neighborsSet ) {
    if (debug) {
      cout << "check for unprocessed neighbor " << neighborTag << endl;
    }
    
    if (map.count(neighborTag) > 0) {
      if (debug) {
        cout << "already processed neighbor tag " << neighborTag << endl;
      }
    } else {
      siblings.insert(neighborTag);
    }
  }
  
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
    
    recurseSuperpixelContainmentImpl(spImage, map, neighborTag);
    
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
  const bool debug = true;

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

  unordered_map<int32_t,int32_t> rootTagToSize;
  
  for ( int32_t tag : rootSet ) {
    rootTagToSize[tag] = 0;
  }
  
  for ( int32_t tag : sortedSuperpixelTags ) {
    if (rootTagToSize.count(tag) > 0) {
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
    
    recurseSuperpixelContainmentImpl(spImage, map, tag);
    
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
