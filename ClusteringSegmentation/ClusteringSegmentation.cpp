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

Mat generateSRM(Mat &inputImg, double Q)
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
      
      vector<uint32_t> pixelsThisBlock;
      
      bool isAllSamePixel = true;
      uint32_t firstPixel = 0x0;
      
      for (int y = actualY; y <= max.y; y++) {
        for (int x = actualX; x <= max.x; x++) {
          if ((debugOutput) && false) {
            char buffer[1024];
            snprintf(buffer, sizeof(buffer), "(%4d,%4d)", x, y);
            cout << buffer << endl;
          }
          
          if (x >= width-1) {
            continue;
          }
          if (y >= height-1) {
            continue;
          }
          
          Coord c(x, y);
          uint32_t pi = (y * width) + x;
          uint32_t quantPixel = outPixels[pi];
          
          if (y == actualY && x == actualX) {
            // First pixel in block
            firstPixel = quantPixel;
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
