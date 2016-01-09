//
//  main.cpp
//  ClusteringSegmentation
//
//  Created by Mo DeJong on 12/30/15.
//  Copyright © 2015 helpurock. All rights reserved.
//

// clusteringsegmentation IMAGE TAGS_IMAGE
//
// This logic reads input pixels from an image and segments the image into different connected
// areas based on growing area of alike pixels. A set of pixels is determined to be alike
// if the pixels are near to each other in terms of 3D space via a fast clustering method.
// The TAGS_IMAGE output file is written with alike pixels being defined as having the same
// tag color.

#include <opencv2/opencv.hpp>

#include "Superpixel.h"
#include "SuperpixelEdge.h"
#include "SuperpixelImage.h"

#include "OpenCVUtil.h"
#include "Util.h"

#include "quant_util.h"

using namespace cv;
using namespace std;

bool clusteringCombine(Mat &inputImg, Mat &resultImg);

//void generateStaticColortable(Mat &inputImg, SuperpixelImage &spImage);

//void writeTagsWithGraytable(SuperpixelImage &spImage, Mat &origImg, Mat &resultImg);

//void writeTagsWithMinColortable(SuperpixelImage &spImage, Mat &origImg, Mat &resultImg);

int main(int argc, const char** argv) {
  const char *inputImgFilename = NULL;
  const char *outputTagsImgFilename = NULL;

  if (argc == 2) {
    inputImgFilename = argv[1];
    // Default to "outtags.png"
    outputTagsImgFilename = "outtags.png";
    
    // In the special case where the inputImgFilename is fully qualified, cd to the directory
    // indicated by the path. This is useful so that just a fully qualified image path can
    // be passed as the first argument without having to explicitly set the process working dir
    // since Xcode tends to get that detail wrong when invoking profiling tools.
    
    bool containsSlash = false;
    int lastSlashOffset = -1;
    
    for ( char *ptr = (char*)inputImgFilename; *ptr != '\0' ; ptr++ ) {
      if (*ptr == '/') {
        containsSlash = true;
        lastSlashOffset = int(ptr - (char*)inputImgFilename);
      }
    }
    
    if (containsSlash) {
      char *dirname = strdup((char*)inputImgFilename);
      assert(lastSlashOffset >= 0);
      dirname[lastSlashOffset] = '\0';
      
      inputImgFilename = inputImgFilename + lastSlashOffset + 1;
      
      cout << "cd \"" << dirname << "\"" << endl;
      chdir(dirname);
      
      free(dirname);
    }
  } else if (argc != 3) {
    cerr << "usage : " << argv[0] << " IMAGE ?TAGS_IMAGE?" << endl;
    exit(1);
  } else if (argc == 3) {
    inputImgFilename = argv[1];
    outputTagsImgFilename = argv[2];
  }

  cout << "read \"" << inputImgFilename << "\"" << endl;
  
  Mat inputImg = imread(inputImgFilename, CV_LOAD_IMAGE_COLOR);
  if( inputImg.empty() ) {
    cerr << "could not read \"" << inputImgFilename << "\" as image data" << endl;
    exit(1);
  }
  
  Mat resultImg;
  
  bool worked = clusteringCombine(inputImg, resultImg);
  if (!worked) {
    cerr << "seeds combine failed " << endl;
    exit(1);
  }
  
  imwrite(outputTagsImgFilename, resultImg);
  
  cout << "wrote " << outputTagsImgFilename << endl;
  
  exit(0);
}

// Given an input image and a pixel buffer that is of the same dimensions
// write the buffer of pixels out as an image in a file.

static
void dumpQuantImage(string filename, Mat inputImg, uint32_t *pixels) {
  Mat quantOutputMat = inputImg.clone();
  quantOutputMat = (Scalar) 0;
  
  const bool debugOutput = false;
  
  int pi = 0;
  for(int y = 0; y < quantOutputMat.rows; y++) {
    for(int x = 0; x < quantOutputMat.cols; x++) {
      uint32_t pixel = pixels[pi++];
      
      if ((debugOutput)) {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "for (%4d,%4d) pixel is %d\n", x, y, pixel);
        cout << buffer;
      }
      
      Vec3b vec = PixelToVec3b(pixel);
      
      quantOutputMat.at<Vec3b>(y, x) = vec;
    }
  }
  
  imwrite(filename, quantOutputMat);
  cout << "wrote " << filename << endl;
  return;
}

bool clusteringCombine(Mat &inputImg, Mat &resultImg)
{
  const bool debugWriteIntermediateFiles = true;
  
  // Alloc object on stack
  SuperpixelImage spImage;
  //
  // Ref to object allocated on heap
//  Ptr<SuperpixelImage> spImagePtr = new SuperpixelImage();
//  SuperpixelImage &spImage = *spImagePtr;
  
  // Generate a "tags" input that contains 1 tag for each 4x4 block of input, so that
  // large regions of the exact same fill color can be detected and processed early.
  
  Mat tagsImg = inputImg.clone();
  tagsImg = (Scalar) 0;
  
  const bool debugOutput = false;
  
  int numXStepsinWidth = (inputImg.cols >> 2);
  
  const int superpixelWidth = 4;
  
  for(int y = 0; y < inputImg.rows; y++) {
    int yStep = y >> 2;
    
    for(int x = 0; x < inputImg.cols; x++) {
      int xStep = x >> 2;

      uint32_t tag = (yStep * numXStepsinWidth) + xStep;
      
      if ((debugOutput)) {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "for (%4d,%4d) tag is %d\n", x, y, tag);
        cout << buffer;
      }
      
      Vec3b vec = PixelToVec3b(tag);
      
      tagsImg.at<Vec3b>(y, x) = vec;
    }
    
    if (debugOutput) {
    cout << endl;
    }
  }

  
  bool worked = SuperpixelImage::parse(tagsImg, spImage);
  
  if (!worked) {
    return false;
  }
  
  // Dump image that shows the input superpixels written with a colortable
  
  resultImg = inputImg.clone();
  resultImg = (Scalar) 0;
  
  sranddev();
  
  if (debugWriteIntermediateFiles) {
    generateStaticColortable(inputImg, spImage);
  }
  
  if (debugWriteIntermediateFiles) {
    writeTagsWithStaticColortable(spImage, resultImg);
    imwrite("tags_init.png", resultImg);
  }
  
  cout << "started with " << spImage.superpixels.size() << " superpixels" << endl;
  
  // Identical
  
  spImage.mergeIdenticalSuperpixels(inputImg);
  
  if ((
#if defined(DEBUG)
       1
#else
       0
#endif // DEBUG
       )) {
    auto vec = spImage.sortSuperpixelsBySize();
    assert(vec.size() > 0);
  }
  
  if (debugWriteIntermediateFiles) {
    writeTagsWithStaticColortable(spImage, resultImg);
    imwrite("tags_after_identical_merge.png", resultImg);
  }
  
  // Do initial invocation of quant logic with an N that depends on the number
  // of large identical regions.
  
  if ((1)) {
    const bool debugOutput = false;
    
    int numPixels = inputImg.rows * inputImg.cols;
    
    uint32_t *pixels = new uint32_t[numPixels];
    assert(pixels);
    uint32_t pi = 0;
    
    for(int y = 0; y < inputImg.rows; y++) {
      for(int x = 0; x < inputImg.cols; x++) {
        Vec3b vec = inputImg.at<Vec3b>(y, x);
        uint32_t pixel = Vec3BToUID(vec);
        
        if ((debugOutput)) {
          char buffer[1024];
          snprintf(buffer, sizeof(buffer), "for (%4d,%4d) pixel is %d\n", x, y, pixel);
          cout << buffer;
        }
        
        pixels[pi++] = pixel;
      }
      
      if (debugOutput) {
        cout << endl;
      }
    }
    
    uint32_t *inPixels = pixels;
    uint32_t *outPixels = new uint32_t[numPixels];
    assert(outPixels);
    
    // Determine a good N (number of clusters)
    
    vector<int32_t> largestSuperpixelResults;
    spImage.scanLargestSuperpixels(largestSuperpixelResults, 0);
    
    assert(largestSuperpixelResults.size() > 0);
    
    int32_t largestSuperpixelTag = largestSuperpixelResults[0];
    
    // Typically the largest superpixel is the background, so pop the first
    // element and then run the stddev logic again.
    
    largestSuperpixelResults = spImage.getSuperpixelsVec();
    
    for ( int offset = 0; offset < largestSuperpixelResults.size(); offset++ ) {
      if (largestSuperpixelResults[offset] == largestSuperpixelTag) {
        largestSuperpixelResults.erase(largestSuperpixelResults.begin() + offset);
        break;
      }
    }
    
    spImage.scanLargestSuperpixels(largestSuperpixelResults, (superpixelWidth*superpixelWidth)); // min is 16 pixels
    
 //   int32_t largestSuperpixelTag = largestSuperpixelResults[0];
    //    vector<int32_t> sortedSuperpixels = spImage.sortSuperpixelsBySize();
    
    const int numClusters = 256;
//    int numClusters = 1 + (int)largestSuperpixelResults.size();
    
    cout << "numClusters detected as " << numClusters << endl;
    
    uint32_t *colortable = new uint32_t[numClusters];
    
    uint32_t numActualClusters = numClusters;
    
    int allPixelsUnique = 0;
    
    quant_recurse(numPixels, inPixels, outPixels, &numActualClusters, colortable, allPixelsUnique );
    
    // Write quant output where each original pixel is replaced with the closest
    // colortable entry.
    
    dumpQuantImage("quant_output.png", inputImg, outPixels);
    
    // Write image that contains one color in each row in a N x 1 image
    
    Mat qtableOutputMat = Mat(numActualClusters, 1, CV_8UC3);
    qtableOutputMat = (Scalar) 0;
    
    for (int i = 0; i < numActualClusters; i++) {
      uint32_t pixel = colortable[i];
      Vec3b vec = PixelToVec3b(pixel);
      qtableOutputMat.at<Vec3b>(i, 0) = vec;
    }
    
    char *outQuantTableFilename = (char*)"quant_table.png";
    imwrite(outQuantTableFilename, qtableOutputMat);
    cout << "wrote " << outQuantTableFilename << endl;
    
    // Generate color sorted clusters
    
    {
      vector<uint32_t> clusterCenterPixels;
      
      for ( int i = 0; i < numClusters; i++) {
        uint32_t pixel = colortable[i];
        clusterCenterPixels.push_back(pixel);
      }
      
      if ((1)) {
        fprintf(stdout, "numClusters %5d\n", numClusters);
        
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
      
      for (int i = 0; i < numActualClusters; i++) {
        int si = (int) sortedOffsets[i];
        uint32_t pixel = colortable[si];
        Vec3b vec = PixelToVec3b(pixel);
        sortedQtableOutputMat.at<Vec3b>(i, 0) = vec;
      }
      
      char *outQuantTableFilename = (char*)"quant_table_sorted.png";
      imwrite(outQuantTableFilename, sortedQtableOutputMat);
      cout << "wrote " << outQuantTableFilename << endl;
      
      // Map pixels to sorted colortable offset
      
      unordered_map<uint32_t, uint32_t> pixel_to_sorted_offset;
      
      assert(numActualClusters <= 256);
      
      for (int i = 0; i < numActualClusters; i++) {
        int si = (int) sortedOffsets[i];
        uint32_t pixel = colortable[si];
        pixel_to_sorted_offset[pixel] = si;
      }
      
      Mat sortedQuantOutputMat = inputImg.clone();
      sortedQuantOutputMat = (Scalar) 0;
      
      pi = 0;
      for(int y = 0; y < sortedQuantOutputMat.rows; y++) {
        for(int x = 0; x < sortedQuantOutputMat.cols; x++) {
          uint32_t pixel = outPixels[pi++];
          assert(pixel_to_sorted_offset.count(pixel) > 0);
          uint32_t offset = pixel_to_sorted_offset[pixel];
          
          if ((debugOutput)) {
            char buffer[1024];
            snprintf(buffer, sizeof(buffer), "for (%4d,%4d) pixel is %d -> offset %d\n", x, y, pixel, offset);
            cout << buffer;
          }
          
          assert(offset <= 256);
          uint32_t grayscalePixel = (offset << 16) | (offset << 8) | offset;
          Vec3b vec = PixelToVec3b(grayscalePixel);
          sortedQuantOutputMat.at<Vec3b>(y, x) = vec;
        }
      }
      
      char *outQuantFilename = (char*)"quant_sorted_offsets.png";
      imwrite(outQuantFilename, sortedQuantOutputMat);
      cout << "wrote " << outQuantFilename << endl;
      
      // Create table that maps input pixels to the table of 256 quant pixels,
      // this basically indicates how often certain colors "bunch" up in the
      // image being scanned. The goal here is to create a simple 1D table
      // so that a more focused N can be determined for the largest "color" blocks.
      
      unordered_map<uint32_t, uint32_t> pixel_to_quant_count;
      
      //int numPixels = inputImg.rows * inputImg.cols;
      for (int i = 0; i < numPixels; i++) {
        uint32_t pixel = outPixels[i++];
        pixel_to_quant_count[pixel] += 1;
      }
      
      cout << "pixel_to_quant_count() = " << pixel_to_quant_count.size() << endl;
      
      if ((0)) {
        for ( auto it = pixel_to_quant_count.begin(); it != pixel_to_quant_count.end(); ++it) {
          uint32_t pixel = it->first;
          uint32_t count = it->second;
          printf("count table[0x%08X] = %6d\n", pixel, count);
        }
      }
      
      // This more focused histogram of just 256 values can now be clustered into a smaller
      // subset of colors in an effort to find the best N or number of clusters to apply
      // to the original image.
      
      for (int i = 0; i < numActualClusters; i++) {
        int si = (int) sortedOffsets[i];
        uint32_t pixel = colortable[si];
        uint32_t count = pixel_to_quant_count[pixel];
        
        printf("count table[0x%08X] = %6d\n", pixel, count);
      }
      
      // Repeated invocation of quant logic as a method of reducing a 256 table of values
      // to a smaller N that best splits the global colorspace.
      
      uint32_t *quantResultPixels = new uint32_t[numPixels];
      uint32_t *quantColortable = new uint32_t[numActualClusters];
      
      int numClustersThisIteration = numActualClusters - 1;
      
      for (int i = 0; i < numActualClusters; i++) {
        int allPixelsUnique = 0;
        
        if (numClustersThisIteration < 2) {
          break;
        }
        
        uint32_t numActualClusters = numClustersThisIteration;
        
        quant_recurse(numPixels, outPixels, quantResultPixels, &numActualClusters, quantColortable, allPixelsUnique );
       
        std::stringstream fnameStream;
        fnameStream << "quant_output_N" << numClustersThisIteration << ".png";
        string fname = fnameStream.str();
        
        dumpQuantImage(fname, inputImg, quantResultPixels);
        
        numClustersThisIteration -= 1;
      }
      
      delete [] quantResultPixels;
      delete [] quantColortable;
    }
    
    // dealloc
    
    delete [] pixels;
    delete [] outPixels;
    delete [] colortable;
  }
    
  if ((0)) {
  Mat minImg;
  writeTagsWithMinColortable(spImage, inputImg, minImg);
  imwrite("tags_min_color.png", minImg);
  cout << "wrote " << "tags_min_color.png" << endl;
  }
  
  // Done
  
  cout << "ended with " << spImage.superpixels.size() << " superpixels" << endl;
  
  return true;
}

