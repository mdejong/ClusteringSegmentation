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

#include "MergeSuperpixelImage.h"

#include "srm.h"

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

// Dump N x 1 image that contains pixels

static
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

// Implement merge of superpixels based on coordinates gather from SRM process

#import "SuperpixelMergeManager.h"


class SRMMergeManager : public SuperpixelMergeManager {
public:
  // Once a superpixel has been processed it is marked as locked
  unordered_map<int32_t, int32_t> locked;
  
  // Other tags is a read-only set of tags that indicate which
  // superpixels were found to be in the region.
  
  set<int32_t> *otherTagsSetPtr;
  
  // The all sorted list makes it possible to hold the sorted
  // list across multiple invocations of the merge logic using
  // the same manager.
  
  vector<int32_t> allSortedSuperpixels;
  
  int32_t mergeStepAtStart;
  
  // Set to true to enable debug step dump
  const bool debugDumpImages = true;
  
  SRMMergeManager(SuperpixelImage & _spImage, Mat &_inputImg)
  : SuperpixelMergeManager(_spImage, _inputImg), mergeStepAtStart(0)
  {}
  
  // Invoked before the merge operation starts, useful to setup initial
  // state and cache values.
  
  void setup() {
    if (allSortedSuperpixels.size() == 0) {
      allSortedSuperpixels = spImage.sortSuperpixelsBySize();
    }
    
    unordered_map<int32_t,int32_t> map;
    
    for ( int32_t tag : *otherTagsSetPtr ) {
      map[tag] = tag;
    }
    
    for ( int32_t tag : allSortedSuperpixels ) {
      if (map.count(tag) > 0) {
        superpixels.push_back(tag);
      }
    }
    
    return;
  }
  
  // Invoked at the end of the processing operation
  
  void finish() {
    return;
  }
  
  // The check method accepts a superpixel tag and returns false
  // if the specific superpixel has already been processed.
  
  bool check(int32_t tag) {
    if (locked.count(tag) > 0) {
      return false;
    }
    return true;
  }
  
  // Invoked when a valid superpixel tag is being processed
  
  void startProcessing(int32_t tag) {
    mergeStepAtStart = mergeStep;
  }
  
  // When the iterator loop is finished processing a specific superpixel
  // this method is invoked to indicate the superpixel tag.
  
  void doneProcessing(int32_t tag) {
    locked[tag] = mergeStep;
    
    if (mergeStepAtStart == mergeStep) {
      // No merges done for this superpixel
    } else {
      if (debugDumpImages) {
        // Determine if a merge was done on this iter
        
        Mat tmpResultImg = inputImg.clone();
        tmpResultImg = (Scalar) 0;
        
        writeTagsWithStaticColortable(spImage, tmpResultImg);
        
        std::stringstream fnameStream;
        fnameStream << "merge_global_step_" << mergeStep << ".png";
        string fname = fnameStream.str();
        
        imwrite(fname, tmpResultImg);
        cout << "wrote " << fname << endl;
      }
    }
  }
  
  // The checkEdge method accepts two superpixel tags, the dst tag
  // indicates the region typically merged into while the src
  // represents a neighbor of dst that is being checked.
  // This method should return true if the edge between the superpixels
  // should be merged and false otherwise.
  
  bool checkEdge(int32_t dst, int32_t src) {
    // dst was already verified, so just check to see if src is in otherTagsSet
    
    if ( otherTagsSetPtr->find(src) != otherTagsSetPtr->end() ) {
      return true;
    } else {
      return false;
    }
  }
  
  // This method actually does the merge operation
  
  void mergeEdge(SuperpixelEdge &edge) {
    SuperpixelMergeManager::mergeEdge(edge);
    
    if (debugDumpImages) {
      // Determine if a merge was done on this iter
      
      Mat tmpResultImg = inputImg.clone();
      tmpResultImg = (Scalar) 0;
      
      writeTagsWithStaticColortable(spImage, tmpResultImg);
      
      std::stringstream fnameStream;
      fnameStream << "merge_step_" << mergeStep << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, tmpResultImg);
      cout << "wrote " << fname << endl;
    }
    
    return;
  }
  
};

// Main method that implements the cluster combine logic

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
  
  const int superpixelDim = 4;
  int blockWidth = inputImg.cols / superpixelDim;
  if ((inputImg.cols % superpixelDim) != 0) {
    blockWidth++;
  }
  int blockHeight = inputImg.rows / superpixelDim;
  if ((inputImg.rows % superpixelDim) != 0) {
    blockHeight++;
  }
  
  assert((blockWidth * superpixelDim) >= inputImg.cols);
  assert((blockHeight * superpixelDim) >= inputImg.rows);
  
  for(int y = 0; y < inputImg.rows; y++) {
    int yStep = y >> 2;
    
    for(int x = 0; x < inputImg.cols; x++) {
      int xStep = x >> 2;

      uint32_t tag = (yStep * blockWidth) + xStep;
      
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
    
    if (largestSuperpixelResults.size() > 0) {
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
      
      spImage.scanLargestSuperpixels(largestSuperpixelResults, (superpixelDim*superpixelDim)); // min is 16 pixels
    }
    
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
    
    dumpQuantTableImage("quant_table.png", inputImg, colortable, numActualClusters);
    
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

      if ((0)) {
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
    }
    
    // dealloc
    
    delete [] pixels;
    delete [] outPixels;
    delete [] colortable;
  }
  
  if ((0)) {
    // Attempt to merge based on a likeness predicate
    
    spImage.mergeSuperpixelsWithPredicate(inputImg);
    
    if (debugWriteIntermediateFiles) {
      writeTagsWithStaticColortable(spImage, resultImg);
      imwrite("tags_after_predicate_merge.png", resultImg);
    }
  }

  if ((0)) {
    // Attempt to merge regions that are very much alike
    // based on a histogram comparison. When the starting
    // point is identical regions then the regions near
    // identical regions are likely to be very alike
    // up until a hard edge.

    int mergeStep = 0;
    
    MergeSuperpixelImage::mergeBackprojectSuperpixels(spImage, inputImg, 1, mergeStep, BACKPROJECT_HIGH_FIVE8);
    
    if (debugWriteIntermediateFiles) {
      writeTagsWithStaticColortable(spImage, resultImg);
      imwrite("tags_after_histogram_merge.png", resultImg);
    }
  }
  
  if ((0)) {
  Mat minImg;
  writeTagsWithMinColortable(spImage, inputImg, minImg);
  imwrite("tags_min_color.png", minImg);
  cout << "wrote " << "tags_min_color.png" << endl;
  }
  
  if ((1)) {
    // SRM

//    double Q = 16.0;
//    double Q = 32.0;
//    double Q = 64.0; // Not too small
    double Q = 128.0; // keeps small circles together
//    double Q = 256.0;
//    double Q = 512.0;
    
    Mat srmTags = generateSRM(inputImg, Q);
    
    // Scan the tags generated by SRM and create superpixels of vario
    
    SuperpixelImage srmSpImage;
    
    bool worked = SuperpixelImage::parse(srmTags, srmSpImage);
    
    if (!worked) {
      return false;
    }
    
    if (debugWriteIntermediateFiles) {
      generateStaticColortable(inputImg, srmSpImage);
    }

    if (debugWriteIntermediateFiles) {
      Mat tmpResultImg = resultImg.clone();
      tmpResultImg = (Scalar) 0;
      writeTagsWithStaticColortable(srmSpImage, tmpResultImg);
      imwrite("srm_tags.png", tmpResultImg);
    }
    
    cout << "srm generated superpixels N = " << srmSpImage.superpixels.size() << endl;
    
    // Scan the largest superpixel regions in largest to smallest order and find
    // overlap between the SRM generated superpixels.
    
    vector<int32_t> srmSuperpixels = srmSpImage.sortSuperpixelsBySize();
    
    unordered_map<int32_t, set<int32_t> > srmSuperpixelToExactMap;
    
    Mat renderedTagsMat = resultImg.clone();
    renderedTagsMat = (Scalar) 0;
    
    spImage.fillMatrixWithSuperpixelTags(renderedTagsMat);

    for ( int32_t tag : srmSuperpixels ) {
      Superpixel *spPtr = srmSpImage.getSuperpixelPtr(tag);
      assert(spPtr);
    
      // Find all the superpixels that are all inside a larger superpixel
      // and then process the contained elements.
      
      // Find overlap between largest superpixels and the known all same superpixels
      
      set<int32_t> &otherTagsSet = srmSuperpixelToExactMap[tag];
      
      for ( Coord coord : spPtr->coords ) {
        Vec3b vec = renderedTagsMat.at<Vec3b>(coord.y, coord.x);
        uint32_t otherTag = Vec3BToUID(vec);
        
        if (otherTagsSet.find(otherTag) == otherTagsSet.end()) {
          if ((1)) {
            fprintf(stdout, "coord (%4d,%4d) = found tag 0x%08X aka %8d\n", coord.x, coord.y, otherTag, otherTag);
          }
          
          otherTagsSet.insert(otherTagsSet.end(), otherTag);
        }
        
        if ((0)) {
          fprintf(stdout, "coord (%4d,%4d) = 0x%08X aka %8d\n", coord.x, coord.y, otherTag, otherTag);
        }
        
        // Lookup a superpixel with this specific tag just to make sure it exists
#if defined(DEBUG)
        Superpixel *otherSpPtr = spImage.getSuperpixelPtr(otherTag);
        assert(otherSpPtr);
        assert(otherSpPtr->tag == otherTag);
#endif // DEBUG
      }
      
      cout << "for SRM superpixel " << tag << " : other tags ";
      for ( int32_t otherTag : otherTagsSet ) {
        cout << otherTag << " ";
      }
      cout << endl;
    } // end foreach srmSuperpixels
    
    // Loop over the otherTagsSet and find any tags that appear in
    // multiple regions.
    
    if ((1)) {
      vector<int32_t> tagsToRemove;
      
      set<int32_t> allSet;
      
      for ( int32_t tag : srmSuperpixels ) {
        set<int32_t> &otherTagsSet = srmSuperpixelToExactMap[tag];
        
        for ( int32_t otherTag : otherTagsSet ) {
          if (allSet.find(otherTag) != allSet.end()) {
            tagsToRemove.push_back(otherTag);
          }
          allSet.insert(allSet.end(), otherTag);
        }
      }
      
      for ( int32_t tag : srmSuperpixels ) {
        set<int32_t> &otherTagsSet = srmSuperpixelToExactMap[tag];
        
        for ( int32_t tag : tagsToRemove ) {
          if ( otherTagsSet.find(tag) != otherTagsSet.end() ) {
            otherTagsSet.erase(tag);
          }
        }
      }
      
      // Dump the removed regions as a mask
      
      if (debugWriteIntermediateFiles) {
        Mat tmpResultImg = resultImg.clone();
        tmpResultImg = (Scalar) 0;
        
        for ( int32_t tag : tagsToRemove ) {
          Superpixel *spPtr = spImage.getSuperpixelPtr(tag);
          assert(spPtr);
          
          Vec3b whitePixel(0xFF, 0xFF, 0xFF);
          
          for ( Coord c : spPtr->coords ) {
            tmpResultImg.at<Vec3b>(c.y, c.x) = whitePixel;
          }
        }
        
        std::stringstream fnameStream;
        fnameStream << "merge_removed_union" << ".png";
        string fname = fnameStream.str();
        
        imwrite(fname, tmpResultImg);
        cout << "wrote " << fname << endl;
      }
    }
    
    // Foreach SRM superpixel, find the set of superpixels
    // in the identical tags that correspond to a union of
    // the pixels in the SRM region and the identical region.
    
    for ( int32_t tag : srmSuperpixels ) {
      set<int32_t> &otherTagsSet = srmSuperpixelToExactMap[tag];
      
      if ((1)) {
      cout << "srm superpixels " << tag << " corresponds to other tags : ";
      for ( int32_t otherTag : otherTagsSet ) {
        cout << otherTag << " ";
      }
      cout << endl;
      }
      
      // For the large SRM superpixel determine the set of superpixels
      // contains in the region by looking at the other tags image.
      
      Mat regionMat = Mat(resultImg.rows, resultImg.cols, CV_8UC1);

      regionMat = (Scalar) 0;
      
      int numCoords = 0;

      vector<Coord> unprocessedCoords;
      
      for ( int32_t otherTag : otherTagsSet ) {
        Superpixel *spPtr = spImage.getSuperpixelPtr(otherTag);
        assert(spPtr);
        
        if ((1)) {
          cout << "superpixel " << otherTag << " with N = " << spPtr->coords.size() << endl;
        }
        
        for ( Coord c : spPtr->coords ) {
          regionMat.at<uint8_t>(c.y, c.x) = 0xFF;
          // Slow bbox calculation simply records all the (X,Y) coords in all the
          // superpixels and then does a bbox using these coords. A faster method
          // would be to do a bbox on each superpixel and then save the upper left
          // and upper right coords only.
          unprocessedCoords.push_back(c);
          numCoords++;
        }
      }
      
      if (numCoords == 0) {
        cout << "zero unprocessed pixels for SRM superpixel " << tag << endl;
      } else {
        std::stringstream fnameStream;
        fnameStream << "srm" << "_N_" << numCoords << "_tag_" << tag << ".png";
        string fname = fnameStream.str();
        
        imwrite(fname, regionMat);
        cout << "wrote " << fname << endl;
      }
      
      if (false && (numCoords != 0)) {
        // The same type of logic implemented as a morphological operation in terms of 4x4 blocks
        // represented as pixels.
        
        Mat morphBlockMat = Mat(blockHeight, blockWidth, CV_8U);
        morphBlockMat = (Scalar) 0;
        
        // Get the first coord for each block that is indicated as inside the SRM superpixel
        
        for ( int32_t otherTag : otherTagsSet ) {
          Superpixel *spPtr = spImage.getSuperpixelPtr(otherTag);
          assert(spPtr);
          
          if ((1)) {
            cout << "unprocessed superpixel " << otherTag << " with N = " << spPtr->coords.size() << endl;
          }
          
          for ( Coord c : spPtr->coords ) {
            // Convert (X,Y) to block (X,Y)
            
            int blockX = c.x / superpixelDim;
            int blockY = c.y / superpixelDim;
            
            if ((0)) {
              cout << "block with tag " << otherTag << " cooresponds to (X,Y) (" << c.x << "," << c.y << ")" << endl;
              cout << "maps to block (X,Y) (" << blockX << "," << blockY << ")" << endl;
            }
            
            // FIXME: optimize for case where (X,Y) is exactly the same as in the previous iteration and avoid
            // writing to the Mat in that case. This shift is cheap.
            
            morphBlockMat.at<uint8_t>(blockY, blockX) = 0xFF;
          }
        }
        
        Mat expandedBlockMat;
        
        for (int expandStep = 0; expandStep < 8; expandStep++ ) {
          if (expandStep == 0) {
            expandedBlockMat = morphBlockMat;
          } else {
            expandedBlockMat = expandWhiteInRegion(expandedBlockMat, 1, tag);
          }
          
          int nzc = countNonZero(expandedBlockMat);
          
          Mat morphBlockMat = Mat(blockHeight, blockWidth, CV_8U);
          
          if (nzc == (blockHeight * blockWidth)) {
            cout << "all pixels in Mat now white " << endl;
            break;
          }
          
          if ((1)) {
            std::stringstream fnameStream;
            fnameStream << "srm" << "_tag_" << tag << "_morph_block_" << expandStep << ".png";
            string fname = fnameStream.str();
            
            imwrite(fname, expandedBlockMat);
            cout << "wrote " << fname << endl;
          }
          
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
            minMaxCoords.push_back(max);
          }
          
          int32_t originX, originY, width, height;
          Superpixel::bbox(originX, originY, width, height, minMaxCoords);
          Rect expandedRoi(originX, originY, width, height);
          
          Mat roiInputMat = inputImg(expandedRoi);
          
          if ((1)) {
            std::stringstream fnameStream;
            fnameStream << "srm" << "_tag_" << tag << "_morph_block_input_" << expandStep << ".png";
            string fname = fnameStream.str();
            
            imwrite(fname, roiInputMat);
            cout << "wrote " << fname << endl;
          }
          
        } // for expandStep
        
      } // end if numCoords
      
    } // end foreach srmSuperpixels

    // Merge manager will iterate over the superpixels found by
    // doing a union of the SRM regions and the superpixels.
    
    if (debugWriteIntermediateFiles) {
      generateStaticColortable(inputImg, spImage);
    }
    
    if (debugWriteIntermediateFiles) {
      Mat tmpResultImg = resultImg.clone();
      tmpResultImg = (Scalar) 0;
      
      writeTagsWithStaticColortable(spImage, tmpResultImg);
      
      std::stringstream fnameStream;
      fnameStream << "merge_step_" << 0 << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, tmpResultImg);
      cout << "wrote " << fname << endl;
    }
    
    SRMMergeManager mergeManager(spImage, inputImg);
    
    for ( int32_t tag : srmSuperpixels ) {
      set<int32_t> &otherTagsSet = srmSuperpixelToExactMap[tag];
      
      if ((1)) {
        cout << "srm superpixels " << tag << " corresponds to other tags : ";
        for ( int32_t otherTag : otherTagsSet ) {
          cout << otherTag << " ";
        }
        cout << endl;
      }
      
      mergeManager.otherTagsSetPtr = &otherTagsSet;
      
      SuperpixelMergeManagerFunc<SRMMergeManager>(mergeManager);
    }
    
    // With the overall merge completed, generate a block Mat
    // for each large superpixel so that specific pixel values
    // for the area around the region can be queried.
    
    for ( int32_t tag : spImage.sortSuperpixelsBySize() ) {
      auto &coords = spImage.getSuperpixelPtr(tag)->coords;
      
      if (coords.size() <= (superpixelDim*superpixelDim)) {
        // Don't bother with non-expanded blocks
        break;
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
        minMaxCoords.push_back(max);
      }
      
      int32_t originX, originY, width, height;
      Superpixel::bbox(originX, originY, width, height, minMaxCoords);
      Rect expandedRoi(originX, originY, width, height);
      
      if ((1)) {
        Mat roiInputMat = inputImg(expandedRoi);
        
        std::stringstream fnameStream;
        fnameStream << "srm" << "_tag_" << tag << "_morph_block_input" << ".png";
        string fname = fnameStream.str();
        
        imwrite(fname, roiInputMat);
        cout << "wrote " << fname << endl;
      }
      
      if ((1)) {
        Mat expandedBlockMat(inputImg.rows, inputImg.cols, CV_8U);
        
        expandedBlockMat = (Scalar) 0;
        
        for ( Point p : locations ) {
          int actualX = p.x * superpixelDim;
          int actualY = p.y * superpixelDim;
          
          Coord min(actualX, actualY);
          Coord max(actualX+superpixelDim-1, actualY+superpixelDim-1);
          
          for ( int y = min.y; y <= max.y; y++ ) {
            for ( int x = min.x; x <= max.x; x++ ) {
              expandedBlockMat.at<uint8_t>(y, x) = 0xFF;
            }
          }
        }
        
        std::stringstream fnameStream;
        fnameStream << "srm" << "_tag_" << tag << "_morph_block_bw" << ".png";
        string fname = fnameStream.str();
        
        imwrite(fname, expandedBlockMat);
        cout << "wrote " << fname << endl;
      }
      
      // Generate a collection of pixels from the blocks included in the
      // expanded mask.
      
      if ((1)) {
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
              regionCoords.push_back(c);
            }
          }
        }
        
        Mat tmpResultImg = inputImg.clone();
        tmpResultImg = Scalar(0,0,0xFF);
        
        int numPixels = (int) regionCoords.size();
        
        uint32_t *inPixels = new uint32_t[numPixels];
        uint32_t *outPixels = new uint32_t[numPixels];
        
        for ( int i = 0; i < numPixels; i++ ) {
          Coord c = regionCoords[i];
          Vec3b vec = inputImg.at<Vec3b>(c.y, c.x);
          uint32_t pixel = Vec3BToUID(vec);
          inPixels[i] = pixel;
          tmpResultImg.at<Vec3b>(c.y, c.x) = vec;
        }
        
        {
          std::stringstream fnameStream;
          fnameStream << "srm" << "_tag_" << tag << "_morph_masked_input" << ".png";
          string fname = fnameStream.str();
          
          imwrite(fname, tmpResultImg);
          cout << "wrote " << fname << endl;
        }
        
        {
          // Generate quant based on the input
          
          const int numClusters = 256;
          
          cout << "numClusters detected as " << numClusters << endl;
          
          uint32_t *colortable = new uint32_t[numClusters];
          
          uint32_t numActualClusters = numClusters;
          
          int allPixelsUnique = 0;
          
          quant_recurse(numPixels, inPixels, outPixels, &numActualClusters, colortable, allPixelsUnique );
          
          // Write quant output where each original pixel is replaced with the closest
          // colortable entry.
          
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
            
            if ((0)) {
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
            
            for (int i = 0; i < numActualClusters; i++) {
              int si = (int) sortedOffsets[i];
              uint32_t pixel = colortable[si];
              Vec3b vec = PixelToVec3b(pixel);
              sortedQtableOutputMat.at<Vec3b>(i, 0) = vec;
            }
           
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
              
              if ((debugOutput)) {
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
              fnameStream << "srm" << "_tag_" << tag << "_quant_table_offsets" << ".png";
              string filename = fnameStream.str();
              
              char *outQuantFilename = (char*)filename.c_str();
              imwrite(outQuantFilename, sortedQuantOutputMat);
              cout << "wrote " << outQuantFilename << endl;
            }
          }
            
            // dealloc
          
          delete [] inPixels;
          delete [] outPixels;
          delete [] colortable;
          
        }

      }

    }
    
  }
  
  // Generate result image after region based merging
  
  if (debugWriteIntermediateFiles) {
    generateStaticColortable(inputImg, spImage);
    writeTagsWithStaticColortable(spImage, resultImg);
    imwrite("tags_after_region_merge.png", resultImg);
  }
  
  // Done
  
  cout << "ended with " << spImage.superpixels.size() << " superpixels" << endl;
  
  return true;
}

