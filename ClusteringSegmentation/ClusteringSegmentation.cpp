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
    uint32_t pixels[16];
    
    pixels[0]  = 0x00EBC58B;
    pixels[1]  = 0x00DAD4E7;
    pixels[2]  = 0x00D7779D;
    pixels[3]  = 0x007E393D;
    pixels[4]  = 0x00ABA4BA;
    pixels[5]  = 0x00CF4B53;
    pixels[6]  = 0x00C49AC7;
    pixels[7]  = 0x00AC7292;
    pixels[8]  = 0x00ECEFE7;
    pixels[9]  = 0x00DC789D;
    pixels[10] = 0x00A8ABC4;
    pixels[11] = 0x00906E9E;
    pixels[12] = 0x00B54748;
    pixels[13] = 0x00A24F44;
    pixels[14] = 0x00857E77;
    pixels[15] = 0x007F654B;
    
    const int numPixels = 16;
    uint32_t *inPixels = pixels;
    uint32_t outPixels[numPixels];
    
    const int numClusters = 256;
    uint32_t colortable[numClusters];
    
    uint32_t numActualClusters = numClusters;
    
    quant_recurse(numPixels, inPixels, outPixels, &numActualClusters, colortable );
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

