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

using namespace cv;
using namespace std;

bool clusteringCombine(Mat &inputImg, Mat &resultImg);

void generateStaticColortable(Mat &inputImg, SuperpixelImage &spImage);

void writeTagsWithStaticColortable(SuperpixelImage &spImage, Mat &resultImg);

void writeTagsWithGraytable(SuperpixelImage &spImage, Mat &origImg, Mat &resultImg);

void writeTagsWithMinColortable(SuperpixelImage &spImage, Mat &origImg, Mat &resultImg);

int main(int argc, const char** argv) {
  if (argc != 3) {
    cerr << "usage : " << argv[0] << " IMAGE TAGS_IMAGE" << endl;
    exit(1);
  }
  
  const char *inputImgFilename = argv[1];
  const char *outputTagsImgFilename = argv[2];
  
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
  
  SuperpixelImage spImage;
  
  // Before any superpixel parsing is started, need to take care of weird edge case in
  // output of Seeds superpixel segmentation.
  
  //Superpixel::splitSplayPixels(tagsImg);
  
  //bool worked = SuperpixelImage::parse(tagsImg, spImage);
  
  //if (!worked) {
  //  return false;
  //}
  
  // Dump image that shows the input superpixels written with a colortable
  
  resultImg = inputImg.clone();
  resultImg = (Scalar) 0;
  
  sranddev();
  
  if (debugWriteIntermediateFiles) {
    generateStaticColortable(inputImg, spImage);
  }
  
  if (debugWriteIntermediateFiles) {
    writeTagsWithStaticColortable(spImage, resultImg);
    imwrite("tags_input.png", resultImg);
  }
  
  cout << "started with " << spImage.superpixels.size() << " superpixels" << endl;
  
  // Identical
  
  spImage.mergeIdenticalSuperpixels(inputImg);
  
  if (debugWriteIntermediateFiles) {
    writeTagsWithStaticColortable(spImage, resultImg);
    imwrite("tags_after_identical_merge.png", resultImg);
  }
  
  int mergeStep = 0;
  
  // RGB
  
  mergeStep = spImage.mergeBackprojectSuperpixels(inputImg, 0, mergeStep, BACKPROJECT_HIGH_FIVE);
  
  if (debugWriteIntermediateFiles) {
    writeTagsWithStaticColortable(spImage, resultImg);
    imwrite("tags_after_backproject_merge_RGB.png", resultImg);
  }
  
  cout << "RGB merge edges count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
  
  // LAB
  
  mergeStep = spImage.mergeBackprojectSuperpixels(inputImg, CV_BGR2Lab, mergeStep, BACKPROJECT_HIGH_FIVE);
  
  if (debugWriteIntermediateFiles) {
    writeTagsWithStaticColortable(spImage, resultImg);
    imwrite("tags_after_backproject_merge_LAB.png", resultImg);
  }
  
  cout << "LAB merge edges count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
  
  // After the initial processing above, identical and very alike regions have been joined as blobs.
  // The next stage does back projection again but in away that compares edge weights between the
  // superpixels to detect when to stop searching for histogram alikeness.
  
  mergeStep = spImage.mergeBredthFirstRecursive(inputImg, 0, mergeStep, NULL, 16);
  
  if (debugWriteIntermediateFiles) {
    writeTagsWithStaticColortable(spImage, resultImg);
    
    imwrite("tags_after_backproject_fill_merge_RGB.png", resultImg);
  }
  
  cout << "RGB fill backproject merge edges count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
  
  // Select only small superpixels and then merge away from largest neighbors
  
  mergeStep = spImage.mergeSmallSuperpixels(inputImg, 0, mergeStep);
  
  if (debugWriteIntermediateFiles) {
    writeTagsWithStaticColortable(spImage, resultImg);
    imwrite("tags_after_small_merge_RGB.png", resultImg);
  }
  
  cout << "RGB merge small count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
  
  // Get large superpixels list at this point. It is generally cleaned up and closer to well
  // defined hard edges.
  
  vector<int32_t> veryLargeSuperpixels;
  spImage.scanLargestSuperpixels(veryLargeSuperpixels);
  
  // Merge edgy superpixel into edgy neighbor(s)
  
  mergeStep = spImage.mergeEdgySuperpixels(inputImg, 0, mergeStep, &veryLargeSuperpixels);
  
  if (debugWriteIntermediateFiles) {
    writeTagsWithStaticColortable(spImage, resultImg);
    imwrite("tags_after_edgy_RGB.png", resultImg);
  }
  
  cout << "RGB merge edgy count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
  
  /*
   
   // Cleanup by doing another BFS in RGB and then a LAB merge with the original large superpixels locked
   // but use a much smaller histogram with only 8 bins per channel.
   
   mergeStep = spImage.mergeBredthFirstRecursive(inputImg, 0, mergeStep, &veryLargeSuperpixels, 8);
   
   if (debugWriteIntermediateFiles) {
   writeTagsWithStaticColortable(spImage, resultImg);
   imwrite("tags_after_cleanup_backproject_fill_merge_RGB.png", resultImg);
   }
   
   cout << "RGB fill backproject merge edges count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
   
   */
  
  // LAB
  
  mergeStep = spImage.mergeBackprojectSuperpixels(inputImg, CV_BGR2Lab, mergeStep, BACKPROJECT_HIGH_FIVE);
  
  if (debugWriteIntermediateFiles) {
    writeTagsWithStaticColortable(spImage, resultImg);
    imwrite("tags_after_cleanup_backproject_merge_LAB.png", resultImg);
  }
  
  cout << "LAB merge count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
  
  /*
   
   // Lossy YCrCb (this combined color farther away than LAB does)
   
   mergeStep = spImage.mergeBackprojectSuperpixels(inputImg, CV_BGR2YCrCb, mergeStep, BACKPROJECT_HIGH_50);
   
   writeTagsWithStaticColortable(spImage, resultImg);
   
   imwrite("tags_after_backproject_merge_YCrCb.png", resultImg);
   
   cout << "YCrCb merge edges count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
   
   */
  
  // Apply bilinear filtering and then run the DFS logic again. This is a tiny amount of filtering
  // it should make very little diff.
  
  //Mat bilinear;
  
  /*
   
   spImage.applyBilinearFiltering(inputImg, bilinear, 5, 1);
   imwrite("bilinear_filter_5_1.png", bilinear);
   
   // This small merge further cleans up around the edges. Small superpixels right on the edge are consumed
   // by the larger neighbor but without collapsing in on edgey regions next to the background.
   
   mergeStep = spImage.mergeBackprojectSuperpixels(bilinear, CV_BGR2Lab, mergeStep, BACKPROJECT_HIGH_FIVE);
   
   writeTagsWithStaticColortable(spImage, resultImg);
   
   imwrite("tags_after_bilinear_backproject_merge1_LAB.png", resultImg);
   
   cout << "LAB bilinear merge1 edges count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
   
   mergeStep = spImage.fillMergeBackprojectSuperpixels(bilinear, CV_BGR2Lab, mergeStep);
   
   writeTagsWithStaticColortable(spImage, resultImg);
   
   imwrite("tags_after_bilinear_backproject_merge2_LAB.png", resultImg);
   
   cout << "RGB fill backproject 2 merge edges count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
   
   // Do another bilinear filtering and then DFS
   
   spImage.applyBilinearFiltering(inputImg, bilinear, 9, 5);
   imwrite("bilinear_filter_9_5.png", bilinear);
   
   mergeStep = spImage.fillMergeBackprojectSuperpixels(bilinear, CV_BGR2Lab, mergeStep);
   
   writeTagsWithStaticColortable(spImage, resultImg);
   
   imwrite("tags_after_bilinear_backproject_merge3_LAB.png", resultImg);
   
   cout << "RGB fill backproject 3 merge edges count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
   
   */
  
  /*
   
   spImage.applyBilinearFiltering(inputImg, bilinear, 17, 50);
   imwrite("bilinear_filter_17_50.png", bilinear);
   
   // This small merge further cleans up around the edges. Small superpixels right on the edge are consumed
   // by the larger neighbor but without collapsing in on edgey regions next to the background.
   
   mergeStep = spImage.mergeBackprojectSuperpixels(bilinear, CV_BGR2Lab, mergeStep, BACKPROJECT_HIGH_FIVE);
   
   writeTagsWithStaticColortable(spImage, resultImg);
   
   imwrite("tags_after_bilinear_backproject_merge1_LAB.png", resultImg);
   
   cout << "LAB bilinear merge1 edges count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
   
   mergeStep = spImage.mergeBackprojectSuperpixels(bilinear, 0, mergeStep, BACKPROJECT_HIGH_FIVE);
   
   writeTagsWithStaticColortable(spImage, resultImg);
   
   imwrite("tags_after_bilinear_backproject_merge2_RGB.png", resultImg);
   
   cout << "RGB bilinear merge2 edges count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
   
   mergeStep = spImage.fillMergeBackprojectSuperpixels(bilinear, 0, mergeStep);
   
   writeTagsWithStaticColortable(spImage, resultImg);
   
   imwrite("tags_after_bilinear_backproject_merge3_RGB.png", resultImg);
   
   cout << "RGB bilinear merge3 edges count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
   
   // RGB and the LAB 5% again after DFS
   
   mergeStep = spImage.mergeBackprojectSuperpixels(bilinear, CV_BGR2Lab, mergeStep, BACKPROJECT_HIGH_FIVE);
   
   writeTagsWithStaticColortable(spImage, resultImg);
   
   imwrite("tags_after_bilinear_backproject_merge4_LAB.png", resultImg);
   
   cout << "LAB bilinear merge4 edges count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
   
   mergeStep = spImage.mergeBackprojectSuperpixels(bilinear, 0, mergeStep, BACKPROJECT_HIGH_FIVE);
   
   writeTagsWithStaticColortable(spImage, resultImg);
   
   imwrite("tags_after_bilinear_backproject_merge5_RGB.png", resultImg);
   
   cout << "RGB bilinear merge5 edges count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
   
   // Switch back to using original input and merge in LAB and RGB again
   
   mergeStep = spImage.mergeBackprojectSmallestSuperpixels(inputImg, CV_BGR2Lab, mergeStep, BACKPROJECT_HIGH_TEN);
   
   writeTagsWithStaticColortable(spImage, resultImg);
   
   imwrite("tags_after_normal_backproject_merge6_LAB.png", resultImg);
   
   cout << "LAB normal merge6 edges count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
   
   mergeStep = spImage.mergeBackprojectSuperpixels(inputImg, 0, mergeStep, BACKPROJECT_HIGH_TEN);
   
   writeTagsWithStaticColortable(spImage, resultImg);
   
   imwrite("tags_after_normal_backproject_merge7_RGB.png", resultImg);
   
   cout << "RGB normal merge7 edges count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
   
   mergeStep = spImage.fillMergeBackprojectSuperpixels(inputImg, 0, mergeStep);
   
   writeTagsWithStaticColortable(spImage, resultImg);
   
   imwrite("tags_after_normal_backproject_merge8_RGB.png", resultImg);
   
   cout << "RGB normal merge8 edges count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
   
   mergeStep = spImage.mergeBackprojectSmallestSuperpixels(bilinear, CV_BGR2Lab, mergeStep, BACKPROJECT_HIGH_TEN);
   
   writeTagsWithStaticColortable(spImage, resultImg);
   
   imwrite("tags_after_normal_backproject_merge9_LAB.png", resultImg);
   
   cout << "LAB normal merge9 edges count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
   
   */
  
  /*
   
   // Apply bilinear filtering and then run the histogram combination logic again
   
   Mat bilinear;
   
   spImage.applyBilinearFiltering(inputImg, bilinear, 5, 10);
   imwrite("bilinear_filter_5_10.png", bilinear);
   
   // This small merge further cleans up around the edges. Small superpixels right on the edge are consumed
   // by the larger neighbor but without collapsing in on edgey regions next to the background.
   
   mergeStep = spImage.mergeBackprojectSuperpixels(bilinear, CV_BGR2Lab, mergeStep, BACKPROJECT_HIGH_TEN);
   
   writeTagsWithStaticColortable(spImage, resultImg);
   
   imwrite("tags_after_bilinear_backproject_merge1_LAB.png", resultImg);
   
   cout << "LAB bilinear merge1 edges count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
   
   spImage.applyBilinearFiltering(inputImg, bilinear, 9, 15);
   imwrite("bilinear_filter_9_15.png", bilinear);
   
   // This small merge further cleans up around the edges. Small superpixels right on the edge are consumed
   // by the larger neighbor but without collapsing in on edgey regions next to the background.
   
   mergeStep = spImage.mergeBackprojectSuperpixels(bilinear, CV_BGR2Lab, mergeStep, BACKPROJECT_HIGH_20);
   
   writeTagsWithStaticColortable(spImage, resultImg);
   
   imwrite("tags_after_bilinear_backproject_merge2_LAB.png", resultImg);
   
   cout << "LAB bilinear merge2 edges count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
   
   // Identify edge pixels between each superpixel and highlight them in RED
   
   // Reverse order merge starting from the smallest superpixel and ending with the largest
   
   mergeStep = spImage.mergeBackprojectSmallestSuperpixels(bilinear, 0, mergeStep, BACKPROJECT_HIGH_FIVE);
   
   writeTagsWithStaticColortable(spImage, resultImg);
   
   imwrite("tags_after_bilinear_backproject_merge3_RGB.png", resultImg);
   
   cout << "RGB bilinear merge3 edges count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
   
   mergeStep = spImage.mergeBackprojectSmallestSuperpixels(bilinear, CV_BGR2Lab, mergeStep, BACKPROJECT_HIGH_FIVE);
   
   writeTagsWithStaticColortable(spImage, resultImg);
   
   imwrite("tags_after_bilinear_backproject_merge4_LAB.png", resultImg);
   
   cout << "LAB bilinear merge4 edges count " << mergeStep << " result num superpixels " << spImage.superpixels.size() << endl;
   
   */
  
  // FIXME: if there are few actual superpixels (like < 256) then it should be possible to emit as grayscale,
  // otherwise emit as RGB but have all values be near zero and have the largest sized superpixels be nearest
  // to zero ?
  
  if (spImage.superpixels.size() < 256) {
    Mat grayImg;
    writeTagsWithGraytable(spImage, inputImg, grayImg);
    imwrite("tags_grayscale.png", grayImg);
    
    cout << "wrote " << "tags_grayscale.png" << endl;
  }
  
  Mat minImg;
  
  writeTagsWithMinColortable(spImage, inputImg, minImg);
  imwrite("tags_min_color.png", minImg);
  cout << "wrote " << "tags_min_color.png" << endl;
  
  // Done
  
  cout << "ended with " << spImage.superpixels.size() << " superpixels" << endl;
  
  return true;
}

