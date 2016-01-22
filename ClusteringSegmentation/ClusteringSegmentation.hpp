//
//  ClusteringSegmentation.hpp
//  ClusteringSegmentation
//
//  Created by Mo DeJong on 1/17/16.
//  Copyright © 2016 helpurock. All rights reserved.
//

#ifndef ClusteringSegmentation_hpp
#define ClusteringSegmentation_hpp

#include <opencv2/opencv.hpp>

#include <string>
#include <unordered_map>

class SuperpixelImage;
class Coord;

using cv::Mat;
using std::string;
using std::unordered_map;
using std::vector;

void generateStaticColortable(Mat &inputImg, SuperpixelImage &spImage);

void writeTagsWithGraytable(SuperpixelImage &spImage, Mat &origImg, Mat &resultImg);

void writeTagsWithMinColortable(SuperpixelImage &spImage, Mat &origImg, Mat &resultImg);

Mat dumpQuantImage(string filename, const Mat &inputImg, uint32_t *pixels);

void dumpQuantTableImage(string filename, const Mat &inputImg, uint32_t *colortable, uint32_t numColortableEntries);

typedef struct {
  // What is the overall most common pixel that the region would quant to
  uint32_t regionQuantPixel;
  
  // Counts for each pixel in the block
  unordered_map<uint32_t, uint32_t> pixelToCountTable;
} HistogramForBlock;

Mat genHistogramsForBlocks(const Mat &inputImg,
                           unordered_map<Coord, HistogramForBlock> &blockMap,
                           int blockWidth,
                           int blockHeight,
                           int superpixelDim);

// Generate a tags Mat from the original input pixels based on SRM algo.

Mat generateSRM(const Mat &inputImg, double Q);

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
                  Mat &outBlockMask);

// Foreach pixel in a colortable determine the "inside/outside" status of that
// pixel based on a stats test as compared to the current known region.

typedef struct {
  int inside;
  int outside;
  float confidence;
  bool isInside;
} InsideOutsideRecord;

void insideOutsideTest(int32_t width,
                       int32_t height,
                       const vector<Coord> &coords,
                       int32_t tag,
                       const vector<Coord> &regionCoords,
                       const uint32_t *outPixels,
                       const vector<uint32_t> &sortedColortable,
                       unordered_map<uint32_t, InsideOutsideRecord> &pixelToInsideMap);

// Recurse into each superpixel and determine the children of each superpixel.

std::vector<int32_t>
recurseSuperpixelContainment(SuperpixelImage &spImage,
                             const Mat &tagsImg,
                             unordered_map<int32_t, std::vector<int32_t> > &map);

// Inlined typed recursion method to iterate over tree structure contained
// in root tags and a map that maps the tag to a vector of children.

static inline
void recurseSuperpixelIterate(const vector<int32_t> &tags,
                              unordered_map<int32_t, vector<int32_t> > &map,
                              std::function<void(int32_t, const vector<int32_t> &)> f)
{
  for ( int32_t tag : tags ) {
    vector<int32_t> &children = map[tag];
    
    f(tag, children);
    
    recurseSuperpixelIterate(children, map, f);
  }
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

  int32_t mergedIntoTag;
  
  // Set to true to enable debug global step dump
  const bool debugDumpImages = true;
  
  // Set to true to enable debug step dump
  const bool debugDumpEachStepImages = false;
  
  SRMMergeManager(SuperpixelImage & _spImage, Mat &_inputImg)
  : SuperpixelMergeManager(_spImage, _inputImg), mergeStepAtStart(0), mergedIntoTag(0)
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
  
  bool checkProcessed(int32_t tag) {
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
      mergedIntoTag = tag;
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
  
  // This merged into method is invoked when a merge operation is completed
  // and the merge logic wishes to record the tag for the merge dst.
  
  void mergedInto(int32_t tag) {
    mergedIntoTag = tag;
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
    
    if (debugDumpEachStepImages) {
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

#endif /* ClusteringSegmentation_hpp */
