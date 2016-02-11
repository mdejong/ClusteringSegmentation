//
//  RegionRemerger.hpp
//  ClusteringSegmentation
//
//  Created by Mo DeJong on 2/11/16.
//  Copyright © 2016 helpurock. All rights reserved.
//

#ifndef RegionRemerger_hpp
#define RegionRemerger_hpp

#include <opencv2/opencv.hpp>

#include <string>
#include <unordered_map>

#include "OpenCVUtil.h"
#include "OpenCVIter.hpp"

using cv::Mat;
using std::string;
using std::unordered_map;
using std::vector;

// This class handles details of non-neighbor region changes for tags.
// For example, if the tag ids of certain regions is going to be
// changed in a way that differs from simple merging two regions
// that are neighbors then the results will need to be reparsed
// in order to regenerate properly defined tag neighbors.

class RegionRemerger {
public:
  CvSize size;
  Mat maskMat;
  Mat mergeMat;
  
  // tag that should be used next. This tag increases in value
  // each time a new set of pixels is merged in.
  
  int32_t mergedTag = 1;
  
  uint32_t tagsAdlerBeforeMerge = 0;
  
  RegionRemerger(const Mat &_tagsImg)
  {
    mergeMat = _tagsImg.clone();
    mergeMat = Scalar(0,0,0);
    size = _tagsImg.size();
    maskMat = Mat(size, CV_8UC1, Scalar(0));
  }
  
  // Reset the state of maskMat to be 0xFF for each pixel that is non-zero in mergeMat
  
  void mergeMatToMask() {

    for_each_byte_const_bgr(maskMat, mergeMat, [](uint8_t *maskBPtr, uint8_t B, uint8_t G, uint8_t R) {
      if (B == 0 && G == 0 && R == 0) {
        *maskBPtr = 0;
      } else {
        *maskBPtr = 0xFF;
      }
    });
    
    return;
  }

  // When maskMat is written then scan for non-zero values in maskMat and then generate a new
  // tag and set each corresponding pixel in mergeMat.
  
  void mergeFromMask() {
    vector<Point> locations;
    findNonZero(maskMat, locations);
    
    assert(locations.size() > 0);
    
    Vec3b mergedVec = Vec3BToUID(mergedTag);
    
    for ( Point p : locations ) {
      int x = p.x;
      int y = p.y;
      
      Vec3b vec = mergeMat.at<Vec3b>(y, x);
      
      if (vec[0] == 0x0 && vec[1] == 0x0 && vec[2] == 0x0) {
        // This pixel has not been seen before, define new merge tag value
        
        if (false) {
          printf("set merge mat (%5d, %5d) = 0x%08X aka %d\n", x, y, mergedTag, mergedTag);
        }
        
        mergeMat.at<Vec3b>(y, x) = mergedVec;
      } else {
        // A region must not attempt to include pixels from a previously merged region ever!
        
        uint32_t alreadySetTag = Vec3BToUID(vec);
        
        printf("coord (%5d, %5d) = attempted remerge when tag already set to 0x%08X aka %d\n", x, y, alreadySetTag, alreadySetTag);
        assert(0);
      }
    } // foreach locations
    
    // Update merge tag after setting all pixel values
    mergedTag += 1;
  }
  
  void mergeLeftovers(const Mat &tagMat) {
    // Gather any remaining tags that have not been merged
    // and add these as new sets of pixels.
    
    unordered_map<uint32_t, vector<Coord>> mergeTagsToCoords;
    
    for ( int y = 0; y < mergeMat.rows; y++ ) {
      for ( int x = 0; x < mergeMat.cols; x++ ) {
        Vec3b vec = mergeMat.at<Vec3b>(y, x);
        
        if (vec[0] == 0x0 && vec[1] == 0x0 && vec[2] == 0x0) {
          vec = tagMat.at<Vec3b>(y, x);
          uint32_t srmTag = Vec3BToUID(vec);
          
          vector<Coord> &vecRef = mergeTagsToCoords[srmTag];
          vecRef.push_back(Coord(x, y));
        }
      }
    }
    
    for ( auto & pair : mergeTagsToCoords ) {
      vector<Coord> &vecRef = pair.second;
      
      for ( Coord c : vecRef ) {
        Vec3b mergedVec = Vec3BToUID(mergedTag); // FIXME: do outside loop
        mergeMat.at<Vec3b>(c.y, c.x) = mergedVec;
        
        if (true) {
          fprintf(stdout, "merge unmerged srm tag at (%5d, %5d) = 0X%08X\n", c.x, c.y, mergedTag);
        }
      }
      
      mergedTag += 1;
    }

    return;
  }
  
};

#endif /* RegionRemerger_hpp */
