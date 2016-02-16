//
//  RegionVectors.hpp
//  ClusteringSegmentation
//
//  Created by Mo DeJong on 2/15/16.
//  Copyright © 2016 helpurock. All rights reserved.
//
//  A set of region vectors is built from a contour and a set of vectors that
//  point outwards at each contour point. The vectors subdivide region coordinates
//  so that all (X, Y) coordinates are represented.

#ifndef RegionVectors_hpp
#define RegionVectors_hpp

#include <assert.h>
#include <string>
#include <vector>
#include <set>
#include <ostream>
#include <unordered_map>

#include "Coord.h"

using std::string;
using std::unordered_map;
using std::vector;

// The amount of space to leave between each contour so that additional
// vectors can be added between existing contour points.

const int32_t RegionVectorsSpaceSkip = 1000;

class RegionVectors {
public:
  RegionVectors() {}
  
  // Vectors from the region edge to the region center
  
  unordered_map<int32_t, vector<Coord>> insideVectorsMap;
  
  // Vectors from the region edge to the outside
  
  unordered_map<int32_t, vector<Coord>> outsideVectorsMap;
  
  // The coords on the contour
  
  vector<Coord> contourCoords;

  // The list of keys foreach vector is maintained as a sorted
  // list of keys defined in a set. This is useful so that
  // new int keys can be inserted between existing vectors
  // without having to copy around the original data.
  
  set<int32_t> orderedKeys;
  
  int32_t getUidForContour(int contouri) {
    return (contouri * RegionVectorsSpaceSkip);
  }
  
  // Given a contour that contains all the points along the
  // contour edge, generate vector entries for each point
  // along the contour.
  
  void setContour(const vector<Coord> &_contourCoords) {
    int32_t i = 0;
    
    orderedKeys.clear();
    insideVectorsMap.clear();
    outsideVectorsMap.clear();
    
    contourCoords = std::move(_contourCoords);
    
    for ( Coord c : contourCoords ) {
      int32_t vecUid = i;
      
      orderedKeys.insert(end(orderedKeys), vecUid);
      outsideVectorsMap[vecUid] = vector<Coord>();
      
      i += RegionVectorsSpaceSkip;
    }
    
    return;
  }
  
  // Get or init a ref to outside vector of coords with a specific vector UID
  
  vector<Coord>& getOutsideVector(int32_t vecUid) {
    if (orderedKeys.count(vecUid) == 0) {
      orderedKeys.insert(end(orderedKeys), vecUid);
    }
    return outsideVectorsMap[vecUid];
  }
  
  // Get or init a ref to inside vector of coords with a specific vector UID

  vector<Coord>& getInsideVector(int32_t vecUid) {
    if (orderedKeys.count(vecUid) == 0) {
      orderedKeys.insert(end(orderedKeys), vecUid);
    }
    return insideVectorsMap[vecUid];
  }
  
  // Add a new vector between two existing vectors

  vector<Coord>& getOutsideVectorBetween(int32_t leftUid, int32_t rightUid) {
    assert(leftUid != rightUid);
    assert(orderedKeys.count(leftUid) > 0);
    assert(orderedKeys.count(rightUid) > 0);
    
    // leftUid and rightUid must be directly next to each other
    
    // FIXME: evenly split the region based on how many values there are between
    // specific contour pixels.
    
    int32_t newUid = -1;
    
    if (leftUid < rightUid) {
      assert(leftUid == (rightUid - RegionVectorsSpaceSkip));
      
      for ( int i = rightUid; i > 0; i-- ) {
        if (i == leftUid) {
          break;
        }
        newUid = i;
        if (orderedKeys.count(newUid) > 0) {
          break;
        }
      }
    } else {
      // Wraps around the end of the contour at zero
      assert(rightUid == 0);
      
      for ( int i = leftUid; i < (leftUid + RegionVectorsSpaceSkip); i++ ) {
        newUid = i;
        
        if (orderedKeys.count(newUid) > 0) {
          break;
        }
      }
    }

    // Got unused newUid
    orderedKeys.insert(end(orderedKeys), newUid);
    return outsideVectorsMap[newUid];
  }
  
};

#endif // RegionVectors_hpp
