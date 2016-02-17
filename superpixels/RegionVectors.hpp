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
  
  // Define N vectors between two existing contour points and return
  // the vecUid for each new in between vector.

  vector<int32_t> makeVectorsBetween(int32_t leftUid, int32_t rightUid, int N) {
#if defined(DEBUG)
    assert(leftUid != rightUid);
    assert(orderedKeys.count(leftUid) > 0);
    assert(orderedKeys.count(rightUid) > 0);
#endif // DEBUG
    
    // leftUid and rightUid must be directly next to each other
    
    int stepBetween = RegionVectorsSpaceSkip / (N + 1);
    
    vector<int32_t> newUids;
    newUids.reserve(N);
    
    for ( int i = 0; i < N; i++ ) {
      int newUid = leftUid + ((i+1) * stepBetween);
#if defined(DEBUG)
      assert(newUid != leftUid);
      assert(newUid != rightUid);
#endif // DEBUG
      newUids.push_back(newUid);
    }
    
    return newUids;
  }
  
  // If vectors are defined between contour points then this method returns
  // those vectors in order.
  
  vector<int32_t> getVectorsBetween(int32_t leftUid, int32_t rightUid) {
#if defined(DEBUG)
    assert(leftUid != rightUid);
    assert(orderedKeys.count(leftUid) > 0);
    assert(orderedKeys.count(rightUid) > 0);
    if (rightUid != 0) {
      assert((leftUid + RegionVectorsSpaceSkip) == rightUid);
    }
#endif // DEBUG

    // FIXME: sucky impl, should sort be done on overall list at start ?
    
    vector<int32_t> vecUids;
    
    const int lastUid = leftUid + RegionVectorsSpaceSkip;
    
    // FIXME: slightly better impl would be to loop through each key and
    // filter out and that are LTEQ leftUid or GTEQ lastUid so that
    // only a small set of numbers had to be checked.
    
    for ( int vecUid = leftUid + 1; vecUid < lastUid; vecUid++ ) {
      if (outsideVectorsMap.count(vecUid) > 0) {
        vecUids.push_back(vecUid);
      }
    }
    
    return vecUids;
  }
};

#endif // RegionVectors_hpp
