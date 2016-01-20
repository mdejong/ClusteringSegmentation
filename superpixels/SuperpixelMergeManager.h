//
//  SuperpixelMergeManager.h
//  ClusteringSegmentation
//
//  Created by Mo DeJong on 1/12/16.
//  Copyright © 2016 helpurock. All rights reserved.
//

#ifndef SuperpixelMergeManager_hpp
#define SuperpixelMergeManager_hpp

#include "SuperpixelImage.h"


// An instance of SuperpixelMergeManager should extend this class and implement any
// methods that are required for a specific type of merge operation.

class SuperpixelMergeManager
{
public:
  SuperpixelImage &spImage;
  Mat &inputImg;
  
  int mergeStep;
  
  // The superpixels will be processed in order as identified by this vector of tags
  
  vector<int32_t> superpixels;
  
  SuperpixelMergeManager(SuperpixelImage & _spImage, Mat &_inputImg)
  : spImage(_spImage), inputImg(_inputImg), mergeStep(0)
  {}
  
  // begin() returns a superpixel tag iterator
  
  //  vector<int32_t>::iterator begin() { return begin(superpixels); }
  
  //  vector<int32_t> superpixelsVec = sortSuperpixelsBySize();
  
  //  for (auto it = mergeManager.begin(); it != mergeManager.end(); ) {
  
  // Invoked before the merge operation starts, useful to setup initial
  // state and cache values. The mergeStep value is always zero when
  // this method is invoked.
  
  void setup() {}
  
  // Invoked at the end of the processing operation
  
  void finish() {}
  
  // Invoked when a valid superpixel tag is being processed

  void startProcessing(int32_t tag) {}
  
  // When the iterator loop is finished processing a specific superpixel
  // this method is invoked to indicate the superpixel tag.
  
  void doneProcessing(int32_t tag) {}
  
  // The check method accepts a superpixel tag and returns false
  // if the specific superpixel has already been processed.
  
  bool checkProcessed(int32_t tag) {
    return true;
  }
  
  // The checkEdge method accepts two superpixel tags, the dst tag
  // indicates the region typically merged into while the src
  // represents a neighbor of dst that is being checked.
  // This method should return true if the edge between the superpixels
  // should be merged and false otherwise.
  
  bool checkEdge(int32_t dstTag, int32_t srcTag) {
    return false;
  }
  
  // This method actually does the merge operation

  void mergeEdge(SuperpixelEdge &edge) {
    mergeStep += 1;
    
    spImage.mergeEdge(edge);
    
    return;
  }
  
};

// The process of merging superpixels depends on some tricky state
// and looping logic. This generic templated class make use of
// a class known to implement the abstract API defined by
// SuperpixelMergeManager to process merge operations.

template <class T>
int SuperpixelMergeManagerFunc(T & mergeManager) {
  const bool debug = true;

  // Setup does one time init and cache logic
  
  mergeManager.setup();
  
  // Scan superpixels and compare to neighbor superpixels using predicate.
  // Note that a copy of the superpixels table must be scanned since the
  // superpixels table can have elements deleted from it as part of a
  // merge during the loop.

  auto it = begin(mergeManager.superpixels);
  auto endIter = end(mergeManager.superpixels);
  
  int32_t currentTag = -1;
  
  for ( ; it != endIter; ) {
    int32_t tag = *it;
    
    if (mergeManager.checkProcessed(tag) == false) {
      if (debug) {
        cout << "superpixel " << tag << " is already processed" << endl;
      }
      
      ++it;
      continue;
    }
    
    Superpixel *spPtr = mergeManager.spImage.getSuperpixelPtr(tag);
    
    if (spPtr == NULL) {
      // Check for the edge case of this superpixel being merged into a neighbor as a result
      // of a previous iteration.
      
      if (debug) {
        cout << "superpixel " << tag << " was merged away already" << endl;
      }
      
      ++it;
      continue;
    }
    
    if (tag != currentTag) {
      currentTag = tag;
      mergeManager.startProcessing(tag);
    }
    
    // Iterate over all neighbor superpixels and do merge based on criteria
    
    auto &neighborsSet = mergeManager.spImage.edgeTable.getNeighborsSet(tag);
    
    if (debug) {
      cout << "found " << neighborsSet.size() << " neighbors of superpixel " << tag << endl;
      
      for ( int32_t neighborTag : neighborsSet ) {
        cout << "neighbor " << neighborTag << endl;
      }
    }
    
    bool mergedNeighbor = false;
    
    for ( auto neighborIter = neighborsSet.begin(); neighborIter != neighborsSet.end(); ) {
      int32_t neighborTag = *neighborIter;
      // Advance the iterator to the next neighbor before a possible merge
      ++neighborIter;
      
      bool doMerge = mergeManager.checkEdge(tag, neighborTag);
      
      if (debug) {
        cout << "neighbor " << neighborTag << " doMerge -> " << doMerge << endl;
      }
      
      if (doMerge) {
        if (debug) {
          cout << "found superpixels " << tag << " and " << neighborTag << " (merging)" << endl;
        }

        SuperpixelEdge edge(tag, neighborTag);
        mergeManager.mergeEdge(edge);
        
        if (mergeManager.spImage.getSuperpixelPtr(tag) == NULL) {
          // In the case where the identical superpixel was merged into a neighbor then
          // the neighbors have changed and this iteration has to end.
          
          if (debug) {
            cout << "ending neighbors iteration since " << tag << " was merged into identical neighbor" << endl;
          }
          
          break;
        } else {
          // Successfully merged neighbor into this superpixel
          mergedNeighbor = true;
        }
      }
    } // end foreach neighbors loop
    
    if (mergedNeighbor) {
      if (debug) {
        cout << "repeating merge loop for superpixel " << tag << " since neighbor was merged" << endl;
      }
    } else {
      if (debug) {
        cout << "advance iterator from superpixel " << tag << " since no neighbor was merged" << endl;
      }

      mergeManager.doneProcessing(tag);
      
      ++it;
    }
  } // end for superpixelsVec loop
  
  // Setup does one time init and cache logic
  
  mergeManager.finish();
  
  return mergeManager.mergeStep;
}

#endif /* SuperpixelMergeManager_hpp */
