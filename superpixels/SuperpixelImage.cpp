// A superpixel image is a matrix that contains N superpixels and N superpixel edges between superpixels.
// A superpixel image is typically parsed from a source of tags, modified, and then written as a new tags
// image.

#include "SuperpixelImage.h"
#include "SuperpixelEdgeFuncs.h"

#include "Superpixel.h"

#include "SuperpixelEdge.h"

#include "SuperpixelEdgeTable.h"

#include "Util.h"

#include "OpenCVUtil.h"

#include <iomanip>      // setprecision

const int MaxSmallNumPixelsVal = 10;

void parse3DHistogram(Mat *histInputPtr,
                      Mat *histPtr,
                      Mat *backProjectInputPtr,
                      Mat *backProjectPtr,
                      int conversion,
                      int numBins);

bool pos_sample_within_bound(vector<float> &weights, float currentWeight);

void generateStaticColortable(Mat &inputImg, SuperpixelImage &spImage);

void writeTagsWithStaticColortable(SuperpixelImage &spImage, Mat &resultImg);

// Note that the valid range for tags is 0 -> 0x00FFFFFF so that
// -1 can be used to indicate no tag. There will never be so many
// tags that 24bits is not enough tags.

// Sort by size with largest superpixel first

typedef
struct SuperpixelSortStruct
{
  Superpixel *spPtr;
} SuperpixelSortStruct;

static
bool CompareSuperpixelSizeDecreasingFunc (SuperpixelSortStruct &s1, SuperpixelSortStruct &s2) {
  Superpixel *sp1Ptr = s1.spPtr;
  Superpixel *sp2Ptr = s2.spPtr;
  int sp1N = (int) sp1Ptr->coords.size();
  int sp2N = (int) sp2Ptr->coords.size();
  if (sp1N == sp2N) {
    // In case of a tie, sort by tag so that edges are processed in
    // increasing tag order to help with edge auto ordering.
    int32_t tag1 = sp1Ptr->tag;
    int32_t tag2 = sp2Ptr->tag;
    return (tag1 < tag2);
  } else {
    return (sp1N > sp2N);
  }
}

template <class ForwardIterator, class T>
ForwardIterator binary_search_iter (ForwardIterator first, ForwardIterator last, const T& val)
{
  first = std::lower_bound(first,last,val);
  if (first != last) {
    if (val < *first) {
      return last;
    } else {
      return first;
    }
  } else {
    return last;
  }
}

bool SuperpixelImage::parse(Mat &tags, SuperpixelImage &spImage) {
  const bool debug = false;
  
  assert(tags.channels() == 3);
  
  TagToSuperpixelMap &tagToSuperpixelMap = spImage.tagToSuperpixelMap;
  
  auto &superpixels = spImage.superpixels;
  
  for( int y = 0; y < tags.rows; y++ ) {
    for( int x = 0; x < tags.cols; x++ ) {
      Vec3b tagVec = tags.at<Vec3b>(y, x);
      int32_t tag = Vec3BToUID(tagVec);
      
      // Note that each tag value is modified here so that no superpixel
      // will have the tag zero.
      
      if (1) {
        // Note that an input tag value must always be smaller than 0x00FFFFFF
        // since this logic will implicitly add 1 to each pixel value to make
        // sure that zero is not used as a valid tag value while processing.
        // This means that the image cannot use the value for all white as
        // a valid tag value, but that is not a big deal since every other value
        // can be used.
        
        if (tag == 0xFFFFFF) {
          cerr << "error : tag pixel has the value 0xFFFFFF which is not supported" << endl;
          return false;
        }
        assert(tag < 0x00FFFFFF);
        tag += 1;
        tagVec[0] = tag & 0xFF;
        tagVec[1] = (tag >> 8) & 0xFF;
        tagVec[2] = (tag >> 16) & 0xFF;
        tags.at<Vec3b>(y, x) = tagVec;
      }
      
      auto iter = tagToSuperpixelMap.find(tag);
      
      if (iter == tagToSuperpixelMap.end()) {
        // A Superpixel has not been created for this UID since no key
        // exists in the table. Create a superpixel and wrap into a
        // unique smart pointer so that the table contains the only
        // live object reference to the Superpixel object.
        
        if (debug) {
          cout << "create Superpixel for UID " << tag << endl;
        }
        
        Superpixel *spPtr = new Superpixel(tag);
        iter = tagToSuperpixelMap.insert(iter, make_pair(tag, spPtr));
        superpixels.insert(tag);
      } else {
        if (debug) {
          cout << "exists  Superpixel for UID " << tag << endl;
        }
      }

      Superpixel *spPtr = iter->second;
      assert(spPtr->tag == tag);
      
      spPtr->appendCoord(x, y);
    }
  }

  assert(superpixels.size() == tagToSuperpixelMap.size());
  
  // Print superpixel info
  
  if (debug) {
    cout << "added " << (tags.rows * tags.cols) << " pixels as " << superpixels.size() << " superpixels" << endl;
    
    for (auto it = superpixels.begin(); it != superpixels.end(); ++it) {
      int32_t tag = *it;
      
      assert(tagToSuperpixelMap.count(tag) > 0);
      Superpixel *spPtr = tagToSuperpixelMap[tag];
      
      cout << "superpixel UID = " << tag << " contains " << spPtr->coords.size() << " coords" << endl;
      
      for (auto coordsIt = spPtr->coords.begin(); coordsIt != spPtr->coords.end(); ++coordsIt) {
        int32_t X = coordsIt->x;
        int32_t Y = coordsIt->y;
        cout << "X,Y" << " " << X << "," << Y << endl;
      }
    }
  }
  
  // Generate edges for each superpixel by looking at superpixel UID's around a given X,Y coordinate
  // and determining the other superpixels that are connected to each superpixel.
  
  bool worked = SuperpixelImage::parseSuperpixelEdges(tags, spImage);
  
  if (!worked) {
    return false;
  }
  
  // Deallocate original input tags image since it could be quite large
  
  //tags.release();
  
  return true;
}

// Examine superpixels in an image and parse edges from the superpixel coords

bool SuperpixelImage::parseSuperpixelEdges(Mat &tags, SuperpixelImage &spImage) {
  const bool debug = false;
  
#if defined(DEBUG)
  auto &superpixels = spImage.superpixels;
#endif // DEBUG
  
  SuperpixelEdgeTable &edgeTable = spImage.edgeTable;
  
  // For each (X,Y) tag value find all the other tags defined in suppounding 8 pixels
  // and dedup the list of neighbor tags for each superpixel.

  int32_t neighborOffsetsArr[] = {
    -1, -1, // UL
     0, -1, // U
     1, -1, // UR
    -1,  0, // L
     1,  0, // R
    -1,  1, // DL
     0,  1, // D
     1,  1  // DR
  };
  
  vector<pair<int32_t, int32_t> > neighborOffsets;
  
  for (int i = 0; i < sizeof(neighborOffsetsArr)/sizeof(int32_t); i += 2) {
    pair<int32_t,int32_t> p(neighborOffsetsArr[i], neighborOffsetsArr[i+1]);
    neighborOffsets.push_back(p);
  }
  
  assert(neighborOffsets.size() == 8);
  
  unordered_map<int32_t, set<int32_t> > &tagToNeighborMap = edgeTable.getNeighborsRef();

  for( int y = 0; y < tags.rows; y++ ) {
    for( int x = 0; x < tags.cols; x++ ) {
      Vec3b tagVec = tags.at<Vec3b>(y, x);
      
      int32_t centerTag = Vec3BToUID(tagVec);
      
      if (debug) {
      cout << "center (" << x << "," << y << ") with tag " << centerTag << endl;
      }
      
      auto iter = tagToNeighborMap.find(centerTag);
      
      if (iter == tagToNeighborMap.end()) {
        // A Superpixel has not been created for this UID since no key
        // exists in the table.
        
        if (debug) {
          cout << "create neighbor vector for UID " << centerTag << endl;
        }
        
        set<int32_t> neighbors;
        iter = tagToNeighborMap.insert(iter, make_pair(centerTag, neighbors));
      } else {
        if (debug) {
          cout << "exits  neighbor vector for UID " << centerTag << endl;
        }
      }
      
      set<int32_t> &neighborUIDsSet = iter->second;

      // Loop over each neighbor around (X,Y) and lookup tag
      
      for (auto pairIter = neighborOffsets.begin() ; pairIter != neighborOffsets.end(); ++pairIter) {
        int dX = pairIter->first;
        int dY = pairIter->second;
        
        int foundNeighborUID;
        
        int nX = x + dX;
        int nY = y + dY;
        
        if (nX < 0 || nX >= tags.cols) {
          foundNeighborUID = -1;
        } else if (nY < 0 || nY >= tags.rows) {
          foundNeighborUID = -1;
        } else {
          Vec3b neighborTagVec = tags.at<Vec3b>(nY, nX);
          foundNeighborUID = Vec3BToUID(neighborTagVec);
        }

        if (foundNeighborUID == -1 || foundNeighborUID == centerTag) {
          if (debug) {
            cout << "ignoring (" << nX << "," << nY << ") with tag " << foundNeighborUID << " since invalid or identity" << endl;
          }
        } else {
          if (debug) {
            cout << "checking (" << nX << "," << nY << ") with tag " << foundNeighborUID << " to see if known neighbor" << endl;
          }
          
          // if (!found) add_to_set()
          
          auto findIter = neighborUIDsSet.find(foundNeighborUID);
          
          if (findIter == neighborUIDsSet.end()) {
            neighborUIDsSet.insert(findIter, foundNeighborUID);
            
            if (debug) {
            cout << "added new neighbor tag " << foundNeighborUID << endl;
            }
          }
        }
      }
      
      if (debug) {
        cout << "after searching all neighbors of (" << x << "," << y << ") the neighbors array (len " << neighborUIDsSet.size() << ") is:" << endl;
        
        for ( int32_t knownNeighborUID : neighborUIDsSet ) {
          cout << knownNeighborUID << endl;
        }
      }
      
    }
  }
  
#if defined(DEBUG)
  
  // Each superpixel now has a vector of values for each neighbor. Create unique list of edges by
  // iterating over the superpixels and only creating an edge object when a pair (A, B) is
  // found where A < B.
  
  for (auto it = superpixels.begin(); it != superpixels.end(); ++it) {
    int32_t tag = *it;
    
    // Foreach superpixel, collect all the connected neighbors and generate edges
    
    assert(tagToNeighborMap.count(tag) > 0);
    
    auto &neighborUIDsSet = tagToNeighborMap[tag];

    if (debug) {
    cout << "superpixel UID = " << tag << " neighbors len = " << neighborUIDsSet.size() << endl;
    }
    
    // Every superpixel must have at least 1 neighbor or the input is invalid, unless there
    // is only 1 superpixel to begin with which could happen if all input was same pixel.
    
    if (superpixels.size() > 1) {
      assert(neighborUIDsSet.size() > 0);
    }
    
    // edgeTable.setNeighbors(tag, neighborUIDsSet);
  }
  
  if (debug) {
    cout << "created " << edgeTable.getAllEdges().size() << " edges in edge table" << endl;
  }
  
#endif // DEBUG

  return true;
}

void SuperpixelImage::mergeEdge(SuperpixelEdge &edgeToMerge) {
  const bool debug = false;
  
  if (debug) {
    cout << "mergeEdge (" << edgeToMerge.A << " " << edgeToMerge.B << ")" << endl;
  }
  
  assert(edgeToMerge.A != edgeToMerge.B);
  
#if defined(DEBUG)
  mergeOrder.push_back(edgeToMerge);
#endif
  
  // Get Superpixel object pointers (not copies of the objects)
  
  Superpixel *spAPtr = getSuperpixelPtr(edgeToMerge.A);
  assert(spAPtr);
  Superpixel *spBPtr = getSuperpixelPtr(edgeToMerge.B);
  assert(spBPtr);

  Superpixel *srcPtr;
  Superpixel *dstPtr;
  
  size_t numCoordsA;
  size_t numCoordsB;
  
  numCoordsA = spAPtr->coords.size();
  numCoordsB = spBPtr->coords.size();
  
  if (numCoordsA >= numCoordsB) {
    // Merged B into A since A is larger

    srcPtr = spBPtr;
    dstPtr = spAPtr;
    
    if (debug) {
      cout << "merge B -> A : " << srcPtr->tag << " -> " << dstPtr->tag << " : " << srcPtr->coords.size() << " <= " << dstPtr->coords.size() << endl;
    }
  } else {
    // Merge A into B since B is larger
    
    srcPtr = spAPtr;
    dstPtr = spBPtr;
    
    if (debug) {
      cout << "merge A -> B : " << srcPtr->tag << " -> " << dstPtr->tag << " : " << srcPtr->coords.size() << " < " << dstPtr->coords.size() << endl;
    }
  }
  
  if (debug) {
    cout << "will merge " << srcPtr->coords.size() << " coords from smaller into larger superpixel" << endl;
  }

  append_to_vector(dstPtr->coords, srcPtr->coords);
  srcPtr->coords.resize(0);
  
  // This logic assumes that the superpixels list is in increasing int order since the
  // parse logic explicitly sorts the generated tags. As superpixels are merged the
  // coords can be consumed by a previous superpixel, but the list should remain ordered
  // in int increasing order so that a binary search can be implemented.
  
#if defined(DEBUG)
  {
    int32_t prevTag = 0;
    
    for (auto it = superpixels.begin(); it != superpixels.end(); ++it) {
      int32_t tag = *it;
      assert(tag > prevTag);
      prevTag = tag;
    }
  }
#endif // DEBUG
  
  // Find entry for srcPtr->tags in superpixels and remove the UID
  
  int32_t tag = srcPtr->tag;
  int numErased = (int) superpixels.erase(tag);
  assert (numErased == 1);

  bool hasEdgeStrengthMap;
  int numRemoved;
  
  // Remove edge between src and dst by removing src from dst neighbors set

  hasEdgeStrengthMap = (edgeTable.edgeStrengthMap.size() > 0);
  
  if (hasEdgeStrengthMap) {
    // Clear edge strength cache of dst->src edge
    SuperpixelEdge cachedKey(dstPtr->tag, srcPtr->tag);
    edgeTable.edgeStrengthMap.erase(cachedKey);
  }
  
  set<int32_t> &neighborsOfDst = edgeTable.getNeighborsSet(dstPtr->tag);
  
  if (debug) {
    cout << "initial dst neighbor set :" << endl;
    for ( int32_t neighborTag : neighborsOfDst ) {
      cout << neighborTag << endl;
    }
  }
  
  // Remove src tag from neighbors of dst set
  
  numRemoved = (int) neighborsOfDst.erase(srcPtr->tag);
  assert(numRemoved == 1);
  
#if defined(DEBUG)
  if (superpixels.size() > 1) {
    assert(neighborsOfDst.size() > 0);
  }
#endif // DEBUG
  
  if (debug) {
    cout << "final dst neighbor set :" << endl;
    for ( int32_t neighborTag : neighborsOfDst ) {
      cout << neighborTag << endl;
    }
  }
  
  // Update neighbors of src by adding dst as a neighbor.
  // In the case where dst is already a neighbor,
  // the duplicate entry in the set is ignored.
  
  if (hasEdgeStrengthMap) {
    // Clear edge strength cache of src->dst edge
    SuperpixelEdge cachedKey(srcPtr->tag, dstPtr->tag);
    edgeTable.edgeStrengthMap.erase(cachedKey);
  }
  
  set<int32_t> &neighborsOfSrc = edgeTable.getNeighborsSet(srcPtr->tag);
  
  if (debug) {
    cout << "all neighbors of src = " << srcPtr->tag << endl;
    for ( int32_t neighborTag : neighborsOfSrc ) {
      cout << neighborTag << endl;
    }
  }
  
  for ( int32_t neighborOfSrcTag : neighborsOfSrc ) {
    if (debug) {
      cout << "iter neighbor of src = " << neighborOfSrcTag << endl;
    }
   
    if (neighborOfSrcTag == dstPtr->tag) {
      // Ignore dst so that src is deleted as a neighbor
      
      if (debug) {
        cout << "ignore neighbor of src since it is the dst node" << endl;
      }
    } else {
      set<int32_t> &neighbors = edgeTable.getNeighborsSet(neighborOfSrcTag);
      
      if (debug) {
        cout << "update neighbor of src " << neighborOfSrcTag << endl;
        cout << "initial neighbor of src neighbor set :" << endl;
        for ( int32_t neighborTag : neighbors ) {
          cout << neighborTag << endl;
        }
      }
      
      // Add edge between neighbor and dst (if it does not exist)
      
      neighbors.insert(dstPtr->tag);
      
      // Remove edge between neighbor and src
      
      numRemoved = (int) neighbors.erase(srcPtr->tag);
      assert(numRemoved == 1);
      
#if defined(DEBUG)
      assert(neighbors.size() > 0);
#endif // DEBUG
      
      if (debug) {
        cout << "final neighbor of src neighbor set :" << endl;
        for ( int32_t neighborTag : neighbors ) {
          cout << neighborTag << endl;
        }
      }
      
      // If this neighbor is not currently a neighbor of dst then
      // add it now with an add that is a nop for duplicate entries
      
      neighborsOfDst.insert(neighborOfSrcTag);
      
      if (debug) {
        cout << "final neighbors dst set :" << endl;
        for ( int32_t neighborTag : neighborsOfDst ) {
          cout << neighborTag << endl;
        }
      }
    }
  }
  
  if (debug) {
    cout << "final edge results for merged UID " << dstPtr->tag << endl;
    
    set<int32_t> &neighbors = edgeTable.getNeighborsSet(dstPtr->tag);
    
    cout << "final dst neighbor set :" << endl;
    for ( int32_t neighborTag : neighbors ) {
      cout << neighborTag << endl;
    }
  }
    
  edgeTable.removeNeighbors(srcPtr->tag);
  
  // Move edge weights from src to dst
  
  if (srcPtr->mergedEdgeWeights.size() > 0) {
    append_to_vector(dstPtr->mergedEdgeWeights, srcPtr->mergedEdgeWeights);
  }

  if (srcPtr->unmergedEdgeWeights.size() > 0) {
    append_to_vector(dstPtr->unmergedEdgeWeights, srcPtr->unmergedEdgeWeights);
  }
  
  // Finally remove the Superpixel object from the lookup table and free the memory
  
  int32_t tagToRemove = srcPtr->tag;
  tagToSuperpixelMap.erase(tagToRemove);
  delete srcPtr;
  
#if defined(DEBUG)
  // When compiled in DEBUG mode in Xcode enable additional runtime checks that
  // ensure that each neighbor of the merged node is also a neighbor of the other.

  srcPtr = getSuperpixelPtr(tagToRemove);
  assert(srcPtr == NULL);
  dstPtr = getSuperpixelPtr(dstPtr->tag);
  assert(dstPtr != NULL);
  
  for ( int32_t neighborTag : edgeTable.getNeighborsSet(dstPtr->tag)) {
    // Make sure that each neighbor of the merged superpixel also has the merged superpixel
    // as a neighbor.
    
    Superpixel *neighborPtr = getSuperpixelPtr(neighborTag);
    assert(neighborPtr != NULL);
    
    bool found = false;
    
    for ( int32_t nnTag : edgeTable.getNeighborsSet(neighborTag) ) {
      if (nnTag == dstPtr->tag) {
        found = true;
        break;
      }
    }
    
    assert(found);
  }
  
  // Check that merge src no longer appers in superpixels list
  
  for (auto it = superpixels.begin(); it != superpixels.end(); ++it) {
    int32_t tag = *it;
    
    if (tagToRemove == tag) {
      assert(0);
    }
    
    // Check that src is not a neighbor of any superpixel
    
    set<int32_t> &neighbors = edgeTable.getNeighborsSet(tag);
    
    for ( int32_t neighborTag : neighbors ) {
      if (neighborTag == tagToRemove) {
        assert(0);
      }
    }
  }
#endif // DEBUG
  
  return;
}

// Lookup Superpixel* given a UID, checks to make sure key is defined in table in DEBUG mode

Superpixel* SuperpixelImage::getSuperpixelPtr(int32_t uid)
{
  TagToSuperpixelMap::iterator iter = tagToSuperpixelMap.find(uid);
  
  if (iter == tagToSuperpixelMap.end()) {
    // In the event that a superpixel was merged into a neighbor during an iteration
    // over a list of known superpixel then this method could be invoked with a uid
    // that is no longer valid. Return NULL to make it possible to detect this case.
    
    return NULL;
  } else {
    // Otherwise the key exists in the table, return the cached pointer to
    // avoid a second lookup because this method is invoked a lot.
    
    return iter->second;
  }
}

// Scan superpixels looking for the case where all pixels in one superpixel exactly match all
// the superpixels in a neighbor superpixel. This exact matching situation can happen in flat
// image areas so removing the duplication can significantly simplify the graph before the
// more compotationally expensive histogram comparison logic is run on all edges. Reducing the
// number of edges via this more simplified method executes on N superpixels and then
// comparison to neighbors need only be done when the exact same pixels condition is found.

void SuperpixelImage::mergeIdenticalSuperpixels(Mat &inputImg) {
  const bool debug = false;
  
  // Scan list of superpixels and extract a list of the superpixels where all the
  // pixels have the exact same value. Doing this initial scan means that we
  // create a new list that will not be mutated in the case of a superpixel
  // merge.
  
  vector<int32_t> identicalSuperpixels;
  identicalSuperpixels.reserve(4096);
  
  for (auto it = superpixels.begin(); it != superpixels.end(); ++it) {
    int32_t tag = *it;
    
    bool isAllSame = isAllSamePixels(inputImg, tag);
    
    Superpixel *spPtr = getSuperpixelPtr(tag);
    
    if (isAllSame) {
      spPtr->setAllSame();
      identicalSuperpixels.push_back(tag);
    } else {
      spPtr->setNotAllSame();
    }
  }
  
  if (debug) {
    cout << "found " << identicalSuperpixels.size() << " superpixels with all identical pixel values" << endl;
  }
  
  for (auto it = identicalSuperpixels.begin(); it != identicalSuperpixels.end(); ) {
    int32_t tag = *it;
    
    // Do a single table lookup for the src superpixel before iterating over neighbors.
    
    Superpixel *spPtr = getSuperpixelPtr(tag);
    
    if (spPtr == NULL) {
      // Check for the edge case of this superpixel being merged into a neighbor as a result
      // of a previous iteration.
      
      if (debug) {
        cout << "identical superpixel " << tag << " was merged away already" << endl;
      }
      
      ++it;
      continue;
    }
    
    // Iterate over all neighbor superpixels and verify that all those pixels match
    // the first pixel value from the known identical superpixel. This loop invokes
    // merge during the loop so the list of neighbors needs to be a copy of the
    // neighbors list since the neighbors list can be changed by the merge.
    
    auto &neighborsSet = edgeTable.getNeighborsSet(tag);
    
    if (debug) {
      cout << "found neighbors of known identical superpixel " << tag << endl;
      
      for ( int32_t neighborTag : neighborsSet ) {
        cout << "neighbor " << neighborTag << endl;
      }
    }
    
    bool mergedNeighbor = false;
    
    for ( auto neighborIter = neighborsSet.begin(); neighborIter != neighborsSet.end(); ) {
      int32_t neighborTag = *neighborIter;
      // Advance the iterator to the next neighbor before a possible merge
      ++neighborIter;

      bool isAllSame = isAllSamePixels(inputImg, spPtr, neighborTag);
      
      if (debug) {
        cout << "neighbor " << neighborTag << " isAllSamePixels() -> " << isAllSame << endl;
      }
    
      if (isAllSame) {
        if (debug) {
          cout << "found identical superpixels " << tag << " and " << neighborTag << " (merging)" << endl;
        }
        
        SuperpixelEdge edge(tag, neighborTag);
        mergeEdge(edge);
        
        if (getSuperpixelPtr(tag) == NULL) {
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
        cout << "repeating merge loop for superpixel " << tag << " since identical neighbor was merged" << endl;
      }
    } else {
      if (debug) {
        cout << "advance iterator from superpixel " << tag << " since no neighbor was merged" << endl;
      }
      
      ++it;
    }
  } // end for identicalSuperpixels loop
  
  return;
}

// checkPredicate given 

bool
SuperpixelImage::checkPredicate(Mat &input, Superpixel *spPtr, int32_t otherTag)
{
  const bool debug = false;
  
  Superpixel *otherSpPtr = getSuperpixelPtr(otherTag);
  if (otherSpPtr == NULL) {
    // Return false when the neighbor was already merged away
    return false;
  }
  
  // Predicate test examines superpixels (S1, S2) to determine if each of
  // the pixels on the border between the regions is exactly the same.
  
  SuperpixelEdgeFuncs::checkNeighborEdgeWeights(*this, input, spPtr->tag, NULL, edgeTable.edgeStrengthMap, 0);
  
  if (debug) {
    for ( auto it = edgeTable.edgeStrengthMap.begin(); it != edgeTable.edgeStrengthMap.end(); ++it ) {
      SuperpixelEdge edge = it->first;
      float strength = it->second;
      
      cout << "edge " << edge << " has strength " << strength << endl;
    }
  }
  
  // If the edge between (S1, S2) is zero then merge
  
  SuperpixelEdge edge(spPtr->tag, otherSpPtr->tag);
  
  float strength = edgeTable.edgeStrengthMap[edge];
  
  if (strength == 0.0) {
    if (debug) {
      cout << "will merge edge " << edge << " with strength " << strength << endl;
    }
    
    return true;
  }
  
  return false;
}

// Scan superpixels looking for the case where one region of superpixels can be merged
// with a neighbor based on a merge predicate.

void SuperpixelImage::mergeSuperpixelsWithPredicate(Mat &inputImg) {
  const bool debug = true;
  
  // Do initial scan of all the superpixels looking for superpixels that
  // are known to be identical so that an optimal branch in the predicate
  // search logic need not scan every pixel in this special case.
  
  for (auto it = superpixels.begin(); it != superpixels.end(); ++it) {
    int32_t tag = *it;
    
    Superpixel *spPtr = getSuperpixelPtr(tag);
    
    if (spPtr->isAllSame() || spPtr->isNotAllSame()) {
      // This superpixel has already been scanned and flags set accordingly
      //continue;
      break;
    }
    
    bool isAllSame = isAllSamePixels(inputImg, tag);
    
    if (isAllSame) {
      spPtr->setAllSame();
    } else {
      spPtr->setNotAllSame();
    }
  }
  
  // Scan superpixels and compare to neighbor superpixels using predicate.
  // Note that a copy of the superpixels table must be scanned since the
  // superpixels table can have elements deleted from it as part of a
  // merge during the loop.
  
  auto superpixelsVec = getSuperpixelsVec();
  
  for (auto it = superpixelsVec.begin(); it != superpixelsVec.end(); ) {
    int32_t tag = *it;
    
    Superpixel *spPtr = getSuperpixelPtr(tag);
    
    if (spPtr == NULL) {
      // Check for the edge case of this superpixel being merged into a neighbor as a result
      // of a previous iteration.
      
      if (debug) {
        cout << "identical superpixel " << tag << " was merged away already" << endl;
      }
      
      ++it;
      continue;
    }
    
    // Iterate over all neighbor superpixels and do merge based on criteria
    
    auto &neighborsSet = edgeTable.getNeighborsSet(tag);
    
    if (debug) {
      cout << "found neighbors of superpixel " << tag << endl;
      
      for ( int32_t neighborTag : neighborsSet ) {
        cout << "neighbor " << neighborTag << endl;
      }
    }
    
    bool mergedNeighbor = false;
    
    for ( auto neighborIter = neighborsSet.begin(); neighborIter != neighborsSet.end(); ) {
      int32_t neighborTag = *neighborIter;
      // Advance the iterator to the next neighbor before a possible merge
      ++neighborIter;
      
      bool doMerge = checkPredicate(inputImg, spPtr, neighborTag);
      
      if (debug) {
        cout << "neighbor " << neighborTag << " doMerge -> " << doMerge << endl;
      }
      
      if (doMerge) {
        if (debug) {
          cout << "found superpixels " << tag << " and " << neighborTag << " (merging)" << endl;
        }
        
        SuperpixelEdge edge(tag, neighborTag);
        mergeEdge(edge);
        
        if (getSuperpixelPtr(tag) == NULL) {
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
      
      ++it;
    }
  } // end for superpixelsVec loop
  
  return;
}

// Sort superpixels by size and sort ties so that smaller pixel tag values appear
// before larger tag values.

vector<int32_t>
SuperpixelImage::sortSuperpixelsBySize()
{
  const bool debug = false;
  
  vector<SuperpixelSortStruct> sortedSuperpixels;
  sortedSuperpixels.reserve(superpixels.size());
  
  for (auto it = superpixels.begin(); it != superpixels.end(); ++it) {
    int32_t tag = *it;
    SuperpixelSortStruct ss;
    ss.spPtr = getSuperpixelPtr(tag);
    sortedSuperpixels.push_back(ss);
  }
  
  sort(sortedSuperpixels.begin(), sortedSuperpixels.end(), CompareSuperpixelSizeDecreasingFunc);
  
  assert(sortedSuperpixels.size() == superpixels.size());
  
  int i = 0;
  
  vector<int32_t> retVec;
  retVec.reserve(superpixels.size());
  
  for (auto it = sortedSuperpixels.begin(); it != sortedSuperpixels.end(); ++it, i++) {
    SuperpixelSortStruct ss = *it;
    int32_t tag = ss.spPtr->tag;

    retVec.push_back(tag);
    
    if (debug) {
      cout << "sorted superpixel at offset " << i << " now has tag " << tag << " with N = " << ss.spPtr->coords.size() << endl;
    }
  }
  
#if defined(DEBUG)
  // Loop over sorted pixels and verify that sizes decrease as list is iterated
  int prevSize = 0;
  
  for (auto it = retVec.begin(); it != retVec.end(); ++it) {
    int32_t tag = *it;
    Superpixel *spPtr = getSuperpixelPtr(tag);
    int currentSize = (int) spPtr->coords.size();
    // Ignore first element
    if (prevSize != 0) {
      assert(currentSize <= prevSize);
    }
    assert(currentSize > 0);
    prevSize = currentSize;
  }
#endif // DEBUG
  
  return retVec;
}

// This util method scans the current list of superpixels and returns the largest superpixels
// using a stddev measure. These largest superpixels are highly unlikely to be useful when
// scanning for edges on smaller elements, for example. This method should be run after
// initial joining has identified the largest superpxiels.

void
SuperpixelImage::scanLargestSuperpixels(vector<int32_t> &results)
{
  const bool debug = false;
  
  const int maxSmallNum = MaxSmallNumPixelsVal;
  
  vector<float> superpixelsSizes;
  vector<uint32_t> superpixelsForSizes;
  
  results.clear();
  
  // First, scan for very small superpixels and treat them as edges automatically so that
  // edge pixels scanning need not consider these small pixels.
  
  for (auto it = superpixels.begin(); it != superpixels.end(); ++it) {
    int32_t tag = *it;
    Superpixel *spPtr = getSuperpixelPtr(tag);
    assert(spPtr);
    
    int numCoords = (int) spPtr->coords.size();
    
    if (numCoords < maxSmallNum) {
      // Ignore really small superpixels in the stats
    } else {
      superpixelsSizes.push_back((float)numCoords);
      superpixelsForSizes.push_back(tag);
    }
  }
  
  if (debug) {
    cout << "found " << superpixelsSizes.size() << " non-small superpixel sizes" << endl;
    
    vector<float> copySizes = superpixelsSizes;
    
    // Sort descending
    
    sort(copySizes.begin(), copySizes.end(), greater<float>());
    
    for (auto it = copySizes.begin(); it != copySizes.end(); ++it) {
      cout << *it << endl;
    }
  }
  
  float mean, stddev;
  
  sample_mean(superpixelsSizes, &mean);
  sample_mean_delta_squared_div(superpixelsSizes, mean, &stddev);
  
  if (debug) {
    char buffer[1024];
    
    snprintf(buffer, sizeof(buffer), "mean %0.4f stddev %0.4f", mean, stddev);
    cout << (char*)buffer << endl;

    snprintf(buffer, sizeof(buffer), "1 stddev %0.4f", (mean + (stddev * 0.5f * 1.0f)));
    cout << (char*)buffer << endl;
    
    snprintf(buffer, sizeof(buffer), "2 stddev %0.4f", (mean + (stddev * 0.5f * 2.0f)));
    cout << (char*)buffer << endl;

    snprintf(buffer, sizeof(buffer), "3 stddev %0.4f", (mean + (stddev * 0.5f * 3.0f)));
    cout << (char*)buffer << endl;
  }
  
  // If the stddev is not at least 100 then these pixels are very small and it is unlikely
  // than any one would be significantly larger than the others. Simply return an empty
  // list as results in this case.
  
  const float minStddev = 100.0f;
  if (stddev < minStddev) {
    if (debug) {
      cout << "small stddev " << stddev << " found so returning empty list of largest superpixels" << endl;
    }
    
    return;
  }
  
  float upperLimit = mean + (stddev * 0.5f * 3.0f); // Cover 99.7 percent of the values
  
  if (debug) {
    char buffer[1024];
    
    snprintf(buffer, sizeof(buffer), "upperLimit %0.4f", upperLimit);
    cout << (char*)buffer << endl;
  }
  
  int offset = 0;
  for (auto it = superpixelsSizes.begin(); it != superpixelsSizes.end(); ++it, offset++) {
    float numCoords = *it;
    
    if (numCoords <= upperLimit) {
      // Ignore this element
      
      if (debug) {
        uint32_t tag = superpixelsForSizes[offset];
        cout << "ignore superpixel " << tag << " with N = " << (int)numCoords << endl;
      }
    } else {
      uint32_t tag = superpixelsForSizes[offset];
      
      if (debug) {
        cout << "keep superpixel " << tag << " with N = " << (int)numCoords << endl;
      }
      
      results.push_back(tag);
    }
  }
  
  return;
}

// This method will examine the bounds of the largest superpixels and then use a backprojection
// to recalculate the exact bounds where the larger smooth area runs into edges defined by
// the smaller superpixels. For example, in many images with an identical background the primary
// edge is defined between the background color and the foreground item(s). This method will
// write a new output image

void SuperpixelImage::rescanLargestSuperpixels(Mat &inputImg, Mat &outputImg, vector<int32_t> *largeSuperpixelsPtr)
{
  const bool debug = false;
  const bool debugDumpSuperpixels = false;
  const bool debugDumpBackprojections = false;
  
  vector<int32_t> largeSuperpixels;
  if (largeSuperpixelsPtr != NULL) {
    largeSuperpixels = *largeSuperpixelsPtr;
  } else {
    scanLargestSuperpixels(largeSuperpixels);
  }

  // Gather superpixels that are larger than the upper limit

  outputImg.create(inputImg.size(), CV_8UC(3));
  outputImg = (Scalar)0;

  for (auto it = largeSuperpixels.begin(); it != largeSuperpixels.end(); ++it) {
    int32_t tag = *it;
    Superpixel *spPtr = getSuperpixelPtr(tag);
    assert(spPtr);
    
    // Do back projection after trimming the large superpixel range. First, simply emit the large
    // superpixel as an image.
    
    Mat srcSuperpixelMat;
    Mat srcSuperpixelHist;
    Mat srcSuperpixelBackProjection;
    
    // Read RGB pixel data from main image into matrix for this one superpixel and then gen histogram.
    
    fillMatrixFromCoords(inputImg, tag, srcSuperpixelMat);
    
    parse3DHistogram(&srcSuperpixelMat, &srcSuperpixelHist, NULL, NULL, 0, -1);
    
    if (debugDumpSuperpixels) {
      std::ostringstream stringStream;
      stringStream << "superpixel_" << tag << ".png";
      
      std::string str = stringStream.str();
      const char *filename = str.c_str();
      
      Mat revMat = outputImg.clone();
      revMat = (Scalar) 0;
      
      reverseFillMatrixFromCoords(srcSuperpixelMat, false, tag, revMat);
      
      cout << "write " << filename << " ( " << revMat.cols << " x " << revMat.rows << " )" << endl;
      imwrite(filename, revMat);
    }

    // Generate back projection for entire image
    
    parse3DHistogram(NULL, &srcSuperpixelHist, &inputImg, &srcSuperpixelBackProjection, 0, -1);
    
    // srcSuperpixelBackProjection is a grayscale 1 channel image

    if (debugDumpBackprojections) {
      std::ostringstream stringStream;
      stringStream << "backproject_from_" << tag << ".png";
      
      std::string str = stringStream.str();
      const char *filename = str.c_str();
      
      cout << "write " << filename << " ( " << srcSuperpixelBackProjection.cols << " x " << srcSuperpixelBackProjection.rows << " )" << endl;
      imwrite(filename, srcSuperpixelBackProjection);
    }
    
    // A back projection is more efficient if we actually know a range to indicate how near to the edge the detected edge line is.
    // But, if the amount it is off is unknown then how to determine the erode range?
    
    // Doing a back projection is more effective if the actualy
    
    // Erode the superpixel shape a little bit to draw it back from the likely edge.
    
    Mat erodeBWMat(inputImg.size(), CV_8UC(1), Scalar(0));
    Mat bwPixels(srcSuperpixelMat.size(), CV_8UC(1), Scalar(255));
    
    // FIXME: rework fill to write to the kind of Mat either color or BW
    
    reverseFillMatrixFromCoords(bwPixels, true, tag, erodeBWMat);
    
    if (debugDumpSuperpixels) {
      std::ostringstream stringStream;
      stringStream << "superpixel_" << tag << "_bw.png";
      
      std::string str = stringStream.str();
      const char *filename = str.c_str();
      
      cout << "write " << filename << " ( " << erodeBWMat.cols << " x " << erodeBWMat.rows << " )" << endl;
      imwrite(filename, erodeBWMat);
    }
    
    // Erode to pull superpixel edges back by a few pixels
    
//    Mat erodeMinBWMat = erodeBWMat.clone();
//    erodeMinBWMat = (Scalar) 0;
//    Mat erodeMaxBWMat = erodeBWMat.clone();
//    erodeMaxBWMat = (Scalar) 0;

    Mat minBWMat, maxBWMat;
    
    //int erosion_type = MORPH_ELLIPSE;
    int erosion_type = MORPH_RECT;
    
    // FIXME: should the erode size depend on the image dimensions?
    
    int erosion_size = 1;
    
//    if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
//    else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
//    else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
    
    Mat element = getStructuringElement( erosion_type,
                                        Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                        Point( erosion_size, erosion_size ) );
    
    // Apply the erosion operation to reduce the white area
    
    erode( erodeBWMat, minBWMat, element );
    
    if (debugDumpSuperpixels) {
      std::ostringstream stringStream;
      stringStream << "superpixel_" << tag << "_bw_erode.png";
      
      std::string str = stringStream.str();
      const char *filename = str.c_str();
      
      cout << "write " << filename << " ( " << minBWMat.cols << " x " << minBWMat.rows << " )" << endl;
      imwrite(filename, minBWMat);
    }
    
    // Apply a dilate to expand the white area
    
    dilate( erodeBWMat, maxBWMat, element );

    if (debugDumpSuperpixels) {
      std::ostringstream stringStream;
      stringStream << "superpixel_" << tag << "_bw_dilate.png";
      
      std::string str = stringStream.str();
      const char *filename = str.c_str();
      
      cout << "write " << filename << " ( " << maxBWMat.cols << " x " << maxBWMat.rows << " )" << endl;
      imwrite(filename, maxBWMat);
    }
    
    // Calculate gradient x 2 which is erode and dialate and then intersection.
    // This area is slightly fuzzy to account for merging not getting exactly
    // on the edge.
    
    Mat gradMat;
    
    morphologyEx(erodeBWMat, gradMat, MORPH_GRADIENT, element, Point(-1,-1), 1);

    if (debugDumpSuperpixels) {
      std::ostringstream stringStream;
      stringStream << "superpixel_" << tag << "_bw_gradient.png";
      
      std::string str = stringStream.str();
      const char *filename = str.c_str();
      
      cout << "write " << filename << " ( " << gradMat.cols << " x " << gradMat.rows << " )" << endl;
      imwrite(filename, gradMat);
    }
    
    // The white pixels indicate where histogram backprojection should be examined, a mask on whole image.
    // A histogram could be computed from the entire background area, or it could be computed from the
    // area around the identified but known to not be the edge.
    
    int numNonZero = countNonZero(gradMat);
    
    Mat backProjectInputFlatMat(1, numNonZero, CV_8UC(3));
    Mat backProjectOutputFlatMat(1, numNonZero, CV_8UC(1));
    
    vector <pair<int32_t, int32_t>> coords;
    
    int offset = 0;
    
    for( int y = 0; y < gradMat.rows; y++ ) {
      for( int x = 0; x < gradMat.cols; x++ ) {
        uint8_t bVal = gradMat.at<uint8_t>(y, x);
        
        if (bVal) {
          Vec3b pixelVec = inputImg.at<Vec3b>(y, x);
          backProjectInputFlatMat.at<Vec3b>(0, offset++) = pixelVec;
        }
      }
    }
    
    assert(offset == numNonZero);
    
    // Generate back projection for just the mask area.
    
    if (debug) {
      backProjectOutputFlatMat = (Scalar) 0;
    }
    
    parse3DHistogram(NULL, &srcSuperpixelHist, &backProjectInputFlatMat, &backProjectOutputFlatMat, 0, -1);
    
    // copy back projection pixels back into full size image, note that any pixels not in the mask
    // identified by gradMat are ignored.
    
    Mat maskedGradientMat = erodeBWMat.clone();
    maskedGradientMat = (Scalar) 0;
    
    offset = 0;
    
    for( int y = 0; y < gradMat.rows; y++ ) {
      for( int x = 0; x < gradMat.cols; x++ ) {
        uint8_t bVal = gradMat.at<uint8_t>(y, x);
        
        if (bVal) {
          uint8_t per = backProjectOutputFlatMat.at<uint8_t>(0, offset++);
          maskedGradientMat.at<uint8_t>(y, x) = per;
        }
      }
    }
    
    assert(offset == numNonZero);
    
    if (debugDumpSuperpixels) {
      std::ostringstream stringStream;
      stringStream << "superpixel_" << tag << "_bw_gradient_backproj.png";
      
      std::string str = stringStream.str();
      const char *filename = str.c_str();
      
      cout << "write " << filename << " ( " << maskedGradientMat.cols << " x " << maskedGradientMat.rows << " )" << endl;
      imwrite(filename, maskedGradientMat);
    }
    
    // The generated back projection takes the existing edge around the foreground object into
    // account with this approach since the histogram was created from edge defined by the
    // superpixel segmentation. If the goal is to have the edge right on the detected edge
    // then this would seem to be best. If instead the background is defined by creating a
    // histogram from the area pulled away from the edge then the background histogram would
    // not get as close to the foreground object.
    
  }
  
  return;
}

// Return vector of all edges
vector<SuperpixelEdge> SuperpixelImage::getEdges()
{
  return edgeTable.getAllEdges();
}

// Read RGB values from larger input image and create a matrix that is the width
// of the superpixel and contains just the pixels defined by the coordinates
// contained in the superpixel. The caller passes in the tag from the superpixel
// in question in order to find the coords.

void SuperpixelImage::fillMatrixFromCoords(Mat &input, int32_t tag, Mat &output) {
  Superpixel *spPtr = getSuperpixelPtr(tag);
  spPtr->fillMatrixFromCoords(input, tag, output);
}

// This method is the inverse of fillMatrixFromCoords(), it reads pixel values from a matrix
// and writes them back to the corresponding X,Y values location in an image. This method is
// very useful when running an image operation on all the pixels in a superpixel but without
// having to process all the pixels in a bbox area. The dimensions of the input must be
// NUM_COORDS x 1. The caller must init the matrix values and the matrix size. This method
// can be invoked multiple times to write multiple superpixel values to the same output
// image.

void SuperpixelImage::reverseFillMatrixFromCoords(Mat &input, bool isGray, int32_t tag, Mat &output) {
  Superpixel *spPtr = getSuperpixelPtr(tag);
  spPtr->reverseFillMatrixFromCoords(input, isGray, tag, output);
}

// Read RGB values from larger input image based on coords defined for the superpixel
// and return true only if all the pixels have the exact same value.

bool SuperpixelImage::isAllSamePixels(Mat &input, int32_t tag) {
  const bool debug = false;
  
  Superpixel *spPtr = getSuperpixelPtr(tag);
  
  auto &coords = spPtr->coords;
  
  if (debug) {
    int numCoords = (int) coords.size();
    
    cout << "checking for superpixel all same pixels for " << tag << " with coords N=" << numCoords << endl;
  }
  
  Coord coord = coords[0];
  int32_t X = coord.x;
  int32_t Y = coord.y;
  Vec3b pixelVec = input.at<Vec3b>(Y, X);
  uint32_t knownFirstPixel = (uint32_t) (Vec3BToUID(pixelVec) & 0x00FFFFFF);
  
  return isAllSamePixels(input, knownFirstPixel, coords);
}

// When a superpixel is known to have all identical pixel values then only the first
// pixel in that superpixel needs to be compared to all the other pixels in a second
// superpixel. This optimized code accepts a pointer to the first superpixel in
// order to avoid repeated table lookups.

bool SuperpixelImage::isAllSamePixels(Mat &input, Superpixel *spPtr, int32_t otherTag) {
  Superpixel *otherSpPtr = getSuperpixelPtr(otherTag);
  if (otherSpPtr == NULL) {
    // In the case where a neighbor superpixel was already merged then
    // just return false for the all same test.
    return false;
  }
  
  // The current superpixel is assumed to contain all the same pixels
#if defined(DEBUG)
  assert(spPtr->isAllSame() == true);
#endif // DEBUG
  
  // Special case, spPtr is known to contain all same but otherSpPtr does
  // not contain all the same values. Return false early.
  
  if (otherSpPtr->isNotAllSame()) {
    // Shortcut when pixels in other superpixel are known to not be all the same
    return false;
  }
  
  // Get pixel value from first coord in first superpixel
  
  Coord coord = spPtr->coords[0];
  int32_t X = coord.x;
  int32_t Y = coord.y;
  Vec3b pixelVec = input.at<Vec3b>(Y, X);
  uint32_t knownFirstPixel = (uint32_t) (Vec3BToUID(pixelVec) & 0x00FFFFFF);
  
  // Special case when otherSpPtr is known to contain pixels that are all
  // exactly the same. In this case, code need only check knownFirstPixel
  // against the first value in otherSpPtr->coords.
  
  if (otherSpPtr->isAllSame()) {
    Coord coord = otherSpPtr->coords[0];
    int32_t X = coord.x;
    int32_t Y = coord.y;
    
    Vec3b pixelVec = input.at<Vec3b>(Y, X);
    uint32_t otherPixel = (uint32_t) (Vec3BToUID(pixelVec) & 0x00FFFFFF);
    
    if (knownFirstPixel == otherPixel) {
      return true;
    }
  }
  
  // Compare first value to all other values in the other superpixel.
  // Performance is critically important here so this code takes care
  // to avoid making a copy of the coords vector or the elements inside
  // the vector since there can be many coordinates.
  
  return isAllSamePixels(input, knownFirstPixel, otherSpPtr->coords);
}

// Read RGB values from larger input image based on coords defined for the superpixel
// and return true only if all the pixels have the exact same value. This method
// accepts a knownFirstPixel value which is a pixel value known to be the first
// value for matching purposes. In the case where two superpixels are being compared
// then determine the known pixel from the first group and pass the coords for the
// second superpixel which will then be checked one by one. This optimization is
// critically important when a very large number of oversegmented superpixel were
// parsed from the original image.

bool SuperpixelImage::isAllSamePixels(Mat &input, uint32_t knownFirstPixel, vector<Coord> &coords) {
  const bool debug = false;
  
  int numCoords = (int) coords.size();
  assert(numCoords > 0);
  
  if (debug) {
    char buffer[6+1];
    snprintf(buffer, 6+1, "%06X", knownFirstPixel);
    
    cout << "checking for all same pixels with coords N=" << numCoords << " and known first pixel " << buffer << endl;
  }
  
  // FIXME: 32BPP support
  
  for (auto it = coords.begin(); it != coords.end(); ++it) {
    Coord coord = *it;
    int32_t X = coord.x;
    int32_t Y = coord.y;
    
    Vec3b pixelVec = input.at<Vec3b>(Y, X);
    uint32_t pixel = (uint32_t) (Vec3BToUID(pixelVec) & 0x00FFFFFF);
    
    if (debug) {
      // Print BGRA format
      
      int32_t pixel = Vec3BToUID(pixelVec);
      char buffer[6+1];
      snprintf(buffer, 6+1, "%06X", pixel);
      
      char ibuf[3+1];
      snprintf(ibuf, 3+1, "%03d", (int)distance(coords.begin(), it));
      
      cout << "pixel i = " << ibuf << " 0xFF" << (char*)&buffer[0] << endl;
    }
    
    if (pixel != knownFirstPixel) {
      if (debug) {
        cout << "pixel differs from known first pixel" << endl;
      }
      
      return false;
    }
  }
  
  if (debug) {
    cout << "all pixels the same after processing " << numCoords << " coordinates" << endl;
  }
  
  return true;
}

// Gen a static colortable of a fixed size that contains enough colors to support
// the number of superpixels defined in the tags.

static vector<uint32_t> staticColortable;

// This table maps a superpixel tag to the offset in the staticColortable.

static unordered_map<int32_t,int32_t> staticTagToOffsetTable;

void generateStaticColortable(Mat &inputImg, SuperpixelImage &spImage)
{
  // Count number of superpixel to get max num colors
  
  int max = (int) spImage.superpixels.size();
  
  staticColortable.erase(staticColortable.begin(), staticColortable.end());
  
  for (int i = 0; i < max; i++) {
    uint32_t pixel = 0;
    pixel |= (rand() % 256);
    pixel |= ((rand() % 256) << 8);
    pixel |= ((rand() % 256) << 16);
    pixel |= (0xFF << 24);
    staticColortable.push_back(pixel);
  }
  
  // Lookup table from UID to offset into colortable
  
  staticTagToOffsetTable.erase(staticTagToOffsetTable.begin(), staticTagToOffsetTable.end());
  
  int32_t offset = 0;
  
  for (auto it = spImage.superpixels.begin(); it!=spImage.superpixels.end(); ++it) {
    int32_t tag = *it;
    staticTagToOffsetTable[tag] = offset;
    offset += 1;
  }
}

void writeTagsWithStaticColortable(SuperpixelImage &spImage, Mat &resultImg)
{
  for (auto it = spImage.superpixels.begin(); it != spImage.superpixels.end(); ++it) {
    int32_t tag = *it;
    
    Superpixel *spPtr = spImage.getSuperpixelPtr(tag);
    assert(spPtr);
    
    auto &coords = spPtr->coords;
    
    for (auto coordsIter = coords.begin(); coordsIter != coords.end(); ++coordsIter) {
      Coord coord = *coordsIter;
      int32_t X = coord.x;
      int32_t Y = coord.y;
      
      uint32_t offset = staticTagToOffsetTable[tag];
      uint32_t pixel = staticColortable[offset];
      
      Vec3b tagVec;
      tagVec[0] = pixel & 0xFF;
      tagVec[1] = (pixel >> 8) & 0xFF;
      tagVec[2] = (pixel >> 16) & 0xFF;
      resultImg.at<Vec3b>(Y, X) = tagVec;
    }
  }
}

// Write tags but use a passed in colortable to map superpixel UIDs to colors

void writeTagsWithDymanicColortable(SuperpixelImage &spImage, Mat &resultImg, unordered_map<int32_t,int32_t> map)
{
  for (auto it = spImage.superpixels.begin(); it != spImage.superpixels.end(); ++it) {
    int32_t tag = *it;
    
    Superpixel *spPtr = spImage.getSuperpixelPtr(tag);
    assert(spPtr);
    
    auto &coords = spPtr->coords;
    
    for (auto coordsIter = coords.begin(); coordsIter != coords.end(); ++coordsIter) {
      Coord coord = *coordsIter;
      int32_t X = coord.x;
      int32_t Y = coord.y;
      
      assert(map.count(tag) > 0);
      uint32_t pixel = (uint32_t) map[tag];
      
      Vec3b tagVec;
      tagVec[0] = pixel & 0xFF;
      tagVec[1] = (pixel >> 8) & 0xFF;
      tagVec[2] = (pixel >> 16) & 0xFF;
      resultImg.at<Vec3b>(Y, X) = tagVec;
    }
  }
}

// Assuming that there are N < 256 superpixels then the output can be writting as 8 bit grayscale.

void writeTagsWithGraytable(SuperpixelImage &spImage, Mat &origImg, Mat &resultImg)
{
  resultImg.create(origImg.rows, origImg.cols, CV_8UC(1));
  
  int gray = 0;
  
  // Sort superpixel by size
  
  vector<SuperpixelSortStruct> sortedSuperpixels;
  
  for (auto it = spImage.superpixels.begin(); it != spImage.superpixels.end(); ++it) {
    int32_t tag = *it;
    SuperpixelSortStruct ss;
    ss.spPtr = spImage.getSuperpixelPtr(tag);
    sortedSuperpixels.push_back(ss);
  }
  
  sort(sortedSuperpixels.begin(), sortedSuperpixels.end(), CompareSuperpixelSizeDecreasingFunc);
  
  for (auto it = sortedSuperpixels.begin(); it != sortedSuperpixels.end(); ++it) {
    SuperpixelSortStruct ss = *it;
    Superpixel *spPtr = ss.spPtr;
    assert(spPtr);
    
    //cout << "N = " << (int)spPtr->coords.size() << endl;
    
    auto &coords = spPtr->coords;
    
    for (auto coordsIter = coords.begin(); coordsIter != coords.end(); ++coordsIter) {
      Coord coord = *coordsIter;
      int32_t X = coord.x;
      int32_t Y = coord.y;
      
      resultImg.at<uint8_t>(Y, X) = gray;
    }
    
    gray++;
  }
}

// Generate gray table and the write pixels as int BGR.

void writeTagsWithMinColortable(SuperpixelImage &spImage, Mat &origImg, Mat &resultImg)
{
  resultImg.create(origImg.rows, origImg.cols, CV_8UC(3));
  
  int gray = 0;
  
  // Sort superpixel by size
  
  vector<SuperpixelSortStruct> sortedSuperpixels;
  
  for (auto it = spImage.superpixels.begin(); it != spImage.superpixels.end(); ++it) {
    int32_t tag = *it;
    SuperpixelSortStruct ss;
    ss.spPtr = spImage.getSuperpixelPtr(tag);
    sortedSuperpixels.push_back(ss);
  }
  
  sort(sortedSuperpixels.begin(), sortedSuperpixels.end(), CompareSuperpixelSizeDecreasingFunc);
  
  for (auto it = sortedSuperpixels.begin(); it != sortedSuperpixels.end(); ++it) {
    SuperpixelSortStruct ss = *it;
    Superpixel *spPtr = ss.spPtr;
    assert(spPtr);
    
    //cout << "N[" << gray << "] = " << (int)spPtr->coords.size() << endl;
    
    auto &coords = spPtr->coords;
    
    for (auto coordsIter = coords.begin(); coordsIter != coords.end(); ++coordsIter) {
      Coord coord = *coordsIter;
      int32_t X = coord.x;
      int32_t Y = coord.y;
      
      uint8_t B = gray & 0xFF;
      uint8_t G = (gray >> 8) & 0xFF;
      uint8_t R = (gray >> 16) & 0xFF;
      
      Vec3b pixelVec(B,G,R);
      
      resultImg.at<Vec3b>(Y, X) = pixelVec;
    }
    
    gray++;
  }
}




