//
//  SuperpixelEdgeFuncs.cpp
//  ClusteringSegmentation
//
//  Created by Mo DeJong on 1/7/16.
//  Copyright © 2016 helpurock. All rights reserved.
//

#include "SuperpixelEdgeFuncs.h"

// Compare method for CompareNeighborTuple type, in the case of a tie the second column
// is sorted in terms of decreasing int values.

static
bool CompareNeighborTupleFunc (CompareNeighborTuple &elem1, CompareNeighborTuple &elem2) {
  double hcmp1 = get<0>(elem1);
  double hcmp2 = get<0>(elem2);
  if (hcmp1 == hcmp2) {
    int numPixels1 = get<1>(elem1);
    int numPixels2 = get<1>(elem2);
    return (numPixels1 > numPixels2);
  }
  return (hcmp1 < hcmp2);
}

void
SuperpixelEdgeFuncs::checkNeighborEdgeWeights(SuperpixelImage &spImage,
                                              Mat &inputImg,
                                              int32_t tag,
                                              vector<int32_t> *neighborsPtr,
                                              unordered_map<SuperpixelEdge, float> &edgeStrengthMap,
                                              int step)
{
  const bool debug = false;
  
  // If any edges of this superpixel do not have an edge weight then store
  // the neighbor so that it will be considered in a call to compare
  // neighbor edge weights.
  
  vector<int32_t> neighborsVec;
  
  if (neighborsPtr == NULL) {
    neighborsVec = spImage.edgeTable.getNeighbors(tag);
    neighborsPtr = &neighborsVec;
  }
  
  bool doNeighborsEdgeCalc = false;
  vector<int32_t> neighborsThatHaveEdgeWeights;
  
  for (auto neighborIter = neighborsPtr->begin(); neighborIter != neighborsPtr->end(); ++neighborIter) {
    int32_t neighborTag = *neighborIter;
    
#if defined(DEBUG)
    Superpixel *neighborSpPtr = spImage.getSuperpixelPtr(neighborTag);
    assert(neighborSpPtr);
#endif // DEBUG
    
    SuperpixelEdge edge(tag, neighborTag);
    
    if (edgeStrengthMap.find(edge) == edgeStrengthMap.end()) {
      // Edge weight does not yet exist for this edge
      
      if (debug) {
        cout << "calculate edge weight for " << edge << endl;
      }
      
      doNeighborsEdgeCalc = true;
    } else {
      if (debug) {
        cout << "edge weight already calculated for " << edge << endl;
      }
      
      neighborsThatHaveEdgeWeights.push_back(neighborTag);
    }
  }
  
  // For each neighbor to calculate, do the computation and then save as edge weight
  
  if (doNeighborsEdgeCalc) {
    vector<CompareNeighborTuple> compareNeighborEdgesVec;
    
    unordered_map<int32_t, bool> lockedNeighbors;
    
    for (auto neighborIter = neighborsThatHaveEdgeWeights.begin(); neighborIter != neighborsThatHaveEdgeWeights.end(); ++neighborIter) {
      int32_t neighborTag = *neighborIter;
      lockedNeighbors[neighborTag] = true;
      
      if (debug) {
        cout << "edge weight search locked neighbor " << neighborTag << " since it already has an edge weight" << endl;
      }
    }
    
    unordered_map<int32_t, bool> *lockedPtr = NULL;
    
    if (lockedNeighbors.size() > 0) {
      lockedPtr = &lockedNeighbors;
    }
    
    compareNeighborEdges(spImage, inputImg, tag, compareNeighborEdgesVec, lockedPtr, step, false);
    
    // Create edge weight table entry for each neighbor that was compared
    
    for (auto it = compareNeighborEdgesVec.begin(); it != compareNeighborEdgesVec.end(); ++it) {
      CompareNeighborTuple tuple = *it;
      float edgeWeight = get<0>(tuple);
      int32_t neighborTag = get<2>(tuple);
      
      SuperpixelEdge edge(tag, neighborTag);
      
      edgeStrengthMap[edge] = edgeWeight;
      
      if (debug) {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "for edge %14s calc and saved edge weight %8.4f", edge.toString().c_str(), edgeStrengthMap[edge]);
        cout << (char*)buffer << endl;
      }
    }
  } // end if neighbors block
  
  return;
}

// This compare method will examine the pixels on an edge between a superpixel and each neighbor
// and then the neighbors will be returned in sorted order from smallest to largest in terms
// of the color diff between nearest pixels returned in a tuple of (NORMALIZE_DIST NUM_PIXELS TAG).
// Color compare operations are done in the LAB colorspace. The histogram compare looks at all
// the pixels in two superpixels, but this method is more useful when the logic wants to look
// at the values at the edges as opposed to the whole superpixel.

void
SuperpixelEdgeFuncs::compareNeighborEdges(SuperpixelImage &spImage,
                                          Mat &inputImg,
                                          int32_t tag,
                                          vector<CompareNeighborTuple> &results,
                                          unordered_map<int32_t, bool> *lockedTablePtr,
                                          int32_t step,
                                          bool normalize)
{
  const bool debug = false;
  const bool debugShowSorted = false;
  const bool debugDumpSuperpixelEdges = false;
  
  if (!results.empty()) {
    results.erase (results.begin(), results.end());
  }
  
  Superpixel *srcSpPtr = spImage.getSuperpixelPtr(tag);
  assert(srcSpPtr);
  
  for ( int32_t neighborTag : spImage.edgeTable.getNeighborsSet(tag) ) {
    if (lockedTablePtr && (lockedTablePtr->count(neighborTag) != 0)) {
      // If a locked down table is provided then do not consider a neighbor that appears
      // in the locked table.
      
      if (debug) {
        cout << "skipping consideration of locked neighbor " << neighborTag << endl;
      }
      
      continue;
    }
    
    Superpixel *neighborSpPtr = spImage.getSuperpixelPtr(neighborTag);
    assert(neighborSpPtr);
    
    if (debug) {
      cout << "compare edge between " << tag << " and " << neighborTag << endl;
    }
    
    // Get edge coordinates that are shared between src and neighbor
    
    vector<Coord> edgeCoords1;
    vector<Coord> edgeCoords2;
    
    Superpixel::filterEdgeCoords(srcSpPtr, edgeCoords1, neighborSpPtr, edgeCoords2);
    
    // Gather pixels based on the edge coords only
    
    Mat srcEdgeMat;
    
    Superpixel::fillMatrixFromCoords(inputImg, edgeCoords1, srcEdgeMat);
    
    // Note that inputImg is assumed to be in BGR colorspace here
    
    cvtColor(srcEdgeMat, srcEdgeMat, CV_BGR2Lab);
    
    Mat neighborEdgeMat;
    
    Superpixel::fillMatrixFromCoords(inputImg, edgeCoords2, neighborEdgeMat);
    
    cvtColor(neighborEdgeMat, neighborEdgeMat, CV_BGR2Lab);
    
    if (debugDumpSuperpixelEdges) {
      std::ostringstream stringStream;
      stringStream << "edge_between_" << tag << "_" << neighborTag << ".png";
      std::string str = stringStream.str();
      const char *filename = str.c_str();
      
      Mat outputMat = inputImg.clone();
      
      outputMat = Scalar(255, 0, 0);
      
      Mat srcEdgeRed = srcEdgeMat.clone();
      srcEdgeRed = Scalar(0, 0, 255);
      
      Mat neighborEdgeGreen = neighborEdgeMat.clone();
      neighborEdgeGreen = Scalar(0, 255, 0);
      
      Superpixel::reverseFillMatrixFromCoords(srcEdgeRed, false, edgeCoords1, outputMat);
      Superpixel::reverseFillMatrixFromCoords(neighborEdgeGreen, false, edgeCoords2, outputMat);
      
      cout << "write " << filename << " ( " << outputMat.cols << " x " << outputMat.rows << " )" << endl;
      imwrite(filename, outputMat);
    }
    
    // Determine smaller num coords of the two and use that as the N
    
    int numCoordsToCompare = mini((int) edgeCoords1.size(), (int) edgeCoords2.size());
    
    if (debug) {
      cout << "will compare " << numCoordsToCompare << " coords on edge" << endl;
    }
    
    assert(numCoordsToCompare >= 1);
    
    // One each iteration, select the closest coord and mark that slot as used.
    
    uint8_t neighborEdgeMatUsed[numCoordsToCompare];
    
    for (int j = 0; j < numCoordsToCompare; j++) {
      neighborEdgeMatUsed[j] = false;
    }
    
    // FIXME: use float instead of double here. The hypot(float, float) func should
    // be faster since the arith has limited range. Also possible to use an optimized
    // impl
    // https://en.wikipedia.org/wiki/Alpha_max_plus_beta_min_algorithm
    // http://www.dspguru.com/dsp/tricks/magnitude-estimator
    
    double distSum = 0.0;
    int numSum = 0;
    
    for (int i = 0; i < numCoordsToCompare; i++) {
      // FIXME: this is doing a costly pair copy, use iterator instead, same as below.
      Coord srcCoord = edgeCoords1[i];
      Vec3b srcVec = srcEdgeMat.at<Vec3b>(0, i);
      
      // Determine which is the dst coordinates is the closest to this src coord via a distance measure.
      // This should give slightly better results.
      
      double minCoordDist = (double) 0xFFFFFFFF;
      int minCoordOffset = -1;
      
      for (int j = 0; j < numCoordsToCompare; j++) {
        if (neighborEdgeMatUsed[j]) {
          // Already compared to this coord
          continue;
        }
        
        Coord neighborCoord = edgeCoords2[j];
        
        double coordDist = hypot( neighborCoord.x - srcCoord.x, neighborCoord.y - srcCoord.y );
        
        if (debug) {
          char buffer[1024];
          snprintf(buffer, sizeof(buffer), "coord dist from (%5d, %5d) to (%5d, %5d) is %12.4f",
                   srcCoord.x, srcCoord.y,
                   neighborCoord.x, neighborCoord.y,
                   coordDist);
          cout << (char*)buffer << endl;
        }
        
        if (coordDist < minCoordDist) {
          minCoordDist = coordDist;
          minCoordOffset = j;
        }
      }
      assert(minCoordOffset != -1);
      
      if (debug) {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "closest to (%5d, %5d) found as (%5d, %5d) dist is %0.4f",
                 srcCoord.x, srcCoord.y,
                 edgeCoords2[minCoordOffset].x, edgeCoords2[minCoordOffset].y, minCoordDist);
        cout << (char*)buffer << endl;
      }
      
      if (debugDumpSuperpixelEdges) {
        std::ostringstream stringStream;
        stringStream << "edge_between_" << tag << "_" << neighborTag << "_step" << i << ".png";
        std::string str = stringStream.str();
        const char *filename = str.c_str();
        
        Mat outputMat = inputImg.clone();
        
        outputMat = Scalar(255, 0, 0);
        
        Mat srcEdgeRed = srcEdgeMat.clone();
        srcEdgeRed = Scalar(0, 0, 255);
        
        Mat neighborEdgeGreen = neighborEdgeMat.clone();
        neighborEdgeGreen = Scalar(0, 255, 0);
        
        srcEdgeRed.at<Vec3b>(0, i) = Vec3b(255, 255, 255);
        neighborEdgeGreen.at<Vec3b>(0, minCoordOffset) = Vec3b(128, 128, 128);
        
        Superpixel::reverseFillMatrixFromCoords(srcEdgeRed, false, edgeCoords1, outputMat);
        Superpixel::reverseFillMatrixFromCoords(neighborEdgeGreen, false, edgeCoords2, outputMat);
        
        cout << "write " << filename << " ( " << outputMat.cols << " x " << outputMat.rows << " )" << endl;
        imwrite(filename, outputMat);
      }
      
      if (minCoordDist > 1.5) {
        // Not close enough to an available pixel to compare, just skip this src pixel
        // and use the next one without adding to the sum.
        continue;
      }
      
      Vec3b dstVec = neighborEdgeMat.at<Vec3b>(0, minCoordOffset);
      neighborEdgeMatUsed[minCoordOffset] = true;
      
      // Calc color Delta-E distance in 3D vector space
      
      double distance = delta_e_1976(srcVec[0], srcVec[1], srcVec[2],
                                     dstVec[0], dstVec[1], dstVec[2]);
      
      if (debug) {
        int32_t srcPixel = Vec3BToUID(srcVec);
        int32_t dstPixel = Vec3BToUID(dstVec);
        
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "LAB dist between pixels 0x%08X and 0x%08X = %0.12f", srcPixel, dstPixel, distance);
        cout << (char*)buffer << endl;
      }
      
      distSum += distance;
      numSum += 1;
    }
    
    assert(numSum > 0);
    
    double distAve = distSum / numSum;
    
    if (debug) {
      char buffer[1024];
      snprintf(buffer, sizeof(buffer), "Ave LAB dist %0.12f calculated from %0.12f / %d", distAve, distSum, numSum);
      cout << (char*)buffer << endl;
    }
    
    // tuple : DIST NUM_COORDS TAG
    
    CompareNeighborTuple tuple = make_tuple(distAve, neighborSpPtr->coords.size(), neighborTag);
    
    results.push_back(tuple);
  }
  
  // Normalize DIST
  
  if (normalize) {
    double maxDist = 0.0;
    
    for (auto it = results.begin(); it != results.end(); ++it) {
      CompareNeighborTuple tuple = *it;
      
      double dist = get<0>(tuple);
      
      if (dist > maxDist) {
        maxDist = dist;
      }
    }
    
    for (int i=0; i < results.size(); i++) {
      CompareNeighborTuple tuple = results[i];
      
      double normDist;
      
      if (maxDist == 0.0) {
        // Special case of only 1 edge pixel and identical RGB values
        normDist = 1.0;
      } else {
        normDist = get<0>(tuple) / maxDist;
      }
      
      CompareNeighborTuple normTuple(normDist, get<1>(tuple), get<2>(tuple));
      
      results[i] = normTuple;
    }
  }
  
  if (debug) {
    cout << "unsorted tuples from src superpixel " << tag << endl;
    
    for (auto it = results.begin(); it != results.end(); ++it) {
      CompareNeighborTuple tuple = *it;
      
      char buffer[1024];
      snprintf(buffer, sizeof(buffer), "( %12.4f %5d %5d )", get<0>(tuple), get<1>(tuple), get<2>(tuple));
      cout << (char*)buffer << endl;
    }
  }
  
  // Sort tuples by DIST value with ties in increasing order
  
  if (results.size() > 1) {
    sort(results.begin(), results.end(), CompareNeighborTupleFunc);
  }
  
  if (debug || debugShowSorted) {
    cout << "sorted tuples from src superpixel " << tag << endl;
    
    for (auto it = results.begin(); it != results.end(); ++it) {
      CompareNeighborTuple tuple = *it;
      
      char buffer[1024];
      snprintf(buffer, sizeof(buffer), "( %12.4f %5d %5d )", get<0>(tuple), get<1>(tuple), get<2>(tuple));
      cout << (char*)buffer << endl;
    }
  }
  
  return;
}
