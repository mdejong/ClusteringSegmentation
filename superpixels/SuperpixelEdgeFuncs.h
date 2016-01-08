// SuperpixelEdgeFuncs is a module that contains functions related to edges between superpixel nodes.

#ifndef SUPERPIXEL_EDGE_FUNCS_H
#define	SUPERPIXEL_EDGE_FUNCS_H

#include <vector>
#include <set>
#include <unordered_map>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#include "Util.h"
#include "Superpixel.h"
#include "SuperpixelImage.h"

class SuperpixelEdgeFuncs {
  
public:

  // This method iterates over all the neighbors of a specific superpixel in order
  // to determine the edge weights.
  
  static
  void checkNeighborEdgeWeights(SuperpixelImage &spImage,
                                Mat &inputImg,
                                int32_t tag,
                                vector<int32_t> *neighborsPtr,
                                unordered_map<SuperpixelEdge, float> &edgeStrengthMap,
                                int step);

  // Compare function that examines neighbor edges
  
  static
  void compareNeighborEdges(SuperpixelImage &spImage,
                            Mat &inputImg,
                            int32_t tag,
                            vector<CompareNeighborTuple> &results,
                            unordered_map<int32_t, bool> *lockedTablePtr,
                            int32_t step,
                            bool normalize);
  
};

#endif // SUPERPIXEL_EDGE_FUNCS_H
