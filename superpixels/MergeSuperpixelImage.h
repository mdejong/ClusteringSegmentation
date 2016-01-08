// A superpixel image is a matrix that contains N superpixels and N superpixel edges between superpixels.
// A superpixel image is typically parsed from a source of tags, modified, and then written as a new tags
// image.

#ifndef SUPERPIXEL_METHODS_H
#define	SUPERPIXEL_METHODS_H

#include <opencv2/opencv.hpp>

#include <unordered_map>

using namespace std;
using namespace cv;

#include "SuperpixelImage.h"

typedef enum {
  BACKPROJECT_HIGH_FIVE, // top 95% with gray = 200
  BACKPROJECT_HIGH_FIVE8, // top 95% with gray = 200 (8 bins per channel)
  BACKPROJECT_HIGH_TEN,  // top 90% with gray 200
  BACKPROJECT_HIGH_15,  // top 85% with gray 200
  BACKPROJECT_HIGH_20,  // top 80% with gray 200
  BACKPROJECT_HIGH_50,  // top 80% with gray 200
} BackprojectRange;

class MergeSuperpixelImage : public SuperpixelImage {
  
  public:

  // Repeated merge of the largest superpixels up until the
  // easily merged superpixels have been merged.
  
  void mergeAlikeSuperpixels(Mat &inputImg);
  
  int mergeBackprojectSuperpixels(Mat &inputImg, int colorspace, int startStep, BackprojectRange range);
  
  int mergeBackprojectSmallestSuperpixels(Mat &inputImg, int colorspace, int startStep, BackprojectRange range);

  int fillMergeBackprojectSuperpixels(Mat &inputImg, int colorspace, int startStep);

  // Merge small superpixels away from the largest neighbor.
  
  int mergeSmallSuperpixels(Mat &inputImg, int colorspace, int startStep);
  
  // Merge superpixels detected as "edges" away from the largest neighbor.

  int mergeEdgySuperpixels(Mat &inputImg, int colorspace, int startStep, vector<int32_t> *largeSuperpixelsPtr);
  
  void scanLargestSuperpixels(vector<int32_t> &results);
  
  void rescanLargestSuperpixels(Mat &inputImg, Mat &outputImg, vector<int32_t> *largeSuperpixelsPtr);
  
  // Compare function that does histogram compare for each neighbor of superpixel tag
  
  void compareNeighborSuperpixels(Mat &inputImg,
                                  int32_t tag,
                                  vector<CompareNeighborTuple> &results,
                                  unordered_map<int32_t, bool> *lockedTablePtr,
                                  int32_t step);
    
  // Evaluate backprojection of superpixel to the connected neighbors

  void backprojectNeighborSuperpixels(Mat &inputImg,
                                      int32_t tag,
                                      vector<CompareNeighborTuple> &results,
                                      unordered_map<int32_t, bool> *lockedTablePtr,
                                      int32_t step,
                                      int conversion,
                                      int numPercentRanges,
                                      int numTopPercent,
                                      bool roundPercent,
                                      int minGraylevel,
                                      int numBins);
  
  void backprojectDepthFirstRecurseIntoNeighbors(Mat &inputImg,
                                                 int32_t tag,
                                                 vector<int32_t> &results,
                                                 unordered_map<int32_t, bool> *lockedTablePtr,
                                                 int32_t step,
                                                 int conversion,
                                                 int numPercentRanges,
                                                 int numTopPercent,
                                                 int minGraylevel,
                                                 int numBins);
  
  // Recursive bredth first search to fully expand the largest superpixel in a BFS order
  // and then lock the superpixel before expanding in terms of smaller superpixels. This
  // logic looks for possible expansion using back projection but it keeps track of
  // edge weights so that an edge will not be collapsed when it has a very high weight
  // as compared to the other edge weights for this specific superpixel.
  
  int mergeBredthFirstRecursive(Mat &inputImg, int colorspace, int startStep, vector<int32_t> *largeSuperpixelsPtr, int numBins);
  
  void filterOutVeryLargeNeighbors(int32_t tag, vector<int32_t> &neighbors);
  
  bool shouldMergeEdge(int32_t tag, float edgeWeight);
  
  void addUnmergedEdgeWeights(int32_t tag, vector<float> &edgeWeights);
  
  void addMergedEdgeWeight(int32_t tag, float edgeWeight);
    
  void recurseTouchingSuperpixels(int32_t rootUID,
                                  int32_t rootValue,
                                  unordered_map<int32_t, int32_t> &touchingTable);

};

#endif // SUPERPIXEL_METHODS_H
