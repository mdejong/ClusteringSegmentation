// This class represents a collection of pixels from an image that is defined such that each pixel is touching
// one of the other pixels in the superpixel.

#ifndef SUPERPIXEL_H
#define	SUPERPIXEL_H

#include "OpenCVUtil.h"
#include "Coord.h"
#include "SuperpixelEdge.h"

using namespace std;
using namespace cv;

#include <unordered_map>

// Define this to add a table of associated objects to the superpixel object.
// This can be a handy way to store data that code will access later, but
// it adds 32 or 64 bits to each Superpixel.

#define ENABLE_SUPERPIXEL_ASSOC_DATA

typedef enum {
  SuperpixelFlagsNotAllSame = (1 << 0),
  SuperpixelFlagsAllSame = (1 << 1),
} SuperpixelFlags;

class Superpixel {
  
  public:

  Superpixel();
  Superpixel(int32_t tag);
  ~Superpixel();

  int32_t tag;
  
  vector<Coord> coords;

  // This vector stores superpixel edges that have been successfully merged.
  
  vector<float> mergedEdgeWeights;
  
  // This vector stores superpixel edges that were not merged and are seen
  // as hard edges. These values could chang
  
  vector<float> unmergedEdgeWeights;
  
  // Flags that apply to all pixels in the superpixel grouping, 0 when no flags set.
  uint32_t flags;
  
  void setAllSame() {
    this->flags = SuperpixelFlagsAllSame;
  }
  
  bool isAllSame() {
    return (this->flags == SuperpixelFlagsAllSame);
  }
  
  void setNotAllSame() {
    this->flags = SuperpixelFlagsNotAllSame;
  }

  bool isNotAllSame() {
    return (this->flags == SuperpixelFlagsNotAllSame);
  }
  
#if defined(ENABLE_SUPERPIXEL_ASSOC_DATA)
  
  // A user of this Superpixel class may have specific data that should be
  // associated with a specific superpixel, this member is typically NULL
  // but can initialized as needed to point to a table that maps UID
  // integers to a generic pointer of any type.
  
  unordered_map<uint32_t, void*> *assocDataPtr;
  
  unordered_map<uint32_t, void*>& getAssocData() {
    if (assocDataPtr == NULL) {
      assocDataPtr = new unordered_map<uint32_t, void*>();
    }
    return *assocDataPtr;
  }
  
#endif // ENABLE_SUPERPIXEL_ASSOC_DATA
  
  void appendCoord(int x, int y);
  
  // Read RGB values from larger input image and create a matrix that is the width
  // of the superpixel and contains just the pixels defined by the coordinates
  // contained in the superpixel. The caller passes in the tag from the superpixel
  // in question in order to find the coords.
  
  void fillMatrixFromCoords(Mat &input, int32_t tag, Mat &output);
  
  static
  void fillMatrixFromCoords(Mat &input, vector<Coord> &coords, Mat &output);
  
  // This method is the inverse of fillMatrixFromCoords(), it reads pixel values from a matrix
  // and writes them back to the corresponding X,Y values location in an image. This method is
  // very useful when running an image operation on all the pixels in a superpixel but without
  // having to process all the pixels in a bbox area. The dimensions of the input must be
  // NUM_COORDS x 1. The caller must init the matrix values and the matrix size. This method
  // can be invoked multiple times to write multiple superpixel values to the same output
  // image.
  
  void reverseFillMatrixFromCoords(Mat &input, bool isGray, int32_t tag, Mat &output);
  
  static
  void reverseFillMatrixFromCoords(Mat &input, bool isGray, vector<Coord> &coords, Mat &output);
  
  // Filter the coords and return a vector that contains only the coordinates that share
  // an edge with the other superpixel.
  
  static
  void filterEdgeCoords(Superpixel *superpixe1Ptr,
                        vector<Coord> &edgeCoords1,
                        Superpixel *superpixe2Ptr,
                        vector<Coord> &edgeCoords2);
  
  // Get bounding box of superpixel.
  
  void bbox(int32_t &originX, int32_t &originY, int32_t &width, int32_t &height);
  
  static void splitSplayPixels(Mat &inOutTagImg);
  
  bool shouldMergeEdge(float edgeWeight);
};

// Find bounding box of a superpixel. This is the (X,Y) of the upper right corner and the width and height.

static inline
cv::Rect Superpixel_opencv_bbox(Superpixel *spPtr)
{
  int32_t originX, originY, width, height;
  spPtr->bbox(originX, originY, width, height);
  return cv::Rect(originX, originY, width, height);
}

#endif // SUPERPIXEL_H
