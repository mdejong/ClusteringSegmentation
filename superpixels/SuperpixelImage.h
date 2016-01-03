// A superpixel image is a matrix that contains N superpixels and N superpixel edges between superpixels.
// A superpixel image is typically parsed from a source of tags, modified, and then written as a new tags
// image.

#ifndef SUPERPIXEL_IMAGE_H
#define	SUPERPIXEL_IMAGE_H

#include <vector>
#include <unordered_map>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Superpixel;
class SuperpixelEdge;

#include "Coord.h"
#include "SuperpixelEdgeTable.h"

typedef unordered_map<int32_t, Superpixel*> TagToSuperpixelMap;

typedef tuple<double, int32_t, int32_t> CompareNeighborTuple;

class SuperpixelImage {
  
  public:
  
  // This map contains the actual pointers to Superpixel objects.
  
  TagToSuperpixelMap tagToSuperpixelMap;
  
  // The superpixels list contains the UIDs for superpixels
  // in UID sorted order.
  
  vector<int32_t> superpixels;

  // The edge table contains a representation of "edges" in terms
  // of adjacent nodes lists.
  
  SuperpixelEdgeTable edgeTable;
  
  // This superpixel edge merge order list is only active in DEBUG.

#if defined(DEBUG)
  vector<SuperpixelEdge> mergeOrder;
#endif
  
  // Lookup Superpixel* given a UID

  Superpixel* getSuperpixelPtr(int32_t uid);
  
  // Return vector of all edges
  vector<SuperpixelEdge> getEdges();
  
  // Parse tags image and construct superpixels. Note that this method will modify the
  // original tag values by adding 1 to each original tag value.
  
  static
  bool parse(Mat &tags, SuperpixelImage &spImage);

  static
  bool parseSuperpixelEdges(Mat &tags, SuperpixelImage &spImage);
  
  // Merge superpixels defined by edge in this image container
  
  void mergeEdge(SuperpixelEdge &edge);
  
  // Merge superpixels where all pixels are the same pixel.
  
  void mergeIdenticalSuperpixels(Mat &inputImg);
  
  void scanLargestSuperpixels(vector<int32_t> &results);
  
  void rescanLargestSuperpixels(Mat &inputImg, Mat &outputImg, vector<int32_t> *largeSuperpixelsPtr);
  
  void fillMatrixFromCoords(Mat &input, int32_t tag, Mat &output);
  
  // This method is the inverse of fillMatrixFromCoords(), it reads pixel values from a matrix
  // and writes them back to X,Y values that correspond to the original image. This method is
  // very useful when running an image operation on all the pixels in a superpixel but without
  // having to process all the pixels in a bbox area. The dimensions of the input must be
  // NUM_COORDS x 1.
  
  void reverseFillMatrixFromCoords(Mat &input, bool isGray, int32_t tag, Mat &output);
  
  // true when all pixels in a superpixel are exactly identical
  
  bool isAllSamePixels(Mat &input, int32_t tag);
  
  // When a superpixel is known to have all identical pixel values then only the first
  // pixel in that superpixel needs to be compared to all the other pixels in a second
  // superpixel.
  
  bool isAllSamePixels(Mat &input, Superpixel *spPtr, int32_t otherTag);
  
  bool isAllSamePixels(Mat &input, uint32_t knownFirstPixel, vector<Coord> &coords);
  
  void sortSuperpixelsBySize();

};

// Util functions

void writeTagsWithStaticColortable(SuperpixelImage &spImage, Mat &resultImg);

void generateStaticColortable(Mat &inputImg, SuperpixelImage &spImage);

void writeTagsWithGraytable(SuperpixelImage &spImage, Mat &origImg, Mat &resultImg);

void writeTagsWithMinColortable(SuperpixelImage &spImage, Mat &origImg, Mat &resultImg);

#endif // SUPERPIXEL_IMAGE_H
