// This file contains utility functions for general purpose use and interaction with OpenCV.

#ifndef OPENCV_HULL_H
#define	OPENCV_HULL_H

#include <opencv2/opencv.hpp>
#include <unordered_map>

using namespace std;
using namespace cv;

#include "Coord.h"

// This utility method does the nasty job of parsing a binary shape from an input Mat
// where the non-zero pixels are treated as 0xFF. This logic is very tricky because
// of the special case where the contour pixel is right up against the left/right/top/bottom
// edge of the image. This logic must parse the shape as a contour with an extra pixel
// of padding around the binary image data to account for this possible input. Then,
// the coordinates of the resulting points are generated without considering the extra
// padding pixels. If anything goes wrong, this method will just print an error msg
// and exit.

void findContourOutline(const cv::Mat &binMat, vector<Point2i> &contour, bool simplify);

// This function scans a region and returns the hull coords split
// into convex and concave regions.

typedef struct {
  Coord defectPoint; // Set to interior defect point if concave
  vector<Coord> coords;
  bool isConcave; // Region curves in on itself at this segment
} TypedHullCoords;

vector<TypedHullCoords>
clockwiseScanOfHullCoords(
                          const Mat & tagsImg,
                          int32_t tag,
                          const vector<Coord> &regionCoords);

// This util class represents either a line segment or a curve segment. An instance of this class
// would be used in a vector that could contain either line segments or curve segments. Also,
// note that a segment best represented by a curve might be better represented by 2 lines
// if it was split into 2.

class HullLineOrCurveSegment {
public:
  bool isLine;
  
  // This points list is the raw points for a line segment. For a curve segment,
  // this list contains the control points.
  
  std::vector<cv::Point2i> points;
  
  // Slope for known line segment
  
  cv::Point2f slope;
  
  // Generic cost value
  
  int32_t cost;
  
  HullLineOrCurveSegment() : isLine(false), slope(), cost(0)
  {
  }
  
  ~HullLineOrCurveSegment()
  {
  }
  
  friend std::ostream& operator<<(std::ostream& os, const HullLineOrCurveSegment& loc) {
    os << "isLine=" << loc.isLine;
    os << ",";
    for ( auto it  = begin(loc.points); it != end(loc.points); it++) {
      os << *it << ", ";
    }
    return os;
  }
};

// This method accepts a contour that is not simplified and detects straight lines
// as compared to the non-straight curves.

vector<HullLineOrCurveSegment>
splitContourIntoLinesSegments(int32_t tag, CvSize size, CvRect roi, const vector<Coord> &contourCoords, double epsilon);

#endif // OPENCV_HULL_H
