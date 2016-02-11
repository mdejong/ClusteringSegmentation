// This file contains utility functions for general purpose use and interaction with OpenCV

#include "OpenCVHull.hpp"

#include "OpenCVUtil.h"

#include "Superpixel.h"

#include "OpenCVIter.hpp"

#include "Util.h"

#define HULL_DUMP_IMAGE_PREFIX "srm_tag_"

// Get a range of contour values given a starting point and and ending point.
// Note that in the case where the ending point wraps around are treated as
// a continuation.

static
void appendCoordsInRangeOnContour(int startIdx,
                                  int endIdx,
                                  const vector<Point2i> &contour,
                                  vector<Coord> &outCoords)
{
  int maxOffset;
  
  bool wrapsAround = false;
  
  if (endIdx < startIdx) {
    if (endIdx == 0) {
      wrapsAround = false;
    } else {
      wrapsAround = true;
    }
    
    maxOffset = (int)contour.size();
  } else {
    maxOffset = endIdx;
  }
  
  for ( int contouri = startIdx; contouri < maxOffset; contouri++ ) {
    Point2i p = contour[contouri];
    Coord c(p.x, p.y);
    outCoords.push_back(c);
  }
  
  if (wrapsAround) {
    maxOffset = endIdx;
    
    for ( int contouri = 0; contouri < maxOffset; contouri++ ) {
      Point2i p = contour[contouri];
      Coord c(p.x, p.y);
      outCoords.push_back(c);
    }
  }
  
  return;
}

// This utility method does the nasty job of parsing a binary shape from an input Mat
// where the non-zero pixels are treated as 0xFF. This logic is very tricky because
// of the special case where the contour pixel is right up against the left/right/top/bottom
// edge of the image. This logic must parse the shape as a contour with an extra pixel
// of padding around the binary image data to account for this possible input. Then,
// the coordinates of the resulting points are generated without considering the extra
// padding pixels. If anything goes wrong, this method will just print an error msg
// and exit.

void findContourOutline(const cv::Mat &binMat, vector<Point2i> &contour, bool simplify) {
  const bool debug = true;
  const bool debugDumpImages = true;
  
  if (debug) {
    cout << "findContourOutline" << endl;
  }
  
  assert(contour.size() == 0);
  
  vector<vector<Point2i> > contours;
  vector<Vec4i> hierarchy;
  
  if (debugDumpImages) {
    writeWroteImg("find_contour_input.png", binMat);
  }
  
  // FIXME: detect ROI+1 and use that rect as the ROI size for new detection Mat
  
  // The findContours() method returns funny results when an on pixel is on one
  // of the edges of the Mat. So, allocate a Mat that has one additional pixel
  // of spacing and then copy the existing pixels into the larger mat via a ROI
  // copy.
  
  Mat largerMat(binMat.rows + 2, binMat.cols + 2, CV_8UC1, Scalar(0));
  assert(largerMat.cols == binMat.cols+2);
  assert(largerMat.rows == binMat.rows+2);
  
  Rect borderedROI(1, 1, binMat.cols, binMat.rows);
  
  Mat largerROIMat = largerMat(borderedROI);
  
  assert(largerROIMat.size() == binMat.size());
  
  binMat.copyTo(largerROIMat);
  
  if (debugDumpImages) {
    writeWroteImg("find_contour_uncropped_input.png", largerMat);
  }
  
  // Extract contour from Mat with known 1 pixel padding on all sides w no simplification

  int flags = CV_CHAIN_APPROX_NONE;
  
  if (simplify) {
//    flags = CV_CHAIN_APPROX_SIMPLE;
    
//    flags = CV_CHAIN_APPROX_TC89_L1;
    
    flags = CV_CHAIN_APPROX_TC89_KCOS;
  }
  
  findContours( largerMat, contours, hierarchy, CV_RETR_LIST, flags );
  
  // Contour detection logic will only be successful when 1 single connected regions is detected.
  // Bomb out if the code detects more than 1 region.
  
  if (contours.size() == 0) {
    // Did not detect any contour regions at all, could be all black input pixels.
    
    fprintf(stderr, "error: no contour could be parsed from in input image\n");
    //exit(1);
    assert(0);
  }
  
  if (contours.size() > 1) {
    // Emit contours as N outlines
    
    int ci = 0;
    for ( auto contour : contours ) {
      Mat outMat = largerMat.clone();
      
      drawContours( outMat, contours, ci, Scalar(0xFF), 1, 4, hierarchy );
      
      std::stringstream fnameStream;
      fnameStream << "contour_failed" << ci << ".png";
      string fname = fnameStream.str();
      
      writeWroteImg(fname, outMat);
      
      ci++;;
    }
    
    // Multiple contour regions cannot be supported since only 1 single closed shape is
    // supported as the input. Emit a visual representation of the input image with
    // a bbox around each detected region to make it clear why the input failed.
    
    Mat imageWithBbox(largerMat.size(), CV_8UC3);
    
    imageWithBbox = Scalar(0, 0, 0);
    
    for_each_bgr_const_byte(imageWithBbox, largerMat, [](uint8_t B, uint8_t G, uint8_t R, uint8_t bVal)->Vec3b {
      const Vec3b white3b(0xFF, 0xFF, 0xFF);
      const Vec3b black3b(0, 0, 0);
      
      if (bVal != 0) {
        return white3b;
      } else {
        return black3b;
      }
    });
    
    // Now draw detected contour bbox over input image to make it clear when the
    // N > 1 detected regions are.
    
    for( int i = 0; i< contours.size(); i++ ) {
      Rect rect = boundingRect(contours[i]); // Find the bounding rectangle for contour
      rectangle(imageWithBbox, rect, CV_RGB(255-(i*8),0,0));
    }
    
    writeWroteImg("contour_failed_n_bbox.png", imageWithBbox);
    
    fprintf(stderr, "error: found %d distinct contour regions in input image\n", (int)contours.size());
//    exit(1);
    assert(0);
  }
  
  // Copy contour points into ref passed in from caller scope
  
  contour = contours[0];
  
  if (debugDumpImages) {
    // Zero out largerMat and render the contour outline
    
    largerMat = Scalar(0);
    
    int thickness = 1;
    
    Scalar color = Scalar(0xFF);
    
    //int lineType = 4; // 4 connected
    int lineType = 8; // 8 connected
    //int lineType = CV_AA; // antialiased line
    
    drawContours( largerMat, contours, 0, color, thickness, lineType );
    
    writeWroteImg("find_contour_output.png", largerMat);
  }
  
  // The ROI region is defined such that (1,1) is actually (0,0) from the
  // original image.
  
  // At this point the contour is a vector of points, but (1,1) must be subtracted
  // from each point to account for the ROI region.
  
  const Point2i offset11(1, 1);
  
  if (debug) {
    cout << "points before offset adjust:" << endl;
    
    for( int i = 0; i < contour.size(); i++ ) {
      Point2i p = contour[i];
      cout << p << ", ";
    }
    cout << endl;
  }
  
  int contourMax = (int)contour.size();
  for( int i = 0; i < contourMax; i++ ) {
    Point2i p = contour[i];
#if defined(DEBUG)
    assert(p.x > 0);
    assert(p.y > 0);
#endif // DEBUG
    p = p - offset11;
    contour[i] = p;
  }
  
  if (debug) {
    cout << "points after offset adjust:" << endl;
    
    for( int i = 0; i < contour.size(); i++ ) {
      Point2i p = contour[i];
      cout << p << ", ";
    }
    cout << endl;
  }
  
  if (debugDumpImages) {
    // Rerender contour into original sized Mat after crop
    
    Mat croppedBinMat = binMat.clone();
    croppedBinMat = Scalar(0);
    
    contours.clear();
    contours.push_back(contour);
    
    int thickness = 1;
    
    Scalar color = Scalar(0xFF);
    
    //int lineType = 4; // 4 connected
    int lineType = 8; // 8 connected
    //int lineType = CV_AA; // antialiased line
    
    drawContours( croppedBinMat, contours, 0, color, thickness, lineType );
    
    writeWroteImg("find_contour_output_cropped.png", croppedBinMat);
  }
    
  if (debug) {
    cout << "findContourOutline returning " << contour.size() << " points" << endl;
  }
  
  return;
}

// Given a set of coordinates that make up all the points of a region, calculate
// a contour region and return the contour coordinates split up into convex vs
// concave parts.

vector<TypedHullCoords>
clockwiseScanOfHullCoords(
                          const Mat & tagsImg,
                          int32_t tag,
                          const vector<Coord> &regionCoords)
{
  const bool debug = true;
  const bool debugDumpImages = true;
  
  if (debug) {
    cout << "clockwiseScanOfHullCoords " << tag << endl;
  }
  
  // If the shape is convex then wrap it in a convex hull and simplify the shape with
  // smallish straight lines so that perpendicular lines can be computed as compared
  // to each line segment in order to find the shape normals.
  
  Mat binMat(tagsImg.size(), CV_8UC1, Scalar(0));
  
  for ( Coord c : regionCoords ) {
    binMat.at<uint8_t>(c.y, c.x) = 0xFF;
  }
  
  if (debugDumpImages) {
    std::stringstream fnameStream;
    fnameStream << HULL_DUMP_IMAGE_PREFIX << tag << "_contour_detect" << ".png";
    string fname = fnameStream.str();
    
    imwrite(fname, binMat);
    cout << "wrote " << fname << endl;
    cout << "" << endl;
  }
  
  // Convert point into a simplified contour of straight lines
  
  // CHAIN_APPROX_NONE or CV_CHAIN_APPROX_SIMPLE
  
  // Note that in cases of a tight angle, a certain coord can be
  // repeated in the generated contour.
  
  //findContours(binMat, contours, CV_RETR_LIST, CHAIN_APPROX_NONE);
  //  assert(contours.size() >= 1);
  //  vector<Point> contour = contours[0];
  
  vector<Point> contour;
  findContourOutline(binMat, contour, false);
  
  assert(contour.size() > 0);
  
  // Invert the default counter clockwise contour orientation
  
  vector<Point2i> tmp = contour;
  contour.clear();
  for ( auto it = tmp.rbegin(); it != tmp.rend(); ++it ) {
    Point2i p = *it;
    contour.push_back(p);
  }
  tmp.clear();
  
  if (debug) {
    int i = 0;
    for ( Point2i p : contour ) {
      cout << "contour[" << i << "] = " << "(" << p.x << "," << p.y << ")" << endl;
      i++;
    }
  }
  
  if (debug) {
    cout << "contour as Point2i" << endl;
    
    cout << "\tPoint2i contour[" << contour.size() << "];" << endl;
    
    int i = 0;
    for ( Point2i p : contour ) {
      cout << "\tcontour[" << i << "] = " << "Point2i(" << p.x << "," << p.y << ");" << endl;
      i++;
    }
  }
 
  return clockwiseScanOfHullContour(tagsImg.size(), tag, contour);
}

// Given a contour that is already parsed into a clockwise set points, split
// the split up into convex vs
// concave parts.

vector<TypedHullCoords>
clockwiseScanOfHullContour(CvSize size,
                           int32_t tag,
                           const vector<Point2i> &contour)
{
  const bool debug = true;
  const bool debugDumpImages = true;

  vector<vector<Point> > contours;
  
  Mat binMat(size, CV_8UC1, Scalar(0));
  
  // Render as contour
  
  if (debugDumpImages) {
    binMat = Scalar(0);
    
    drawContours(binMat, contours, 0, Scalar(0xFF));
    
    std::stringstream fnameStream;
    fnameStream << HULL_DUMP_IMAGE_PREFIX << tag << "_contour" << ".png";
    string fname = fnameStream.str();
    
    imwrite(fname, binMat);
    cout << "wrote " << fname << endl;
    cout << "" << endl;
  }
  
  // Dump contour points rendered with incresing grayscale values in the range 0xFF/4 -> 0xFF
  // so that the ordering is clear.
  
  if (debugDumpImages) {
    binMat = Scalar(0);
    
    int iMax = (int) contour.size();
    uint8_t grayLevel[iMax];
    
    const int minLevel = 0x7F;
    int delta = (0xFF - minLevel);
    
    for ( int i = iMax-1; i >= 0; i-- ) {
      int offsetFromZero = (iMax - 1) - i;
      float percent = float(offsetFromZero) / float(iMax);
      printf("offsetFromZero %d -> %0.4f\n", offsetFromZero, percent);
      
      int numToSubtract = round(delta * percent);
      
      grayLevel[i] = 0xFF - numToSubtract;
      
      printf("grayLevel[%5d] = %d\n", i, grayLevel[i]);
    }
    
    int i = 0;
    for ( Point2i p : contour ) {
      uint8_t gray = grayLevel[i];
      
      printf("i is %d : grey is %d\n", i, gray);
      
      binMat.at<uint8_t>(p.y, p.x) = gray;
      
      i++;
    }
    
    std::stringstream fnameStream;
    fnameStream << HULL_DUMP_IMAGE_PREFIX << tag << "_contour_order" << ".png";
    string fname = fnameStream.str();
    
    imwrite(fname, binMat);
    cout << "wrote " << fname << endl;
    cout << "" << endl;
  }
  
  // hull around contour
  
  Mat contourMat(contour);
  
  vector<int> hull;
  
  convexHull(contourMat, hull, false, false);
  
  int hullCount = (int)hull.size();
  
  if (debugDumpImages) {
    binMat = Scalar(0);
    
    vector<Point2i> hullPoints;
    
    for ( int offset : hull ) {
      Point2i p = contour[offset];
      hullPoints.push_back(p);
    }
    
    vector<vector<Point2i> > contours;
    
    contours.push_back(hullPoints);
    
    drawContours(binMat, contours, 0, Scalar(0xFF), 1, 8 );
    
    std::stringstream fnameStream;
    fnameStream << HULL_DUMP_IMAGE_PREFIX << tag << "_hull_lines" << ".png";
    string fname = fnameStream.str();
    
    writeWroteImg(fname, binMat);
    cout << "" << endl;
  }
  
  // Generate hull defects to indicate where shape becomes convex
  
  vector<Vec4i> defectVec;
  
  assert(hull.size() > 2);
  assert(contour.size() > 3);
  
  convexityDefects(contour, hull, defectVec);
  
  binMat = Scalar(0);
  
  Mat colorMat;
  
  if (debugDumpImages) {
    colorMat = Mat(size, CV_8UC3);
  }
  
  if (debugDumpImages) {
    colorMat = Scalar(0, 0, 0);
    drawContours(colorMat, contours, 0, Scalar(0xFF,0xFF,0xFF), CV_FILLED); // Draw contour as white filled region
  }
  
  for (int cDefIt = 0; cDefIt < defectVec.size(); cDefIt++) {
    int startIdx = defectVec[cDefIt].val[0];
    int endIdx = defectVec[cDefIt].val[1];
    int defectPtIdx = defectVec[cDefIt].val[2];
    double depth = (double)defectVec[cDefIt].val[3]/256.0f;  // see documentation link below why this
    
    Point2i startP = contour[startIdx];
    Point2i endP = contour[endIdx];
    Point2i defectP = contour[defectPtIdx];
    
    if (debug) {
    printf("start  %8d = (%4d,%4d)\n", startIdx, startP.x, startP.y);
    printf("end    %8d = (%4d,%4d)\n", endIdx, endP.x, endP.y);
    printf("defect %8d = (%4d,%4d)\n", defectPtIdx, defectP.x, defectP.y);
    printf("depth  %0.3f\n", depth);
    }
    
    if (debugDumpImages) {
      line(binMat, startP, defectP, Scalar(255), 1, 0);
      line(binMat, endP, defectP, Scalar(128), 1, 0);
    }
    
    if (debugDumpImages) {
      line(colorMat, startP, endP, Scalar(0xFF,0,0), 1, 0);
      circle(colorMat, defectP, 4, Scalar(0,0,0xFF), 2);
    }
  }
  
  if (debugDumpImages) {
    std::stringstream fnameStream;
    fnameStream << HULL_DUMP_IMAGE_PREFIX << tag << "_hull_defectpoints" << ".png";
    string fname = fnameStream.str();
    
    imwrite(fname, binMat);
    cout << "wrote " << fname << endl;
    cout << "" << endl;
  }
  
  if (debugDumpImages) {
    std::stringstream fnameStream;
    fnameStream << HULL_DUMP_IMAGE_PREFIX << tag << "_hull_defect_render" << ".png";
    string fname = fnameStream.str();
    
    imwrite(fname, colorMat);
    cout << "wrote " << fname << endl;
    cout << "" << endl;
  }
  
  // Filter the defect results obtained above to remove the detected defects that are
  // actually not defects but are overdetected when the line slope changes only a tiny
  // bit.
  
  // Map offset into contour to defect offset
  
  unordered_map<int, int> defectStartOffsetMap;
  
  // Map the contour offset to the 3 points that make up the defect
  // so that the defect triangle can be rendered.
  
  unordered_map<int, vector<Coord>> defectStartOffsetToTripleMap;
  
  if (debugDumpImages) {
    colorMat = Scalar(0, 0, 0);
    drawContours(colorMat, contours, 0, Scalar(0xFF,0xFF,0xFF), CV_FILLED); // Draw contour as white filled region
  }
  
  if (debugDumpImages) {
    binMat = Scalar(0);
  }
  
  Mat colorMat2;
  
  if (debugDumpImages) {
    colorMat2 = Mat(size, CV_8UC3);
  }
  
  for (int cDefIt = 0; cDefIt < defectVec.size(); cDefIt++) {
    int startIdx = defectVec[cDefIt].val[0];
    int endIdx = defectVec[cDefIt].val[1];
    int defectPtIdx = defectVec[cDefIt].val[2];
    double depth = (double)defectVec[cDefIt].val[3]/256.0f;  // see documentation link below why this
    
    Point2i startP = contour[startIdx];
    Point2i endP = contour[endIdx];
    Point2i defectP = contour[defectPtIdx];
    
    if (debug) {
    printf("start  %8d = (%4d,%4d)\n", startIdx, startP.x, startP.y);
    printf("end    %8d = (%4d,%4d)\n", endIdx, endP.x, endP.y);
    printf("defect %8d = (%4d,%4d)\n", defectPtIdx, defectP.x, defectP.y);
    printf("depth  %0.3f\n", depth);
    }
    
    // The initial filtering checks abs(dx,dy) for the delta between
    // startP and defectP. If these points are within 2 pixels of each
    // other then short circult a "closeness" test since it is clear
    // that the points are very near each other.
    
    Point2i startToDefectDelta = startP - defectP;
    Point2i endToDefectDelta = endP - defectP;
    
    Point2i startToEndDelta = endP - startP;
    startToEndDelta /= 2;
    
    Point2i midP = startP + startToEndDelta;
    Point2i midToDefectDelta = midP - defectP;
    
    if (debugDumpImages) {
      colorMat2 = Scalar(0,0,0);
      
      line(colorMat2, startP, midP, Scalar(0xFF,0,0), 1, 0); // blue
      line(colorMat2, midP, endP, Scalar(0,0xFF,0), 1, 0); // green
      
      line(colorMat2, midP, defectP, Scalar(0,0,0xFF), 1, 0);
      
      std::stringstream fnameStream;
      fnameStream << HULL_DUMP_IMAGE_PREFIX << tag << "_hull_defect_" << cDefIt << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, colorMat2);
      cout << "wrote " << fname << endl;
      cout << "" << endl;
    }
    
    if (debug) {
      printf("start -> defect  (%4d,%4d)\n", startToDefectDelta.x, startToDefectDelta.y);
      printf("end   -> defect  (%4d,%4d)\n", endToDefectDelta.x, endToDefectDelta.y);
      printf("mid   -> defect  (%4d,%4d)\n", midToDefectDelta.x, midToDefectDelta.y);
    }
    
    const int minDistanceCutoff = 2;
    
    // Fast test to determine if a delta vector is small
    
    auto isCloseFunc = [](Point2i delta)->bool {
      if ((abs(delta.x) <= minDistanceCutoff) && (abs(delta.y) <= minDistanceCutoff)) {
        return true;
      } else {
        return false;
      }
    };
    
    bool startToDefectDeltaClose = isCloseFunc(startToDefectDelta);
    bool endToDefectDeltaClose   = isCloseFunc(endToDefectDelta);
    bool midToDefectDeltaClose   = isCloseFunc(midToDefectDelta);
    
    if (startToDefectDeltaClose) {
      continue;
    }
    if (endToDefectDeltaClose) {
      continue;
    }
    if (midToDefectDeltaClose) {
      continue;
    }
    
    // FIXME: fast integer distance could filter out any points where
    // start{ and defectP and within +-2 or is endP and defectP are close.
    
    // Determine distance in pixels from the defect point to the midpoint of the
    // (startP, endP) hull line.
    
    Point2f startF(startP.x, startP.y);
    Point2f endF(endP.x, endP.y);
    
    Point2f midF = endF - startF;
    midF = startF + (midF * 0.5);
    
    Point2f defectF(defectP.x, defectP.y);
    
    Point2f defectToMidF = midF - defectF;
    
    //int rX = round(defectToMidF.x);
    //int rY = round(defectToMidF.y);
    float rX = defectToMidF.x;
    float rY = defectToMidF.y;
    float defectDelta = sqrt(rX*rX + rY*rY);
    
    if (debug) {
      printf("defectDelta %0.3f\n", defectDelta);
    }
    
    if (debugDumpImages) {
      Point2i midP(round(midF.x), round(midF.y));
      line(binMat, defectP, midP, Scalar(0xFF), 1);
    }
    
    // Get angle between midpoint and defect point
    
    auto isSmallAngleFunc = [](Point2f v1, Point2f v2, int32_t &degrees)->bool {
      float radAngleBetween = angleBetween(v1, v2);
      degrees = radAngleBetween * 180.0f / M_PI;
      
      const int plusMinus = 80;
      const int degreeLow = (90-plusMinus);
      const int degreeHigh = (90+plusMinus);
      
      if (degrees < degreeLow || degrees > degreeHigh) {
        return true;
      } else {
        return false;
      }
    };
    
    Point2f midToDefect = defectF - midF;
    Point2f midToStart = startF - midF;
    Point2f midToEnd = endF - midF;
    
    int32_t angleBetweenStartAndDefectDegrees;
    int32_t angleBetweenEndAndDefectDegrees = -1;
    
    bool isAngleBetweenStartAndDefectSmall = isSmallAngleFunc(midToDefect, midToStart, angleBetweenStartAndDefectDegrees);
    bool isAngleBetweenEndAndDefectSmall = false;
    
    if (!isAngleBetweenStartAndDefectSmall) {
      isAngleBetweenEndAndDefectSmall = isSmallAngleFunc(midToDefect, midToEnd, angleBetweenEndAndDefectDegrees);
    }
    
    if (debug) {
      printf("midToDefect (%0.3f, %0.3f)\n", midToDefect.x, midToDefect.y );
      printf("midToStart (%0.3f, %0.3f)\n", midToStart.x, midToStart.y );
      printf("midToEnd (%0.3f, %0.3f)\n", midToEnd.x, midToEnd.y );
      
      printf("isAngleBetweenStartAndDefectSmall %s : angleBetweenStartAndDefectDegrees %3d \n", (isAngleBetweenStartAndDefectSmall ? "true " : "false"), angleBetweenStartAndDefectDegrees );
      
      printf("isAngleBetweenEndAndDefectSmall   %s : angleBetweenStartAndDefectDegrees %3d \n", (isAngleBetweenEndAndDefectSmall ? "true " : "false"), angleBetweenEndAndDefectDegrees );
      printf("\n");
    }
    
    if (debugDumpImages) {
      colorMat2 = Scalar(0, 0, 0);
      
      Point2i midP(round(midF.x), round(midF.y));
      
      line(colorMat2, startP, midP, Scalar(0xFF,0,0), 1, 0); // blue
      line(colorMat2, midP, endP, Scalar(0,0xFF,0), 1, 0); // green
      
      line(colorMat2, midP, defectP, Scalar(0,0,0xFF), 1, 0);
      
      
      std::stringstream fnameStream;
      fnameStream << HULL_DUMP_IMAGE_PREFIX << tag << "_hull_defect_" << cDefIt << "_angle_" << angleBetweenStartAndDefectDegrees << "_and_" << angleBetweenEndAndDefectDegrees << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, colorMat2);
      cout << "wrote " << fname << endl;
      cout << "" << endl;
    }
    
    // If the angle between the line from (startP, midP) and (midP, defectP) is very
    // small then consider this hull segment as convex.
    
    // FIXME: if the number of pixels inside the contour from (start, defect, end)
    // polygon pixel inside the region is not small, then consider this region to
    // be a reasonable size. But the size should be related to the size of the shape
    // of the original shape.
    
    // FIXME: depth must depend on relative size of contour
    
    float minDefectDepth = 2.0f; // Defect must be more than just a little bit
    
    bool isVerySmallAngle;
    
    if (isAngleBetweenStartAndDefectSmall || isAngleBetweenEndAndDefectSmall) {
      if (debug) {
        printf("SKIP very small angle in degrees : %d and %d\n", angleBetweenStartAndDefectDegrees, angleBetweenEndAndDefectDegrees);
      }
      
      isVerySmallAngle = true;
    } else {
      isVerySmallAngle = false;
    }
    
    bool isHullConcaveDefect;
    
    if (defectDelta <= minDefectDepth) {
      isHullConcaveDefect = false;
    } else if (isVerySmallAngle) {
      isHullConcaveDefect = false;
    } else {
      isHullConcaveDefect = true;
    }
    
    // In the case where the detected angle is not super small, check the number of pixels that would be contained in
    // the convex region that are not exactly on the hull line. This checks for odd cases like 15 degrees, but where
    // the rounding to pixels means that just a few pixels are not actually on the hull line.
    
    if (isHullConcaveDefect && !isVerySmallAngle) {
      vector<Coord> hullAndDefectCoords;
      
      Coord startC(startP.x, startP.y);
      Coord endC(endP.x, endP.y);
      Coord defectC(defectP.x, defectP.y);
      
      hullAndDefectCoords.push_back(startC);
      hullAndDefectCoords.push_back(defectC);
      hullAndDefectCoords.push_back(endC);
      
      Rect roiRect = bboxPlusN(hullAndDefectCoords, size, 1);
      
      Mat roiMat(roiRect.size(), CV_8UC1);
      
      roiMat = Scalar(0);
      
      // Capture points on line between start and end
      
      Point2i originP(roiRect.x, roiRect.y);
      
      line(roiMat, startP - originP, endP - originP, Scalar(0xFF));
      
      if (debugDumpImages) {
        std::stringstream fnameStream;
        fnameStream << HULL_DUMP_IMAGE_PREFIX << tag << "_hull_defect_" << cDefIt << "_only_line" << ".png";
        string fname = fnameStream.str();
        
        imwrite(fname, roiMat);
        cout << "wrote " << fname << endl;
        cout << "" << endl;
      }
      
      vector<Point2i> hullLinePoints;
      findNonZero(roiMat, hullLinePoints);
      unordered_map<Coord, bool> lineLookupTable;
      
      for ( Point2i p : hullLinePoints ) {
        int x = p.x;
        int y = p.y;
        Coord c(x, y);
        lineLookupTable[c] = true;
      }
      
      // Capture filled contour points in the region
      
      roiMat = Scalar(0);
      
      vector<Point2i> contour;
      
      // (start, defect, end)
      
      for ( Coord c : hullAndDefectCoords ) {
        Point2i p(c.x, c.y);
        Point2i roiP = p - originP;
        contour.push_back(roiP);
      }
      
      vector<vector<Point2i> > contours;
      contours.push_back(contour);
      
      drawContours(roiMat, contours, 0, Scalar(0xFF), CV_FILLED, 8);
      
      vector<Point2i> filledContourPoints;
      findNonZero(roiMat, filledContourPoints);
      
      assert(filledContourPoints.size() > 0);
      
      vector<Coord> nonHullLinePoints;
      
      for ( Point2i p : filledContourPoints ) {
        int x = p.x;
        int y = p.y;
        Coord c(x, y);
        
        if (lineLookupTable.count(c) == 0) {
          // Add coord if it was not on the line
          nonHullLinePoints.push_back(c);
        }
      }
      
      // Render points that are not on the line
      
      if (debugDumpImages) {
        binMat = Scalar(0);
        
        Coord originC(roiRect.x, roiRect.y);
        
        for ( Coord c : nonHullLinePoints ) {
          Coord gC = originC + c;
          binMat.at<uint8_t>(gC.y, gC.x) = 0xFF;
        }
        
        std::stringstream fnameStream;
        fnameStream << HULL_DUMP_IMAGE_PREFIX << tag << "_hull_defect_" << cDefIt << "_not_on_line" << ".png";
        string fname = fnameStream.str();
        
        imwrite(fname, binMat);
        cout << "wrote " << fname << endl;
        cout << "" << endl;
      }
      
      if (debugDumpImages) {
        colorMat = Scalar(0, 0, 0);
        
        Vec3b greenVec(0x0, 0xFF, 0);
        
        for ( Point2i p : hullLinePoints ) {
          Coord gC(originP.x + p.x, originP.y + p.y);
          colorMat.at<Vec3b>(gC.y, gC.x) = greenVec;
        }
        
        Vec3b whiteVec(0xFF, 0xFF, 0xFF);
        
        for ( Coord c : nonHullLinePoints ) {
          Coord gC(originP.x + c.x, originP.y + c.y);
          colorMat.at<Vec3b>(gC.y, gC.x) = whiteVec;
        }
        
        std::stringstream fnameStream;
        fnameStream << HULL_DUMP_IMAGE_PREFIX << tag << "_hull_defect_" << cDefIt << "_combined_line_defect" << ".png";
        string fname = fnameStream.str();
        
        imwrite(fname, colorMat);
        cout << "wrote " << fname << endl;
        cout << "" << endl;
      }
      
      // Ifthere are more points on the line that points that are not on the
      // line then skip this region.
      
      if (debug) {
        printf("num line points %d : num region non-line points %d\n", (int) lineLookupTable.size(), (int) nonHullLinePoints.size());
      }
      
      if (nonHullLinePoints.size() <= lineLookupTable.size()) {
        if (debug) {
          printf("setting isHullConcaveDefect to false since very few coords in region\n");
        }
        
        isHullConcaveDefect = false;
      }
    }
    
    if (isHullConcaveDefect) {
      if (debug) {
        printf("KEEP defectDelta  %0.3f\n", defectDelta);
      }
      
      assert(defectStartOffsetMap.count(startIdx) == 0);
      defectStartOffsetMap[startIdx] = cDefIt;
      
      vector<Coord> triple;
      triple.push_back(Coord(startP.x, startP.y));
      triple.push_back(Coord(endP.x, endP.y));
      triple.push_back(Coord(defectP.x, defectP.y));
      
      defectStartOffsetToTripleMap[startIdx] = triple;
    } else {
      if (debug) {
        printf("SKIP defectDelta  %0.3f\n", defectDelta);
      }
    }
    
    if (debug) {
      printf("\n");
    }
  }
  
  // Render contours points by looking up offsets in defectStartOffsetMap
  
  if (debugDumpImages) {
    for (int cDefIt = 0; cDefIt < defectVec.size(); cDefIt++) {
      int startIdx = defectVec[cDefIt].val[0];
      int endIdx = defectVec[cDefIt].val[1];
      int defectPtIdx = defectVec[cDefIt].val[2];
      double depth = (double)defectVec[cDefIt].val[3]/256.0f;  // see documentation link below why this
      
      Point2i startP = contour[startIdx];
      Point2i endP = contour[endIdx];
      Point2i defectP = contour[defectPtIdx];
      
      if (defectStartOffsetMap.count(startIdx) > 0) {
        printf("start  %8d = (%4d,%4d)\n", startIdx, startP.x, startP.y);
        printf("end    %8d = (%4d,%4d)\n", endIdx, endP.x, endP.y);
        printf("defect %8d = (%4d,%4d)\n", defectPtIdx, defectP.x, defectP.y);
        printf("depth  %0.3f\n", depth);
        
        line(colorMat, startP, endP, Scalar(0xFF,0,0), 1, 0);
        circle(colorMat, defectP, 4, Scalar(0,0,0xFF), 2);
      } else {
        printf("SKIP depth  %0.3f\n", depth);
      }
    }
  }
  
  if (debugDumpImages) {
    std::stringstream fnameStream;
    fnameStream << HULL_DUMP_IMAGE_PREFIX << tag << "_hull_defect_normals" << ".png";
    string fname = fnameStream.str();
    
    imwrite(fname, binMat);
    cout << "wrote " << fname << endl;
    cout << "" << endl;
  }
  
  if (debugDumpImages) {
    std::stringstream fnameStream;
    fnameStream << HULL_DUMP_IMAGE_PREFIX << tag << "_hull_defect_filtered_render" << ".png";
    string fname = fnameStream.str();
    
    writeWroteImg(fname, colorMat);
    cout << "" << endl;
  }
  
  vector<TypedHullCoords> hullCoords;
  
  // Determine hull point iteration order that is clockwise and
  // increases in terms of offsets as the iteration progresses.
  
  vector<pair<int, int> > orderedOffsetsPairs;
  
  // Iterate over clockwise hull ordering so that result pairs start from N
  // and progress clockwise around the shape.
  
  int prevOffset = 0;
  const uint32_t maxUint = 0xFFFFFFFF;
  uint32_t minOffset = maxUint;
  
  for ( int i = 0; i < hullCount; i++ ) {
    if (i == 0) {
      prevOffset = hull[hull.size() - 1];
    }
    
    int offset = hull[i];
    
    orderedOffsetsPairs.push_back(make_pair(prevOffset, offset));
    
    if (prevOffset < minOffset) {
      minOffset = prevOffset;
    }
    
    prevOffset = offset;
  }
  assert(minOffset != maxUint);
  
  if (debug) {
    for ( pair<int, int> &pairRef : orderedOffsetsPairs ) {
      int offset1 = pairRef.first;
      int offset2 = pairRef.second;
      Point2i p1 = contour[offset1];
      Point2i p2 = contour[offset2];
      cout << "pair (" << offset1 << " , " << offset2 << ") -> (" << Coord(p1.x,p1.y) << ", " <<  Coord(p2.x,p2.y) << ")" << endl;
      
      // FIXME: check prev
      //assert(offset1 <= offset2);
    }
  }
  
  // Reorder orderedOffsetsPairs by placing the element that starts at
  // offset 0 at the front of the vector.
  
  auto midIt = begin(orderedOffsetsPairs);
  auto endIt = end(orderedOffsetsPairs);
  bool atFirstIterPos = true;
  
  for ( ; midIt != endIt; midIt++ ) {
    int offset1 = midIt->first;
    //int offset2 = midIt->second;
    
    if (offset1 == minOffset) {
      break;
    }
    
    atFirstIterPos = false;
  }
  assert(midIt != endIt);
  
  if (debug) {
    cout << "rotate so that " << minOffset << " is at the front" << endl;
  }
  
  if (atFirstIterPos == false) {
    
    cout << "rotate()" << endl << endl;
    
    int numBefore = (int) orderedOffsetsPairs.size();
    
    rotate(begin(orderedOffsetsPairs), midIt, end(orderedOffsetsPairs));
    
    int numAfter = (int) orderedOffsetsPairs.size();
    
    assert(numBefore == numAfter);
    
    if (debug) {
      for ( pair<int, int> &pairRef : orderedOffsetsPairs ) {
        int offset1 = pairRef.first;
        int offset2 = pairRef.second;
        Point2i p1 = contour[offset1];
        Point2i p2 = contour[offset2];
        cout << "pair (" << offset1 << " , " << offset2 << ") -> (" << Coord(p1.x,p1.y) << ", " <<  Coord(p2.x,p2.y) << ")" << endl;
      }
    }
  }
  
  int iMax = (int) orderedOffsetsPairs.size();
  
  // Verify that the end of each pair is the same offset as the start of the next pair.
  
  for ( int i = 0; i < (iMax - 1); i++ ) {
    pair<int, int> &pairRef1 = orderedOffsetsPairs[i];
    pair<int, int> &pairRef2 = orderedOffsetsPairs[i+1];
    
    int offsetA1 = pairRef1.first;
    int offsetA2 = pairRef1.second;
    
    int offsetB1 = pairRef2.first;
    int offsetB2 = pairRef2.second;
    
    //        cout << "" << endl;
    //        cout << "pair 1 " << offsetA1 << " , " << offsetA2 << endl;
    //        cout << "pair 2 " << offsetB1 << " , " << offsetB2 << endl;
    
    if (i > (iMax - 2)) {
      assert(offsetA1 < offsetA2);
    }
    
    assert(offsetA2 == offsetB1);
  }
  
  // Iterate over hull offset pairs in clockwise order
  
  for ( int i = 0; i < iMax; i++ ) {
    pair<int, int> &pairRef = orderedOffsetsPairs[i];
    
    int offset1 = pairRef.first;
    int offset2 = pairRef.second;
    
    // Gather pair of points that indicate a hull line.
    
    Point2i pt1 = contour[offset1];
    Point2i pt2 = contour[offset2];
    
    Coord ct1(pt1.x, pt1.y);
    Coord ct2(pt2.x, pt2.y);
    
    if (debug) {
      cout << "hull offsets between " << offset1 << " to " << offset2 << endl;
      cout << "hull line between " << ct1 << " to " << ct2 << endl;
    }
    
    // Gather all the coords from contour that in the set (pt1, pt2)
    
    hullCoords.push_back(TypedHullCoords());
    TypedHullCoords &typedHullCoords = hullCoords[hullCoords.size() - 1];
    auto &coordsVec = typedHullCoords.coords;
    
    appendCoordsInRangeOnContour(offset1, offset2, contour, coordsVec);
    
    // Check hull segment start and end coords. A hull segment
    // should start at ct1 aka offset1. Note that since a coordinate
    // could be duplicated in the contour the offset must be used.
    
    if (defectStartOffsetMap.count(offset1) > 0) {
      // (start, end) indicates a concave range
      typedHullCoords.isConcave = true;
      
      assert(defectStartOffsetToTripleMap.count(offset1) > 0);
      vector<Coord> &triple = defectStartOffsetToTripleMap[offset1];
      Coord defectC = triple[2];
      typedHullCoords.defectPoint = defectC;
    } else {
      typedHullCoords.defectPoint = Coord(0xFFFF, 0xFFFF);
      typedHullCoords.isConcave = false;
    }
  }
  
  // All coordinates from original contour must be accounted for in hullCoords
  
#if defined(DEBUG)
  int coordCount = 0;
  
  vector<Coord> combined;
  
  for ( TypedHullCoords &typedHullCoords : hullCoords ) {
    auto vec = typedHullCoords.coords;
    assert(vec.size() > 0);
    coordCount += vec.size();
    
    append_to_vector(combined, vec);
  }
  
  if (coordCount != contour.size()) {
    for ( int i = 0; i < mini(coordCount, (int)contour.size()); i++ ) {
      Point2i p = contour[i];
      Coord c1(p.x, p.y);
      
      Coord c2 = combined[i];
      
      cout << "c1 == c2 : " << i << " : " << c1 << " <-> " << c2 << endl;
      
      if (c1 != c2) { assert(0); }
    }
    
    assert(coordCount == contour.size());
  }
  assert(coordCount == contour.size());
  
  // hullCoords could contain duplicates
  unordered_map<Coord, bool> seen;
  for ( TypedHullCoords &typedHullCoords : hullCoords ) {
    auto vec = typedHullCoords.coords;
    for ( Coord c : vec ) {
      seen[c] = true;
    }
  }
  // every coord in contour must be in hullCoords
  for ( Point2i p : contour ) {
    Coord c(p.x, p.y);
    assert(seen.count(c) == 1);
  }
#endif // DEBUG
  
  if (debug) {
    for ( TypedHullCoords &typedHullCoords : hullCoords ) {
      if (debug) {
        cout << "typedHullCoords.isConcave " << typedHullCoords.isConcave << " and coords " << endl;
        
        for ( Coord c : typedHullCoords.coords ) {
          cout << c << " ";
        }
        cout << endl;
      }
    }
    cout << "";
  }
  
  // Render hull lines as 0xFF for convex and 0x7F for concave
  
  if (debugDumpImages) {
    binMat = Scalar(0);
    
    for ( TypedHullCoords &typedHullCoords : hullCoords ) {
      uint8_t gray;
      if (typedHullCoords.isConcave) {
        gray = 0x7F;
      } else {
        gray = 0xFF;
      }
      
      for ( Coord c : typedHullCoords.coords ) {
        binMat.at<uint8_t>(c.y, c.x) = gray;
      }
    }
    
    std::stringstream fnameStream;
    fnameStream << HULL_DUMP_IMAGE_PREFIX << tag << "_hull_type" << ".png";
    string fname = fnameStream.str();
    
    imwrite(fname, binMat);
    cout << "wrote " << fname << endl;
    cout << "" << endl;
  }
  
  // Render hull lines as increasing UID values so that the specific regions
  // appear more clearly separated from the others. This makes the specific
  // order of the hull segments more clear as color values.
  
  if (debugDumpImages) {
    Mat colorMat(binMat.size(), CV_8UC3, Scalar(0,0,0));
    
    for ( TypedHullCoords &typedHullCoords : hullCoords ) {
      uint32_t pixel = 0;
      pixel |= (rand() % 256);
      pixel |= ((rand() % 256) << 8);
      pixel |= ((rand() % 256) << 16);
      pixel |= (0xFF << 24);
      
      Vec3b vec = PixelToVec3b(pixel);
      
      for ( Coord c : typedHullCoords.coords ) {
        colorMat.at<Vec3b>(c.y, c.x) = vec;
      }
    }
    
    std::stringstream fnameStream;
    fnameStream << HULL_DUMP_IMAGE_PREFIX << tag << "_hull_segments" << ".png";
    string fname = fnameStream.str();
    
    imwrite(fname, colorMat);
    cout << "wrote " << fname << endl;
    cout << "" << endl;
  }
  
  // Dump hull lines with diff colors for each segment render
  
  if (debugDumpImages) {
    Mat colorMat(binMat.size(), CV_8UC3, Scalar(0,0,0));
    
    for ( TypedHullCoords &typedHullCoords : hullCoords ) {
      auto vecOfCoords = typedHullCoords.coords;
      auto vecOfPoints = convertCoordsToPoints(vecOfCoords);
      auto startP = vecOfPoints[0];
      auto endP = vecOfPoints[vecOfPoints.size() - 1];
      
      Scalar color = Scalar((rand() % 256), (rand() % 256), (rand() % 256));
      line(colorMat, startP, endP, color, 1, 8);
    }
    
    std::stringstream fnameStream;
    fnameStream << HULL_DUMP_IMAGE_PREFIX << tag << "_hull_lines_segments" << ".png";
    string fname = fnameStream.str();
    
    imwrite(fname, colorMat);
    cout << "wrote " << fname << endl;
    cout << "" << endl;
  }
  
  // Implement merge of hull line segments to simplify the regions that were
  // detected as concave segments but were actually just part of a regular
  // convex region. These regions need to be joined together to simplify
  // the shape and make the grouping into typed hull coords make more logical
  // sense.
  
  if ((1)) {
    vector<int> angles;
    Point2i lastP;
    
    // Count number of coords before operations
    
    int coordCountBefore = 0;
    
    for ( TypedHullCoords &typedHullCoords : hullCoords ) {
      auto vecOfCoords = typedHullCoords.coords;
      coordCountBefore += (int) vecOfCoords.size();
    }
    
    // In the first iteration, gather (start, end) coords for each hull line
    // and make sure that there is one additional entry at the end of the
    // set of pairs that corresponds to the first segment.
    
    vector<pair<Point2i,Point2i> > vecOfStartEndPointPairs;
    vector<bool> vecOfIsConcave;
    
    vector<TypedHullCoords*> vecOfHullCoordsPtrs;
    
    for ( TypedHullCoords &typedHullCoords : hullCoords ) {
      auto vecOfCoords = typedHullCoords.coords;
      Point2i startP = coordToPoint(vecOfCoords[0]);
      Point2i endP = coordToPoint(vecOfCoords[vecOfCoords.size() - 1]);
      vecOfStartEndPointPairs.push_back(make_pair(startP, endP));
      
      vecOfIsConcave.push_back(typedHullCoords.isConcave);
      
      vecOfHullCoordsPtrs.push_back(&typedHullCoords);
    }
    
    {
      TypedHullCoords &typedHullCoords = hullCoords[0];
      auto vecOfCoords = typedHullCoords.coords;
      Point2i firstStartP = coordToPoint(vecOfCoords[0]);
      Point2i firstEndP = coordToPoint(vecOfCoords[vecOfCoords.size() - 1]);
      
      vecOfStartEndPointPairs.push_back(make_pair(firstStartP, firstEndP));
      
      vecOfIsConcave.push_back(typedHullCoords.isConcave);
      
      vecOfHullCoordsPtrs.push_back(&typedHullCoords);
    }
    
    // Iterate from (1, end) knowing that each element will always have
    
    int iMax = (int) hullCoords.size();
    
    for ( int i = 0; i < iMax; i++ ) {
      auto currentPair = vecOfStartEndPointPairs[i];
      auto nextPair = vecOfStartEndPointPairs[i+1];
      
      bool currentIsConcave = vecOfIsConcave[i];
      bool nextIsConcave = vecOfIsConcave[i+1];
      
      cout << "(i, i+1): (" << i << ", " << i+1 << ")" << endl;
      cout << "currentPair: " << pointToCoord(currentPair.first) << " " << pointToCoord(currentPair.second) << endl;
      cout << "nextPair: " << pointToCoord(nextPair.first) << " " << pointToCoord(nextPair.second) << endl;
      
      // Use nextPair.first as anchor point and then calculate vectors to currentPair.first
      // and to nextPair.second
      
      Point2i anchorP = nextPair.first;
      Point2i prevP = currentPair.first;
      Point2i nextP = nextPair.second;
      
      Point2i d1 = anchorP - prevP;
      Point2i d2 = anchorP - nextP;
      
      int angleDeg;
      
      if ((d1.x == 0 && d1.y == 0) || (d2.x == 0 && d2.y == 0)) {
        // Unlikely case where 2 points are identical, avoid angle calculation
        angleDeg = 0;
      } else {
        float radAngleBetween = angleBetween(d1, d2);
        angleDeg = radAngleBetween * 180.0f / M_PI;
      }
      
      // FIXME: id d < 3 then unlikely to be a big angle change, treat as part of same line?
      
      int angleMinus = 30;
      bool isNearStraight = angleDeg > (180 - angleMinus);
      
      if (debug) {
        printf("lp (%3d, %3d)\n", prevP.x, prevP.y );
        printf("p1 (%3d, %3d)\n", anchorP.x, anchorP.y );
        printf("p2 (%3d, %3d)\n", nextP.x, nextP.y );
        
        printf("d1 (%3d, %3d)\n", d1.x, d1.y );
        printf("d2 (%3d, %3d)\n", d2.x, d2.y );
        
        printf("angle %d\n", angleDeg);
        printf("angle isNearStraight %d\n", isNearStraight);
        printf("\n");
      }
      
      bool mergeLineSegments = (isNearStraight && (!currentIsConcave && !nextIsConcave));
      
      if (debugDumpImages) {
        Mat colorMat(binMat.size(), CV_8UC3, Scalar(0,0,0));
        
        Scalar c1 = Scalar(0, 0xFF, 0);
        line(colorMat, prevP, anchorP, c1, 1, 8);
        
        Scalar c2 = Scalar(0, 0, 0xFF);
        line(colorMat, anchorP, nextP, c2, 1, 8);
        
        std::stringstream fnameStream;
        fnameStream << HULL_DUMP_IMAGE_PREFIX << tag << "_hull_lines_segment_" << i << "_angle_" << angleDeg << (mergeLineSegments ? "_merged" : "_notmerged" ) << ".png";
        string fname = fnameStream.str();
        
        imwrite(fname, colorMat);
        cout << "wrote " << fname << endl;
        cout << "" << endl;
      }
      
      if (mergeLineSegments) {
        if (debug) {
          cout << "combine convex segments " << i << " and " << (i+1) << endl;
        }
        
        // FIXME: what happens if join around 0 would combine non-concave regions?
        
        TypedHullCoords &typedHullCoords1 = *(vecOfHullCoordsPtrs[i]);
        TypedHullCoords &typedHullCoords2 = *(vecOfHullCoordsPtrs[i+1]);
        
#if defined(DEBUG)
        assert(typedHullCoords1.isConcave == false);
        assert(typedHullCoords2.isConcave == false);
        assert(typedHullCoords1.coords.size() > 0);
#endif // DEBUG
        
        auto &coordsVecSrc = typedHullCoords1.coords;
        auto &coordsVecDst = typedHullCoords2.coords;
        
        // Insert at front of coordsVecDst to maintain contour order
        coordsVecDst.insert(begin(coordsVecDst), begin(coordsVecSrc), end(coordsVecSrc));
        coordsVecSrc.clear();
      }
    }
    
    // After processing all hull line segments, filter out segments with zero coords.
    
    vector<TypedHullCoords> combinedHullCoords;
    
    int coordPairsOffset = 0;
    int lastCoordPairsOffset = (int) vecOfHullCoordsPtrs.size() - 1;
    
    for ( TypedHullCoords *typedHullCoordsPtr : vecOfHullCoordsPtrs ) {
      if (coordPairsOffset == lastCoordPairsOffset) {
        if (debug) {
          cout << "ignore repeated first combinedHullCoords at offset " << coordPairsOffset << endl;
        }
      } else if ((coordPairsOffset != lastCoordPairsOffset) && (typedHullCoordsPtr->coords.size() > 0)) {
        if (debug) {
          cout << "add to combinedHullCoords at offset " << coordPairsOffset << " N = " << typedHullCoordsPtr->coords.size() << endl;
        }
        
        combinedHullCoords.push_back(*typedHullCoordsPtr);
      } else {
        if (debug) {
          cout << "ignore zero length combinedHullCoords at offset " << coordPairsOffset << endl;
        }
      }
      coordPairsOffset++;
    }
    
    cout << "filtered hull coords down to " << combinedHullCoords.size() << " elements" << endl;
    
    hullCoords = combinedHullCoords;
    
#if defined(DEBUG)
    int coordCountAfter = 0;
    
    for ( TypedHullCoords &typedHullCoords : hullCoords ) {
      auto vecOfCoords = typedHullCoords.coords;
      coordCountAfter += (int) vecOfCoords.size();
    }
    
    assert(coordCountAfter == coordCountBefore);
#endif // DEBUG
    
    cout << "" << endl;
  }
  
  // Dump hull lines with diff colors for each segment render
  
  if (debugDumpImages) {
    Mat colorMat(binMat.size(), CV_8UC3, Scalar(0,0,0));
    
    for ( TypedHullCoords &typedHullCoords : hullCoords ) {
      auto vecOfCoords = typedHullCoords.coords;
      auto vecOfPoints = convertCoordsToPoints(vecOfCoords);
      auto startP = vecOfPoints[0];
      auto endP = vecOfPoints[vecOfPoints.size() - 1];
      
      Scalar color = Scalar((rand() % 256), (rand() % 256), (rand() % 256));
      line(colorMat, startP, endP, color, 1, 8);
    }
    
    std::stringstream fnameStream;
    fnameStream << HULL_DUMP_IMAGE_PREFIX << tag << "_hull_lines_combined_segments" << ".png";
    string fname = fnameStream.str();
    
    imwrite(fname, colorMat);
    cout << "wrote " << fname << endl;
    cout << "" << endl;
  }
  
  if (debug) {
    cout << "clockwiseScanOfHullCoords return" << endl;
  }
  
  return hullCoords;
}

// For existing test cases

vector<HullLineOrCurveSegment>
splitContourIntoLinesSegments(int32_t tag, CvSize size, CvRect roi, const vector<Coord> &contourCoords, double epsilon)
{
  vector<Point2i> contour = convertCoordsToPoints(contourCoords);
  return splitContourIntoLinesSegments(tag, size, roi, contour, epsilon);
}

// This method accepts a contour that is not simplified and detects straight lines
// as compared to the non-straight curves.

vector<HullLineOrCurveSegment>
splitContourIntoLinesSegments(int32_t tag, CvSize size, CvRect roi, const vector<Point2i> &contour, double epsilon)
{
  const bool debug = true;
  const bool debugDumpImages = true;
  
  if (debug) {
    cout << "splitContourIntoLinesSegments" << endl;
  }
  
  if (debugDumpImages) {
    Mat binMat(size, CV_8UC1, Scalar(0));
    
    vector<vector<Point2i> > contours;
    contours.push_back(contour);
    drawContours(binMat, contours, 0, Scalar(0xFF), CV_FILLED); // Draw contour as white filled region
    
    std::stringstream fnameStream;
    fnameStream << HULL_DUMP_IMAGE_PREFIX << tag << "_hull_approx_contour_original" << ".png";
    string fname = fnameStream.str();
    
    writeWroteImg(fname, binMat);
    cout << "" << endl;
  }
  
  // Verify that points on contour are 8 connected and not simplified.
  
  auto is8Connected = [](const Point2i &p1, const Point2i &p2)->bool {
    Point2i delta = p2 - p1;
    int dx = abs(delta.x);
    int dy = abs(delta.y);
    
    if ((dx < 2) && (dy < 2)) {
      // 0 or +-1 means next point is 8 connected
      return true;
    } else {
      return false;
    }
  };
  
#if defined(DEBUG)
  if (1) {
    for ( int i = 0; i < contour.size(); i++ ) {
      Point2i p1 = contour[vecOffsetAround((int)contour.size(), i-1)];
      Point2i p2 = contour[i];
      
      if (is8Connected(p1, p2) == false) {
        cout << "not 8 connected :" << pointToCoord(p1) << " " << pointToCoord(p2) << endl;
        assert(0);
      }
    }
  }
#endif // DEBUG
  
  vector<Point2i> approxContour;
  
  //  double epsilon = 1.4; // Max dist between original curve and approx
  
  approxPolyDP(Mat(contour), approxContour, epsilon, true);
  
  if (debugDumpImages) {
    Mat colorMat(size, CV_8UC3, Scalar(0,0,0));
    
    vector<vector<Point2i> > contours;
    contours.push_back(approxContour);
    
    drawContours(colorMat, contours, 0, Scalar(0xFF,0xFF,0xFF), CV_FILLED); // Draw contour as white filled region
    
    // Draw each contour line as a different color
    
    for ( int i = 0; i < approxContour.size(); i++ ) {
      Point2i p1 = approxContour[i];
      Point2i p2 = approxContour[vecOffsetAround((int)approxContour.size(), i+1)];
      
      Scalar color = Scalar((rand() % 256), (rand() % 256), (rand() % 256));
      
      line(colorMat, p1, p2, color, 2, 8);
    }
    
    std::stringstream fnameStream;
    fnameStream << HULL_DUMP_IMAGE_PREFIX << tag << "_hull_approx_contour" << ".png";
    string fname = fnameStream.str();
    
    writeWroteImg(fname, colorMat);
    cout << "" << endl;
  }
  
  // Iterate over contour points and determine which ones are inside the approx
  // simplified lines represented by approxContour.
  
  vector<HullLineOrCurveSegment> segments;
  bool lastSegmentIsLine = false;
  
  int contouri = 0;
  
  // This offset indicates where a forward iteration must stop, note that in
  // the case where a line wraps around the end this max value can go past
  // the end of the contours array and that will wrap around.
  
  int contouriMax = (int) contour.size();
  
  for ( int i = 0; i < approxContour.size(); i++ ) {
    Point2i p1 = approxContour[i];
    Point2i p2 = approxContour[vecOffsetAround((int)approxContour.size(), i+1)];
    
    if (i == 0) {
      // On the first iteration, check for the case where the very first coordinate
      // in a line does not match the first coordinate in contour. In this case,
      // the elements need to be read back from the front of the contour array
      // an skipped here.
      
      while (1) {
#if defined(DEBUG)
        assert(contouri < contour.size());
#endif // DEBUG
        
        if (contour[contouri] == p1) {
          if (debug) {
            printf("stop advance past at initial contouri %d (%5d, %5d)\n", contouri, contour[contouri].x, contour[contouri].y);
          }
          
          break;
        } else {
          if (debug) {
            printf("advance starting point past (%5d, %5d)\n", contour[contouri].x, contour[contouri].y);
          }
          
          contouri++;
          contouriMax++;
        }
      }
      
#if defined(DEBUG)
      assert(contour[contouri] == p1);
#endif // DEBUG
      
      if (debug) {
        printf("starting point at initial contouri %d (%5d, %5d)\n", contouri, contour[contouri].x, contour[contouri].y);
      }
    }
    
    Point2f vec = p2 - p1;
    
    if (debug) {
      printf("i %d of %d\n", i, (int)approxContour.size());
      printf("line points (%5d, %5d) -> (%5d, %5d)\n", p1.x, p1.y, p2.x, p2.y);
      printf("line vector (%0.3f, %0.3f)\n", vec.x, vec.y);
    }
    
    // If the points are not 8 connected then this indicates the start of a new
    // line segment that will consume N points from the original contour.
    
    if (is8Connected(p1, p2)) {
      // Non line element, append points to current non-line segment or
      // create one if the last segment is a line.
      
      if (lastSegmentIsLine || (segments.size() == 0)) {
        HullLineOrCurveSegment locSeg;
        locSeg.isLine = false;
        segments.push_back(std::move(locSeg));
        lastSegmentIsLine = false;
      }
      
#if defined(DEBUG)
      {
        assert(contouri < contouriMax);
        Point2i contourP = contour[vecOffsetAround((int)contour.size(), contouri)];
        assert(contourP == p1);
      }
#endif // DEBUG
      
      vector<Point2i> &consumedPointsVec = segments[segments.size() - 1].points;
      consumedPointsVec.push_back(p1);
      
      if (debug) {
        printf("consume contouri %d = (%d, %d)\n", contouri, p1.x, p1.y);
      }
      
#if defined(DEBUG)
      assert(consumedPointsVec.size() > 0);
#endif // DEBUG
      
      contouri++;
    } else {
      // Consume contour points leading up to but not including p2 when a line is found
      
      // Note that multiple lines are not combined since they would not
      // have the same common slope.
      
      HullLineOrCurveSegment locSeg;
      locSeg.isLine = true;
      
      // Determine common slope for the whole line
      
      float scale = makeUnitVector(vec);
      scale = scale;
      
      locSeg.slope = vec;
      
      segments.push_back(locSeg);
      lastSegmentIsLine = true;
      
      vector<Point2i> &consumedPointsVec = segments[segments.size() - 1].points;
      
#if defined(DEBUG)
      {
        assert(contouri < contouriMax);
        Point2i contourP = contour[vecOffsetAround((int)contour.size(), contouri)];
        assert(contourP == p1);
      }
#endif // DEBUG
      
      while (1) {
        if (contouri == contouriMax) {
          if (debug) {
            printf("stop consume at last contouri %d\n", contouri);
          }
          break;
        }
        
        // FIXME: adjust ahead so that max is set to the number skipped over in iter1!
        // Then wrap the array access around.
        
        Point2i contourP = contour[vecOffsetAround((int)contour.size(), contouri)];
        
        if (contourP == p2) {
          if (debug) {
            printf("stop consume at contouri %d = (%d, %d)\n", contouri, contourP.x, contourP.y);
          }
          break;
        } else {
          if (debug) {
            printf("consume contouri %d = (%d, %d)\n", contouri, contourP.x, contourP.y);
          }
          consumedPointsVec.push_back(contourP);
          contouri++;
        }
      } // end while !is8Connected
      
#if defined(DEBUG)
      assert(consumedPointsVec.size() > 0);
#endif // DEBUG
      
    } // end line block
  }
  
  // Make sure final point gets added
  assert(approxContour.size() > 0);
  if (contouri == (contour.size() - 1)) {
    Point2i lastPoint = contour[contour.size() - 1];
    segments[segments.size() - 1].points.push_back(lastPoint);
    
    if (debug) {
      cout << "append lastPoint " << pointToCoord(lastPoint) << endl;
    }
  }
  
  if (debug) {
    cout << "splitContourIntoLinesSegments return " << segments.size() << " segments" << endl;
    
    for ( auto &locSeg : segments ) {
      cout << "isLine " << locSeg.isLine << endl;
      cout << "slope " << locSeg.slope.x << "," << locSeg.slope.y << endl;
      
      cout << "N coords = " << locSeg.points.size() << endl;
      
      for ( Point2i p : locSeg.points ) {
        cout << p.x << "," << p.y << endl;
      }
    }
  }
  
#if defined(DEBUG)
  // Each point in contour should appear in the segments in order
  
  vector<Point2i> combinedSegmentPoints;
  
  for ( auto &locSeg : segments ) {
    assert(locSeg.points.size() > 0);
    append_to_vector(combinedSegmentPoints, locSeg.points);
  }
  
  while (combinedSegmentPoints.size() < contour.size()) {
    combinedSegmentPoints.push_back(Point2i(-1, -1));
  }
  
  // Note that before an element by element compare can be executed
  // the coordinates in contour need to be rotated so that the
  // very first element is the one at contouriMax (wraps around)
  
  vector<Point2i> inPoints = contour;
  
  if (contouriMax != contour.size()) {
    auto midIt = begin(inPoints) + vecOffsetAround((int)contour.size(), contouriMax);
    rotate(begin(inPoints), midIt, end(inPoints));
    assert(inPoints.size() == contour.size());
  }
  
  for ( int i = 0; i < combinedSegmentPoints.size(); i++ ) {
    Point2i pOrig = inPoints[i];
    Point2i pOut = combinedSegmentPoints[i];
    if (pOrig != pOut) {
      assert(0);
    }
  }
  
  // Totals
  int totalNumPointsOut = 0;
  
  for ( auto &locSeg : segments ) {
    totalNumPointsOut += locSeg.points.size();
  }
  
  assert(contour.size() == totalNumPointsOut);
#endif // DEBUG
  
  // Each line segment is represented as a different color and
  // each curve segment is represented as a different color.
  
  if (debugDumpImages) {
    Mat colorMat(size, CV_8UC3, Scalar(0,0,0));
    
    for ( auto &locSeg : segments ) {
      Vec3b color;
      
      if (locSeg.isLine) {
        color = Vec3b(0,0,0xFF); // Red for line
      } else {
        color = Vec3b(0x7F,0x7F,0x7F); // Gray for curve pixels
      }
      
      for ( Point2i p : locSeg.points ) {
        colorMat.at<Vec3b>(p.y, p.x) = color;
      }
    }
    
    std::stringstream fnameStream;
    fnameStream << HULL_DUMP_IMAGE_PREFIX << tag << "_hull_approx_line_or_curve_type" << ".png";
    string fname = fnameStream.str();
    
    writeWroteImg(fname, colorMat);
    cout << "" << endl;
  }
  
  // Render each line as a different color with width 1
  
  if (debugDumpImages) {
    Mat colorMat(size, CV_8UC3, Scalar(0,0,0));
    
    for ( auto &locSeg : segments ) {
      Vec3b color((rand() % 256), (rand() % 256), (rand() % 256));
      
      for ( Point2i p : locSeg.points ) {
        colorMat.at<Vec3b>(p.y, p.x) = color;
      }
    }
    
    std::stringstream fnameStream;
    fnameStream << HULL_DUMP_IMAGE_PREFIX << tag << "_hull_approx_line_or_curve" << ".png";
    string fname = fnameStream.str();
    
    writeWroteImg(fname, colorMat);
    cout << "" << endl;
  }
  
  return segments;
}
