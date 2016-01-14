// This file contains utility functions for general purpose use and interaction with OpenCV

#include "OpenCVUtil.h"

#include "Superpixel.h"

#include "vf_DistanceTransform.h"

// Print SSIM for two images to cout

int printSSIM(Mat inImage1, Mat inImage2)
{
  // default settings
  double C1 = 6.5025, C2 = 58.5225;
  
  IplImage
		*img1=NULL, *img2=NULL, *img1_img2=NULL,
		*img1_temp=NULL, *img2_temp=NULL,
		*img1_sq=NULL, *img2_sq=NULL,
		*mu1=NULL, *mu2=NULL,
		*mu1_sq=NULL, *mu2_sq=NULL, *mu1_mu2=NULL,
		*sigma1_sq=NULL, *sigma2_sq=NULL, *sigma12=NULL,
		*ssim_map=NULL, *temp1=NULL, *temp2=NULL, *temp3=NULL;

  IplImage tmpCopy1 = inImage1;
  IplImage tmpCopy2 = inImage2;
  
  img1_temp = &tmpCopy1;
  img2_temp = &tmpCopy2;
  
  int x=img1_temp->width, y=img1_temp->height;
  int nChan=img1_temp->nChannels, d=IPL_DEPTH_32F;
  CvSize size = cvSize(x, y);
  
  img1 = cvCreateImage(size, d, nChan);
  img2 = cvCreateImage(size, d, nChan);
  
  cvConvert(img1_temp, img1);
  cvConvert(img2_temp, img2);
  cvReleaseImage(&img1_temp);
  img1_temp = NULL;
  cvReleaseImage(&img2_temp);
  img2_temp = NULL;
  
  img1_sq = cvCreateImage( size, d, nChan);
  img2_sq = cvCreateImage( size, d, nChan);
  img1_img2 = cvCreateImage( size, d, nChan);
  
  cvPow( img1, img1_sq, 2 );
  cvPow( img2, img2_sq, 2 );
  cvMul( img1, img2, img1_img2, 1 );
  
  mu1 = cvCreateImage( size, d, nChan);
  mu2 = cvCreateImage( size, d, nChan);
  
  mu1_sq = cvCreateImage( size, d, nChan);
  mu2_sq = cvCreateImage( size, d, nChan);
  mu1_mu2 = cvCreateImage( size, d, nChan);
  
  
  sigma1_sq = cvCreateImage( size, d, nChan);
  sigma2_sq = cvCreateImage( size, d, nChan);
  sigma12 = cvCreateImage( size, d, nChan);
  
  temp1 = cvCreateImage( size, d, nChan);
  temp2 = cvCreateImage( size, d, nChan);
  temp3 = cvCreateImage( size, d, nChan);
  
  ssim_map = cvCreateImage( size, d, nChan);
  
  // ************************** END INITS **********************************
  
  //////////////////////////////////////////////////////////////////////////
  // PRELIMINARY COMPUTING
  cvSmooth( img1, mu1, CV_GAUSSIAN, 11, 11, 1.5 );
  cvSmooth( img2, mu2, CV_GAUSSIAN, 11, 11, 1.5 );
  
  cvPow( mu1, mu1_sq, 2 );
  cvPow( mu2, mu2_sq, 2 );
  cvMul( mu1, mu2, mu1_mu2, 1 );
  
  
  cvSmooth( img1_sq, sigma1_sq, CV_GAUSSIAN, 11, 11, 1.5 );
  cvAddWeighted( sigma1_sq, 1, mu1_sq, -1, 0, sigma1_sq );
  
  cvSmooth( img2_sq, sigma2_sq, CV_GAUSSIAN, 11, 11, 1.5 );
  cvAddWeighted( sigma2_sq, 1, mu2_sq, -1, 0, sigma2_sq );
  
  cvSmooth( img1_img2, sigma12, CV_GAUSSIAN, 11, 11, 1.5 );
  cvAddWeighted( sigma12, 1, mu1_mu2, -1, 0, sigma12 );
  
  
  //////////////////////////////////////////////////////////////////////////
  // FORMULA
  
  // (2*mu1_mu2 + C1)
  cvScale( mu1_mu2, temp1, 2 );
  cvAddS( temp1, cvScalarAll(C1), temp1 );
  
  // (2*sigma12 + C2)
  cvScale( sigma12, temp2, 2 );
  cvAddS( temp2, cvScalarAll(C2), temp2 );
  
  // ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
  cvMul( temp1, temp2, temp3, 1 );
  
  // (mu1_sq + mu2_sq + C1)
  cvAdd( mu1_sq, mu2_sq, temp1 );
  cvAddS( temp1, cvScalarAll(C1), temp1 );
  
  // (sigma1_sq + sigma2_sq + C2)
  cvAdd( sigma1_sq, sigma2_sq, temp2 );
  cvAddS( temp2, cvScalarAll(C2), temp2 );
  
  // ((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2))
  cvMul( temp1, temp2, temp1, 1 );
  
  // ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2))
  cvDiv( temp3, temp1, ssim_map, 1 );
  
  CvScalar index_scalar = cvAvg( ssim_map );
  
  // through observation, there is approximately
  // 1% error max with the original matlab program
  
  cout << "(R, G & B SSIM index)" << endl ;
  cout << index_scalar.val[2] * 100 << "%" << endl ;
  cout << index_scalar.val[1] * 100 << "%" << endl ;
  cout << index_scalar.val[0] * 100 << "%" << endl ;
  
  // Release the images
  
  if (img1) {
    cvReleaseImage(&img1);
  }
  if (img2) {
    cvReleaseImage(&img2);
  }
  if (img1_img2) {
    cvReleaseImage(&img1_img2);
  }
  if (img1_temp) {
    cvReleaseImage(&img1_temp);
  }
  if (img2_temp) {
    cvReleaseImage(&img2_temp);
  }
  if (img1_sq) {
    cvReleaseImage(&img1_sq);
  }
  if (img2_sq) {
    cvReleaseImage(&img2_sq);
  }
  if (mu1) {
    cvReleaseImage(&mu1);
  }
  if (mu2) {
    cvReleaseImage(&mu2);
  }
  if (mu1_sq) {
    cvReleaseImage(&mu1_sq);
  }
  if (mu2_sq) {
    cvReleaseImage(&mu2_sq);
  }
  if (mu1_mu2) {
    cvReleaseImage(&mu1_mu2);
  }
  if (sigma1_sq) {
    cvReleaseImage(&sigma1_sq);
  }
  if (sigma2_sq) {
    cvReleaseImage(&sigma2_sq);
  }
  if (sigma12) {
    cvReleaseImage(&sigma12);
  }
  if (ssim_map) {
    cvReleaseImage(&ssim_map);
  }
  if (temp1) {
    cvReleaseImage(&temp1);
  }
  if (temp2) {
    cvReleaseImage(&temp2);
  }
  if (temp3) {
    cvReleaseImage(&temp3);
  }
  
  return 0;
}

// Find a single "center" pixel in region of interest matrix. This logic
// accepts an input matrix that contains binary pixel values (0x0 or 0xFF)
// and computes a consistent center pixel. When this method returns the
// region binMat is unchanged. The orderMat is set to the size of the roi and
// it is filled with distance transformed gray values. Note that this method
// has to create a buffer zone of 1 pixel so that pixels on the edge have
// a very small distance.

Coord findRegionCenter(Mat &binMat, cv::Rect roi, Mat &outDistMat, int tag)
{
  const bool debug = true;
  const bool debugDumpAllImages = true;
  
  assert(binMat.channels() == 1);
  
  Point2i center(-1, -1);
  
  Mat binROIMat = binMat(roi);
  
  outDistMat.create(roi.height, roi.width, CV_8UC1);
  
  Mat regionMat(roi.height+2, roi.width+2, CV_8UC1, Scalar(0));
  
  assert(regionMat.cols == binROIMat.cols+2);
  assert(regionMat.rows == binROIMat.rows+2);
  
  Mat distMat = regionMat.clone();
  distMat = Scalar(0);
  
  // Copy values from roi into regionMat taking +1 -1 into account so that
  // distance values have a black border around them
  
  Rect borderedROI(1, 1, roi.width, roi.height);
  
  Mat regionCopyROIMat = regionMat(borderedROI);
  
  assert(regionCopyROIMat.size() == binROIMat.size());
  
  binROIMat.copyTo(regionCopyROIMat);
  
  if (debug) {
    cout << "copied ( " << regionCopyROIMat.cols << " x " << regionCopyROIMat.rows << " )" << endl;
    cout << "into ( " << borderedROI.width << " x " << borderedROI.height << " ) at off " << borderedROI.x << "," << borderedROI.y << endl;
  }
  
  // Dump ROI mat
  
  if (debugDumpAllImages) {
    std::ostringstream stringStream;
    stringStream << "superpixel_roi_" << tag << ".png";
    std::string str = stringStream.str();
    const char *filename = str.c_str();
    
    cout << "write " << filename << " ( " << regionMat.cols << " x " << regionMat.rows << " )" << endl;
    imwrite(filename, regionMat);
  }
  
  // Run distance transform
  
  // Get an upper radius bound, basically this is the largest
  // value that a distance could be for this specific bbox
  // with a little extra floting point range to adjust for
  // the case where the number is an exact integer and the
  // computed distance has a rounding error.
  
  // FIXME: Distance transform should support a non-linear distance metric, so that
  // the caller can pass in a function to do the scaling so that (0,1,2,3) and (254, 255)
  // could be outside the linear scaling. The edges are much more critical than the
  // ranges in the middle. For example, a total distance range of 500 could scale a
  // like a double side log or a single linear stretch from (4, 254). The problem with
  // the current code is that a very large region could round what should be a 1 edge
  // down to zero.
  
  double radius = round(hypot(regionMat.cols * 0.5, regionMat.rows * 0.5) + 0.5) + 0.01;
  
  if (debug) {
    cout << "calc radius as " << radius << " for " << " ( " << regionMat.cols << " x " << regionMat.rows << " )" << endl;
  }
  
  // Read from regionMat, write distance transformed pixels to distMat
  
  vf::DistanceTransform::WhiteTest whiteTest(regionMat);
  vf::DistanceTransform::OutputDistancePixels distMatOut(distMat, radius);
  
  // Use fast ManhattanMetric for distance transform since it seems
  // to provide better region center estimation than ChessMetric
  // and it is faster than EuclideanMetric.
  
  //vf::DistanceTransform::Meijster::EuclideanMetric metric;
  vf::DistanceTransform::Meijster::ManhattanMetric metric;
  
  vf::DistanceTransform::Meijster::calculate(distMatOut, whiteTest, distMat.cols, distMat.rows, metric);
  
  if (debugDumpAllImages) {
    std::ostringstream stringStream;
    stringStream << "superpixel_dist_" << tag << ".png";
    std::string str = stringStream.str();
    const char *filename = str.c_str();
    
    cout << "write " << filename << " ( " << distMat.cols << " x " << distMat.rows << " )" << endl;
    imwrite(filename, distMat);
  }
  
  // Check the distance transform matrix here, the number of non-zero values must
  // match the number of white pixels in regionMat. If any funky rounding issues
  // cause fewer non-zero pixels to appear then the distance transform is not
  // working as expected since white pixels should have a min dist value of 1.
  
#if defined(DEBUG)
  
  assert(regionMat.size() == distMat.size());
  
  int regionMatNumNonZero = 0;
  
  for(int y = 0; y < regionMat.rows; y++) {
    for(int x = 0; x < regionMat.cols; x++) {
      uint8_t val = regionMat.at<uint8_t>(y, x);
      if (val > 0) {
        regionMatNumNonZero++;
      }
    }
  }
  
  int distMatNumNonZero = 0;
  
  for(int y = 0; y < distMat.rows; y++) {
    for(int x = 0; x < distMat.cols; x++) {
      uint8_t val = distMat.at<uint8_t>(y, x);
      if (val > 0) {
        distMatNumNonZero++;
      }
    }
  }
  
  assert(regionMatNumNonZero == distMatNumNonZero);
  
#endif // DEBUG
  
  // Normalize so that the largest value in the distance transform mat becomes 255
  
  normalize(distMat, distMat, 0, 255.0, NORM_MINMAX);
  
  if (debugDumpAllImages) {
    std::ostringstream stringStream;
    stringStream << "superpixel_dist_normalized_" << tag << ".png";
    std::string str = stringStream.str();
    const char *filename = str.c_str();
    
    cout << "write " << filename << " ( " << distMat.cols << " x " << distMat.rows << " )" << endl;
    imwrite(filename, distMat);
  }
  
  // Threshold so that only those pixels with the value 255 are left and save into regionMat
  // This threshold does not change the valus in distMat since those will be returned to
  // the caller of this function.
  
  threshold(distMat, regionMat, 254.0, 255.0, THRESH_BINARY);
  
  if (debugDumpAllImages) {
    std::ostringstream stringStream;
    stringStream << "superpixel_mask_dist_thresh_max_" << tag << ".png";
    std::string str = stringStream.str();
    const char *filename = str.c_str();
    
    cout << "write " << filename << " ( " << regionMat.cols << " x " << regionMat.rows << " )" << endl;
    imwrite(filename, regionMat);
  }
  
  // If the distance transform returns more than 1 pixel with the maximum
  // threshold value then simply iterate from the top left of the ROI
  // to find the first non-zero value as use that as the center.
  
  int nonZero = countNonZero(regionMat);
  
  if (debug) {
    cout << "found " << nonZero << " non-zero pixels with max thresh" << endl;
  }
  
  assert(nonZero > 0);
  
  if (0 && (nonZero > 2)) {
    // Find bbox of all the pixels still on and then choose the pixel that
    // is the closest to the center.
    
    int maxY = regionMat.rows;
    int maxX = regionMat.cols;
    
    vector<Coord> coords;
    
    for(int y = 0; y < maxY; y++) {
      for(int x = 0; x < maxX; x++) {
        uint8_t val = regionMat.at<uint8_t>(y, x);
        if (val > 0) {
          Coord coord(x, y);
          coords.push_back(coord);
          
          if (debug) {
            cout << "non-zero coord (" << x << "," << y << ")" << endl;
          }
        }
      }
    }
    
    int32_t originX, originY, width, height;
    
    Superpixel::bbox(originX, originY, width, height, coords);
    
    // Calculate bbox center coordinate
    
    Point2i bboxCenter = Point2i(originX + width/2, originY + height/2);
    Point2i closestToCenter(-1, -1);
    
    // Calculate distance to bbox center
    
    double minDist = 0xFFFF;
    
    for ( Coord coord : coords ) {
      int x = coord.x;
      int y = coord.y;
      
      float dist = hypot(float(bboxCenter.x - x), float(bboxCenter.y - y));
      
      if (debug) {
        cout << "dist from (" << x << "," << y << ") to bbox center (" << bboxCenter.x << "," << bboxCenter.y << ") is " << dist << endl;
      }
      
      if (dist < minDist) {
        closestToCenter = Point2i(x, y);
        minDist = dist;
        
        if (debug) {
          cout << "new min" << endl;
        }
        
        if (minDist == 0.0) {
          // Quite the loop once a zero distance has been found
          break;
        }
      }
    }
    
    assert(closestToCenter.x != -1);
    assert(closestToCenter.y != -1);
    
    // Have point out of N possible points that is closest to the bbox center
    
    center = closestToCenter;
  } else if (nonZero > 2) {
    // Calculate center of mass in terms of the pixels that are on
    // and the choose an actualy starting coordinate that is closest
    // to the center coord.
    
    int32_t cX = 0;
    int32_t cY = 0;
    int32_t N = 0;
    
    for (int y = 0; y < regionMat.rows; y++) {
      for (int x = 0; x < regionMat.cols; x++) {
        uint8_t val = regionMat.at<uint8_t>(y, x);
        if (val > 0) {
          cX += x;
          cY += y;
          N += 1;
        }
      }
    }
    
    cX = cX / N;
    cY = cY / N;
    
    // If this center of mass coordinate is on, then use it now
    
    if (debug) {
      cout << "center of mass calculated as (" << cX << "," << cY << ")" << endl;
    }
    
    uint8_t isOn = regionMat.at<uint8_t>(cY, cX);
    
    if (isOn) {
      center = Point2i(cX, cY);
    } else {
      // Center of mass coordinate is not actually on, could be a shape
      // with a hole in it. Scan to find the coordinate with the smallest
      // distance as compared to this center point.
      
      vector<Coord> coords;
      
      for (int y = 0; y < regionMat.rows; y++) {
        for (int x = 0; x < regionMat.cols; x++) {
          uint8_t val = regionMat.at<uint8_t>(y, x);
          if (val > 0) {
            Coord coord(x, y);
            coords.push_back(coord);
            
            if (debug) {
              cout << "non-zero coord (" << x << "," << y << ")" << endl;
            }
          }
        }
      }
      
      Point2i closestToCenter(-1, -1);
      
      // Calculate distance to bbox center
      
      float minDist = 0xFFFF;
      
      if (debug) {
        cout << "min dist " << minDist << endl;
      }
      
      for ( Coord coord : coords ) {
        int32_t x = coord.x;
        int32_t y = coord.y;
        
        float dist = hypot(float(cX - x), float(cY - y));
        
        if (debug) {
          cout << "dist from (" << x << "," << y << ") to center of mass (" << cX << "," << cY << ") is " << dist << endl;
        }
        
        if (dist < minDist) {
          closestToCenter = Point2i(x, y);
          minDist = dist;
          
          if (debug) {
            cout << "new min" << endl;
          }
          
          if (minDist == 0.0) {
            // Quite the loop once a zero distance has been found
            break;
          }
        }
      }
      
      assert(closestToCenter.x != -1);
      assert(closestToCenter.y != -1);
      
      // Have point out of N possible points that is closest to the bbox center
      
      center = closestToCenter;
    }
  } else {
    // Iterate over all binary values and save the first on pixel if 1 or 2 to choose from
    
    int maxY = regionMat.rows;
    int maxX = regionMat.cols;
    
    for(int y = 0; y < maxY; y++) {
      for(int x = 0; x < maxX; x++) {
        uint8_t val = regionMat.at<uint8_t>(y, x);
        if (val > 0) {
          center = Point2i(x, y);
          break;
        }
      }
    }
    assert(center.x != -1);
  }
  
  if (debug) {
    cout << "found center point (" << center.x << "," << center.y << ") in ROI region of size " << regionMat.rows << " x " << regionMat.cols << endl;
  }
  
  // The center point should not be on the border
  
  assert(center.x != 0 && center.x != (regionMat.cols-1));
  assert(center.y != 0 && center.y != (regionMat.rows-1));
  
  center.x -= 1;
  center.y -= 1;
  
  // Adjust
  
  // Dump center point image as BGR image over original size image
  
  if (debugDumpAllImages) {
    Mat colorMat(binMat.size(), CV_8UC3);
    cvtColor(binMat, colorMat, COLOR_GRAY2BGR);
    
    Point2i notROICenter(center.x + roi.x, center.y + roi.y);
    
    if (debug) {
      cout << "roi (" << roi.x << "," << roi.y << ") " << roi.width << " x " << roi.height << " in region " << binMat.cols << " x " << binMat.rows << endl;
      
      cout << "will write non-ROI center point (" << notROICenter.x << "," << notROICenter.y << ") in size " << colorMat.rows << " x " << colorMat.cols << endl;
    }
    
    colorMat.at<Vec3b>(notROICenter) = Vec3b(0, 0, 0xFF);
    
    std::ostringstream stringStream;
    stringStream << "superpixel_mask_center_" << tag << ".png";
    std::string str = stringStream.str();
    const char *filename = str.c_str();
    
    cout << "write " << filename << " ( " << colorMat.cols << " x " << colorMat.rows << " )" << endl;
    imwrite(filename, colorMat);
  }
  
  // Copy the dist values back into distMat taking the border into account
  
  Mat distCopyROIMat = distMat(borderedROI);
  
  assert(distCopyROIMat.size() == outDistMat.size());
  
  distCopyROIMat.copyTo(outDistMat);
  
  if (debug) {
    cout << "copy ROI from dist buffer of size " << borderedROI.width << " x " << borderedROI.height << endl;
    cout << "copy ROI into out dist buffer of size " << distCopyROIMat.cols << " x " << distCopyROIMat.rows << endl;
  }
  
  // Verify that the region center is in the ROI region
  
  assert(center.x >= 0);
  assert(center.x < distCopyROIMat.cols);
  assert(center.y >= 0);
  assert(center.y < distCopyROIMat.rows);
  
  Coord centerPair(center.x, center.y);
  return centerPair;
}

// Given an input binary Mat (0x0 or 0xFF) perform a dilate() operation that will expand
// the white region inside a black region. This makes use of a circular operator and
// an expansion size indicated by the caller.

Mat expandWhiteInRegion(Mat &binMat, int expandNumPixelsSize, int tag)
{
  assert(binMat.channels() == 1);
  
  Mat outBinMat = binMat.clone();
 
  int dilation_type = MORPH_ELLIPSE;
  int dilation_size = expandNumPixelsSize;
  
  Mat element = getStructuringElement( dilation_type,
                                      Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                      Point( dilation_size, dilation_size ) );
  
  dilate( binMat, outBinMat, element );
  
  return outBinMat;
}

// Given a superpixel tag that indicates a region segmented into 4x4 squares
// map (X,Y) coordinates to a minimized Mat representation that can be
// quickly morphed with minimal CPU and memory usage.

Mat expandBlockRegion(int32_t tag,
                      const vector<Coord> &coords,
                      int expandNum,
                      int blockWidth, int blockHeight,
                      int superpixelDim)
{
  const bool debug = true;
  const bool debugDumpImages = true;
  
  Mat morphBlockMat = Mat(blockHeight, blockWidth, CV_8U);
  morphBlockMat = (Scalar) 0;
  
  // Iterate over input coords and calcualte block coord to activate
  
//  int lastOffset = -1;
  
  for ( Coord c : coords ) {
    // Convert (X,Y) to block (X,Y)
    
    int blockX = c.x / superpixelDim;
    int blockY = c.y / superpixelDim;
    
    if (debug) {
      cout << "block with tag " << tag << " cooresponds to (X,Y) (" << c.x << "," << c.y << ")" << endl;
      cout << "maps to block (X,Y) (" << blockX << "," << blockY << ")" << endl;
    }
    
    // FIXME: optimize for case where (X,Y) is exactly the same as in the previous iteration and avoid
    // writing to the Mat in that case. This shift is cheap.
    
    morphBlockMat.at<uint8_t>(blockY, blockX) = 0xFF;
  }
  
  Mat expandedBlockMat;
      
  for (int expandStep = 0; expandStep <= expandNum; expandStep++ ) {
    if (expandStep == 0) {
      expandedBlockMat = morphBlockMat;
    } else {
      expandedBlockMat = expandWhiteInRegion(expandedBlockMat, 1, tag);
    }
    
    int nzc = countNonZero(expandedBlockMat);
    
    if (nzc == (blockHeight * blockWidth)) {
      if (debug) {
      cout << "all pixels in Mat now white " << endl;
      }
      break;
    }
    
    if (debugDumpImages) {
      std::stringstream fnameStream;
      fnameStream << "srm" << "_tag_" << tag << "_morph_block_" << expandStep << ".png";
      string fname = fnameStream.str();
      
      imwrite(fname, expandedBlockMat);
      cout << "wrote " << fname << endl;
    }
    
  } // for expandStep
  
  return expandedBlockMat;
}

// Given a Mat that contains pixels count each pixel and return a histogram
// of the number of times each pixel is found in the image.

void generatePixelHistogram(const Mat & inQuantPixels,
                            unordered_map<uint32_t, uint32_t> &pixelToCountTable)
{
  const bool debugOutput = false;
  
  Mat quantOutputMat = inQuantPixels.clone();
  quantOutputMat = (Scalar) 0;
  
  if (inQuantPixels.channels() == 3) {
    for(int y = 0; y < quantOutputMat.rows; y++) {
      for(int x = 0; x < quantOutputMat.cols; x++) {
        Vec3b vec = inQuantPixels.at<Vec3b>(y, x);
        uint32_t pixel = Vec3BToUID(vec);
        pixel &= 0x00FFFFFF; // Opaque 24BPP
        
        if ((debugOutput)) {
          char buffer[1024];
          snprintf(buffer, sizeof(buffer), "for (%4d,%4d) pixel is 0x%08X (%d) -> count %d", x, y, pixel, pixel, pixelToCountTable[pixel]);
          cout << buffer << endl;
        }
        
        pixelToCountTable[pixel] += 1;
      }
    }
  } else if (inQuantPixels.channels() == 4) {
    for(int y = 0; y < quantOutputMat.rows; y++) {
      for(int x = 0; x < quantOutputMat.cols; x++) {
        Vec4b vec = inQuantPixels.at<Vec4b>(y, x);
        uint32_t pixel = Vec4BToPixel(vec);
        
        if ((debugOutput)) {
          char buffer[1024];
          snprintf(buffer, sizeof(buffer), "for (%4d,%4d) pixel is 0x%08X (%d) -> count %d", x, y, pixel, pixel, pixelToCountTable[pixel]);
          cout << buffer << endl;
        }
        
        pixelToCountTable[pixel] += 1;
      }
    }
    
  } else {
    assert(0);
  }
  
  return;
}

// Given a Mat that contains quant pixels and a colortable, map the quant
// pixels to indexes in the colortable. If the asGreyscale) flag is true
// then each index is assumed to be a byte and is written as a greyscale pixel.

Mat mapQuantPixelsToColortableIndexes(const Mat & inQuantPixels, const vector<uint32_t> &colortable, bool asGreyscale)
{
  const bool debugOutput = false;
  
  // Map pixels to sorted colortable offset
  
  unordered_map<uint32_t, uint32_t> pixel_to_sorted_offset;
  
  for (int i = 0; i < colortable.size(); i++) {
    uint32_t pixel = colortable[i];
    pixel &= 0x00FFFFFF; // Opaque 24BPP
    pixel_to_sorted_offset[pixel] = i;
#if defined(DEBUG)
    assert(pixel_to_sorted_offset[pixel] <= 0xFF);
    assert(pixel_to_sorted_offset[pixel] == i);
#endif // DEBUG
    
    if ((debugOutput)) {
      char buffer[1024];
      snprintf(buffer, sizeof(buffer), "colortable[%4d] = 0x%08X", i, pixel);
      cout << buffer << endl;
    }
  }
  
  Mat quantOutputMat = inQuantPixels.clone();
  quantOutputMat = (Scalar) 0;
  
  for(int y = 0; y < quantOutputMat.rows; y++) {
    for(int x = 0; x < quantOutputMat.cols; x++) {
      Vec3b vec = inQuantPixels.at<Vec3b>(y, x);
      uint32_t pixel = Vec3BToUID(vec);
      pixel &= 0x00FFFFFF; // Opaque 24BPP
      
      auto it = pixel_to_sorted_offset.find(pixel);
      
      if (it == pixel_to_sorted_offset.end()) {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "for (%4d,%4d) pixel is 0x%08X (%d) but this pixel has no matching colortable entry \n", x, y, pixel, pixel);
        cerr << buffer;
        assert(0);
      }
      
      uint32_t offset = it->second;
      
      if ((debugOutput)) {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "for (%4d,%4d) pixel is 0x%08X (%d) -> offset %d", x, y, pixel, pixel, offset);
        cout << buffer << endl;
      }
      
      if (asGreyscale) {
        assert(offset < 256);
        uint32_t grayscalePixel = (offset << 16) | (offset << 8) | offset;
        vec = PixelToVec3b(grayscalePixel);
      } else {
        vec = PixelToVec3b(offset);
      }
      quantOutputMat.at<Vec3b>(y, x) = vec;
    }
  }
  
  return quantOutputMat;
}

