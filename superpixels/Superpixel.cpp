// This class represents a collection of pixels from an image that is defined such that each pixel is touching
// one of the other pixels in the superpixel.

#include "Superpixel.h"

#include "Util.h"

#include <opencv2/opencv.hpp>

#include "OpenCVUtil.h"

Superpixel::Superpixel()
:tag(0), assocDataPtr(NULL), flags(0)
{
  ;
}

Superpixel::Superpixel(int32_t tag)
{
  this->tag = tag;
  this->assocDataPtr = NULL;
  this->flags = 0;
}

Superpixel::~Superpixel()
{
#if defined(ENABLE_SUPERPIXEL_ASSOC_DATA)
  if (this->assocDataPtr) {
    // Release any pointers inside this map assuming that each object
    // has been allocated with delete.
    
    for ( auto iter = assocDataPtr->begin(); iter != assocDataPtr->end(); ++iter) {
      //uint32_t key = iter->first;
      void *valueObj = iter->second;
      if (valueObj) {
        // Note that this delete will release the memory but it should not
        // be used with objects that require special destruction since the
        // destrutor might not be called.
        //delete [] valueObj;
        ::operator delete(valueObj);
      }
    }
    
    delete assocDataPtr;
    assocDataPtr = NULL;
  }
#endif // ENABLE_SUPERPIXEL_ASSOC_DATA
}

void Superpixel::appendCoord(int x, int y)
{
#if defined(DEBUG)
  assert(x >= 0);
  assert(y >= 0);
#endif // DEBUG
  
  Coord coord(x, y);
  coords.push_back(coord);
}

// Read RGB values from larger input image and create a matrix that is the width
// of the superpixel and contains just the pixels defined by the coordinates
// contained in the superpixel. The caller passes in the tag from the superpixel
// in question in order to find the coords.

void Superpixel::fillMatrixFromCoords(Mat &input, int32_t tag, Mat &output) {
  const bool debug = false;
  
  auto &coords = this->coords;
  
  if (debug) {
    int numCoords = (int) coords.size();
    cout << "filling matrix from superpixel " << tag << " with coords N=" << numCoords << endl;
  }
  
  fillMatrixFromCoords(input, coords, output);
}

void Superpixel::fillMatrixFromCoords(Mat &input, vector<Coord> &coords, Mat &output) {
  const bool debug = false;
  
  int numCoords = (int) coords.size();
  
  output.create(1, numCoords, CV_8UC(3));
  
  output = Scalar(0, 0, 0);
  
  int i = 0;
  
  for (auto coordsIter = coords.begin(); coordsIter != coords.end(); ++coordsIter, i++) {
    Coord coord = *coordsIter;
    int32_t X = coord.x;
    int32_t Y = coord.y;
    
    Vec3b pixelVec = input.at<Vec3b>(Y, X);
    output.at<Vec3b>(0, i) = pixelVec;
    
    if (debug) {
      // Print BGRA format
      
      int32_t pixel = Vec3BToUID(pixelVec);
      char buffer[6+1];
      snprintf(buffer, 6+1, "%06X", pixel);
      
      char ibuf[3+1];
      snprintf(ibuf, 3+1, "%03d", i);
      
      cout << "pixel copy from X,Y (" << X << "," << Y << ") i = " << ibuf << " 0xFF" << (char*)&buffer[0] << endl;
    }
  }
}

// This method is the inverse of fillMatrixFromCoords(), it reads pixel values from a matrix
// and writes them back to the corresponding X,Y values location in an image. This method is
// very useful when running an image operation on all the pixels in a superpixel but without
// having to process all the pixels in a bbox area. The dimensions of the input must be
// NUM_COORDS x 1. The caller must init the matrix values and the matrix size. This method
// can be invoked multiple times to write multiple superpixel values to the same output
// image.

void Superpixel::reverseFillMatrixFromCoords(Mat &input, bool isGray, int32_t tag, Mat &output) {
  const bool debug = false;
  
  auto &coords = this->coords;
  
  if (debug) {
    int numCoords = (int) coords.size();
    cout << "reverse filling matrix from superpixel " << tag << " with coords N=" << numCoords << endl;
  }
  
  reverseFillMatrixFromCoords(input, isGray, coords, output);
}

void Superpixel::reverseFillMatrixFromCoords(Mat &input, bool isGray, vector<Coord> &coords, Mat &output) {
  const bool debug = false;
  
  int numCoords = (int) coords.size();
  
  assert(input.rows == 1);
  assert(input.cols == numCoords);
  
  int numChannels = output.channels();
  bool writeGrayscale = (numChannels == 1);
  
  if (writeGrayscale) {
    assert(isGray);
  }
  
  int i = 0;
  
  for (auto coordsIter = coords.begin(); coordsIter != coords.end(); ++coordsIter, i++) {
    Coord coord = *coordsIter;
    int32_t X = coord.x;
    int32_t Y = coord.y;
    
    Vec3b pixelVec;
    
    if (isGray) {
      uint8_t gray = input.at<uint8_t>(0, i);
      pixelVec[0] = gray;
      pixelVec[1] = gray;
      pixelVec[2] = gray;
    } else {
      pixelVec = input.at<Vec3b>(0, i);
    }
    
    if (writeGrayscale) {
      output.at<uint8_t>(Y, X) = pixelVec[0];
    } else {
      output.at<Vec3b>(Y, X) = pixelVec;
    }
    
    if (debug) {
      // Print BGRA format
      
      int32_t pixel = Vec3BToUID(pixelVec);
      char buffer[6+1];
      snprintf(buffer, 6+1, "%06X", pixel);
      
      char ibuf[3+1];
      snprintf(ibuf, 3+1, "%03d", i);
      
      cout << "pixel copy to X,Y (" << X << "," << Y << ") i = " << ibuf << " 0xFF" << (char*)&buffer[0] << endl;
    }
  }
}

// Find bounding box of a superpixel. This is the (X,Y) of the upper right corner and the width and height.

void
Superpixel::bbox(int32_t &originX, int32_t &originY, int32_t &width, int32_t &height)
{
  ::bbox(originX, originY, width, height, this->coords);
}

// Filter the coords and return a vector that contains only the coordinates that share
// an edge with the other superpixel.

void
Superpixel::filterEdgeCoords(Superpixel *superpixe1Ptr,
                             vector<Coord> &edgeCoords1,
                             Superpixel *superpixe2Ptr,
                             vector<Coord> &edgeCoords2)
{
  const bool debug = false;
  
  edgeCoords1.clear();
  edgeCoords2.clear();
  
  // Choose the smaller of the two superpixels by bbox and then process all the coordinates
  // from the smaller superpixel checking against the coordinates of the larger superpixel.
  
  int32_t thisSuperpixelOriginX, thisSuperpixelOriginY, thisSuperpixelWidth, thisSuperpixelHeight;
  int32_t otherSuperpixelOriginX, otherSuperpixelOriginY, otherSuperpixelWidth, otherSuperpixelHeight;

  superpixe1Ptr->bbox(thisSuperpixelOriginX, thisSuperpixelOriginY, thisSuperpixelWidth, thisSuperpixelHeight);
  superpixe2Ptr->bbox(otherSuperpixelOriginX, otherSuperpixelOriginY, otherSuperpixelWidth, otherSuperpixelHeight);

  int32_t thisSuperpixelBboxNumCoords = thisSuperpixelWidth * thisSuperpixelHeight;
  int32_t otherSuperpixelBboxNumCoords = otherSuperpixelWidth * otherSuperpixelHeight;
  
  // Generate a target bbox area that is the bbox of the smaller superpixel but x-1,y-1 and w+1,h+1
  // so that neighbors can be calculated.
  
  int32_t bothSuperpixelOriginX, bothSuperpixelOriginY, bothSuperpixelWidth, bothSuperpixelHeight;
  
  Superpixel *smallerPtr;
  Superpixel *largerPtr;
  
  if (thisSuperpixelBboxNumCoords < otherSuperpixelBboxNumCoords) {
    // This superpixel is smaller than the other one
    
    if (debug) {
      cout << "This superpixel is smaller than the other one : " << thisSuperpixelBboxNumCoords << " < " << otherSuperpixelBboxNumCoords << endl;
    }
    
    smallerPtr = superpixe1Ptr;
    largerPtr = superpixe2Ptr;
    
    bothSuperpixelOriginX = thisSuperpixelOriginX;
    bothSuperpixelOriginY = thisSuperpixelOriginY;
    bothSuperpixelWidth = thisSuperpixelWidth;
    bothSuperpixelHeight = thisSuperpixelHeight;
  } else {
    // Other superpixel is smaller than this one
    
    if (debug) {
      cout << "Other superpixel is smaller than this one : " << thisSuperpixelBboxNumCoords << " >= " << otherSuperpixelBboxNumCoords << endl;
    }
    
    smallerPtr = superpixe2Ptr;
    largerPtr = superpixe1Ptr;
    
    bothSuperpixelOriginX = otherSuperpixelOriginX;
    bothSuperpixelOriginY = otherSuperpixelOriginY;
    bothSuperpixelWidth = otherSuperpixelWidth;
    bothSuperpixelHeight = otherSuperpixelHeight;
  }
  
  if (debug) {
    cout << "min size bbox " << bothSuperpixelOriginX << "," << bothSuperpixelOriginY << " " << bothSuperpixelWidth << " x " << bothSuperpixelHeight << endl;
  }
  
  if (bothSuperpixelOriginX > 0) {
    bothSuperpixelOriginX -= 1;
  }
  if (bothSuperpixelOriginY > 0) {
    bothSuperpixelOriginY -= 1;
  }
  
  bothSuperpixelWidth += 2;
  bothSuperpixelHeight += 2;
  
  if (debug) {
    cout << "adj min size bbox " << bothSuperpixelOriginX << "," << bothSuperpixelOriginY << " " << bothSuperpixelWidth << " x " << bothSuperpixelHeight << endl;
  }
  
  // Adjust the coordinates of each set of coords to account for the origin X,Y
  
  vector<Coord> adjSmallerSuperpixelCoords;
  vector<Coord> adjLargerSuperpixelCoords;
  
  for (auto coordsIter = smallerPtr->coords.begin(); coordsIter != smallerPtr->coords.end(); ++coordsIter) {
    Coord coord = *coordsIter;
    
    if (debug) {
      cout << "smaller coord " << coord.x << "," << coord.y << endl;
    }
    
    int32_t adjMinX = coord.x - bothSuperpixelOriginX;
    int32_t adjMinY = coord.y - bothSuperpixelOriginY;
    
    assert(adjMinX >= 0);
    assert(adjMinY >= 0);
    
    Coord adjCoord(adjMinX, adjMinY);
    adjSmallerSuperpixelCoords.push_back(adjCoord);
  }
  
  // Adjust coordinates of larger superpixel, this logic must filter out
  // and coordinates outside of the bbox.

  for (auto coordsIter = largerPtr->coords.begin(); coordsIter != largerPtr->coords.end(); ++coordsIter) {
    Coord coord = *coordsIter;
    
    if (debug) {
      cout << "larger coord " << coord.x << "," << coord.y << endl;
    }
    
    bool skip = true;
    
    if (coord.x < bothSuperpixelOriginX) {
      // Skip
    } else if (coord.y < bothSuperpixelOriginY) {
      // Skip
    } else if (coord.x >= (bothSuperpixelOriginX + bothSuperpixelWidth)) {
      // Skip
    } else if (coord.y >= (bothSuperpixelOriginY + bothSuperpixelHeight)) {
      // Skip
    } else {
      if (debug) {
        cout << "keep larger superpixel coordinate " << coord.x << "," << coord.y << endl;
      }
      
      skip = false;
      
      int32_t adjMinX = coord.x - bothSuperpixelOriginX;
      int32_t adjMinY = coord.y - bothSuperpixelOriginY;
      assert(adjMinX >= 0);
      assert(adjMinY >= 0);
      
      Coord adjCoord(adjMinX, adjMinY);
      adjLargerSuperpixelCoords.push_back(adjCoord);
    }
    
    if (debug) {
      if (skip) {
        cout << "skipped larger superpixel coordinate " << coord.x << "," << coord.y << endl;
      }
    }
  }
  
  // Set UID values for each X,Y in smaller bbox
  
  Mat bboxTags(bothSuperpixelHeight, bothSuperpixelWidth, CV_8UC(3), Scalar(0,0,0));
  
  assert(smallerPtr->tag != 0);
  assert(largerPtr->tag != 0);
  
  assert((adjSmallerSuperpixelCoords.size() + adjLargerSuperpixelCoords.size()) <= (bothSuperpixelHeight * bothSuperpixelWidth));

  int numCoords;
  
  numCoords = (int) adjSmallerSuperpixelCoords.size();
  Mat smallerSuperpixelMat(1, numCoords, CV_8UC(3));
  smallerSuperpixelMat = UIDToScalar(smallerPtr->tag);
  reverseFillMatrixFromCoords(smallerSuperpixelMat, false, adjSmallerSuperpixelCoords, bboxTags);

  numCoords = (int) adjLargerSuperpixelCoords.size();
  Mat largerSuperpixelMat(1, numCoords, CV_8UC(3));
  largerSuperpixelMat = UIDToScalar(largerPtr->tag);
  reverseFillMatrixFromCoords(largerSuperpixelMat, false, adjLargerSuperpixelCoords, bboxTags);

//  cout << bboxTags << endl;
  
  // Scan over all X,Y values in Mat and check for neighbor that is the other superpixel
  
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
  
  for( int y = 0; y < bboxTags.rows; y++ ) {
    for( int x = 0; x < bboxTags.cols; x++ ) {
      Vec3b tagVec = bboxTags.at<Vec3b>(y, x);
      
      int32_t centerTag = Vec3BToUID(tagVec);
      
      if (debug) {
        cout << "center (" << x << "," << y << ") with tag " << centerTag << endl;
      }
      
      // Loop over each neighbor around (X,Y) and lookup tag
      
      for (auto pairIter = neighborOffsets.begin() ; pairIter != neighborOffsets.end(); ++pairIter) {
        int dX = pairIter->first;
        int dY = pairIter->second;
        
        int foundNeighborUID;
        
        int nX = x + dX;
        int nY = y + dY;
        
        if (nX < 0 || nX >= bboxTags.cols) {
          foundNeighborUID = -1;
        } else if (nY < 0 || nY >= bboxTags.rows) {
          foundNeighborUID = -1;
        } else {
          Vec3b neighborTagVec = bboxTags.at<Vec3b>(nY, nX);
          foundNeighborUID = Vec3BToUID(neighborTagVec);
        }
        
        if (foundNeighborUID == -1 || foundNeighborUID == centerTag) {
          if (debug) {
            cout << "ignoring (" << nX << "," << nY << ") with tag " << foundNeighborUID << " since invalid or identity" << endl;
          }
        } else {
          if (debug) {
            cout << "checking (" << nX << "," << nY << ") with tag " << foundNeighborUID << " to see if edge coord" << endl;
          }
          
          if ((centerTag == smallerPtr->tag && foundNeighborUID == largerPtr->tag) ||
              (centerTag == largerPtr->tag && foundNeighborUID == smallerPtr->tag)) {
            
            if (debug) {
              cout << "found edge coord at center (" << x << "," << y << ") with tag " << centerTag << endl;
            }
            
            Coord centerCoord(bothSuperpixelOriginX + x, bothSuperpixelOriginY + y);
            
            if (centerTag == superpixe1Ptr->tag) {
              // Center tag corresponds to edgeCoords1
              edgeCoords1.push_back(centerCoord);
            } else {
              // Center tag corresponds to edgeCoords2
              edgeCoords2.push_back(centerCoord);
            }
            break; // exit neighbor loop once coord is known to be an edge
          } else {
            if (debug) {
              cout << "ignoring (" << nX << "," << nY << ") with tag " << foundNeighborUID << " since unknown" << endl;
            }
          }
        }
      }
    }
  }
  
  return;
}

// This logic executes the heavy process of scanning the initial input tags to remove splay
// pixels Seeds generated superpixels. This process is really just a workaround for what
// appears to be a buggy edge case in the Seeds algo.

void Superpixel::splitSplayPixels(Mat &inOutTagImg)
{
  const bool debug = false;

  Mat mask(inOutTagImg.rows+2, inOutTagImg.cols+2, CV_8UC1);
  Mat croppedMask(inOutTagImg.rows, inOutTagImg.cols, CV_8UC1);
  
  Rect filledRect;

  Rect borderRect(0, 0, inOutTagImg.cols+2, inOutTagImg.rows+2);
  Rect maskROI(1, 1, inOutTagImg.cols, inOutTagImg.rows);

  Scalar scalarZero(0);
  Scalar maskFillColor(255);
  
  int connectivity = 8;
  
  int flags = connectivity | FLOODFILL_FIXED_RANGE | FLOODFILL_MASK_ONLY;
  
  Point seed(0, 0);
  
  // Neighbor iter
  
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
  
  // The seen table will contain an entry for each superpixel tag that has already been
  // processed. When a new superpixel tag is seen, all neighbor pixels will be flood filled
  // and as a result any duplicate pixels seen later on should have been processed by
  // the flood fill.
  
  unordered_map<int32_t,bool> seen;
  
  for( int y = 0; y < inOutTagImg.rows; y++ ) {
    for( int x = 0; x < inOutTagImg.cols; x++ ) {
      Vec3b tagVec = inOutTagImg.at<Vec3b>(y, x);
      
      int32_t centerTag = Vec3BToUID(tagVec);
      
      if (debug) {
        cout << "center (" << x << "," << y << ") with tag " << centerTag << endl;
      }
      
      // If this specific superpixel has not been processed yet then do a BFS
      // to find the bounds of the superpixel.
      
      if (seen.count(centerTag) > 0) {
        if (debug) {
          cout << "tag already processed " << centerTag << endl;
        }
        
        // Already seen so skip rest of processing
        continue;
      }
      
      // Flood fill into mask and determine which pixels are non-zero.
      
      mask = Scalar::all(0);
      rectangle(mask, borderRect, maskFillColor);
      
      seed.x = x;
      seed.y = y;
      
      if (debug) {
        cout << "seed (" << seed.x << "," << seed.y << ") " << endl;
      }
      
      if (debug) {
      imwrite("flood_tags_input.png", inOutTagImg);
      imwrite("flood_mask_input.png", mask);
      }
      
      int numFilled = floodFill(inOutTagImg, mask, seed, maskFillColor, &filledRect, scalarZero, scalarZero, flags);
      
      mask *= 255.0;
      
      // Fill must have at least filled 1 pixel
      
      assert(filledRect.width > 0);
      assert(filledRect.height > 0);

      if (debug) {
        cout << "flood fill bbox (" << filledRect.x << "," << filledRect.y << ") " << filledRect.width << " x " << filledRect.height << endl;
        cout << "numFilled " << numFilled << endl;
      }
      
      // Create an initial cropped ROI that does not include the 1 pixel border
      
      Mat croppedMask = mask(maskROI);
      
      // The filledRect identified pixels are now in terms of the cropped mask image
      
      if (debug) {
      Mat tagsCopy = inOutTagImg.clone();
      rectangle(tagsCopy, filledRect, Scalar(255,255,255));
      imwrite("flood_tags_rectangle_over.png", tagsCopy);
      
      tagsCopy = Scalar(0, 0, 0);
      rectangle(tagsCopy, filledRect, Scalar(255,255,255));
      imwrite("flood_rectangle_over.png", tagsCopy);
      
      imwrite("flood_tags.png", inOutTagImg);
      imwrite("flood_mask2.png", mask);
      imwrite("flood_mask.png", croppedMask);
      }
      
      // Iteration is defined in terms of checking for non-zero pixels in the range of the containing rect

      int maxY = filledRect.y + filledRect.height;
      int maxX = filledRect.x + filledRect.width;
      
      int countNonZero = 0;
      
      vector<pair<int32_t, int32_t> > likelyLineCoords;

      for( int32_t rY = filledRect.y; rY < maxY; rY++ ) {
        for( int32_t rX = filledRect.x; rX < maxX; rX++ ) {
          uint8_t bVal = croppedMask.at<uint8_t>(rY, rX);
          
          if (bVal == 0) {
            continue;
          }
          
          if (debug) {
            cout << "found non-zero pixel " << (int)bVal << " at " << "(" << rX << "," << rY << ")" << endl;
          }
          
          countNonZero += 1;
          
          uint8_t neighborsArr[9] =
          {
            0, 0, 0,
            0, 0, 0,
            0, 0, 0
          };
          int32_t neighborsArrOffset = 0;
          
          // Record any pixels that have just 2 non-zero neighbors. These will typically be straight lines.
          
          int numNeighbors = 0;
          
          // Loop over each neighbor around (X,Y) and lookup tag
          
          for (auto pairIter = neighborOffsets.begin() ; pairIter != neighborOffsets.end(); ++pairIter) {
            int dX = pairIter->first;
            int dY = pairIter->second;
            
            uint8_t neighborVal = 0;
            
            int nX = rX + dX;
            int nY = rY + dY;
            
            if (nX < 0 || nX >= maxX) {
              //neighborVal = 0;
            } else if (nY < 0 || nY >= maxY) {
              //neighborVal = 0;
            } else {
              neighborVal = croppedMask.at<uint8_t>(nY, nX);
            }
            
            neighborsArr[neighborsArrOffset++] = neighborVal;
            
            if (neighborsArrOffset == 4) {
              neighborsArrOffset++; // skip center
            }
            
            if (neighborVal != 0) {
              numNeighbors += 1;
              
              if (debug) {
                cout << "neighbor on  at (" << nX << "," << nY << ")" << endl;
              }
            } else {
              if (debug) {
                cout << "neighbor off at (" << nX << "," << nY << ")" << endl;
              }
            }
          } // end neighbor loop
          
          if (debug) {
            cout << "num neighbors for (" << rX << "," << rY << ") is " << numNeighbors << endl;
          }
          
          if (numFilled == 1) {
            assert(numNeighbors == 0);
          } else {
            assert(numNeighbors > 0);
          }
          
          assert(neighborsArr[4] == 0);
          
          // Detect a coord that is most likely to be either a horizontal or vertical line
          // by looking for N=2 neighbors and (LEFT and RIGHT) or (UP AND DOWN)
          
          if (numNeighbors == 2) {
            
            bool leftAndRight = false;
            bool upAndDown = false;
            
            if (neighborsArr[3] != 0 && neighborsArr[5] != 0) {
              leftAndRight = true;
            } else if (neighborsArr[1] != 0 && neighborsArr[7] != 0) {
              upAndDown = true;
            }
            
            if (leftAndRight || upAndDown) {
              pair<int32_t, int32_t> coord(rX, rY);
              likelyLineCoords.push_back(coord);
            }
          }
          
        }
      } // end rect iter loop
      
      if (debug) {
        cout << "countNonZero " << countNonZero << endl;
      }
      
      if (debug) {
        cout << "done iter looking for likely line coords" << endl;
        cout << "found " << likelyLineCoords.size() << endl;
      }
      
      // Long straight lines typically that SEEDS is running along a simple gradient.
      // Detect only the case of vertical or horizontal long connected lines and then
      // modify the original tags data to add a new tag to replace each pixel that is
      // currently in the straight line.
      
      // The coords were scanned by row and then col so the first one will be the
      // closest to the top left.
      
      seen[centerTag] = true;
    }
  }
  
  
  return;
}

// Return true if edge should be merged based on known edge weights

bool Superpixel::shouldMergeEdge(float edgeWeight)
{
  const bool debug = false;
 
  if (debug) {
    char buffer[1024];
    snprintf(buffer, sizeof(buffer), "shouldMergeEdge check edge weight %16.4f", edgeWeight);
    cout << (char*)buffer << endl;
  }
  
  if (edgeWeight <= 1.0f) {
    // Edge weights are calculated in terms of average LAB differences
    // so always merge 0.0 up to and including 1.0 since it is very
    // likely that values this close will merge.
    
    return true;
  }
  
  float mergedMean, mergedMeanStddev;
  
  sample_mean(mergedEdgeWeights, &mergedMean);
  sample_mean_delta_squared_div(mergedEdgeWeights, mergedMean, &mergedMeanStddev);
  
  float unMergedMean, unMergedMeanStddev;
  
  sample_mean(unmergedEdgeWeights, &unMergedMean);
  sample_mean_delta_squared_div(unmergedEdgeWeights, unMergedMean, &unMergedMeanStddev);
  
  if (debug) {
    char buffer[1024];
    snprintf(buffer, sizeof(buffer), "already merged mean %16.4f and stddev %16.4f", mergedMean, mergedMeanStddev);
    cout << (char*)buffer << endl;
    snprintf(buffer, sizeof(buffer), "un -    merged mean %16.4f and stddev %16.4f", unMergedMean, unMergedMeanStddev);
    cout << (char*)buffer << endl;
  }
  
  // Which of the two mean values is this one closer to?
  
  float distMerged = abs(edgeWeight - mergedMean);
  
  float distUnMerged;
  
  if (mergedMean == 0.0f && unMergedMean == 0.0f) {
    // In this init case no successful merges are done and there are no unmerged
    // edge weights to compare to. Allow the merge in this case since there are
    // no stats to compare to. This case should not be hit in normal use since
    // successful merges should have been seen before a strong edge.
    
    distUnMerged = 0xFFFFFFFF;
  } else if (mergedEdgeWeights.size() > 0 && unMergedMean == 0.0f) {
    // In this second case, at least one merge was successful but
    // no specific hard edge has been found. Use an order of magnitude
    // above the lower limit if larger than 10.0 in LAB delta.
    
    if (mergedMean < 5.0f) {
      unMergedMean = 50.0f;
    } else {
      unMergedMean = mergedMean * 10.0f;
    }
    distUnMerged = abs(edgeWeight - unMergedMean);
  } else {
    // There is at least 1 unmerged weight so calculate distance to it
    distUnMerged = abs(edgeWeight - unMergedMean);
  }
  
  if (debug) {
    char buffer[1024];
    snprintf(buffer, sizeof(buffer), "for edge weight  %16.4f\ndist to unmerged %16.4f\ndist to   merged %16.4f", edgeWeight, distUnMerged, distMerged);
    cout << (char*)buffer << endl;
  }
  
  if (distUnMerged < distMerged) {
    // Superpixel should not be merged since edge weight is nearer to unmerged values
    
    if (debug) {
      cout << "should merge returning false" << endl;
    }
    return false;
  } else {
    // Edge should be merged

    if (debug) {
      cout << "should merge returning true" << endl;
    }
    return true;
  }
}

