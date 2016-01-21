// This file contains utility functions for general purpose use and interaction with OpenCV

#include "Util.h"

#include <ostream>
#include <iostream>

using namespace std;

#define RSHIFT_MASK(num, shift, mask) ((num >> shift) & mask)

#define MASK_LSHIFT(num, mask, shift) ((num & mask) << shift)

#define BOOL int
#define FALSE 0
#define TRUE 1

// mean of N values

void sample_mean(vector<float> &values, float *meanPtr) {
  int len = (int) values.size();
  if (len == 0) {
    *meanPtr = 0.0f;
    return;
  } else if (len == 1) {
    *meanPtr = values[0];
    return;
  }
  
  float sum = 0.0f;
  for (auto it = values.begin(); it != values.end(); ++it) {
    float val = *it;
    sum += val;
  }
  if (sum == 0.0f) {
    *meanPtr = 0.0f;
  } else {
    *meanPtr = sum / len;
  }
}

// The caller must pass in the mean value calculated via sample_mean()

void sample_mean_delta_squared_div(vector<float> &values, float mean, float *stddevPtr) {
  int len = (int) values.size();
  if (len == 0 || len == 1) {
    *stddevPtr = 0.0f;
    return;
  }
  
  float sum = 0.0f;
  for (auto it = values.begin(); it != values.end(); ++it) {
    float value = *it;
    float delta = value - mean;
    sum += (delta * delta);
  }
  if (sum == 0.0f) {
    *stddevPtr = 0.0f;
  } else {
    *stddevPtr = sqrt(sum / (len - 1));
  }
}

// Calculate diffs between float values and return as a vector.
// Return list of (N - (N-1)) elements in the list where the first element
// is always (values[0] - 0.0)

vector<float>
float_diffs(vector<float> &values) {
  const bool debug = false;
  
  if (debug) {
    cout << "float_diffs() for values" << endl;
    for (auto it = values.begin(); it != values.end(); ++it) {
      float value = *it;
      cout << value << endl;
    }
  }
  
  float last = 0.0;
  
  vector<float> deltas;
  
  for (auto it = values.begin(); it != values.end(); ++it) {
    float value = *it;
    float delta = value - last;
    deltas.push_back(delta);
    last = value;
  }
  
  if (debug) {
    cout << "returning deltas" << endl;
    for (auto it = deltas.begin(); it != deltas.end(); ++it) {
      float delta = *it;
      cout << delta << endl;
    }
  }
  
  return deltas;
}

// Util method to return the 8 neighbors of a center point in the order
// R, U, L, D, UR, UL, DL, DR while taking the image bounds into
// account. For example, the point (0, 1) will not return UL, L, or DL.
// The iteration order here is implicitly returning all the pixels
// that are sqrt(dx,dy)=1 first and then the 1.4 distance pixels last.

const vector<Coord>
get8Neighbors(Coord center, int32_t width, int32_t height)
{
  int32_t cX = center.x;
  int32_t cY = center.y;
  
#if defined(DEBUG)
  assert(cX >= 0);
  assert(cX < width);
  assert(cY >= 0);
  assert(cY < height);
#endif // DEBUG

  static
  int32_t neighborOffsetsArr[] = {
    1,  0,  // R
    0, -1,  // U
    -1,  0, // L
    0,  1,  // D
    1, -1,  // UR
    -1, -1, // UL
    -1,  1, // DL
    1,  1   // DR
  };
  
  Coord neighbors[8];
  size_t neighborsOffset = 0;
  
  for (int i = 0; i < sizeof(neighborOffsetsArr)/sizeof(int32_t); i += 2) {
    int32_t dX = neighborOffsetsArr[i];
    int32_t dY = neighborOffsetsArr[i+1];
   
    int32_t X = cX + dX;
    int32_t Y = cY + dY;
    
    if (X < 0 || X >= width) {
      // Ignore this coordinate since it is outside bbox
    } else if (Y < 0 || Y >= height) {
      // Ignore this coordinate since it is outside bbox
    } else {
      Coord neighbor(X, Y);
#if defined(DEBUG)
      assert(neighborsOffset >= 0 && neighborsOffset < 8);
#endif // DEBUG
      neighbors[neighborsOffset++] = neighbor;
    }
  }
  
  // Construct vector object by reading values of known size from stack
  
  const auto it = begin(neighbors);
  vector<Coord> neighborsVec(it, it+neighborsOffset);
#if defined(DEBUG)
  assert(neighborsVec.size() == neighborsOffset);
#endif // DEBUG
  return neighborsVec;
}

static
bool CompareCoordIntWeightTupleDecreasingFunc (CoordIntWeightTuple &elem1, CoordIntWeightTuple &elem2) {
  int32_t weight1 = get<1>(elem1);
  int32_t weight2 = get<1>(elem2);
  return (weight1 > weight2);
}

static
bool CompareCoordIntWeightTupleIncreasingFunc (CoordIntWeightTuple &elem1, CoordIntWeightTuple &elem2) {
  int32_t weight1 = get<1>(elem1);
  int32_t weight2 = get<1>(elem2);
  return (weight1 < weight2);
}

void
sortCoordIntWeightTuples(vector<CoordIntWeightTuple> &tuples, bool decreasing)
{
  if (decreasing) {
    sort(tuples.begin(), tuples.end(), CompareCoordIntWeightTupleDecreasingFunc);
  } else {
    sort(tuples.begin(), tuples.end(), CompareCoordIntWeightTupleIncreasingFunc);
  }
}

// Given a binary image consisting of zero or non-zero values, calculate
// the center of mass for a shape defined by the white pixels.

void centerOfMass(uint8_t *bytePtr, uint32_t width, uint32_t height, uint32_t *xPtr, uint32_t *yPtr)
{
  uint32_t sumX = 0;
  uint32_t sumY = 0;
  uint32_t N = 0;
  
  uint32_t offset = 0;
  
  for (int y=0; y < height; y++) {
    for (int x=0; x < width; x++) {
      uint8_t val = bytePtr[offset];
      if (val) {
        sumX += x;
        sumY += y;
        N += 1;
      }
    }
  }
  
  *xPtr = sumX / N;
  *yPtr = sumY / N;
  return;
}

// Given a vector of pixels and a pixel that may or may not be in the vector, return
// the pixel in the vector that is closest to the indicated pixel.

uint32_t closestToPixel(const vector<uint32_t> &pixels, const uint32_t closeToPixel) {
  const bool debug = false;
  
#if defined(DEBUG)
  assert(pixels.size() > 0);
#endif // DEBUG
  
  unsigned int minDist = (~0);
  uint32_t closestToPixel = 0;
  
  uint32_t cB = closeToPixel & 0xFF;
  uint32_t cG = (closeToPixel >> 8) & 0xFF;
  uint32_t cR = (closeToPixel >> 16) & 0xFF;
  
  for ( uint32_t pixel : pixels ) {
    uint32_t B = pixel & 0xFF;
    uint32_t G = (pixel >> 8) & 0xFF;
    uint32_t R = (pixel >> 16) & 0xFF;
    
    int dB = (int)cB - (int)B;
    int dG = (int)cG - (int)G;
    int dR = (int)cR - (int)R;
    
    unsigned int d3 = (unsigned int) ((dB * dB) + (dG * dG) + (dR * dR));
    
    if (debug) {
      cout << "d3 from (" << B << "," << G << "," << R << ") to closeToPixel (" << cB << "," << cG << "," << cR << ") is (" << dB << "," << dG << "," << dR << ") = " << d3 << endl;
    }
    
    if (d3 < minDist) {
      closestToPixel = pixel;
      minDist = d3;
      
      if ((debug)) {
        fprintf(stdout, "new    closestToPixel 0x%08X\n", closestToPixel);
      }
      
      if (minDist == 0) {
        // Quit the loop once a zero distance has been found
        break;
      }
    }
  }
  
  if ((debug)) {
    fprintf(stdout, "return closestToPixel 0x%08X\n", closestToPixel);
  }
  
  return closestToPixel;
}

// Given a vector of cluster center pixels, determine a cluster to cluster walk order based on 3D
// distance from one cluster center to the next. This method returns a vector of offsets into
// the cluster table with the assumption that the number of clusters fits into a 16 bit offset.

vector<uint32_t> generate_cluster_walk_on_center_dist(const vector<uint32_t> &clusterCenterPixels)
{
  const bool debugDumpClusterWalk = false;
  
  int numClusters = (int) clusterCenterPixels.size();
  
  unordered_map<uint32_t, uint32_t> clusterCenterToClusterOffsetMap;
  
  int clusteri = 0;
  
  for ( clusteri = 0; clusteri < numClusters; clusteri++) {
    uint32_t closestToCenterOfMassPixel = clusterCenterPixels[clusteri];
    clusterCenterToClusterOffsetMap[closestToCenterOfMassPixel] = clusteri;
  }
  
  // Reorder the clusters so that the first cluster contains the pixel with the value
  // that is closest to zero. Then, the next cluster is determined by checking
  // the distance to the first pixel in the next cluster. The clusters are already
  // ordered in terms of density so this reordering logic basically jumps from one
  // cluster to the next in terms of the shortest distance to the next clsuter.
  
  vector<uint32_t> closestSortedClusterOrder;
  
  closestSortedClusterOrder.reserve(numClusters);
  
  if ((1)) {
    // Choose cluster that is closest to (0,0,0)
    
    uint32_t closestToZeroCenter = closestToPixel(clusterCenterPixels, 0x0);
    
    int closestToZeroClusteri = -1;
    
    clusteri = 0;
    
    for ( uint32_t clusterCenter : clusterCenterPixels ) {
      if (closestToZeroCenter == clusterCenter) {
        closestToZeroClusteri = clusteri;
        break;
      }
      
      clusteri += 1;
    }
    
    assert(closestToZeroClusteri != -1);
    
    if (debugDumpClusterWalk) {
      fprintf(stdout, "closestToZero 0x%08X is in clusteri %d\n", closestToZeroCenter, closestToZeroClusteri);
    }
    
    closestSortedClusterOrder.push_back(closestToZeroClusteri);
    
    // Calculate the distance from the cluster center to the next closest cluster center.
    
    {
      uint32_t closestToCenterOfMassPixel = clusterCenterPixels[closestToZeroClusteri];
      
      auto it = clusterCenterToClusterOffsetMap.find(closestToCenterOfMassPixel);
      
#if defined(DEBUG)
      assert(it != end(clusterCenterToClusterOffsetMap));
#endif // DEBUG
      
      clusterCenterToClusterOffsetMap.erase(it);
    }
    
    // Each remaining cluster is represented by an entry in clusterCenterToClusterOffsetMap.
    // Get the center coord and use the center to lookup the cluster index. Then find
    // the next closest cluster in terms of distance in 3D space.
    
    uint32_t closestCenterPixel = clusterCenterPixels[closestToZeroClusteri];
    
    for ( ; 1 ; ) {
      if (clusterCenterToClusterOffsetMap.size() == 0) {
        break;
      }
      
      vector<uint32_t> remainingClustersCenterPoints;
      
      for ( auto it = begin(clusterCenterToClusterOffsetMap); it != end(clusterCenterToClusterOffsetMap); it++) {
        //uint32_t clusterCenterPixel = it->first;
        uint32_t clusterOffset = it->second;
        
        uint32_t clusterStartPixel = clusterCenterPixels[clusterOffset];
        remainingClustersCenterPoints.push_back(clusterStartPixel);
      }
      
      if (debugDumpClusterWalk) {
        fprintf(stdout, "there are %d remaining center pixel clusters\n", (int)remainingClustersCenterPoints.size());
        
        for ( uint32_t pixel : remainingClustersCenterPoints ) {
          fprintf(stdout, "0x%08X\n", pixel);
        }
      }
      
      uint32_t nextClosestClusterCenterPixel = closestToPixel(remainingClustersCenterPoints, closestCenterPixel);
      
      if (debugDumpClusterWalk) {
        fprintf(stdout, "nextClosestClusterPixel is 0x%08X from current clusterEndPixel 0x%08X\n", nextClosestClusterCenterPixel, closestCenterPixel);
      }
      
      assert(nextClosestClusterCenterPixel != closestCenterPixel);
      
#if defined(DEBUG)
      assert(clusterCenterToClusterOffsetMap.count(nextClosestClusterCenterPixel) > 0);
#endif // DEBUG
      
      uint32_t nextClosestClusteri = clusterCenterToClusterOffsetMap[nextClosestClusterCenterPixel];
      
      closestSortedClusterOrder.push_back(nextClosestClusteri);
      
      {
        // Find index from next closest and then lookup cluster center
        
        uint32_t nextClosestCenterPixel = clusterCenterPixels[nextClosestClusteri];
        
        auto it = clusterCenterToClusterOffsetMap.find(nextClosestCenterPixel);
#if defined(DEBUG)
        assert(it != end(clusterCenterToClusterOffsetMap));
#endif // DEBUG
        clusterCenterToClusterOffsetMap.erase(it);
      }
      
      closestCenterPixel = nextClosestClusterCenterPixel;
    }
    
    assert(closestSortedClusterOrder.size() == clusterCenterPixels.size());
  }
  
  return closestSortedClusterOrder;
}

// my_adler32

// largest prime smaller than 65536
#define BASE 65521L

// NMAX is the largest n such that 255n(n+1)/2 + (n+1)(BASE-1) <= 2^32-1
#define NMAX 5552

#define DO1(buf, i)  { s1 += buf[i]; s2 += s1; }
#define DO2(buf, i)  DO1(buf, i); DO1(buf, i + 1);
#define DO4(buf, i)  DO2(buf, i); DO2(buf, i + 2);
#define DO8(buf, i)  DO4(buf, i); DO4(buf, i + 4);
#define DO16(buf)    DO8(buf, 0); DO8(buf, 8);

uint32_t my_adler32(
                    uint32_t adler,
                    unsigned char const *buf,
                    uint32_t len,
                    uint32_t singleCallMode)
{
  int k;
  uint32_t s1 = adler & 0xffff;
  uint32_t s2 = (adler >> 16) & 0xffff;
  
  if (!buf)
    return 1;
  
  while (len > 0) {
    k = len < NMAX ? len :NMAX;
    len -= k;
    while (k >= 16) {
      DO16(buf);
      buf += 16;
      k -= 16;
    }
    if (k != 0)
      do {
        s1 += *buf++;
        s2 += s1;
      } while (--k);
    s1 %= BASE;
    s2 %= BASE;
  }
  
  uint32_t result = (s2 << 16) | s1;
  
  if (singleCallMode && (result == 0)) {
    // All zero input, use 0xFFFFFFFF instead
    result = 0xFFFFFFFF;
  }
  
  return result;
}

// Sort keys in histogram like table in terms of the count

static
bool CompareKeySizePairFunc (pair<uint32_t,uint32_t> &elem1, pair<uint32_t,uint32_t> &elem2) {
  uint32_t size1 = elem1.second;
  uint32_t size2 = elem2.second;
  return (size1 < size2);
}

static
bool CompareKeySizePairFuncBigToSmall (pair<uint32_t,uint32_t> &elem1, pair<uint32_t,uint32_t> &elem2) {
  uint32_t size1 = elem1.second;
  uint32_t size2 = elem2.second;
  return (size2 < size1);
}

vector<uint32_t>
sort_keys_by_count(unordered_map<uint32_t, uint32_t> &pixelToCountTable, bool biggestToSmallest)
{
  vector<pair<uint32_t,uint32_t> > sizePairs;

  for ( auto &pair : pixelToCountTable ) {
    uint32_t pixel = pair.first;
    uint32_t count = pair.second;
    
    std::pair<uint32_t,uint32_t> sp;
    
    sp.first = pixel;
    sp.second = count;
    
    sizePairs.push_back(sp);
    
    //fprintf(stdout, "0x%08X (%d) -> %d\n", pixel, pixel, count);
  }

  if (biggestToSmallest) {
    sort(begin(sizePairs), end(sizePairs), CompareKeySizePairFuncBigToSmall);
  } else {
    sort(begin(sizePairs), end(sizePairs), CompareKeySizePairFunc);
  }

  vector<uint32_t> keys;
  
  for ( auto &pair : sizePairs ) {
    keys.push_back(pair.first);
    
    //fprintf(stdout, "0x%08X -> %d\n", pair.first, pair.second);
  }
  
  return keys;
}

// Simple signed subtraction of each component in a pixel by comparing the
// current pixel to the one directly on the left.

static inline
uint32_t component_sub_8by4(uint32_t *samplesPtr,
                            uint32_t *decodedSamplesPtr,
                            uint32_t width,
                            uint32_t offset,
                            BOOL doSub) {
  const BOOL debug = FALSE;
  
  BOOL readDecoded = (decodedSamplesPtr != NULL);
  
  // indexes must be signed so that they can be negative at top or left edge
  
  int leftOffset = offset - 1;
  
#if defined(IS_OBJC)
  if (debug) {
    NSLog(@"SUB samplesi=%d, lefti=%d, doSub=%d, readDecoded=%d", offset, leftOffset, (int)doSub, (int)readDecoded);
  }
#endif // IS_OBJC
  
  uint32_t samples;
  uint32_t leftSamples;
  
  samples = samplesPtr[offset];
  
  if (leftOffset < 0) {
    leftSamples = 0x0;
  } else {
    if (!readDecoded) {
      leftSamples = samplesPtr[leftOffset];
    } else {
      leftSamples = decodedSamplesPtr[leftOffset];
    }
  }
  
  uint8_t left_c3 = RSHIFT_MASK(leftSamples, 24, 0xFF);
  uint8_t left_c2 = RSHIFT_MASK(leftSamples, 16, 0xFF);
  uint8_t left_c1 = RSHIFT_MASK(leftSamples, 8, 0xFF);
  uint8_t left_c0 = RSHIFT_MASK(leftSamples, 0, 0xFF);
  
  uint8_t c3 = RSHIFT_MASK(samples, 24, 0xFF);
  uint8_t c2 = RSHIFT_MASK(samples, 16, 0xFF);
  uint8_t c1 = RSHIFT_MASK(samples, 8, 0xFF);
  uint8_t c0 = RSHIFT_MASK(samples, 0, 0xFF);
  
#if defined(IS_OBJC)
  if (debug) {
    NSLog(@"left_c3=%d, left_c2=%d, left_c1=%d, left_c0=%d", left_c3, left_c2, left_c1, left_c0);
    NSLog(@"c3=%d, c2=%d, c1=%d, c0=%d", c3, c2, c1, c0);
  }
#else
  if (debug) {
    fprintf(stdout, "left_c3=%d, left_c2=%d, left_c1=%d, left_c0=%d\n", left_c3, left_c2, left_c1, left_c0);
    fprintf(stdout, "c3=%d, c2=%d, c1=%d, c0=%d\n", c3, c2, c1, c0);
  }
#endif // IS_OBJC
  
  uint32_t r3, r2, r1, r0;
  
  if (doSub) {
    r3 = (c3 - left_c3);
    r2 = (c2 - left_c2);
    r1 = (c1 - left_c1);
    r0 = (c0 - left_c0);
  } else {
    r3 = (c3 + left_c3);
    r2 = (c2 + left_c2);
    r1 = (c1 + left_c1);
    r0 = (c0 + left_c0);
  }
  
#if defined(IS_OBJC)
  if (debug) {
    NSLog(@"r3=%d, r2=%d, r1=%d, r0=%d", r3, r2, r1, r0);
  }
#endif // IS_OBJC
  
  //#if defined(DEBUG)
  //  assert(r3 <= 0xFF);
  //  assert(r2 <= 0xFF);
  //  assert(r1 <= 0xFF);
  //  assert(r0 <= 0xFF);
  //#endif // DEBUG
  
  // Mask to 8bit value and shift into component position
  
  uint32_t components =
  MASK_LSHIFT(r3, 0xFF, 24) |
  MASK_LSHIFT(r2, 0xFF, 16) |
  MASK_LSHIFT(r1, 0xFF, 8) |
  MASK_LSHIFT(r0, 0xFF, 0);
  
#if defined(IS_OBJC)
  if (debug) {
    NSLog(@"r word 0x%0.8X", components);
  }
#endif // IS_OBJC
  
  return components;
}

// Trivial sub operation via (pixel - prev) for each
// component. Returns the prediction error.

uint32_t predict_trivial_component_sub(uint32_t pixel, uint32_t prevPixel)
{
  uint32_t mat[2];
  mat[0] = prevPixel;
  mat[1] = pixel;
  return component_sub_8by4(mat, NULL, 2, 1, 1);
}

// Reverse a trivial prediction, returns the pixel.

uint32_t predict_trivial_component_add(uint32_t prevPixel, uint32_t residual)
{
  uint32_t mat[2];
  mat[0] = residual;
  mat[1] = prevPixel;
  uint32_t pixel = component_sub_8by4(mat, mat, 2, 1, 0);
  return pixel;
}

// Scale a delta vector to a close non-zero approximation for each component.
// For example, (128, 255, 1) would be scaled to (1, 2, 0).

void xyzDeltaToUnitVector(int32_t &dR, int32_t &dG, int32_t &dB) {
  const bool debug = true;
  
  float scale = sqrt(float(dR*dR + dG*dG + dB*dB));
  
  if (scale == 0.0f) {
    dR = 0;
    dG = 0;
    dB = 0;
    return;
  }
  
  assert(scale > 0.0f);
  
  float ratioR = dR / scale;
  float ratioG = dG / scale;
  float ratioB = dB / scale;
  
  if (debug) {
  printf("ratioR %0.2f\n", ratioR);
  printf("ratioG %0.2f\n", ratioG);
  printf("ratioB %0.2f\n", ratioB);
  }
  
  // Choose from 5,4,3,2,1 as the scale.
  
  int first3 = 0;
  int first2 = 0;
  int first1 = 0;
  
  int scaledR = 0;
  int scaledG = 0;
  int scaledB = 0;
  
  const int max = 10;
  
  for ( int i = 1; i < max; i++ ) {
    scaledR = round(ratioR * i);
    scaledG = round(ratioG * i);
    scaledB = round(ratioB * i);
    
    uint32_t numNonZero = 0;
    if (scaledR > 0) {
      numNonZero++;
    }
    if (scaledG > 0) {
      numNonZero++;
    }
    if (scaledB > 0) {
      numNonZero++;
    }
    
    if (debug) {
      printf("(dR dG dB) for %d = (%3d %3d %3d) num-num-zero %d\n", i, scaledR, scaledG, scaledB, numNonZero);
    }
    
    if (numNonZero == 3 && first3 == 0) {
      first3 = i;
    }
    
    if (numNonZero == 2 && first2 == 0) {
      first2 = i;
    }
    
    if (numNonZero == 1 && first1 == 0) {
      first1 = i;
    }
  }
  
  if (first1 == 0) {
    // Multiply by scale so that at least 1 unit is non-zero
    scaledR = round(ratioR * scale);
    scaledG = round(ratioG * scale);
    scaledB = round(ratioB * scale);
  } else if (first2 == 0) {
    scaledR = round(ratioR * first1);
    scaledG = round(ratioG * first1);
    scaledB = round(ratioB * first1);
  } else if (first3 == 0) {
    scaledR = round(ratioR * first2);
    scaledG = round(ratioG * first2);
    scaledB = round(ratioB * first2);
  } else if (first3 != 0) {
    scaledR = round(ratioR * first3);
    scaledG = round(ratioG * first3);
    scaledB = round(ratioB * first3);
  }
  
  dR = scaledR;
  dG = scaledG;
  dB = scaledB;
  
  if (debug) {
    printf("(dR dG dB) = (%3d %3d %3d)\n", scaledR, scaledG, scaledB);
  }
  
  return;
}

