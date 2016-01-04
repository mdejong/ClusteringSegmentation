// An edge table represents edges in a graph using a list of neighbor nodes
// for each node. This implementation is very simple and provides fast
// access to graph components in linear time.

#ifndef SUPERPIXEL_EDGE_TABLE_H
#define	SUPERPIXEL_EDGE_TABLE_H

#include <unordered_map>
#include <set>

#include "SuperpixelEdge.h"

using namespace std;
using namespace cv;

class SuperpixelEdgeTable {
  
  public:
  
  SuperpixelEdgeTable();
  
  // Edge strength map holds float edge weights given an edge key.
  // This table is useful because it contains a value that is the same
  // for an edge that can be used by both superpixels that the edge
  // applies to.
  
  unordered_map<SuperpixelEdge, float> edgeStrengthMap;
    
  // Return the neighbors of a superpixel UID as a vector of int32_t
  
  vector<int32_t> getNeighbors(int32_t tag);
  
  // A neighbor iterator supports fast access to the neighbors list
  // in sorted order. The caller must take care to not hold an
  // iterator during a merge since that can change the neighbor list.
  
  set<int32_t>&
  getNeighborsSet(int32_t tag);
  
  // Set list of neighbor nodes for a given superpixel UID.
  // This method is typically called when initializing the
  // graph.
  
  void setNeighbors(int32_t tag, vector<int32_t> neighborUIDsVec);
  
  void setNeighbors(int32_t tag, set<int32_t> neighborsSet);

  // When deleting a node, remove the neighbor entries with this method
  
  void removeNeighbors(int32_t tag);
  
  // Return all edges as a flat list of SuperpixelEdge objects, this is useful
  // for inspection purposes but should not be called in real code since
  // the entire graphs is iterated over.
  
  vector<SuperpixelEdge> getAllEdges();
  
  vector<int32_t> getAllTagsInNeighborsTable();

  // Accessor for neighbors member
  
  inline
  unordered_map <int32_t, set<int32_t> > &getNeighborsRef()
  {
    return neighbors;
  }
  
  private:
  
  unordered_map <int32_t, set<int32_t> > neighbors;
  
};

#endif // SUPERPIXEL_EDGE_TABLE_H
