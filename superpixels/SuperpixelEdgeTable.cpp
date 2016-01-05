// An edge table represents edges as constant time lookup tables that make it possible to
// iterate over an update graph nodes and edges in constant time.

#include "Superpixel.h"

#include "SuperpixelEdgeTable.h"

SuperpixelEdgeTable::SuperpixelEdgeTable()
{
}

// This implementation iterates over all the edges in the graph and it is not fast
// for a graph with many nodes.

vector<SuperpixelEdge> SuperpixelEdgeTable::getAllEdges()
{
  const bool debug = false;
  
  // Walk over all superpixels and get each neighbor, then create an edge only when the UID
  // of a superpixel is smaller than the UID of the edge. This basically does a dedup of
  // all the edges without creating and checking for dups.
  
  vector<SuperpixelEdge> allEdges;
  
  vector<int32_t> allSuperpixles = getAllTagsInNeighborsTable();
  
  for (auto it = allSuperpixles.begin(); it != allSuperpixles.end(); ++it ) {
    int32_t tag = *it;
    
    if (debug) {
      cout << "for superpixel " << tag << " neighbors:" << endl;
    }
    
    for ( int32_t neighborTag : getNeighborsSet(tag) ) {
      if (debug) {
        cout << neighborTag << endl;
      }
      
      if (tag <= neighborTag) {
        SuperpixelEdge edge(tag, neighborTag);
        allEdges.push_back(edge);
        
        if (debug) {
          cout << "added edge (" << edge.A << "," << edge.B << ")" << endl;
        }
      } else {
        if (debug) {
          SuperpixelEdge edge(tag, neighborTag);
          
          cout << "ignored dup edge (" << edge.A << "," << edge.B << ")" << endl;
        }
      }
    }
  }
  
  return allEdges;
}

// A neighbor iterator supports fast access to the neighbors list
// in sorted order. The caller must take care to not hold an
// iterator during a merge since that can change the neighbor list.

set<int32_t>&
SuperpixelEdgeTable::getNeighborsSet(int32_t tag)
{
  auto iter = neighbors.find(tag);
  
  if (iter == neighbors.end()) {
    // Neighbors key must be defined for this tag
    assert(0);
  } else {
    // Otherwise the key exists in the table, return ref to vector in table
    // with the assumption that the caller will not change it.
    
    return iter->second;
  }
}

// This impl returns a copy of the neighbors vector

vector<int32_t> SuperpixelEdgeTable::getNeighbors(int32_t tag)
{
  vector<int32_t> neighborVec;
  if ( neighbors.find(tag) == neighbors.end()) {
    return neighborVec;
  }
  for ( int32_t neighborTag : getNeighborsSet(tag) ) {
    neighborVec.push_back(neighborTag);
  }
  return neighborVec;
}

// Set initial list of neighbors for a superpixel or rest the list after making
// changes. The neighbor values are sorted only to make the results easier to
// read, there should not be much impact on performance since the list of neighbors
// is typically small.

void SuperpixelEdgeTable::setNeighbors(int32_t tag, vector<int32_t> neighborUIDsVec)
{
  set<int32_t> neighborsSet;
  
  for ( int32_t tag : neighborUIDsVec ) {
    neighborsSet.insert(tag);
  }
  // FIXME: use faster impl of insert
  //neighborsSet.insert(neighbors.begin(), neighbors.end);
  
  //sort (neighborsUIDsVec.begin(), neighborsUIDsVec.end());
  
  neighbors[tag] = neighborsSet;
}

// Set initial list of neighbors for a superpixel or rest the list after making
// changes. The neighbor values are sorted only to make the results easier to
// read, there should not be much impact on performance since the list of neighbors
// is typically small.

void SuperpixelEdgeTable::setNeighbors(int32_t tag, set<int32_t> neighborsSet)
{
  neighbors[tag] = neighborsSet;
}

// When deleting a node, remove the neighbor entries

void SuperpixelEdgeTable::removeNeighbors(int32_t tag)
{
  neighbors.erase(tag);
}

// Return a vector of tags that have an entry in the neighbors table.
// Even if the vector contains zero elements, this method is not fast.

vector<int32_t> SuperpixelEdgeTable::getAllTagsInNeighborsTable()
{
  vector<int32_t> vec;
  for ( auto it = neighbors.begin(); it != neighbors.end(); ++it ) {
    int32_t tag = it->first;
    vec.push_back(tag);
  }
  sort (vec.begin(), vec.end());
  return vec;
}
