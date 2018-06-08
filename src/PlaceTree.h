#ifndef PLACETREE_H
#define PLACETREE_H

#include "PlaceDivider.h"

class PlaceTree {

public:
  PlaceTree(int placeCount);
  std::vector<treeNode> generatePlaceDendrogram();
  void addNode(cv::Mat currentPlaceMean);
  std::vector<int> levels;
  std::vector<cv::Mat> nodeMeans;

private:
  std::vector<int> phi;
  std::vector<double> lambda;
  int placeCount;
  std::vector<cv::Mat> allInvariantMeans;
};
#endif
