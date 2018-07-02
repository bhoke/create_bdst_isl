#ifndef PLACETREE_H
#define PLACETREE_H

#include "PlaceDivider.h"

class PlaceTree {

public:
  PlaceTree(int placeCount = 0);
  void generatePlaceDendrogram();
  void addNode(cv::Mat currentPlaceMean);
  std::vector<int> levels;
  std::vector<cv::Mat> nodeMeans;
  std::vector<treeNode> tree;
  std::vector<std::vector<int>> nodeMembers;
  std::vector<cv::Mat> allInvariantMeans;

private:
  std::vector<int> phi;
  std::vector<double> lambda;
  int placeCount;
};
#endif
