#include "PlaceDivider.h"

class PlaceTree {

public:
  PlaceTree(int placeCount);
  std::vector<treeNode> generatePlaceDendrogram();
  void addNode(cv::Mat currentPlaceMean);

private:
  std::vector<int> phi;
  std::vector<double> lambda;
  int placeCount;
  std::vector<cv::Mat> allInvariantMeans;
};
