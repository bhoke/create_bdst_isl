#include "PlaceTree.h"
#include <opencv2/core.hpp>


PlaceTree::PlaceTree(int plCount)
{
  this->placeCount = plCount;
}

bool nodecompare(treeNode node1, treeNode node2){
  /* Helper function for std::sort. */
  return node1.distance < node2.distance;
}

std::vector<treeNode> PlaceTree::generatePlaceDendrogram(){
  int i,j,k;
  std::vector<treeNode> tree;
  int nodeCount = placeCount - 1;
  int* index = new int[placeCount];
  int* tempLevel = new int[placeCount];
  cv::Mat* tempMeans = new cv::Mat[placeCount];
  treeNode tn;

  for(i = 0; i < placeCount;i++){
    tn.left = i;
    tn.right = phi[i];
    tn.distance = lambda[i];
    tree.push_back(tn);
    index[i] = i;
    tempLevel[i] = 1;
    tempMeans[i] = this-> allInvariantMeans[i];
    std::cout << allInvariantMeans[i] << std::endl;
  }

  std::sort(tree.begin(),tree.end(),nodecompare);

  for (i = 0; i < nodeCount; i++) {
    j = tree[i].left; // j is the node sorted in ascending distance order
    k = phi[j]; // phi [j] is the first node which left node connects to
    tree[i].left = index[j];
    tree[i].right = index[k];
    index[k] = -i-1;
    //tempMeans[k] = tempMeans[k] * (float)tempLevel[k] + tempMeans[j] * (float)tempLevel[j];
    nodeMeans.push_back(tempMeans[k]);
    tempLevel[k] += tempLevel[j];
    levels.push_back(tempLevel[k]);
  }

  delete []tempMeans;
  delete []index;
  delete []tempLevel;
  return tree;
}

void PlaceTree::addNode(cv::Mat currentPlaceMean)
{
  phi.push_back(placeCount);
  lambda.push_back(DBL_MAX);
  std::vector<double> distNewNode;
  for (int i = 0; i < placeCount; i++)
  distNewNode.push_back(cv::norm(currentPlaceMean, allInvariantMeans[i]));
  for (int i = 0; i < placeCount; i++) {
    if (lambda[i] >= distNewNode[i])
    {
      distNewNode[phi[i]] = min(distNewNode[phi[i]],lambda[i]);
      lambda[i] = distNewNode[i];
      phi[i] = placeCount;
    }
    else{
      distNewNode[phi[i]] = min(distNewNode[phi[i]],distNewNode[i]);
    }
  }
  for (int i = 0;i < placeCount;i++)
  if(lambda[i] >= lambda[phi[i]])
  phi[i] = placeCount;

  for(int i = 0; i < placeCount; i ++) std::cout << "phi: " << phi[i]<< std::endl;
  std::cout << "PlaceCount before increment" << placeCount << std::endl;
  placeCount++; //Place count is set to N+1
  std::cout << "We have " << placeCount << " places" << std::endl;
  allInvariantMeans.push_back(currentPlaceMean);
  // std::cout << allInvariantMeans.back() << std::endl;
}
