#include "PlaceTree.h"
#include <opencv2/core.hpp>


PlaceTree::PlaceTree(int plCount)
{
  placeCount = plCount;
}

bool nodecompare(treeNode node1, treeNode node2){
  /* Helper function for std::sort. */
  return node1.distance < node2.distance;
}

void PlaceTree::generatePlaceDendrogram(){
  int j,k;
  int nodeCount = placeCount - 1;
  nodeMembers.clear();
  nodeMeans.clear();
  tree.clear();
  levels.clear();
  int* index = new int[placeCount];
  treeNode tn;
  nodeMembers.resize((size_t)nodeCount);
  levels.resize((size_t)nodeCount);
  nodeMeans.resize((size_t)nodeCount);
  cv::Size invariantSize = allInvariantMeans[0].size();
  std::fill(nodeMeans.begin(), nodeMeans.end(),cv::Mat::zeros(invariantSize,CV_32FC1));
  for(int i = 0; i < placeCount;i++){
    tn.left = i;
    tn.right = phi[i];
    tn.distance = lambda[i];
    tree.push_back(tn);
    index[i] = i;
  }

  std::sort(tree.begin(),tree.end(),nodecompare);
  std::cout<< "Node #\tLeft\tRight\tNumel" << std::endl;
  for (int i = 0; i < nodeCount; i++) {
    j = tree[i].left; // j is the node sorted in ascending distance order
    k = phi[j]; // phi [j] is the first node which left node connects to
    tree[i].left = index[j];
    tree[i].right = index[k];
    int leftLevel = 0,rightLevel = 0;
    if(index[j] < 0 ){
      leftLevel = levels[-index[j] - 1];
      nodeMembers[i].reserve(nodeMembers[i].size() + nodeMembers[-index[j] -1].size());
      nodeMembers[i].insert(nodeMembers[i].end(), nodeMembers[-index[j] -1 ].begin(), nodeMembers[-index[j] -1].end());
      nodeMeans[i] += nodeMeans[-index[j] -1] * levels[-index[j] -1];
    }
    else{
      leftLevel ++;
      nodeMembers[i].push_back(index[j]);
      nodeMeans[i] += allInvariantMeans[index[j]];
    }
    if(index[k] < 0){
      rightLevel = levels[-index[k] - 1];
      nodeMembers[i].reserve(nodeMembers[i].size() + nodeMembers[-index[k] - 1].size());
      nodeMembers[i].insert(nodeMembers[i].end(), nodeMembers[-index[k] -1].begin(), nodeMembers[-index[k] -1].end());
      nodeMeans[i] += nodeMeans[-index[k] -1] * levels[-index[k] -1];
    }
    else{
      rightLevel++;
      nodeMembers[i].push_back(index[k]);
      nodeMeans[i] += allInvariantMeans[index[k]];
    }
    levels[i] = leftLevel + rightLevel;
    nodeMeans[i] = nodeMeans[i] / levels[i];
    index[k] = -i-1;
    std::cout<< -i-1 <<"\t" << tree[i].left << "\t" << tree[i].right<< "\t"
    << levels[i] << "\t" << std::endl;
  }

  delete []index;
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

  placeCount++; //Place count is set to N+1
  std::cout << "We have " << placeCount << " places" << std::endl;
  allInvariantMeans.push_back(currentPlaceMean);
  // std::cout << allInvariantMeans.back() << std::endl;
}
