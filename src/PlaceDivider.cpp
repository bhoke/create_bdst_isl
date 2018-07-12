#include "PlaceDivider.h"

double* nodeDiff(treeNode *tn,int nnodes);

void PlaceDivider::clusterPlace(Place currentPlace)
{
  //Function which clusters a single place into subplaces
  cv::Mat currentInvariants = currentPlace.memberInvariants;
  int nrows = currentInvariants.rows;
  int ncols = currentInvariants.cols;
  double** data = new double*[nrows];
  int* clusterid = new int[ncols];

  std::cout << nrows << "   " << ncols << std::endl;
  for(int i = 0; i < nrows; ++i)
  {
    data[i] = new double[ncols];
    for (int j = 0; j < ncols; ++j)
    data[i][j] = currentInvariants.at<float>(i,j);
  }

  treeNode* placeTree = treecluster(nrows,ncols,data,1,'s',NULL);
  double *differences = nodeDiff(placeTree,ncols-2);
  int clusterCount = 2 + (int)std::distance(differences,std::max_element(differences,differences + ncols-2));
  std::cout <<"Cluster Count is: "<<  clusterCount << std::endl;
  int* placeClusters[clusterCount];
  int* count = new int[clusterCount];
  cuttree(ncols,placeTree,clusterCount,clusterid,placeClusters,count);
  std::vector <subPlace> currentSPs;

  std::cout << "Cols = " << currentPlace.memberInvariants.cols << " Rows = " << currentPlace.memberInvariants.rows << std::endl ;

  for (int i = 0; i < clusterCount; i++){
    subPlace temp_sp;
    int k = 0;
    for (int j = 0; j < count[i]; j++){
      temp_sp.memberInvariants = cv::Mat::zeros(HARMONIC1 * HARMONIC2 * 6,count[i], CV_64F);
      currentPlace.memberInvariants.col(placeClusters[i][j]).copyTo(temp_sp.memberInvariants.col(k));
      k++;
    }
    temp_sp.calculateMeanInvariant();
  }
  std::cout << "placeTree for place ID " << currentPlace.id << ": " << std::endl;
  for (int i = 0; i < ncols - 1 ; i++)
  std::cout << placeTree[i].left << "\t" << placeTree[i].right << "\t"
  << placeTree[i].distance << "\t" << std::endl;

  delete []clusterid;
}

// HELPER FUNCTIONS
double* nodeDiff(treeNode *tn,int nnodes)
{
  // Calculation of adjacent nodes to determine maximum difference leap
  // Function starts to calculate from the top of the tree so we can easily detemrine the number of clusters
  double *result = new double[nnodes];
  for(int i = nnodes; i > 0; i--)
  {
    result[nnodes - i] = tn[i].distance - tn[i -1].distance;
  }

  return result;
}
