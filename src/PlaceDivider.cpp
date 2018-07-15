#include "PlaceDivider.h"

double* nodeDiff(treeNode *tn,int nnodes);

std::vector<Place> PlaceDivider::clusterPlace(Place currentPlace)
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
  int clusterCount = 1 + (int)std::distance(differences,std::max_element(differences,differences + ncols-2));
  int* placeClusters[clusterCount];
  int count[clusterCount];
  cuttree(ncols,placeTree,clusterCount,clusterid,placeClusters,count);
  std::vector <Place> currentSPs;

  for (int i = 0; i < clusterCount; i++){
    if(count[i] > 5){
      Place temp_sp;
      temp_sp.memberInvariants = cv::Mat::zeros(HARMONIC1 * HARMONIC2 * 6,count[i], CV_32F);
      for (int j = 0; j < count[i]; j++){
        currentPlace.memberInvariants.col(placeClusters[i][j]).copyTo(temp_sp.memberInvariants.col(j));
        temp_sp.memberBPIDs.push_back(placeClusters[i][j]);
      }
      cv::reduce(temp_sp.memberInvariants,temp_sp.meanInvariant,1,CV_REDUCE_AVG);
      currentSPs.push_back(temp_sp);
    }
  }

  std::cout << "Contains "<<currentSPs.size() << "subplaces" << std::endl;
  delete [] clusterid;
  return currentSPs;
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
