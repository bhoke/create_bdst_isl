#include "bubble/bubbleprocess.h"
#include "Utility/PlaceDetector.h"
#include "imageprocess/imageprocess.h"
#include "PlaceRecognizer.h"
#include <opencv2/ml/ml.hpp>
#include <algorithm>
#include <numeric>

#include <QDir>
#include <QFile>
#include <sys/sysinfo.h>

#include <stdio.h>
#include <stdlib.h>  /* The standard C libraries */

void clusterPlace(Place pl); // Clusters a single place into different sub-places
float calculateCostFunctionv2(float firstDistance, float secondDistance, LearnedPlace closestPlace, Place detected_place);
void addNode();
static int nodecompare(const void* a, const void* b);
//float performSVM(cv::Mat trainingVector, cv::Mat testVector); // Perform the one-Class SVM calculation

ros::Timer timer;

std::vector<LearnedPlace> learnedPlaces;
std::vector<Place> detectedPlaces;

int learnedPlaceCounter = 1;
Place currentPlace;
double tau_h, tau_r;

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "createBDSTISL");

  tau_h = 0;
  tau_r = 2.5;

  std::cout <<"Parameters: "<<tau_h << "  "<<tau_r << std::endl;

  PlaceRecognizer PR;

  ros::Rate loop(50);
  //
  while(ros::ok())
  {
    ros::spinOnce();
    loop.sleep();
  } // while(ros::ok())
  std::vector< treeNode> tree = PR.PT.generatePlaceDendrogram();
  for (int i = 0; i < tree.size(); i ++)
  std::cout << tree[i].left << "   " << tree[i].right << "    " <<  tree[i].distance << std::endl;
  PR.closeDatabases();
  return 0;
}

// static inline float computeSquare (float x) { return x*x; }

// float calculateCostFunctionv2(float firstDistance, float secondDistance, LearnedPlace closestPlace, Place detected_place)
// {
//
//   float result = -1;
//
//   float firstPart = firstDistance;
//   float secondPart = firstDistance/secondDistance;
//   float votePercentage = 0;
//
//   //votePercentage = performSVM(closestPlace.memberInvariants,detected_place.memberInvariants);
//
//   std::cout <<"Vote percentage: "<<votePercentage << std::endl;
//
//   result = firstPart+secondPart+(1-votePercentage);
//   std::cout << "result: " << result<< std::endl;
//   return result;
// }
