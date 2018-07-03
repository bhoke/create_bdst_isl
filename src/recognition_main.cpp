#include "PlaceRecognizer.h"

std::vector<LearnedPlace> learnedPlaces;
std::vector<Place> detectedPlaces;

int learnedPlaceCounter = 1;
Place currentPlace;
double tau_r;

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "create_bdst_isl");

  tau_r = 2.5;

  std::cout <<"tau_r: "<< tau_r << std::endl;

  PlaceRecognizer PR(tau_r);

  ros::Rate loop(50);
  //
  while(ros::ok())
  {
    ros::spinOnce();
    loop.sleep();
  }

  PR.closeDatabases();
  return 0;
}
