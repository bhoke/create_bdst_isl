#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Int8.h>
#include <std_msgs/Int16.h>
#include <vector>

#include <QDebug>
#include <QString>

//#include "PlaceDivider.h"
#include "PlaceTree.h"
#include "database/databasemanager.h"

#define MIN_NO_PLACES 3

class PlaceRecognizer
{
private:
  ros::NodeHandle nh;
  ros::Subscriber plIDSubscriber;
  ros::Subscriber filePathSubscriber;
  void placeCallback (const std_msgs::Int16 PlaceID);
  void mainFilePathCallback (const std_msgs::String mainDir);
  DatabaseManager dbmanager,knowledgedbmanager;
  QString mainFilePath;
  std::vector<Place> detectedPlaces;
  std::vector<LearnedPlace> learnedPlaces;
  void learnCurrentPlace();
  int recognizeCurrentPlace();
  void updateTree();

public:
  PlaceRecognizer();
  Place currentPlace;
  void closeDatabases();
  PlaceTree PT;
};
