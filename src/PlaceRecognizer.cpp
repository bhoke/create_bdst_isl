#include "PlaceRecognizer.h"

PlaceRecognizer::PlaceRecognizer() : PT(0){

  PT = PlaceTree(0);
  this -> plIDSubscriber = this->nh.subscribe<std_msgs::Int16>
  ("placeDetectionISL/placeID", 5, &PlaceRecognizer::placeCallback, this);

  this -> filePathSubscriber = this->nh.subscribe<std_msgs::String>
  ("placeDetectionISL/mainFilePath",2, &PlaceRecognizer::mainFilePathCallback,this);
}

void PlaceRecognizer::placeCallback(std_msgs::Int16 placeId){
  // Callback function for place detection.
  // The input is the place id signal from the place detection node
  int recognized = 0;
  std::cout << "Place Callback Received" << std::endl;
  currentPlace = dbmanager.getPlace((int)placeId.data);
  //PlaceDivider::clusterPlace(this->currentPlace);
  int lpCount = learnedPlaces.size();
  if (lpCount < MIN_NO_PLACES) {
    learnCurrentPlace();
  }
  else {
    recognized = recognizeCurrentPlace();
    if (recognized > 0) updateTree();
    else learnCurrentPlace();
  }
}

void PlaceRecognizer::mainFilePathCallback(std_msgs::String mainDir)
{
  // A callback function for the main file
  // the input is main file path string from the place detection node
  QString mainFilePath = QString::fromStdString(mainDir.data);
  qDebug() <<"Main File Path Callback received" << mainFilePath;
  QString detected_places_dbpath = mainFilePath + "/detected_places.db";
  QString knowledge_dbpath = mainFilePath + "/knowledge.db";

  if(dbmanager.openDB(detected_places_dbpath))
  {
    std::cout <<"Places db opened" << std::endl;
  }
  if(knowledgedbmanager.openDB(knowledge_dbpath,"knowledge"))
  {
    std::cout <<"Knowledge db opened" << std::endl;

    if(knowledgedbmanager.getLearnedPlaceMaxID() == 0)
    {
      std::cout <<"Starting with empty knowledge" << std::endl;
    }
    else
    {
      int previousKnowledgeSize = knowledgedbmanager.getLearnedPlaceMaxID();

      std::cout <<"Starting with previous knowledge. Previous number of places: "<<previousKnowledgeSize << std::endl;

      LearnedPlace::lpCounter = previousKnowledgeSize+1;

      for (int i = 1; i <= previousKnowledgeSize; i++)
      {
        LearnedPlace aPlace = knowledgedbmanager.getLearnedPlace(i);
        learnedPlaces.push_back(aPlace);
      }
    } // Previous knowledge
  } // if(knowledgedbmanager.openDB(knowledge_dbpath,"knowledge"))
}

int PlaceRecognizer::recognizeCurrentPlace(){
  std::cout << "I am in recognizeCurrentPlace()" << std::endl;
  return 0;
}

void PlaceRecognizer::updateTree(){
  std::cout << "I am in updateTree()" << std::endl;
}

void PlaceRecognizer::learnCurrentPlace(){
  LearnedPlace alearnedPlace(this->currentPlace);
  this->learnedPlaces.push_back(alearnedPlace);
  std::cout <<"Place "<< this -> learnedPlaces.back().id << " is learned!" << std::endl;
  knowledgedbmanager.insertLearnedPlace(alearnedPlace);
  PT.addNode(currentPlace.meanInvariant);
}

void PlaceRecognizer::closeDatabases() {
  this->knowledgedbmanager.closeDB();
  this->dbmanager.closeDB();
  std::cout << "Databases are closed!!" << std::endl;
}
