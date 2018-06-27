#include "PlaceRecognizer.h"

PlaceRecognizer::PlaceRecognizer(float tau_r) : PT(0){

  PT = PlaceTree(0);
  this -> plIDSubscriber = this->nh.subscribe<std_msgs::Int16>
  ("placeDetectionISL/placeID", 5, &PlaceRecognizer::placeCallback, this);

  this -> filePathSubscriber = this->nh.subscribe<std_msgs::String>
  ("placeDetectionISL/mainFilePath",2, &PlaceRecognizer::mainFilePathCallback,this);

  this->svm = cv::ml::SVM::create();
  this->svm->setType(cv::ml::SVM::ONE_CLASS);
  this->svm->setKernel(cv::ml::SVM::LINEAR);
  this->svm->setNu(0.9);
  this->svm->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_ITER, 1000, 1e-8));

  this->recognitionThreshold = tau_r;
}

void PlaceRecognizer::placeCallback(std_msgs::Int16 placeId){
  // Callback function for place detection.
  // The input is the place id signal from the place detection node
  std::cout << "Place Callback Received" << std::endl;
  currentPlace = dbmanager.getPlace((int)placeId.data);
  bool recognized;
  //PlaceDivider::clusterPlace(this->currentPlace);
  int lpCount = learnedPlaces.size();
  if (lpCount < MIN_NO_PLACES) {
    learnCurrentPlace();
  }
  else {
    PT.generatePlaceDendrogram();
    recognized = recognizeCurrentPlace();
    if (recognized) updateTree();
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

      LearnedPlace::lpCounter = previousKnowledgeSize + 1;

      for (int i = 1; i <= previousKnowledgeSize; i++)
      {
        LearnedPlace aPlace = knowledgedbmanager.getLearnedPlace(i);
        learnedPlaces.push_back(aPlace);
      }
    } // Previous knowledge
  } // if(knowledgedbmanager.openDB(knowledge_dbpath,"knowledge"))
}

bool PlaceRecognizer::recognizeCurrentPlace(){
  //Recognition part is calculated as in the paper: "An Integrated Model of Autonomous Topological Spatial Cognition"
  float result, votePercentage,
  firstDistance, secondDistance,
  distanceLeft,distanceRight, costResult;
  bool recognized;
  // this -> svm -> train(learnedPlaces,currentPlace)
  std::vector<treeNode> currentTree = PT.tree;
  std::vector<int> leftMembers, rightMembers, closestMembers;
  for (int i = currentTree.size() - 1; i >= 0; i--){
    int leftNode = currentTree[i].left;
    int rightNode = currentTree[i].right;
    if (leftNode < 0){
      distanceLeft = cv::norm(currentPlace.meanInvariant,PT.nodeMeans[-leftNode - 1],cv::NORM_L2SQR);
      leftMembers = PT.nodeMembers[-leftNode - 1];
    }
    else{
      distanceLeft = cv::norm(currentPlace.meanInvariant,PT.allInvariantMeans[leftNode],cv::NORM_L2SQR);
      leftMembers = {leftNode};
    }
    if (rightNode < 0){
      distanceRight = cv::norm(currentPlace.meanInvariant,PT.nodeMeans[-rightNode - 1],cv::NORM_L2SQR);
      rightMembers = PT.nodeMembers[-rightNode - 1];
    }
    else{
      distanceRight = cv::norm(currentPlace.meanInvariant,PT.allInvariantMeans[rightNode],cv::NORM_L2SQR);
      rightMembers = {rightNode};
    }
    std::cout << "Left Node: " << leftNode << std::endl;
    std::cout << "Right Node: " << rightNode << std::endl;

    if(distanceLeft < distanceRight){
      firstDistance = distanceLeft;
      secondDistance = distanceRight;
      closestMembers = leftMembers;
    }
    else{
      firstDistance = distanceRight;
      secondDistance = distanceLeft;
      closestMembers = rightMembers;
    }

    votePercentage = calcVote(closestMembers);
    costResult = (firstDistance + firstDistance / secondDistance + (1 - votePercentage));
    recognized = costResult < recognitionThreshold;
    std::cout << "costResult: " << costResult << std::endl;
  }

  return recognized;
}

float PlaceRecognizer::calcVote(std::vector<int> closestMembers){
  cv::Mat trainVector,testVector,svmResult;
  int currentMember;
  float voteSum, votePercentage;

  for(int i = 0; i < closestMembers.size(); i++)
  {
    currentMember = closestMembers[i];
    if (trainVector.empty())
    {
      trainVector = learnedPlaces[currentMember].memberInvariants;
    }
    else
    {
      cv::hconcat(trainVector,learnedPlaces[currentMember].memberInvariants,trainVector);
    }
  }

  double minVal = 0, maxVal = 0;
  cv::minMaxLoc(trainVector,&minVal,&maxVal);
  trainVector = (trainVector - minVal)/(maxVal - minVal);
  testVector = currentPlace.memberInvariants.clone();
  testVector = (testVector - minVal)/(maxVal - minVal);;

  cv::transpose(testVector,testVector);

  svm -> train(trainVector,cv::ml::COL_SAMPLE,cv::Mat::ones(1,trainVector.cols,CV_32FC1));
  svm -> predict(testVector,svmResult);
  voteSum = cv::sum(svmResult)[0];
  votePercentage = voteSum / (float)svmResult.rows;
  std::cout << "votePercentage: "<<votePercentage << std::endl;
  return votePercentage;
}

void PlaceRecognizer::updateTree(){
  // Place is recognized, we do not add a new node to tree but update the nodes
  // and meanInvariant of the places
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
