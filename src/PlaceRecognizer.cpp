#include "PlaceRecognizer.h"

PlaceRecognizer::PlaceRecognizer(float tau_r){

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
  int previousKnowledgeSize;
  if(dbmanager.openDB(detected_places_dbpath))
  {
    std::cout <<"Places db opened" << std::endl;
  }
  if(knowledgedbmanager.openDB(knowledge_dbpath,"knowledge"))
  {
    std::cout <<"Knowledge db opened" << std::endl;
    previousKnowledgeSize = knowledgedbmanager.getLearnedPlaceMaxID();
    if(previousKnowledgeSize == 0)
    {
      std::cout <<"Starting with empty knowledge" << std::endl;
    }
    else
    {
      std::cout <<"Starting with previous knowledge. Previous number of places: "<< previousKnowledgeSize << std::endl;
      LearnedPlace::lpCounter = previousKnowledgeSize + 1;
      LearnedPlace aPlace;
      for (int i = 1; i <= previousKnowledgeSize; i++){
        aPlace = knowledgedbmanager.getLearnedPlace(i);
        learnedPlaces.push_back(aPlace);
        PT.allInvariantMeans.push_back(aPlace.meanInvariant);
      }
      for (int i = 0; i < previousKnowledgeSize; i++)
        PT.addNode(learnedPlaces[i].meanInvariant);

    } // Previous knowledge
  } // if(knowledgedbmanager.openDB(knowledge_dbpath,"knowledge"))
}

int PlaceRecognizer::recognizeCurrentPlace(){
  //Recognition part is calculated as in the paper: "An Integrated Model of Autonomous Topological Spatial Cognition"
  double result, votePercentage,
  highCorr, secondCorr,
  corrLeft,corrRight, corrResult;
  bool recognized;
  std::vector<treeNode> currentTree = PT.tree;
  std::vector<int> leftMembers, rightMembers, closestMembers;
  int i = 2 - currentTree.size();
  int j,k;
  // this -> svm -> train(learnedPlaces,currentPlace)
  while(i < 0){
    int leftNode = currentTree[-i -1].left;
    int rightNode = currentTree[-i -1].right;
    if (leftNode < 0){
      corrLeft = fabs(cv::compareHist(currentPlace.meanInvariant,PT.nodeMeans[-leftNode - 1],CV_COMP_CORREL));
      leftMembers = PT.nodeMembers[-leftNode - 1];
    }
    else{
      corrLeft = fabs(cv::compareHist(currentPlace.meanInvariant,PT.allInvariantMeans[leftNode],CV_COMP_CORREL));
      leftMembers = {leftNode};
    }
    if (rightNode < 0){
      corrRight = fabs(cv::compareHist(currentPlace.meanInvariant,PT.nodeMeans[-rightNode - 1],CV_COMP_CORREL));
      rightMembers = PT.nodeMembers[-rightNode - 1];
    }
    else{
      corrRight = fabs(cv::compareHist(currentPlace.meanInvariant,PT.allInvariantMeans[rightNode],CV_COMP_CORREL));
      rightMembers = {rightNode};
    }
    j = leftNode; k = rightNode;
    std::cout << "Left Node: " << leftNode << std::endl;
    std::cout << "Right Node: " << rightNode << std::endl;

    if(corrLeft > corrRight){
      highCorr = corrLeft;
      secondCorr = corrRight;
      closestMembers = leftMembers;
      i = j;
    }
    else{
      highCorr = corrRight;
      secondCorr = corrLeft;
      closestMembers = rightMembers;
      i = k;
    }

    votePercentage = calcVote(closestMembers);
    corrResult = highCorr + secondCorr / highCorr + votePercentage;
    recognized = corrResult > recognitionThreshold;
    std::cout << "corrResult: " << corrResult << std::endl;
    if (!recognized) {
      std::cout << "No recognition... Returning" << std::endl;
      learnCurrentPlace();
      return -1;
    }
  }
  std::cout << "Recognized with place id: " << i << std::endl;
  updateTree(i);
  return i;
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

void PlaceRecognizer::updateTree(int i){
  // Place is recognized, we do not add a new node to tree but update the nodes
  // and meanInvariant of the places
  std::cout << "Updating " << i << "th place..." << std::endl;
  learnedPlaces[i].memberPlaceIDs.push_back(currentPlace.id);
  cv::hconcat(learnedPlaces[i].memberInvariants,currentPlace.memberInvariants,learnedPlaces[i].memberInvariants);
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
