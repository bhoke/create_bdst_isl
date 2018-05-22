#include "PlaceRecognizer.h"


PlaceRecognizer::PlaceRecognizer(){

  this -> plIDSubscriber = this->nh.subscribe<std_msgs::Int16>
  ("placeDetectionISL/placeID", 5, &PlaceRecognizer::placeCallback, this);

  this -> filePathSubscriber = this->nh.subscribe<std_msgs::String>
  ("placeDetectionISL/mainFilePath",2, &PlaceRecognizer::mainFilePathCallback,this);
}

void PlaceRecognizer::placeCallback(std_msgs::Int16 placeId){
  // Callback function for place detection.
  // The input is the place id signal from the place detection node

  std::cout << "Place Callback Received" << std::endl;
  this-> currentPlace = dbmanager.getPlace((int)placeId.data);
  PlaceDivider::clusterPlace(this->currentPlace);
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

void PlaceRecognizer::learnCurrentPlace(currentPlace){
  LearnedPlace alearnedPlace(currentPlace);
  this->learnedPlaces.push_back(currentPlace);
  std::cout <<"Places size = "<< learnedPlaces.size() << " < " << MIN_NO_PLACES << " --> No Recognition" << std::endl;
  detectedPlaces.push_back(currentPlace);
}

treeNode* PlaceRecognizer::generatePlaceDendrogram(int* phi,double* lambda,int totalPlaceCount){
  int i,j,k;
  treeNode* tree;
  tree = new treeNode[totalPlaceCount];
  int* index = new int[totalPlaceCount];

  for(i = 0; i< totalPlaceCount;i++){
    tree[i].left = i;
    tree[i].right = phi[i];
    tree[i].distance = lambda[i];
    index[i] = i;
  }

  qsort(tree,totalPlaceCount,sizeof(treeNode),nodecompare);

  for (i = 0; i < totalPlaceCount; i++) {
    j = tree[i].left; // j is the node sorted in ascending distance order
    k = phi[j]; // phi [j] is the first node which left node connects to
    tree[i].left = index[j];
    tree[i].right = index[k];
    index[k] = -i-1;
  }

  return tree;
}

void addNode(double* distNewNode,int placeCount, int* phi,double* lambda)
{
  phi[placeCount] = placeCount;
  lambda[placeCount] = DBL_MAX;
  for (int i = 0; i < placeCount; i++)
  if (lambda[i] >= distNewNode[i])
  {
    if (lambda[i] <= distNewNode[phi[i]])
    distNewNode[phi[i]] = lambda[i];
    lambda[i] = distNewNode[i];
    phi[i] = placeCount;
  }
  else if (distNewNode[i] < distNewNode[phi[i]])
  distNewNode[phi[i]] = distNewNode[i];

  for (int i = 0;i < placeCount;i++)
  if(lambda[i] >= lambda[phi[i]])
  phi[i] = placeCount;
}

static int nodecompare(const void* a, const void* b)
/* Helper function for qsort. */
{ const treeNode* node1 = (const treeNode*)a;
  const treeNode* node2 = (const treeNode*)b;
  const double term1 = node1->distance;
  const double term2 = node2->distance;
  if (term1 < term2) return -1;
  if (term1 > term2) return +1;
  return 0;
}
