#include "bubble/bubbleprocess.h"
#include "Utility/PlaceDetector.h"
#include "imageprocess/imageprocess.h"
#include "database/databasemanager.h"
#include <opencv2/ml/ml.hpp>

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Int8.h>
#include <std_msgs/Int16.h>
#include <vector>
#include <algorithm>
#include <numeric>

#include <QDir>
#include <QDebug>
#include <QFile>
#include <QDateTime>
#include <sys/sysinfo.h>

#include <stdio.h>
#include <stdlib.h>  /* The standard C libraries */
extern "C" {
#include "cluster.h"
}

#define MIN_NO_PLACES 3

// Clusters a single place into different sub-places
void clusterPlace(Place pl);

// Perform the one-Class SVM calculation
//float performSVM(cv::Mat trainingVector, cv::Mat testVector);

float calculateCostFunctionv2(float firstDistance, float secondDistance, LearnedPlace closestPlace, Place detected_place);

// Calculate the mean invariants so that the robot can perform recognition
void calculateMeanInvariantsOfBDST();

// OPENCV modifikasyonu
double compareHistHK( InputArray _H1, InputArray _H2, int method );

ros::Timer timer;

std::vector<LearnedPlace> learnedPlaces;
std::vector<Place> currentPlaces;

int learnedPlaceCounter = 1;
Place currentPlace;
DatabaseManager dbmanager,knowledgedbmanager;
QString mainFilePath;
double tau_h, tau_r;

// Callback function for place detection. The input is the place id signal from the place detection node
void placeCallback(std_msgs::Int16 placeId)
{
    // read the place from the database
    std::cout << "Place Callback Received" << std::endl;
    Place aPlace = dbmanager.getPlace((int)placeId.data);
    clusterPlace(aPlace);

    if(learnedPlaces.size() < MIN_NO_PLACES)
    {
        LearnedPlace alearnedPlace(aPlace);
        learnedPlaces.push_back(alearnedPlace);
        std::cout <<"Places size = "<< learnedPlaces.size() << " < " << MIN_NO_PLACES << " --> No Recognition" << std::endl;
        return;
    }

    currentPlaces.push_back(aPlace);
}

// A callback function for the main file, the input is main file path string from the place detection node
void mainFilePathCallback(std_msgs::String mainfp)
{
    mainFilePath = QString::fromStdString(mainfp.data);
    qDebug()<<"Main File Path Callback received" << mainFilePath;
    QString detected_places_dbpath = mainFilePath + "/detected_places.db";
    QString knowledge_dbpath = mainFilePath + "/knowledge.db";

    if(dbmanager.openDB(detected_places_dbpath))
    {
        qDebug()<<"Places db opened";
    }
    if(knowledgedbmanager.openDB(knowledge_dbpath,"knowledge"))
    {
        qDebug()<<"Knowledge db opened";

        if(knowledgedbmanager.getLearnedPlaceMaxID() == 0)
        {
            qDebug()<<"Starting with empty knowledge";
        }
        else
        {
            int previousKnowledgeSize = knowledgedbmanager.getLearnedPlaceMaxID();

            qDebug()<<"Starting with previous knowledge. Previous number of places: "<<previousKnowledgeSize;

            LearnedPlace::lpCounter = previousKnowledgeSize+1;

            for (int i = 1; i <= previousKnowledgeSize; i++)
            {
                LearnedPlace aPlace = knowledgedbmanager.getLearnedPlace(i);

                learnedPlaces.push_back(aPlace);
            }
        }
    }
}

int main (int argc, char** argv)
{
    // Initialize ROS
    ros::init (argc, argv, "createBDSTISL");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    tau_h = 0;
    tau_r = 2.5;

    // get recognition parameters
    pnh.getParam("tau_h",tau_h);
    pnh.getParam("tau_r",tau_r);
    qDebug()<<"Parameters: "<<tau_h<<tau_r;

    ros::Subscriber plIDSubscriber = nh.subscribe<std_msgs::Int16>("placeDetectionISL/placeID",5, placeCallback);
    ros::Subscriber filePathSubscriber = nh.subscribe<std_msgs::String>("placeDetectionISL/mainFilePath",2,mainFilePathCallback);
    //
    ros::Rate loop(50);
    //
    while(ros::ok())
    {
        ros::spinOnce();
        loop.sleep();
        // if(learnedPlaces.size() >= MIN_NO_PLACES)
        // {
    //         // Perform Top Down BDST recognition to recognize the current place, to use Bottom up recognition use performBottomUpBDSTRecognition function
            int result = -1;//performTopDownBDSTRecognition(tau_r,tau_l,bdst,currentPlace);
    //         // int result= performBottomUpBDSTRecognition(tau_r,tau_l,bdst,currentPlace); //performTopDownBDSTRecognition(1.25,2,bdst,currentPlace);
    //
    //         // if recognized, result is the recognized place id, if -1, the place is not recongized.
    //
            // if(result < 0) // No recognition case
            // {
    //             // convert current place to a learned one and update the topological map
    //             // add learned place to whole places and create tree
            // }
    //
            // else //Recognition case
            // {
    //             // We should just update the place that new place belongs to
    //             // The topological map will not be updated only the last node should be updated
            // }
    //
        // }
    }

    //
    dbmanager.closeDB();

    return 0;
}

double* nodeDiff(treeNode *tn,int nnodes)
{
    double *result = new double[nnodes];
    for(int i = nnodes; i > 0; i--)
    {
        result[nnodes - i] = tn[i].distance - tn[i -1].distance;
    }

    return result;
}

//Function which clusters a single place into subplaces
void clusterPlace(Place pl)
{
        Mat currentInvariants = pl.memberInvariants;
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

        treeNode* placeTree = treecluster(nrows,ncols,data,1,'w',NULL);
        double *differences = nodeDiff(placeTree,ncols-2);
        int clusterCount = 2 + (int)std::distance(differences,std::max_element(differences,differences + ncols-2));
        std::cout <<"Cluster Count is: "<<  clusterCount << std::endl;
        int* placeClusters[clusterCount];
        int* count = new int[clusterCount];
        cuttree(ncols,placeTree,clusterCount,clusterid,placeClusters,count);
        std::vector <subPlace> currentSPs;

        std::cout << "Cols = " << pl.memberInvariants.cols << " Rows = " << pl.memberInvariants.rows << std::endl ;

        for (int i = 0; i < clusterCount; i++){
          subPlace temp_sp;
          int k = 0;
          for (int j = 0; j < count[i]; j++){
            temp_sp.memberInvariants = cv::Mat::zeros(HARMONIC1 * HARMONIC2 * 5,count[i], CV_64F);
            pl.memberInvariants.col(placeClusters[i][j]).copyTo(temp_sp.memberInvariants.col(k));
            k++;
          }
          temp_sp.calculateMeanInvariant();
        }
        std::cout << "placeTree for place ID " << pl.id << ": " << std::endl;
        for (int i = 0; i < ncols - 1 ; i++)
            std::cout << placeTree[i].left << "\t" << placeTree[i].right << "\t"
                      << placeTree[i].distance << "\t" << std::endl;

        delete []clusterid;
}

static inline float computeSquare (float x) { return x*x; }

float calculateCostFunctionv2(float firstDistance, float secondDistance, LearnedPlace closestPlace, Place detected_place)
{

    float result = -1;

    float firstPart = firstDistance;
    float secondPart = firstDistance/secondDistance;
    float votePercentage = 0;

    //  if(DatabaseManager::openDB("/home/hakan/Development/ISL/Datasets/Own/deneme/db1.db"))
    //  {
    // Place aPlace = DatabaseManager::getPlace(closestPlace.id);

    //votePercentage = performSVM(closestPlace.memberInvariants,detected_place.memberInvariants);

    qDebug()<<"Vote percentage"<<votePercentage;

    result = firstPart+secondPart+(1-votePercentage);
    std::cout << "result: " << result<< std::endl;
    return result;
}
