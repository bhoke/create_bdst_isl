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

// A typedef for sorting places based on their distance and index
typedef std::pair<float,int> mypair;

// Comparator function for sorting
bool comparator ( const mypair& l, const mypair& r)
{ return l.first < r.first; }

// Addition operator for std::vector
template <typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b)
{
    assert(a.size() == b.size());

    std::vector<T> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), b.begin(),
                   std::back_inserter(result), std::plus<T>());
    return result;
}

// Clusters a single place into different sub-places
void clusterPlace(Place pl);

// Perform the one-Class SVM calculation
float performSVM(cv::Mat trainingVector, cv::Mat testVector);

float calculateCostFunctionv2(float firstDistance, float secondDistance, LearnedPlace closestPlace, Place detected_place);

// Calculate the mean invariants so that the robot can perform recognition
void calculateMeanInvariantsOfBDST();

//int performTopDownBDSTRecognition(float tau_g, float tau_l, BDST* bdst, Place detected_place);
//int performBottomUpBDSTRecognition(float tau_g, float tau_l, BDST* bdst, Place detected_place);

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

QFile file;
QTextStream strm;
QFile placeTreeFile;
QTextStream placeTreeStream;

// Callback function for place detection. The input is the place id signal from the place detection node
void placeCallback(std_msgs::Int16 placeId)
{
    Place aPlace = dbmanager.getPlace((int)placeId.data);
    clusterPlace(aPlace);

    if(learnedPlaces.size() < MIN_NO_PLACES)
    {
        LearnedPlace alearnedPlace(aPlace);
        learnedPlaces.push_back(alearnedPlace);
        std::cout <<"Places size = "<<learnedPlaces.size() << " < " << MIN_NO_PLACES << " No Recognition" << endl;
        return;
    }

    currentPlaces.push_back(aPlace);
}


// A callback function for the main file, the input is main file path string from the place detection node
void mainFilePathCallback(std_msgs::String mainfp)
{
    std::string tempstr = mainfp.data;

    mainFilePath = QString::fromStdString(tempstr);

    qDebug()<<"Main File Path Callback received"<<mainFilePath;

    QString detected_places_dbpath = mainFilePath;

    detected_places_dbpath.append("/detected_places.db");

    QString knowledge_dbpath = mainFilePath;

    knowledge_dbpath.append("/knowledge.db");

    if(dbmanager.openDB(detected_places_dbpath))
    {
        qDebug()<<"Places db opened";
    }
    // if usePreviousMemory is true in place detection, it constructs previous place invariants and perform bdst calculations
    // the recognition will be based on the previous knowledge
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

    QString placeTreeFileName = mainFilePath;
    placeTreeFileName = placeTreeFileName.append("/placeTree.txt");

    placeTreeFile.setFileName(placeTreeFileName);

    if(placeTreeFile.open(QFile::WriteOnly))
    {
        qDebug() << "Place Tree file is opened";
        placeTreeStream.setDevice(&placeTreeFile);
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

    ros::Subscriber sbc = nh.subscribe<std_msgs::Int16>("placeDetectionISL/placeID",5, placeCallback);
    ros::Subscriber filepathsubscriber = nh.subscribe<std_msgs::String>("placeDetectionISL/mainFilePath",2,mainFilePathCallback);

    ros::Rate loop(50);
    while(ros::ok())
    {
        ros::spinOnce();
        if(learnedPlaces.size() >= MIN_NO_PLACES)
        {
            // Perform Top Down BDST recognition to recognize the current place, to use Bottom up recognition use performBottomUpBDSTRecognition function
            int result = -1;//performTopDownBDSTRecognition(tau_r,tau_l,bdst,currentPlace);
            // int result= performBottomUpBDSTRecognition(tau_r,tau_l,bdst,currentPlace); //performTopDownBDSTRecognition(1.25,2,bdst,currentPlace);

            // if recognized, result is the recognized place id, if -1, the place is not recongized.
            if(result < 0) // No recognition case
            {
                // convert current place to a learned one and update the topological map
                // add learned place to whole places and create tree
            }
            else //Recognition case
            {
                // We should just update the place that new place belongs to
                // The topological map will not be updated only the last node should be updated
            }
        }

        loop.sleep();
    }

    file.close();
    dbmanager.closeDB();

    ros::shutdown();

    qDebug()<< "Close place tree File ";
    placeTreeFile.close();
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
        double **data = new double*[nrows];
        int *clusterid = new int[ncols];

        std::cout << nrows << "   " << ncols << std::endl;
        for(int i = 0; i < nrows; ++i)
        {
            data[i] = new double[ncols];
            for (int j = 0; j < ncols; ++j)
                data[i][j] = currentInvariants.at<float>(i,j);
        }

        treeNode* placeTree = treecluster(nrows,ncols,data,1,'w',NULL);
        double *differences = nodeDiff(placeTree,ncols-2);
        int clusterCount = 1 + (int)std::distance(differences,std::max_element(differences,differences + ncols-2));
        cuttree(ncols,placeTree,clusterCount,clusterid);

        std::cout << "placeTree for place ID " << pl.id << ": " << std::endl;
        for (int i = 0; i < ncols - 1 ; i++)
            std::cout << placeTree[i].left << "\t" << placeTree[i].right << "\t"
                      << placeTree[i].distance << "\t" << differences[i] << std::endl;

        delete []clusterid;
}
static inline float computeSquare (float x) { return x*x; }
float performSVM(cv::Mat trainingVector, cv::Mat testVector)
{
    //  Mat trainingDataMat(4, 2, CV_32FC1, trainingData);
    float result = 0;

    //std::cout << "training vector: " << trainingVector.rows << "x" << trainingVector.cols << std::endl;

    // cv::transpose(trainingVector,trainingVector);

    // cv::transpose(testVector,testVector);

    cv::Mat labelsMat;

    // Set up SVM's parameters
    cv::Ptr<cv::ml::SVM> cvSVM =  cv::ml::SVM::create();
    cvSVM->setType(cv::ml::SVM::ONE_CLASS);
    cvSVM->setKernel(cv::ml::SVM::RBF);
    cvSVM->setGamma((double)1.0/trainingVector.rows);
    cvSVM->setNu(0.15);
    cvSVM->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_ITER, 1000, 1e-8));

    cvSVM->train(trainingVector,cv::ml::ROW_SAMPLE,labelsMat);

    float summ = 0;

    for(int i = 0; i< testVector.rows; i++) summ+=  cvSVM->predict(testVector.row(i));

    result = (float)summ/testVector.rows;

    return result;
}
float calculateCostFunctionv2(float firstDistance, float secondDistance, LearnedPlace closestPlace, Place detected_place)
{

    float result = -1;

    float firstPart = firstDistance;
    float secondPart = firstDistance/secondDistance;
    float votePercentage = 0;

    //  if(DatabaseManager::openDB("/home/hakan/Development/ISL/Datasets/Own/deneme/db1.db"))
    //  {
    // Place aPlace = DatabaseManager::getPlace(closestPlace.id);

    // for KNN results performKNN function can be used, now SVM classification is used
    //votePercentage= performKNN(closestPlace.memberInvariants, secondClosestPlace.memberInvariants, detected_place.memberInvariants);
    votePercentage = performSVM(closestPlace.memberInvariants,detected_place.memberInvariants);

    qDebug()<<"Vote percentage"<<votePercentage;

    result = firstPart+secondPart+(1-votePercentage);
    std::cout << "result: " << result<< std::endl;
    return result;
}
