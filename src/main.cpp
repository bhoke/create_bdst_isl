#include "bubble/bubbleprocess.h"
#include "imageprocess/imageprocess.h"
#include "database/databasemanager.h"
#include "bdst.h"
#include "Utility.h"

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

// Train the SVM classifier
void trainSVM();

// Perform the one-Class SVM calculation
float performSVM(cv::Mat trainingVector, cv::Mat testVector);

float calculateCostFunctionv2(float firstDistance, float secondDistance, LearnedPlace closestPlace, Place detected_place);

// Calculate the mean invariants so that the robot can perform recognition
void calculateMeanInvariantsOfBDST();

int performTopDownBDSTRecognition(float tau_g, float tau_l, BDST* bdst, Place detected_place);
int performBottomUpBDSTRecognition(float tau_g, float tau_l, BDST* bdst, Place detected_place);

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
int tau_l;

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
    tau_r = 1.4;
    tau_r = 2.5;

    // get recognition parameters
    pnh.getParam("tau_h",tau_h);
    pnh.getParam("tau_r",tau_r);
    pnh.getParam("tau_l",tau_l);
    qDebug()<<"Parameters: "<<tau_h<<tau_r<<tau_l;

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

//Function which clusters a single place into subplaces
void clusterPlace(Place pl)
{
        Mat currentInvariants = pl.memberInvariants;
        int nrows = currentInvariants.rows;
        int ncols = currentInvariants.cols;
        double **data = new double*[nrows];
        int *clusterid = new int[ncols];

        std::cout << nrows << "   " << ncols << std::endl;
        for(int i = 0; i < nrows; ++i){
            data[i] = new double[ncols];
            for (int j = 0; j < ncols; ++j)
                data[i][j] = currentInvariants.at<float>(i,j);
        }

        Node* placeTree = treecluster(nrows,ncols,data,1,'w',NULL);

        cuttree(ncols,placeTree,clusterCount,clusterid);
        std::cout << "placeTree for place ID " << pl.id << ": " << std::endl;
        for (int i = 0; i < ncols - 1 ; i++)
            std::cout << placeTree[i].left << "\t" << placeTree[i].right <<
                         " belongs to cluster " << clusterid[i] <<  std::endl;

        delete []clusterid;
}
/*void clusterPlace(Place pl)
//{
//    Mat currentInvariants = pl.memberInvariants;
//    int nrows = currentInvariants.rows;
//    int ncols = currentInvariants.cols;
//    std::cout << ncols << std::endl;

//    cv::Mat sum[ncols-1],treeMean[ncols-1];
//    double var[ncols],sumSq[ncols];
//    int level[ncols];
//    double **data = new double*[nrows];
//    double **distmatrix;
//    int **mask = new int*[nrows];
//    double weight[nrows];

//    for(int i = 0; i < nrows; ++i){
//        mask[i] = new int[ncols];
//        data[i] = new double[ncols];
//        weight[i] = 1.0;
//        for (int j = 0; j < ncols; ++j)
//        {
//            mask[i][j] = 1;
//            data[i][j] = currentInvariants.at<float>(i,j);
//        }
//    }

//    for (int i= 0 ; i < ncols - 1; ++i)
//    {
//        treeMean[i] = cv::Mat::zeros(nrows,1,CV_64F);
//        sum[i]  = cv::Mat::zeros(nrows,1,CV_64F);
//        sumSq[i]  = 0.0;
//        level[i] = 0;
//        var[i] = 0.0;
//    }

//    Node* placeTree = treecluster(nrows,ncols,data,1,'s',distmatrix);

//    for(int i = 0; i < ncols -1 ; ++i)
//    {
//        //        std::cout << "Node " << -i-1 << "\t";
//        int k = placeTree[i].left;
//        std::cout << k << "\t";
//        cv::Mat curInv,sqrInv;
//        if (k < 0)
//        {
//            sum[i] += sum[-k-1];
//            sumSq[i] += sumSq[-k-1];
//            level[i] += level[-k-1];
//        }
//        else
//        {
//            ++level[i];
//            curInv = currentInvariants.col(k).clone();
//            curInv.convertTo(curInv,CV_64F);
//            sum[i] += curInv;
//            cv::mulTransposed(curInv,sqrInv,true);
//            sumSq[i] += sqrInv.at<double>(0,0);
//        }
//        int j = placeTree[i].right;
//        std::cout << j << "\t" << placeTree[i].distance << std::endl;
//        if (j < 0)
//        {
//            sum[i] += sum[-j-1];
//            sumSq[i] += sumSq[-j-1];
//            level[i] += level[-j-1];
//        }
//        else
//        {
//            ++level[i];
//            curInv = currentInvariants.col(j).clone();
//            curInv.convertTo(curInv,CV_64F);
//            sum[i] += curInv;
//            cv::mulTransposed(curInv,sqrInv,true);
//            sumSq[i] += sqrInv.at<double>(0,0);
//        }
//        treeMean[i] = sum[i]/level[i];
//        cv::Mat meanSq;
//        cv::mulTransposed(treeMean[i],meanSq,true);
//        var[i] = sumSq[i] / level[i] - meanSq.at<double>(0,0);
//    }
}*/
static inline float computeSquare (float x) { return x*x; }
float performSVM(cv::Mat trainingVector, cv::Mat testVector)
{
    //  Mat trainingDataMat(4, 2, CV_32FC1, trainingData);
    float result = 0;

    //std::cout << "training vector: " << trainingVector.rows << "x" << trainingVector.cols << std::endl;

    cv::transpose(trainingVector,trainingVector);

    cv::transpose(testVector,testVector);

    Mat labelsMat;

    // Set up SVM's parameters
    CvSVMParams params;
    params.svm_type    = CvSVM::ONE_CLASS;
    params.kernel_type = CvSVM::RBF;
    params.gamma = (double)1.0/trainingVector.rows;
    params.nu = 0.15;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 1e-8);

    // Train the SVM
    CvSVM SVM;

    SVM.train(trainingVector, labelsMat, Mat(), Mat(), params);

    float summ = 0;

    for(int i = 0; i< testVector.rows; i++){
        //   Mat singleTest =
        summ+=  SVM.predict(testVector.row(i));
    }


    ///   cv::Scalar summ = cv::sum(resultsVector);

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
