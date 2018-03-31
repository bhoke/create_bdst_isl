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

using namespace std;
using namespace cv;


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

// Calculates the distance matrix based on mean invariants of places
double** calculateDistanceMatrix(int nrows, int ncols, double** data, int** mask, int transpose);

// Train the SVM classifier
void trainSVM();

// Perform the one-Class SVM calculation
float performSVM(cv::Mat trainingVector, cv::Mat testVector);

// Perform kNN classification between first and second closest places
float performKNN(Mat trainingVector, Mat secondtrainingVector, Mat testVector);

float calculateCostFunctionv2(float firstDistance, float secondDistance, LearnedPlace closestPlace, Place detected_place);

// Perform the BDST operations for creating the BDST
void performBDSTCalculations();

// Calculate the Binary BDST
Node* calculateBinaryBDST(int nrows, int ncols, double** data);

void calculateMergedBDSTv2(float tau_h, int nnodes, int noplaces, Node* tree, BDST* bdst);

// Calculate the mean invariants so that the robot can perform recognition
void calculateMeanInvariantsOfBDST(BDST* aLevel);

int performTopDownBDSTRecognition(float tau_g, float tau_l, BDST* bdst, Place detected_place);

int performBottomUpBDSTRecognition(float tau_g, float tau_l, BDST* bdst, Place detected_place);

//double calculateLevelCostFunction(Place place, std::vector<double> closestMeanInvariant, std::vector<double> secondClosestMeanInvariant, float voteRate);

// OPENCV modifikasyonu
double compareHistHK( InputArray _H1, InputArray _H2, int method );

// FOR debugging
std::vector< std::vector<float> > readInvariantVectors();

ros::Timer timer;

std::vector<LearnedPlace> places;

int learnedPlaceCounter = 1;

std::vector< std::vector<float> > invariants;

TopologicalMap topmap;

Place currentPlace;

BDST* bdst ;

DatabaseManager dbmanager;

DatabaseManager knowledgedbmanager;

uint lastTopMapNodeId;

QString mainFilePath;

std::vector<Place> currentPlaces;


double tau_h, tau_r;
int tau_l;

// a function to construct invariant matrix from learned places
void constructInvariantsMatrix(std::vector<LearnedPlace> plcs)
{
    if(invariants.size() > 0)
        invariants.clear();
    for(uint i = 0 ; i < plcs.size(); i++)
    {
        std::vector<float> invariant = plcs.at(i).meanInvariant;
        invariants.push_back(invariant);
    }
}

// a function to update topological map and insert it to the knowledge database
void updateTopologicalMap(int node1, int node2)
{
    std::pair<int,int> mapNode;


    lastTopMapNodeId = node2;
    // Just to shift the id's from 0 to 1
    mapNode.first = node1;
    mapNode.second = node2;

    topmap.connections.push_back(mapNode);


    // Write relation to the topological map
    knowledgedbmanager.insertTopologicalMapRelation(topmap.connections.size(),mapNode);
}


// A function to convert a place to a learned place
LearnedPlace convertPlacetoLearnedPlace(Place place)
{
    LearnedPlace aplace;

    aplace.id = learnedPlaceCounter;
    aplace.meanInvariant = place.meanInvariant;
    aplace.memberInvariants = place.memberInvariants;
    aplace.memberIds = place.memberIds;

    if(aplace.memberPlaces.empty())
    {
        aplace.memberPlaces = cv::Mat(1,1,CV_16UC1);
        aplace.memberPlaces.at<unsigned short>(0,0) = (unsigned short)place.id;
    }
    // insert learned place to the knowledge dataset
    knowledgedbmanager.insertLearnedPlace(aplace);

    learnedPlaceCounter++;

    return aplace;
}

QFile file;
QTextStream strm;
QFile placeTreeFile;
QTextStream placeTreeStream;

// Callback function for place detection. The input is the place id signal from the place detection node
void placeCallback(std_msgs::Int16 placeId)
{
    // read the place from the database
    Place aPlace = dbmanager.getPlace((int)placeId.data);
    //int m=0,n=0;
    clusterPlace(aPlace);

    if(places.size() < MIN_NO_PLACES && places.size() >= 1)
    {
        updateTopologicalMap(places.back().id,aPlace.id);
    }

    if(places.size() < MIN_NO_PLACES)
    {
        LearnedPlace alearnedPlace = convertPlacetoLearnedPlace(aPlace);
        places.push_back(alearnedPlace);
        qDebug()<<"Places size"<<places.size() << endl;
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

            learnedPlaceCounter = previousKnowledgeSize+1;

            for (int i = 1; i <= previousKnowledgeSize; i++)
            {
                LearnedPlace aPlace = knowledgedbmanager.getLearnedPlace(i);

                places.push_back(aPlace);
            }
            constructInvariantsMatrix(places);
            performBDSTCalculations();
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
        if(places.size() >= MIN_NO_PLACES)
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
    std::cout << ncols << std::endl;

    cv::Mat sum[ncols-1],treeMean[ncols-1];
    double var[ncols],sumSq[ncols];
    int level[ncols];
    double **data = new double*[nrows];
    double **distmatrix;
    int **mask = new int*[nrows];
    double weight[nrows];

    for(int i = 0; i < nrows; ++i){
        mask[i] = new int[ncols];
        data[i] = new double[ncols];
        weight[i] = 1.0;
        for (int j = 0; j < ncols; ++j)
        {
            mask[i][j] = 1;
            data[i][j] = currentInvariants.at<float>(i,j);
        }
    }

    for (int i= 0 ; i < ncols - 1; ++i)
    {
        treeMean[i] = cv::Mat::zeros(nrows,1,CV_64F);
        sum[i]  = cv::Mat::zeros(nrows,1,CV_64F);
        sumSq[i]  = 0.0;
        level[i] = 0;
        var[i] = 0.0;
    }

    Node* placeTree = treecluster(nrows,ncols,data,mask,weight,1,'e','s',distmatrix);

    for(int i = 0; i < ncols -1 ; ++i)
    {
        //        std::cout << "Node " << -i-1 << "\t";
        int k = placeTree[i].left;
        std::cout << k << "\t";
        cv::Mat curInv,sqrInv;
        if (k < 0)
        {
            sum[i] += sum[-k-1];
            sumSq[i] += sumSq[-k-1];
            level[i] += level[-k-1];
        }
        else
        {
            ++level[i];
            curInv = currentInvariants.col(k).clone();
            curInv.convertTo(curInv,CV_64F);
            sum[i] += curInv;
            cv::mulTransposed(curInv,sqrInv,true);
            sumSq[i] += sqrInv.at<double>(0,0);
        }
        int j = placeTree[i].right;
        std::cout << j << "\t" << placeTree[i].distance << std::endl;
        if (j < 0)
        {
            sum[i] += sum[-j-1];
            sumSq[i] += sumSq[-j-1];
            level[i] += level[-j-1];
        }
        else
        {
            ++level[i];
            curInv = currentInvariants.col(j).clone();
            curInv.convertTo(curInv,CV_64F);
            sum[i] += curInv;
            cv::mulTransposed(curInv,sqrInv,true);
            sumSq[i] += sqrInv.at<double>(0,0);
        }
        treeMean[i] = sum[i]/level[i];
        cv::Mat meanSq;
        cv::mulTransposed(treeMean[i],meanSq,true);
        var[i] = sumSq[i] / level[i] - meanSq.at<double>(0,0);
        //        std::cout << std::setprecision(15) <<"variance = " << var[i] << std::endl;
        //        std::cout << std::setprecision(15) <<"sumSq = " << sumSq[i] << std::endl;
        //        std::cout << std::setprecision(15) <<"MeanSq = "<< meanSq.at<double>(0,0) << std::endl;
        //        std::cout << "Level = "<< level[i] << std::endl;
        //        std:: cout << "------------------------------------------------------" << std::endl;
        //        std::cin.get();
    }
}

// a function to construct BDST based on place invariants
void performBDSTCalculations()
{
    const int nrows = invariants.size();
    const int ncols = invariants[0].size();

    double** data = new double*[nrows];

    //  int** mask = new int*[nrows];

    int i;

    //double** distmatrix;

    for (i = 0; i < nrows; i++)
    {
        data[i] = new double[ncols];

        //mask[i] = new int[ncols];
    }

    for(i = 0; i < nrows; i++)
    {
        for(int j = 0 ; j < ncols; j++)
        {
            data[i][j] = invariants[i][j];
            //   qDebug()<<data[i][j];
        }

    }

    // construct binary tree based on the place invariants
    Node* binarytree = calculateBinaryBDST(nrows,ncols,data);

    if(bdst)
        bdst->deleteLater();

    bdst =  new BDST;

    // calculateMergedBDST(tau_h,nrows-1,nrows,binarytree,bdst);

    // construct merged bdst from the contructed binary tree
    calculateMergedBDSTv2(tau_h,nrows-1,nrows,binarytree,bdst);


    free(binarytree);

    for ( i = 0; i < nrows; i++){
        //delete [] mask[i];
        delete [] data[i];
    }
    // delete [] mask;
    delete [] data;

}

double** calculateDistanceMatrix(int nrows, int ncols, double** data, int** mask,int transpose)
/* Calculate the distance matrix between genes using the Euclidean distance. */
{
    int i, j;
    double** distMatrix;
    double* weight = NULL;

    if(transpose == 0)
    {
        weight = new double[ncols];
        for (i = 0; i < ncols; i++) weight[i] = 1.0;
    }
    else
    {
        weight = new double[nrows];
        for (i = 0; i < nrows; i++) weight[i] = 1.0;
    }
    distMatrix = distancematrix(nrows, ncols, data, mask, weight, 'e',transpose);

    if (!distMatrix)
    {
        printf ("Insufficient memory to store the distance matrix\n");
        delete weight;
        return NULL;
    }
    // This part is for changing the values of distMatrix to the MATLAB format. Multiply by the length of the feature vector (600) and take the sqrt
    if(transpose == 0)
    {
        for(i = 0; i < nrows; i++)
            for(j = 0; j< i; j++)
                distMatrix[i][j] = sqrt(distMatrix[i][j]*600);
    }
    else
    {
        for(i = 0; i < ncols; i++)
            for(j = 0; j< i; j++)
                distMatrix[i][j] = sqrt(distMatrix[i][j]*600);
    }

    //    printf("   Place:");
    //    for(i=0; i<nrows-1; i++) printf("%6d", i);
    //    printf("\n");
    //    for(i=0; i<nrows; i++)
    //    { printf("Gene %2d:",i);
    //        for(j=0; j<i; j++) printf(" %5.4f",distMatrix[i][j]);
    //        printf("\n");
    //    }
    //    printf("\n");
    delete weight;
    return distMatrix;
}

Node* calculateBinaryBDST(int nrows, int ncols, double** data)
{
    int** mask = new int*[nrows];

    double** distmatrix;

    for (int i = 0; i < nrows; i++)
    {

        mask[i] = new int[ncols];
    }


    for(int i = 0; i < nrows; i++)
    {
        for(int j = 0; j < ncols; j++ )
        {
            mask[i][j] = 1;

        }
    }

    distmatrix = calculateDistanceMatrix(nrows, ncols, data, mask,0);

    //const int nnodes = nrows-1;

    Node* tree;

    //    printf("\n");
    //    printf("================ Pairwise single linkage clustering ============\n");
    /* Since we have the distance matrix here, we may as well use it. */
    tree = treecluster(nrows, ncols, 0, 0, 0, 0, 'e', 's', distmatrix);
    /* The distance matrix was modified by treecluster, so we cannot use it any
       * more. But we still need to deallocate it here.
       * The first row of distmatrix is a single null pointer; no need to free it.
       */
    for (int i = 1; i < nrows; i++) delete distmatrix[i];
    delete distmatrix;

    if (!tree)
    { // Indication that the treecluster routine failed

        qDebug()<<"treecluster routine failed due to insufficient memory";

        return NULL;
    }

    else
    {
        return tree;
    }

    return NULL;

}

void calculateMergedBDSTv2(float tau_h, int nnodes, int noplaces, Node* tree, BDST* bdst)
{
    //int levelcount = 0;
    //int nodecount = 0;

    std::vector<TreeLeaf> leaves;

    QString homepath = mainFilePath;

    if(homepath.isEmpty())
    {
        homepath = QDir::homePath();
    }

    homepath.append("/mergedbdst.txt");

    QFile file(homepath);
    if(file.open(QFile::WriteOnly))
    {
        //qDebug()<<"BDST File is opened for writing";

    }

    QTextStream txtstr(&file);
    // BDST bdst;

    //  bdst.levels.resize(1);

    // places start from 0 to

    // These are the initial leaves, we will merge them

    for(int i = 0 ; i < nnodes; i++)
    {
        TreeLeaf leaf;

        // The left child
        leaf.left = tree[i].left;

        // If left child is less than 0 that means it is an inner node. MATLAB implementation uses positive index so we switch the index to positive
        if(leaf.left < 0) leaf.left=noplaces-leaf.left;

        // The right child
        leaf.right = tree[i].right;

        // If right child is less than 0 that means it is an inner node. MATLAB implementation uses positive index so we switch the index to positive
        if(leaf.right < 0) leaf.right=noplaces-leaf.right;

        // The value of the leaf
        leaf.val = tree[i].distance;
        // qDebug()<<"Tree distance"<<leaf.val;

        // While building merged BDST we check whether the leaf is used or not
        leaf.isused = false;

        // Each leaf has a connection to the parent node. It is marked by this variable
        leaf.parentConnection = noplaces+i+1;

        leaves.push_back(leaf);

    }

    for(uint i = 0; i < leaves.size(); i++)
    {
        TreeLeaf aLeaf = leaves[i];
        //qDebug()<<"Leaf"<<aLeaf.left<<aLeaf.right<<aLeaf.isused;

        // qDebug()<<"i is"<<i;

        // The leaf has been unused we should check the perimeter
        if(!aLeaf.isused)
        {
            Level aLevel;

            aLevel.members.push_back(aLeaf.left);
            aLevel.members.push_back(aLeaf.right);

            aLevel.parentNodes.push_back(aLeaf.parentConnection);

            aLevel.val = aLeaf.val + tau_h;

            leaves[i].isused = true;

            if(aLeaf.right < noplaces && aLeaf.left < noplaces)
                txtstr<<aLeaf.left+1<<" "<<aLeaf.right+1<<" "<<aLeaf.val+tau_h<<" "<<"\n";
            else if(aLeaf.right < noplaces)
                txtstr<<aLeaf.left<<" "<<aLeaf.right+1<<" "<<aLeaf.val+tau_h<<" "<<"\n";
            else if(aLeaf.left < noplaces)
                txtstr<<aLeaf.left+1<<" "<<aLeaf.right<<" "<<aLeaf.val+tau_h<<" "<<"\n";
            else
                txtstr<<aLeaf.left<<" "<<aLeaf.right<<" "<<aLeaf.val+tau_h<<" "<<"\n";



            // We are looking for the following leaves
            for(uint k = i ; k < leaves.size(); k++)
            {
                // The leaf should be unused and the value should be less than level value
                if(!leaves[k].isused && leaves[k].val <= aLevel.val)
                {
                    if(aLeaf.parentConnection == leaves[k].left && leaves[k].right > noplaces)
                    {

                        //  aLevel.parentNodes.push_back(leaves[k].left);
                        aLevel.parentNodes.push_back(leaves[k].right);

                    }
                    else if(aLeaf.parentConnection == leaves[k].left && leaves[k].right < noplaces)
                    {

                        //  aLevel.parentNodes.push_back(leaves[k].left);
                        aLevel.members.push_back(leaves[k].right);

                    }
                    else if(aLeaf.parentConnection == leaves[k].right && leaves[k].left > noplaces)
                    {
                        aLevel.parentNodes.push_back(leaves[k].left);
                    }
                    else if(aLeaf.parentConnection == leaves[k].right && leaves[k].left < noplaces)
                    {
                        aLevel.members.push_back(leaves[k].left);

                    }


                    if(aLeaf.parentConnection == leaves[k].right || aLeaf.parentConnection == leaves[k].left )
                    {
                        leaves[k].isused = true;

                        aLevel.parentNodes.push_back(leaves[k].parentConnection);

                        if(leaves[k].right < noplaces && leaves[k].left < noplaces)
                            txtstr<<leaves[k].left+1<<" "<<leaves[k].right+1<<" "<<aLevel.val<<" "<<"\n";
                        else if(leaves[k].right < noplaces)
                            txtstr<<leaves[k].left<<" "<<leaves[k].right+1<<" "<<aLevel.val<<" "<<"\n";
                        else if(leaves[k].left < noplaces)
                            txtstr<<leaves[k].left+1<<" "<<leaves[k].right<<" "<<aLevel.val<<" "<<"\n";
                        else
                            txtstr<<leaves[k].left<<" "<<leaves[k].right<<" "<<aLevel.val<<" "<<"\n";

                        aLeaf.parentConnection = leaves[k].parentConnection;
                    }
                }
            }

            // We are looking for the following leaves second time
            for(uint k = i ; k < leaves.size(); k++)
            {
                if(!leaves[k].isused && leaves[k].val <= aLevel.val)
                {
                    // I have found the parent connection so I should add this to leaf to the current level
                    if(std::find(aLevel.parentNodes.begin(), aLevel.parentNodes.end(), leaves[k].parentConnection)!=aLevel.parentNodes.end())
                    {
                        aLevel.members.push_back(leaves[k].right);
                        aLevel.members.push_back(leaves[k].left);

                        leaves[k].isused = true;

                        if(leaves[k].right < noplaces && leaves[k].left < noplaces)
                            txtstr<<leaves[k].left+1<<" "<<leaves[k].right+1<<" "<<aLevel.val<<" "<<"\n";
                        else if(leaves[k].right < noplaces)
                            txtstr<<leaves[k].left<<" "<<leaves[k].right+1<<" "<<aLevel.val<<" "<<"\n";
                        else if(leaves[k].left < noplaces)
                            txtstr<<leaves[k].left+1<<" "<<leaves[k].right<<" "<<aLevel.val<<" "<<"\n";
                        else
                            txtstr<<leaves[k].left<<" "<<leaves[k].right<<" "<<aLevel.val<<" "<<"\n";
                    }
                }
            }
            bdst->levels.append(aLevel);
        }
    }

    for(int j = 0; j < bdst->levels.size(); j++)
    {
        bdst->levels[j].connectionIndex = *std::max_element(bdst->levels.at(j).parentNodes.begin(),bdst->levels.at(j).parentNodes.end());

        if((j < bdst->levels.size()-1 != false) && bdst->levels[j+1].members.size() > 0)
        {

            if(std::find(bdst->levels[j+1].members.begin(), bdst->levels[j+1].members.end(), bdst->levels[j].connectionIndex)==bdst->levels[j+1].members.end())
            {
                bdst->levels[j+1].members.push_back( bdst->levels[j].connectionIndex );

            }
        }
        // calculateMeanInvariantForBDSTLevel( &bdst->levels[j]);
    }

    calculateMeanInvariantsOfBDST( bdst);
    file.close();

    qDebug()<<"Finished";
}

std::vector< std::vector<float> >  readInvariantVectors()
{
    QFile file("/home/hakan/ros_workspace/createBDSTISL/invariants.txt");

    std::vector< std::vector<float> > invariants;


    /****************** THIS PART IS FOR MANUALLY READING INVARIANTS (DEBUGGING ONLY)  ********************************/

    if(file.open(QFile::ReadOnly))
    {
        QTextStream stream(&file);

        // double num1,num2,num3,num4,num5,num6;
        int j = 0;

        while(!stream.atEnd())
        {
            // stream>>num1>>num2>>num3>>num4>>num5>>num6;
            QString line = stream.readLine();
            QStringList list = line.split("\t");
            //    qDebug()<<list<<list.size();
            if(j  == 0)
            {
                invariants.resize(list.size());
            }

            if(j < 600)
            {
                for(int i = 0 ; i < list.size(); i++)
                {
                    invariants[i].push_back(list.at(i).toFloat());
                }

                j++;
            }
        }
        file.close();
    }

    return invariants;
}
// a function to train SVM
void trainSVM()
{
    // Set up training data
    // float labels[4] = {1.0, 1.0, 1.0, 1.0};
    // Mat labelsMat(4, 1, CV_32FC1, labels);

    Mat labelsMat;

    float trainingData[4][2] = { {501, 10}, {502, 12}, {501, 13}, {503, 10} };
    Mat trainingDataMat(4, 2, CV_32FC1, trainingData);

    // Set up SVM's parameters
    CvSVMParams params;
    params.svm_type    = CvSVM::ONE_CLASS;
    params.kernel_type = CvSVM::RBF;
    params.gamma = 0.5;
    params.nu = 0.5;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);

    // Train the SVM
    CvSVM SVM;
    SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

    Mat testData(1,2,CV_32FC1);

    testData.at<float>(0,0) = 501;
    testData.at<float>(0,1) = 10;

    // qDebug()<<testData.at<float>(0,1);

    qDebug()<<SVM.predict(testData,true);

}

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
static inline float computeSquare (float x) { return x*x; }

int performBottomUpBDSTRecognition(float tau_g, float tau_l, BDST *bdst, Place detected_place)
{

    Level currentLevel;
    int levelCount = 0;

    std::vector<int> visitedNodes;

    /** In topological map places start from 1, in bdst they start from 0 **/
    uint previousPlaceID = lastTopMapNodeId;

    previousPlaceID = previousPlaceID - 1;
    /** *******************************************************************/

    int currentLevelCount = 0;

    // qDebug()<<detected_place.<uint>(1,0);

    /**  *********** We should find the level of the previous children ******************/

    for(int i = 0; i < bdst->levels.size(); i++)
    {
        bool isfound = false;

        for(uint j = 0; j< bdst->levels[i].members.size(); j++)
        {

            if(previousPlaceID == bdst->levels[i].members[j])
            {
                currentLevel = bdst->levels[i];
                currentLevelCount = i;
                isfound = true;
                break;
            }
        }

        if(isfound)
            break;

    }

    /** ********************************************************************************/
    // We are going from bottom to up
    while(1)
    {

        //std::cout << "currentLevel.members.size()" << currentLevel.members.size() << std::endl;

        std::vector< mypair> distpairs;
        mypair distpair;

        // qDebug()<<"Current Level member size "<<currentLevel.members.size();

        /** For each member of the current level we calculate the distances **/
        for(uint k = 0; k < currentLevel.members.size(); k++)
        {

            // Get the member number
            uint aMember = currentLevel.members.at(k);

            float sum_of_elems = 0;

            std::vector<float> invariant;

            // If the member is a terminal node
            if(aMember < invariants.size())
                invariant = invariants.at(aMember);

            // If it is not a terminal node, we should get the mean Invariant
            else
            {
                for(int j = 0; j < bdst->levels.size(); j++)
                {
                    if(bdst->levels.at(j).connectionIndex == aMember)
                    {
                        invariant = bdst->levels.at(j).meanInvariant;
                        break;
                    }
                }
            }

            // We get the place's mean invariant and transform to an std::vector
            std::vector<float>  placeInvariant = detected_place.meanInvariant;

            // This is the result string
            std::vector<float> result;

            // Now we take the difference between the member and detected place invariant
            std::transform(invariant.begin(),invariant.end(), placeInvariant.begin(),
                           std::back_inserter(result),
                           std::minus<float>());

            // Now we get the square of the result to eliminate minuses
            std::transform(result.begin(), result.end(), result.begin(), computeSquare);

            // We are summing the elements of the result
            sum_of_elems =std::accumulate(result.begin(),result.end(),0.0);

            sum_of_elems = sqrt(sum_of_elems);


            // We are now collecting the difference and the indexes
            distpair.first = sum_of_elems;
            distpair.second = aMember;

            distpairs.push_back(distpair);

            //   qDebug()<<"result"<<sum_of_elems;

        }

        // Now we are sorting in ascending order the distance and member pairs
        std::sort(distpairs.begin(),distpairs.end(),comparator);

        // We find the closest and second closest members
        mypair firstClosestMember = distpairs.at(0);
        mypair secondClosestMember = distpairs.at(1);

        // If it is not a terminal node, then we should go one level down
        if(firstClosestMember.second >= invariants.size())
        {
            bool isvisited = false;

            for(int j = 0; j < visitedNodes.size(); j++)
            {
                if(firstClosestMember.second == visitedNodes[j])
                {
                    isvisited =true;

                    if(levelCount < tau_l)
                    {
                        bool isfound = false;

                        for(uint kl = currentLevelCount+1; kl < bdst->levels.size(); kl++)
                        {
                            for(uint l = 0; l < bdst->levels[kl].members.size(); l++)
                            {
                                if(bdst->levels.at(kl).members.at(l) == currentLevel.connectionIndex)
                                {
                                    visitedNodes.push_back(currentLevel.connectionIndex);
                                    currentLevel = bdst->levels[kl];
                                    qDebug()<<"Going one level up"<<"new level is"<<currentLevel.val;
                                    levelCount++;
                                    currentLevelCount = kl;
                                    isfound = true;
                                    break;
                                }
                            }

                            if(isfound)
                                break;
                        }

                        if(!isfound)
                        {
                            qDebug()<<"Not Recognized!!";
                            return -1;

                        }

                    }
                    else{
                        qDebug()<<"Not Recognized!!";

                        return -1;
                    }

                }

                if(isvisited)
                    break;

            }

            if(!isvisited)
            {
                for(uint j = 0; j < bdst->levels.size(); j++)
                {
                    if(bdst->levels.at(j).connectionIndex == firstClosestMember.second)
                    {
                        currentLevel = bdst->levels.at(j);
                        break;
                    }
                }

            }
        }
        // We have found the closest terminal node, now we should calculate the cost function and check if it is recognized
        else
        {
            qDebug()<<"Closest terminal node"<<firstClosestMember.second +1;
            qDebug()<<"Second closest terminal node"<<secondClosestMember.second +1;

            //  if(dbmanager.openDB("/home/hakan/Development/ISL/Datasets/Own/deneme/db1.db"))
            //   {
            LearnedPlace aPlace = dbmanager.getLearnedPlace((firstClosestMember.second+1));
            Place aPlace2 = dbmanager.getPlace((secondClosestMember.second+1));

            float costValue =  calculateCostFunctionv2(firstClosestMember.first,secondClosestMember.first,aPlace,detected_place);

            // float costValue =  calculateCostFunction(firstClosestMember.first,secondClosestMember.first,aPlace,detected_place);

            if(costValue <= tau_g)
            {
                qDebug()<<"Recognized";
                return firstClosestMember.second;
            }
            // We should check if we can go further up otherwise we should check if we can go top-down
            else
            {
                if(levelCount < tau_l)
                {
                    bool isfound = false;
                    for(uint j = currentLevelCount+1; j < bdst->levels.size(); j++)
                    {
                        for(uint l = 0; l < bdst->levels[j].members.size(); l++)
                        {
                            if(bdst->levels.at(j).members.at(l) == currentLevel.connectionIndex)
                            {
                                visitedNodes.push_back(currentLevel.connectionIndex);
                                currentLevel = bdst->levels[j];
                                qDebug()<<"Going one level up"<<"new level is"<<currentLevel.val;
                                levelCount++;
                                currentLevelCount = j;
                                isfound = true;
                                break;
                            }
                        }

                        if(isfound)
                            break;
                    }

                    if(!isfound)
                    {
                        qDebug()<<"Not Recognized!!";
                        return -1;

                    }

                }
                else{

                    qDebug()<<"Not Recognized!!";

                    return -1;
                }

            }

        }

    }


    return -1;

}

int performTopDownBDSTRecognition(float tau_g, float tau_l, BDST *bdst, Place detected_place)
{
    Level currentLevel;
    // qDebug()<<detected_place.memberIds.at<uint>(1,0);
    currentLevel = bdst->levels.at(bdst->levels.size()-1);
    int levelIndex = bdst->levels.size()-1;
    qDebug() << "Current Place is : " << detected_place.id;
    // We are going from top to down
    while(1)// for(int i = bdst->levels.size()-1; i >= 0; i--)
    {
        std::vector< mypair> distpairs;
        mypair distpair;

        // For each member of the first level
        for(uint k = 0; k < currentLevel.members.size(); k++)
        {
            // Get the member number
            int aMember = currentLevel.members.at(k);

            float sum_of_elems = 0;

            std::vector<float> invariant;

            // If the member is a terminal node
            if(aMember < invariants.size())
            {
                invariant = invariants.at(aMember);
            }
            // If it is not a terminal node, we should get the mean Invariant
            else
            {
                /*  levelIndex--;
                if(levelIndex< 0) return -1;*/

                for(uint j = 0; j < levelIndex; j++)
                {
                    if(bdst->levels.at(j).connectionIndex == aMember)
                    {
                        invariant = bdst->levels.at(j).meanInvariant;
                        break;
                    }
                }
            }

            // We get the place's mean invariant and transform to an std::vector
            std::vector<float>  placeInvariant = detected_place.meanInvariant;

            // This is the result string
            std::vector<float> result;

            // Now we take the difference between the member and detected place invariant
            std::transform(invariant.begin(),invariant.end(), placeInvariant.begin(),
                           std::back_inserter(result),
                           std::minus<float>());

            // Now we get the square of the result to eliminate minuses
            std::transform(result.begin(), result.end(), result.begin(), computeSquare);

            // We are summing the elements of the result
            sum_of_elems =std::accumulate(result.begin(),result.end(),0.0);//#include <numeric>

            sum_of_elems = sqrt(sum_of_elems);

            // We now take the square root
            // std::transform(result.begin(), result.end(), result.begin(), (float(*)(float)) sqrt);

            // We are now collecting the difference and the indexes
            distpair.first = sum_of_elems;
            distpair.second = aMember;

            distpairs.push_back(distpair);

            //  qDebug()<<"Distance result: "<<sum_of_elems<<aMember;

        }

        // Now we are sorting in ascending order the distance and member pairs
        std::sort(distpairs.begin(),distpairs.end(),comparator);

        // We find the closest and second closest members
        mypair firstClosestMember = distpairs.at(0);
        mypair secondClosestMember = distpairs.at(1);

        // If it is not a terminal node, then we should go one level down
        if(firstClosestMember.second >= invariants.size())
        {
            levelIndex--;
            if(levelIndex < 0) return -1;

            for(uint j = 0; j < bdst->levels.size(); j++)
            {
                if(bdst->levels.at(j).connectionIndex == firstClosestMember.second)
                {
                    currentLevel = bdst->levels.at(j);
                    break;
                }
            }


        }
        // We have found the closest terminal node, now we should calculate the cost function and check if it is recognized
        else
        {
            qDebug()<<"Closest terminal node"<<firstClosestMember.second +1;
            qDebug()<<"Second closest terminal node"<<secondClosestMember.second +1;

            //  if(dbmanager.openDB("/home/hakan/Development/ISL/Datasets/Own/deneme/db1.db"))
            //   {

            LearnedPlace aPlace = dbmanager.getLearnedPlace(firstClosestMember.second+1);//dbmanager.getPlace((firstClosestMember.second+1));

            //Place aPlace = dbmanager.getPlace((firstClosestMember.second+1));

            //std::cout << "aPlace.memberInvariants.size()" << aPlace.memberInvariants.rows << "x" << aPlace.memberInvariants.cols << std::endl;

            float costValue = 100.0;

            if(secondClosestMember.second < invariants.size())
            {

                //LearnedPlace aPlace2 = dbmanager.getLearnedPlace(secondClosestMember.second+1);//dbmanager.getPlace((secondClosestMember.second+1));

                // costValue = calculateCostFunctionv3(firstClosestMember.first,secondClosestMember.first,aPlace,detected_place);
                costValue = calculateCostFunctionv2(firstClosestMember.first,secondClosestMember.first,aPlace,detected_place);
            }
            else
                //costValue =  calculateCostFunctionv3(firstClosestMember.first,secondClosestMember.first,aPlace,detected_place);
                costValue = calculateCostFunctionv2(firstClosestMember.first,secondClosestMember.first,aPlace,detected_place);

            if(costValue <= tau_g)
            {
                qDebug()<<"Recognized";
                qDebug()<<"Distance is: "<< firstClosestMember.first;
                return firstClosestMember.second;
            }

            qDebug()<<"Not Recognized Searching for next level...";

            // Search for the next level;
            bool isNextLevelFound = false;
            levelIndex--;
            if(levelIndex < 0) return -1;
            for(uint k = 0; k < currentLevel.members.size(); k++)
            {
                for(uint ll = 0; ll < levelIndex; ll++)
                {
                    if(currentLevel.members[k] == bdst->levels[ll].connectionIndex)
                    {
                        currentLevel = bdst->levels[ll];
                        isNextLevelFound = true;
                        break;
                    }
                }


            }

            if(!isNextLevelFound)
            {
                qDebug()<<"Not Recognized!! Returning...";
                return -1;
            }

        }

    } //while(true)


    return -1;
}

void calculateMeanInvariantsOfBDST(BDST *bdst)
{
    for(int j =0 ; j < bdst->levels.size(); j++){

        Level* aLevel = &bdst->levels[j];

        std::vector<float> sum;

        for(int i = 0 ; i < aLevel->members.size(); i++)
        {

            if(i == 0)
            {
                if(aLevel->members.at(i) < invariants.size())
                {
                    sum = invariants[aLevel->members.at(i)];
                }
                else
                {
                    //for(int k = 0; k < bdst->levels.size(); k++)
                    for(int k = 0; k < j; k++)
                    {
                        if(bdst->levels.at(k).connectionIndex == aLevel->members.at(i) )
                        {
                            sum = bdst->levels.at(k).meanInvariant;
                            break;

                        }
                    }
                }
            }
            else
            {
                if(aLevel->members.at(i) < invariants.size())
                {
                    sum = sum + invariants[aLevel->members.at(i)];
                }
                else
                {
                    // this part is corrected by Esen, it should go until j, otherwise it tries to reach an uncalculated matrix
                    //for(int k = 0; k < bdst->levels.size(); k++)
                    for(int k = 0; k < j; k++)
                    {

                        if(bdst->levels.at(k).connectionIndex == aLevel->members.at(i))
                        {
                            sum = sum + bdst->levels.at(k).meanInvariant;
                            break;

                        }
                    }
                }
            }

        }

        for(int i = 0; i < sum.size(); i++)
        {
            sum[i] = sum[i]/aLevel->members.size();
        }

        aLevel->meanInvariant = sum;

    }

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
