#pragma once

#include "Eigen/Eigen"
#include "opencv2/core/core.hpp"
#include "Landmark.h"
#include "sensor_msgs/image_encodings.h"
#include "sensor_msgs/Image.h"
#include <vector>

using namespace cv;
using namespace std;

struct Innov
{
    
    Eigen::Matrix4d Del;
    
    Eigen::MatrixXd del;

};

Eigen::Matrix3d skew(const Eigen::Vector3d& x);



class StereoCamera
{

public:
    Eigen::Matrix3d Rotation;
    Eigen::Vector3d Translation;
      
    Mat Image_t0_L;
    Mat Image_t0_R; 
    Mat Image_t1_L;
    Mat Image_t1_R;
    int flag = 0;

    int maxFeatures = 80;
    double featureDist = 20;
    double minHarrisQuality = 0.1;
    double featureSearchThreshold = 1.0;
    
    float fx_left = 458.654;
    float fy_left = 457.296;
    float cx_left = 367.215;
    float cy_left = 248.375;

    float fx_right = 457.587;
    float fy_right = 456.134;
    float cx_right = 379.999;
    float cy_right = 255.238;

    Mat Camera_left = (Mat_<double>(3,3) << 458.654, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0);
    Mat Distortion_coef_left = (Mat_<double>(1,4) << -0.2834, 0.073959, 0.0001936, 0.000017618);

    Mat Camera_right = (Mat_<double>(3,3) << 457.587, 0.0, 379.999, 0.0, 456.134, 255.238, 0.0, 0.0, 1.0);
    Mat Distortion_coef_right = (Mat_<double>(1,4) << -0.28368, 0.07451284, -0.00010473, 0.000035559);;


    Eigen::Matrix4d XL = (Eigen::Matrix4d()<<0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
                                            0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
                                            -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
                                            0.0, 0.0, 0.0, 1.0).finished();

    Eigen::Matrix4d XR = (Eigen::Matrix4d()<<0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
                                            0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
                                            -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
                                            0.0, 0.0, 0.0, 1.0).finished();                                        

    // EqF variables

    vector<Landmark> landmarks;

    Eigen::Matrix4d X_rb = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d P_init = Eigen::Matrix4d::Identity();
    
    
    double P_coef = 0.01;
    double Q_coef = 0.01;

    double Sigma_coef = 5;
    

    double dt = 0.05;




public:


    // Velocity estimation functions


    void TrackLandmarks(const Mat &image_old, const Mat &image_new);



    [[nodiscard]] vector<Point2f> removeDuplicateFeatures(const vector<Point2f> &proposedFeatures) const;
    [[nodiscard]] vector<Point2f> detectNewFeatures(const Mat &image) const;
    [[nodiscard]] vector<Landmark> createNewLandmarks(const vector<Point2f> &newFeatures) const;
    void matchStereoFeatures(vector<Landmark> &proposedLandmarks, const Mat &image_left, const Mat &image_right) const;


    void init3DCoordinates(vector<Landmark> &newLandmarks) const;
    void update3DCoordinate(vector<Landmark> &newLandmarks) const;
    void addNewLandmarks(const vector<Landmark>& newlandmarks);


    // Old feature tracking functions

    vector<Point2f> DetectNewFeatures(const Mat &image);
    vector<Point2f> TrackFeatures_Time(const Mat &image,const Mat &image_new, const vector<Point2f> &lm);
    vector<Point2f> TrackFeatures_LR(const Mat &image_left, const Mat &image_right, const vector<Point2f> &landmarksleft);
    void Triangulation_Euroc(const vector<Point2f> &pl, const vector<Point2f> &pr, vector<Point3f> &p_3d);
    vector<int> GenerateDiffNumber(int min,int max,int num);
    void Save_Point(vector<Point3f>& vs, const string file);
    void Save_t(double t, const string file);

    void Save_Matrix(Eigen::Matrix4d tfmat, const string file);

    int reprojection_gauss_newton(
	    const std::vector<Point2f>& points1,
	    const std::vector<Point3f>& points2,
	    Eigen::Matrix3d&            rotation,
	    Eigen::Vector3d&            translation
    );
    // EqF functions
    void update_vel(const Eigen::Matrix4d vel);
    Eigen::MatrixXd compute_c();
    Eigen::MatrixXd compute_error();
    Eigen::MatrixXd build_Sigma();
    void update_Sigma(Eigen::MatrixXd &C_mat, Eigen::MatrixXd &Sigma);
    Innov Compute_innovation(const Eigen::MatrixXd &C_mat, const Eigen::MatrixXd &err, const Eigen::MatrixXd &Sigma);
    void update_innovation(const Innov &innovation);
    void ProcessImage_EqF(const sensor_msgs::ImageConstPtr& msg_left, const sensor_msgs::ImageConstPtr& msg_right);
};