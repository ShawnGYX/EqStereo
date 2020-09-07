#pragma once

#include "Eigen/Eigen"
#include "opencv2/core/core.hpp"
#include "Landmark.h"
#include "Innov.h"
#include <vector>

using namespace cv;
using namespace std;

Eigen::Matrix3d skew(const Eigen::Vector3d& x);
[[nodiscard]] vector<Point2f> removeDuplicateFeatures(const vector<Point2f> &proposedFeatures, const vector<Landmark>& oldLandmarks, const double& featureDist);


class StereoCamera
{

public:
    Eigen::Matrix3d Rotation;
    Eigen::Vector3d Translation;
      
    Mat Image_t0_L;
    Mat Image_t0_R; 
    Mat Image_t1_L;
    Mat Image_t1_R;

    int maxFeatures = 80;
    double featureDist = 20;
    double minHarrisQuality = 0.1;
    double featureSearchThreshold = 1.0;
    
    static constexpr float fx_left = 458.654;
    static constexpr float fy_left = 457.296;
    static constexpr float cx_left = 367.215;
    static constexpr float cy_left = 248.375;

    static constexpr float fx_right = 457.587;
    static constexpr float fy_right = 456.134;
    static constexpr float cx_right = 379.999;
    static constexpr float cy_right = 255.238;

    const Mat Camera_left = (Mat_<double>(3,3) << 458.654, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0);
    const Mat Distortion_coef_left = (Mat_<double>(1,4) << -0.2834, 0.073959, 0.0001936, 0.000017618);

    const Mat Camera_right = (Mat_<double>(3,3) << 457.587, 0.0, 379.999, 0.0, 456.134, 255.238, 0.0, 0.0, 1.0);
    const Mat Distortion_coef_right = (Mat_<double>(1,4) << -0.28368, 0.07451284, -0.00010473, 0.000035559);;


    const Eigen::Matrix4d XL = (Eigen::Matrix4d()<<0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
                                            0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
                                            -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
                                            0.0, 0.0, 0.0, 1.0).finished();

    const Eigen::Matrix4d XR = (Eigen::Matrix4d()<<0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
                                            0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
                                            -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
                                            0.0, 0.0, 0.0, 1.0).finished();                                        

    // EqF variables

    double Sigma_coef = 5;
    double dt = 0.05;


public:


    // Velocity estimation functions


    void TrackLandmarks(vector<Landmark>& landmarks, const Mat &image_old, const Mat &image_new);
    [[nodiscard]] vector<Point2f> detectNewFeatures(const vector<Landmark>& oldLandmarks, const Mat &image) const;
    [[nodiscard]] vector<Landmark> createNewLandmarks(const vector<Point2f> &newFeatures) const;
    void matchStereoFeatures(vector<Landmark> &proposedLandmarks, const Mat &image_left, const Mat &image_right) const;


    void init3DCoordinates(vector<Landmark> &newLandmarks, const Eigen::Matrix4d& currentPose) const;
    void update3DCoordinate(vector<Landmark> &newLandmarks) const;
    void addNewLandmarks(vector<Landmark>& landmarks, const vector<Landmark>& newlandmarks);


    // Old feature tracking functions
    void Save_t(double t, const string file);
    void Save_Matrix(Eigen::Matrix4d tfmat, const string file);

    int reprojection_gauss_newton(
	    const std::vector<Point2f>& points1,
	    const std::vector<Point3f>& points2,
	    Eigen::Matrix3d&            rotation,
	    Eigen::Vector3d&            translation
    );

    // Main Stereo Camera
    Eigen::Matrix4d processImages(vector<Landmark>& landmarks, const Eigen::Matrix4d& currentPose, const Mat& img_left, const Mat& img_right, const double& t);
};