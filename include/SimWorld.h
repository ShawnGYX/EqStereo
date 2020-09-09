#pragma once

#include <vector>
#include "Eigen/Eigen"
#include "Landmark.h"

using namespace Eigen;
using namespace std;

class SimWorld {
    Eigen::Matrix3d Rotation = Matrix3d::Identity();
    Eigen::Vector3d Translation = Vector3d::Zero();
    vector<Vector3d> worldPoints;

    int maxFeatures = 80;
    double Sigma_coef = 5;
    
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

    const Eigen::Matrix3d KL = (Eigen::Matrix3d()<<458.654, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0).finished();

    const Eigen::Matrix4d XR = (Eigen::Matrix4d()<<0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
                                            0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
                                            -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
                                            0.0, 0.0, 0.0, 1.0).finished();
    const Eigen::Matrix3d KR = (Eigen::Matrix3d()<<457.587, 0.0, 379.999, 0.0, 456.134, 255.238, 0.0, 0.0, 1.0).finished();
    

public:

[[nodiscard]] vector<Landmark> generateRandomLandmarks(const int& number);
[[nodiscard]] Matrix4d simulateMotion(vector<Landmark>& landmarks);
void updateLandmarkMeasurements(vector<Landmark>& landmarks) const;


};