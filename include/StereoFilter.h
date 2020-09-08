#pragma once

#include "Eigen/Eigen"
#include "Landmark.h"
#include "Innov.h"
#include <vector>
#include "yaml-cpp/yaml.h"

using namespace Eigen;
using namespace std;

Matrix3d skew(const Vector3d& x);

class StereoFilter
{
protected:
// Parameters
    float fx_left = 458.654;
    float fy_left = 457.296;
    float cx_left = 367.215;
    float cy_left = 248.375;

    float fx_right = 457.587;
    float fy_right = 456.134;
    float cx_right = 379.999;
    float cy_right = 255.238;

    bool initialisationFlag;


    Matrix4d XL = (Matrix4d()<<0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
                                            0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
                                            -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
                                            0.0, 0.0, 0.0, 1.0).finished();

    Matrix4d XR = (Matrix4d()<<0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
                                            0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
                                            -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
                                            0.0, 0.0, 0.0, 1.0).finished();          

    double P_coef = 0.01;
    double Q_coef = 0.8;
    double dt = 0.01;

public:                              
    // EqF variables
    Matrix4d X_rb = Matrix4d::Identity();
    Matrix4d P_init = Matrix4d::Identity();


public:

    StereoFilter(){};
    StereoFilter(const YAML::Node& configNode){
            P_coef = configNode["P_coef"].as<double>();
            Q_coef = configNode["Q_coef"].as<double>();
            dt     = configNode["dt"].as<double>();
    }

    // EqF functions

    void update_vel(const Matrix4d vel);

    MatrixXd compute_c(vector<Landmark>& landmarks) const;
    MatrixXd compute_error(vector<Landmark>& landmarks);
    static MatrixXd build_Sigma(const vector<Landmark>& landmarks);
    void update_Sigma(MatrixXd &C_mat, MatrixXd &Sigma, vector<Landmark>& landmarks);
    Innov Compute_innovation(const MatrixXd &C_mat, const MatrixXd &err, const MatrixXd &Sigma, vector<Landmark>& landmarks, bool isMoving);
    void update_innovation(const Innov &innovation, vector<Landmark>& landmarks);
    
    void integrateEquations(vector<Landmark>& landmarks, const Matrix4d& velocity);

    Matrix4d getPose() { return P_init * X_rb; }
};