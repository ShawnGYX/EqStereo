#pragma once

#include "opencv2/core/core.hpp"
#include "Eigen/Eigen"

using namespace cv;

struct Landmark 
{
    Point2f camcoor_left_distorted;
    Point2f camcoor_right_distorted;
    Point2f camcoor_left;
    Point2f camcoor_right;
    Point2f camcoor_left_norm;
    Point2f camcoor_right_norm;

    Eigen::Vector3d p_0;
    Point3f p_t_bff;


    Eigen::Vector3d X_lm;
    
    Eigen::Vector2d camcoor_left_hat;
    Eigen::Vector2d camcoor_right_hat;

    Eigen::Matrix3d sig;
};