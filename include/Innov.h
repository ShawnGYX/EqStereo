#pragma once

#include "Eigen/Eigen"

struct Innov
{
    Eigen::Matrix4d Del;   
    Eigen::MatrixXd del;
};