#include "StereoCamera.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/core/eigen.hpp"

#include "eigen3/unsupported/Eigen/MatrixFunctions"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>

typedef Eigen::Vector3d             vec3d;
typedef Eigen::Matrix3d             mat3d;
typedef Eigen::Matrix<double, 6, 1> vec6d;
typedef Eigen::Matrix<double, 6, 6> mat6d;
typedef Eigen::Matrix<double, 3, 6> mat3x6d;
typedef Eigen::Matrix<double, 2, 6> mat2x6d;

Eigen::Matrix3d skew(const Eigen::Vector3d& x)
{
    Eigen::Matrix3d x_s;
    x_s << 0.0,-x.x(),x.y(),
            x.z(),0,-x.x(),
            -x.y(),x.x(),0;
    
    return x_s;
}


// Velocity estimation functions


void StereoCamera::TrackLandmarks(vector<Landmark>& landmarks, const Mat &image_old, const Mat &image_new)
{
    if (landmarks.empty()) return;

    vector<Point2f> oldPoints;
    for (const auto & feature: landmarks)
    {
        oldPoints.emplace_back(feature.camcoor_left_distorted);
    }
    
    vector<Point2f> points;
    vector<uchar> status;
    vector<float> err;
    calcOpticalFlowPyrLK(image_old, image_new, oldPoints, points, status, err);

    vector<Point2f> pointsNorm;
    cv::undistortPoints(points, pointsNorm, Camera_left, Distortion_coef_left);

    vector<Point2f> points_Undistort;

    cv::undistortPoints(points, points_Undistort, Camera_left, Distortion_coef_left, noArray(), Camera_left);

    for (long int i=points.size()-1; i >= 0; --i) {
        if (status[i] == 0) 
        {
            landmarks.erase(landmarks.begin() + i);
            continue;
        }

        landmarks[i].camcoor_left = points_Undistort[i];
        landmarks[i].camcoor_left_distorted = points[i];
        landmarks[i].camcoor_left_norm = pointsNorm[i];

    }


}



vector<Point2f> removeDuplicateFeatures(const vector<Point2f> &proposedFeatures, const vector<Landmark>& oldLandmarks, const double& featureDist)
{
    vector<Point2f> newfeatures;

    for (const auto & proposed : proposedFeatures)
    {
        bool useFlag = true;
        for ( const auto & feature : oldLandmarks)
        {
            if (norm(proposed - feature.camcoor_left_distorted) < featureDist)
            {
                useFlag = false;
                break;
            }
        }

        if (useFlag)
        {
            newfeatures.emplace_back(proposed);
        }
        
    }

    return newfeatures;
}

vector<Point2f> StereoCamera::detectNewFeatures(const vector<Landmark>& oldLandmarks, const Mat &image) const
{
    vector<Point2f> proposedfeatures;
    goodFeaturesToTrack(image,proposedfeatures,maxFeatures,minHarrisQuality,featureDist);
    vector<Point2f> newFeatures = removeDuplicateFeatures(proposedfeatures, oldLandmarks, featureDist);

    return newFeatures;
}

[[nodiscard]] vector<Landmark> StereoCamera::createNewLandmarks(const vector<Point2f> &newFeatures) const
{
    vector<Landmark> newlandmarks;
    if (newFeatures.empty()) return newlandmarks;

    vector<Point2f> newFeaturesNorm;
    vector<Point2f> newFeatures_undistort;
    cv::undistortPoints(newFeatures, newFeaturesNorm, Camera_left, Distortion_coef_left);
    cv::undistortPoints(newFeatures, newFeatures_undistort, Camera_left, Distortion_coef_left, noArray(), Camera_left);

    

    for (int i = 0; i < newFeatures.size(); i++)
    {
        Landmark lm;
        lm.camcoor_left_distorted = newFeatures[i];
        lm.camcoor_left = newFeatures_undistort[i];
        lm.camcoor_left_norm = newFeaturesNorm[i];
        lm.X_lm = Eigen::Vector3d::Zero();
        lm.sig = Eigen::Matrix3d::Identity()*Sigma_coef;
        

        newlandmarks.emplace_back(lm);
    }
    
    return newlandmarks;

}

void StereoCamera::matchStereoFeatures(vector<Landmark> &proposedLandmarks, const Mat &image_left, const Mat &image_right) const
{
    if (proposedLandmarks.empty()) return;

    vector<Point2f> LeftPoints;
    for (const auto & feature: proposedLandmarks)
    {
        LeftPoints.emplace_back(feature.camcoor_left_distorted);
    }
    
    vector<Point2f> points;
    vector<uchar> status;
    vector<float> err;
    calcOpticalFlowPyrLK(image_left, image_right, LeftPoints, points, status, err);

    vector<Point2f> pointsNorm;
    cv::undistortPoints(points, pointsNorm, Camera_right, Distortion_coef_right);

    vector<Point2f> pointsUndistort;
    cv::undistortPoints(points, pointsUndistort, Camera_right, Distortion_coef_right, noArray(), Camera_right);

    for (long int i=points.size()-1; i >= 0; --i) {
        if (status[i] == 0) 
        {
            proposedLandmarks.erase(proposedLandmarks.begin() + i);
            continue;
        }

        proposedLandmarks[i].camcoor_right_norm = pointsNorm[i];
        proposedLandmarks[i].camcoor_right_distorted = points[i];
        proposedLandmarks[i].camcoor_right = pointsUndistort[i];

    }

}


void StereoCamera::init3DCoordinates(vector<Landmark> &newLandmarks, const Eigen::Matrix4d& currentPose) const
{

    const Eigen::Matrix4d XLR = XL.inverse()*XR;
    const Eigen::Matrix3d Rot = XLR.block<3,3>(0,0);
    const Eigen::Vector3d Trans = XLR.block<3,1>(0,3);

    for (int i = 0; i < newLandmarks.size(); i++)
    {
        Eigen::Matrix3d pnt_skew;
        pnt_skew << 0,-1, newLandmarks[i].camcoor_right_norm.y,
                    1,0,-newLandmarks[i].camcoor_right_norm.x,
                    -newLandmarks[i].camcoor_right_norm.y,newLandmarks[i].camcoor_right_norm.x,0;
        
        Eigen::Vector3d pnt_l;
        pnt_l << newLandmarks[i].camcoor_left_norm.x,newLandmarks[i].camcoor_left_norm.y,1;

        Eigen::Vector3d nom = pnt_skew*Rot.transpose()*Trans;
        Eigen::Vector3d den = pnt_skew*Rot.transpose()*pnt_l;

        float z = nom.norm()/den.norm();
        
        float x = z*newLandmarks[i].camcoor_left_norm.x;
        float y = z*newLandmarks[i].camcoor_left_norm.y;
        Eigen::Vector4d pnt_eigen(x,y,z,1);
        

        Eigen::Vector4d pnt_global;
        pnt_global = currentPose*XL*pnt_eigen;           
        newLandmarks[i].p_0 = pnt_global.head(3);
    
    }

}

void StereoCamera::update3DCoordinate(vector<Landmark> &newLandmarks) const
{
    Eigen::Matrix4d R12 = XL.inverse()*XR;

    Eigen::Matrix3d Rot = R12.block<3,3>(0,0);
    Eigen::Vector3d Trans = R12.block<3,1>(0,3);

    for (int i = 0; i < newLandmarks.size(); i++)
    {
        Eigen::Matrix3d pnt_skew;
        pnt_skew << 0,-1, newLandmarks[i].camcoor_right_norm.y,
                    1,0,-newLandmarks[i].camcoor_right_norm.x,
                    -newLandmarks[i].camcoor_right_norm.y,newLandmarks[i].camcoor_right_norm.x,0;
        
        Eigen::Vector3d pnt_l;
        pnt_l << newLandmarks[i].camcoor_left_norm.x,newLandmarks[i].camcoor_left_norm.y,1;

        Eigen::Vector3d nom = pnt_skew*Rot.transpose()*Trans;
        Eigen::Vector3d den = pnt_skew*Rot.transpose()*pnt_l;

        float z = nom.norm()/den.norm();
        
        float x = z*newLandmarks[i].camcoor_left_norm.x;
        float y = z*newLandmarks[i].camcoor_left_norm.y;
        Point3f pnt;
        Eigen::Vector3d pnt_eigen(x,y,z);
        pnt.x = x;
        pnt.y = y;
        pnt.z = z;

        newLandmarks[i].p_t_bff = pnt;
    
    }

}

void StereoCamera::addNewLandmarks(vector<Landmark>& landmarks, const vector<Landmark>& newlandmarks)
{
    for (auto & lm : newlandmarks)
    {
        if (landmarks.size() >= maxFeatures) break;
        landmarks.emplace_back(lm);
    }
}



// Old feature tracking functions

void StereoCamera::Save_t(double t, const string file)
{
    ofstream trajectory;
    trajectory.open(file,ios::app);
    trajectory<<setprecision(14)<<t<<"\n";
    trajectory.close();
}   


void StereoCamera::Save_Matrix(Eigen::Matrix4d tfmat, const string file)
{
    ofstream trajectory;
    trajectory.open(file,ios::app);
    trajectory<<tfmat<<"\n";
    trajectory.close();
}   

int StereoCamera::reprojection_gauss_newton(
    const std::vector<Point2f>& points1,
    const std::vector<Point3f>& points2,
    mat3d&                    rotation,
    vec3d&                    translation
) {
    assert(points1.size() == points2.size());
    size_t n_pairs = points1.size();

    const auto max_iterations = 10;
    double prev_loss = 0.;
    auto iter = 0;

    while (iter < max_iterations) {

        double loss = 0.;
        mat6d h = mat6d::Zero();
        vec6d g = vec6d::Zero();

        for (auto i = 0; i < n_pairs; ++i) {

            Eigen::Vector3d p2;
            Eigen::Vector2d p1;
            p1 << points1[i].x,points1[i].y;
            p2 << points2[i].x,points2[i].y,points2[i].z;
            
            // auto& p1 = points1[i];
            // auto& p2 = points2[i];
            auto p2_trans = rotation * p2 + translation;

            double x = p2_trans[0], y = p2_trans[1], z = p2_trans[2];

            Eigen::Vector3d p2_reproj;
            Eigen::Matrix3d camera_left_eigen;
            cv2eigen(Camera_left, camera_left_eigen);
            p2_reproj = camera_left_eigen*p2_trans;

            Eigen::Vector2d p2_reproj_image;
            p2_reproj_image << p2_reproj.x()/p2_reproj.z(), p2_reproj.y()/p2_reproj.z();


            auto err = p2_reproj_image-p1;
            loss += 0.5 * err.squaredNorm();

            double r = err.norm();
            double weight = 1/(pow(r,2));

            mat2x6d jacobian_mat;
            jacobian_mat <<
                fx_left/z, 0, -fx_left*x/(z*z), -fx_left*x*y/(z*z), fx_left+fx_left*x*x/(z*z), -fx_left*y/z,
                0, fy_left/z, -fy_left*y/(z*z), -fy_left-fy_left*y*y/(z*z), fy_left*x*y/(z*z), fy_left*x/z;

            h += jacobian_mat.transpose() * jacobian_mat * weight;
            g += jacobian_mat.transpose() * -err * weight;
        }

        vec6d dx = h.ldlt().solve(g);

        if (std::isnan(dx[0])) { std::cerr << "result is nan." << std::endl; break; }
        if (dx.norm() < 1e-5) { break; }



        vec3d w = dx.tail<3>();
        vec3d u = dx.head<3>();
        double theta = w.norm();
        double A = (sin(theta))/theta;
        double B = (1-cos(theta))/(theta*theta);
        double C = (1-A)/(theta*theta);
        const mat3d identity = mat3d::Identity();
        mat3d w_hat;
        w_hat <<
                0., -w[2],  w[1],
            w[2],      0., -w[0],
            -w[1],  w[0],      0.;
        mat3d delta_r = identity + A*w_hat + B*w_hat*w_hat;
        mat3d V = identity + B*w_hat + C*w_hat*w_hat;
        vec3d delta_t = V*u; 



        translation = delta_r * translation + delta_t;
        rotation = delta_r * rotation ;

        ++iter;
    }

    return iter;
}


// EqF functions




// Processing image

Eigen::Matrix4d StereoCamera::processImages(vector<Landmark>& landmarks, const Eigen::Matrix4d& currentPose, const Mat& img_left, const Mat& img_right, const double& t) {

    // Estimate the velocity by tracking landmarks
    cout<<setprecision(14)<<t<<endl;

    // Track landmarks to the new images
    Image_t1_L = img_left;
    Image_t1_R = img_right;    
    this->TrackLandmarks(landmarks, Image_t0_L,Image_t1_L);
    this->matchStereoFeatures(landmarks,Image_t1_L,Image_t1_R);

    // Collect data for velocity estimation
    vector<Point3f> pntset_0(landmarks.size());
    vector<Point2f> lm_t1_image_left(landmarks.size());
    transform(landmarks.begin(), landmarks.end(), pntset_0.begin(), [](const Landmark& lm) {return lm.p_t_bff; });
    transform(landmarks.begin(), landmarks.end(), lm_t1_image_left.begin(), [](const Landmark& lm) {return lm.camcoor_left; });

    // Create new landmarks
    vector<Point2f> newFeatures = this->detectNewFeatures(landmarks, Image_t1_L);
    vector<Landmark> newLandmarks = this->createNewLandmarks(newFeatures);
    this->matchStereoFeatures(newLandmarks,Image_t1_L,Image_t1_R);
    this->init3DCoordinates(newLandmarks, currentPose);
    this->addNewLandmarks(landmarks, newLandmarks);
    this->update3DCoordinate(landmarks);

    if (pntset_0.empty()) {
        return Eigen::Matrix4d::Identity();
    }

    // Estimate rotation and translation
    Mat rvec, tvec;
    solvePnPRansac(pntset_0, lm_t1_image_left, Camera_left, noArray(), rvec, tvec, false, 100, 4.0F, 0.98999, noArray(), cv::SOLVEPNP_EPNP);
    cv::Mat R_inbuilt;
    cv::Rodrigues(rvec, R_inbuilt);
    Eigen::Matrix3d r_mat;
    Eigen::MatrixXd t_mat;
    cv2eigen(tvec,t_mat);
    cv2eigen(R_inbuilt, r_mat);
    
    Rotation = r_mat.transpose();
    Translation = -r_mat.transpose()*t_mat;

    for (int iteration_1 = 0; iteration_1 < 6; iteration_1++)
    {
        int iter4 = reprojection_gauss_newton(lm_t1_image_left,pntset_0,Rotation,Translation);
    }

    // TODO: What is going on with rotation here?
    Rotation << Rotation(0,0), -Rotation(0,1), -Rotation(0,2),-Rotation(1,0),Rotation(1,1), -Rotation(1,2),-Rotation(2,0),-Rotation(2,1),Rotation(2,2);
    Translation << -Translation;

    Eigen::Matrix4d tfmat = Eigen::Matrix4d::Identity();
    tfmat.block<3,3>(0,0) << Rotation;
    tfmat.block<3,1>(0,3) << Translation;

    Save_Matrix(tfmat, "/home/shawnge/euroc_test/trajec.txt");
    Save_t(t,"/home/shawnge/euroc_test/time.txt");

    return tfmat;
}
