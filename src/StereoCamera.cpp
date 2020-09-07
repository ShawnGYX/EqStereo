#include "StereoCamera.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/core/eigen.hpp"

#include <cv_bridge/cv_bridge.h>

#include "eigen3/unsupported/Eigen/MatrixFunctions"

#include <fstream>

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


void StereoCamera::TrackLandmarks(const Mat &image_old, const Mat &image_new)
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

vector<Point2f> StereoCamera::detectNewFeatures(const Mat &image) const
{
    vector<Point2f> proposedfeatures;
    goodFeaturesToTrack(image,proposedfeatures,maxFeatures,minHarrisQuality,featureDist);

    vector<Point2f> newFeatures = removeDuplicateFeatures(proposedfeatures, landmarks, featureDist);

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


void StereoCamera::init3DCoordinates (vector<Landmark> &newLandmarks) const
{
    
    const Eigen::Matrix4d Phat = P_init*X_rb;

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
        pnt_global = Phat*XL*pnt_eigen;           
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

void StereoCamera::addNewLandmarks(const vector<Landmark>& newlandmarks)
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

void StereoCamera::update_vel(const Eigen::Matrix4d vel)
{
    X_rb = X_rb * vel;
}

Eigen::MatrixXd StereoCamera::compute_c()
{
    int lm_num = landmarks.size();
    Eigen::Matrix4d Phat;
    Phat = P_init*X_rb;

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(4*lm_num,3*lm_num);

    for (int i = 0; i < lm_num; i++)
    {
        Eigen::Vector3d pi_hat;
        pi_hat = landmarks[i].p_0 + P_init.block<3,3>(0,0)*landmarks[i].X_lm;

        Eigen::Vector4d pi_hat_homo;
        pi_hat_homo << pi_hat.x(),pi_hat.y(),pi_hat.z(),1;

        Eigen::Vector4d yi_hat_left;
        yi_hat_left = XL.inverse()*Phat.inverse()*pi_hat_homo;

        Eigen::Matrix<double,2,3> de_left;
        de_left << fx_left/yi_hat_left.z(),0,-fx_left*yi_hat_left.x()/(yi_hat_left.z()*yi_hat_left.z()),
                    0,fy_left/yi_hat_left.z(),-fy_left*yi_hat_left.y()/(yi_hat_left.z()*yi_hat_left.z());

        Eigen::Matrix<double,2,3> ci_left;
        ci_left = de_left*XL.block<3,3>(0,0).inverse()*X_rb.block<3,3>(0,0).transpose();

        Eigen::Vector4d yi_hat_right;
        yi_hat_right = XR.inverse()*Phat.inverse()*pi_hat_homo;

        Eigen::Matrix<double,2,3> de_right;
        de_right << fx_right/yi_hat_right.z(),0,-fx_right*yi_hat_right.x()/(yi_hat_right.z()*yi_hat_right.z()),
                    0,fy_right/yi_hat_right.z(),-fy_right*yi_hat_right.y()/(yi_hat_right.z()*yi_hat_right.z());

        Eigen::Matrix<double,2,3> ci_right;
        ci_right = de_right*XR.block<3,3>(0,0).inverse()*X_rb.block<3,3>(0,0).transpose();

        C.block<2,3>(4*(i),3*(i)) = ci_left;
        C.block<2,3>(2+4*(i),3*(i)) = ci_right;

        double p_x_l, p_x_r, p_y_l, p_y_r;
        p_x_l = fx_left*yi_hat_left.x()/yi_hat_left.z()+cx_left;
        p_x_r = fx_right*yi_hat_right.x()/yi_hat_right.z()+cx_right;
        p_y_l = fy_left*yi_hat_left.y()/yi_hat_left.z()+cy_left;
        p_y_r = fy_right*yi_hat_right.y()/yi_hat_right.z()+cy_right;

        landmarks[i].camcoor_left_hat << p_x_l,p_y_l;
        landmarks[i].camcoor_right_hat << p_x_r,p_y_r;


    }

    return C;
    
}

Eigen::MatrixXd StereoCamera::compute_error()
{
    int lm_num = landmarks.size();
    Eigen::Vector2d err_left;
    Eigen::Vector2d err_right;


    Eigen::MatrixXd measurement_err = Eigen::MatrixXd::Zero(4*lm_num,1);

    for (int i = 0; i < lm_num; i++)
    {
        Eigen::Vector2d cam_left, cam_right;
        cam_left << landmarks[i].camcoor_left.x,landmarks[i].camcoor_left.y;
        cam_right << landmarks[i].camcoor_right.x,landmarks[i].camcoor_right.y;

        err_left = cam_left-landmarks[i].camcoor_left_hat;
        err_right = cam_right-landmarks[i].camcoor_right_hat;

        measurement_err.block<2,1>(4*(i),0) = err_left;
        measurement_err.block<2,1>(2+4*(i),0) = err_right;
    }
    
    return measurement_err;


}

Eigen::MatrixXd StereoCamera::build_Sigma()
{
    int lm_num = landmarks.size();
    Eigen::MatrixXd Sigma;

    Sigma = Eigen::MatrixXd::Zero(3*lm_num,3*lm_num);
    
    for (int i = 0; i < lm_num; i++)
    {
        Sigma.block<3,3>(3*(i),3*(i)) = landmarks[i].sig;
    }

    return Sigma;

}

void StereoCamera::update_Sigma(Eigen::MatrixXd &C_mat, Eigen::MatrixXd &Sigma)
{
    int lm_num = landmarks.size();

    

    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(3*lm_num,3*lm_num)*P_coef;
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(4*lm_num,4*lm_num)*Q_coef;

    Eigen::MatrixXd s;
    s = C_mat*Sigma*C_mat.transpose()+Q;

    Sigma += dt*(P-Sigma*C_mat.transpose()*s.inverse()*C_mat*Sigma);

    for (int i = 0; i < lm_num; i++)
    {
        landmarks[i].sig = Sigma.block<3,3>(3*(i),3*(i));
    }
}


Innov StereoCamera::Compute_innovation(const Eigen::MatrixXd &C_mat, const Eigen::MatrixXd &err, const Eigen::MatrixXd &Sigma)
{
    
    int lm_num = landmarks.size();
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(4*lm_num,4*lm_num)*Q_coef;

    Eigen::MatrixXd s;
    s = C_mat*Sigma*C_mat.transpose()+Q;
    Eigen::MatrixXd gamma;
    gamma = Sigma*C_mat.transpose()*s.inverse()*err;

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3*lm_num,6);

    for (int i = 0; i < lm_num; i++)
    {
        Eigen::Vector4d q_i;
        Eigen::Vector4d pi_homo;

        pi_homo << landmarks[i].p_0.x(),landmarks[i].p_0.y(),landmarks[i].p_0.z(),1;

        q_i = P_init.inverse()*pi_homo;

        Eigen::Matrix3d q_a;
        q_a = skew(landmarks[i].X_lm+q_i.head(3));

        A.block<3,3>(3*(i),0) = q_a;
        A.block<3,3>(3*(i),3) = -1*Eigen::Matrix3d::Identity();

    }

    Eigen::VectorXd v;
    v = (A.transpose()*A).inverse()*A.transpose()*gamma;

    Eigen::Matrix3d v_skew;
    v_skew = skew(v.head(3));

    Eigen::Matrix4d Delta = Eigen::Matrix4d::Zero();
    Delta.block<3,3>(0,0) = v_skew;
    Delta.block<3,1>(0,3) = v.tail(3);

    Eigen::MatrixXd delta = Eigen::MatrixXd::Zero(3*lm_num,1);

    for (int i = 0; i < lm_num; i++)
    {
        Eigen::Vector4d q_i;
        Eigen::Vector4d pi_homo;

        pi_homo << landmarks[i].p_0.x(),landmarks[i].p_0.y(),landmarks[i].p_0.z(),1;

        q_i = P_init.inverse()*pi_homo;

        delta.block<3,1>(3*(i),0) = gamma.block<3,1>(3*(i),0)+v_skew*q_i.head(3)+v.tail(3);
    }


    Innov Inn;
    Inn.Del = Delta;
    Inn.del = delta;

    return Inn;

}

void StereoCamera::update_innovation(const Innov &innovation)
{
    int lm_num = landmarks.size();

    Eigen::MatrixXd delta = innovation.del;
    Eigen::Matrix4d Delta = innovation.Del;

    X_rb = (dt*Delta).exp()*X_rb;

    for (int i = 0; i < lm_num; i++)
    {
        Eigen::Vector3d d;
        d << delta(3*i,0),delta(1+3*i,0),delta(2+3*i,0);
        landmarks[i].X_lm = landmarks[i].X_lm + dt * d + dt * Delta.block<3,3>(0,0) * landmarks[i].X_lm;
    }
    
}





// Processing image

void StereoCamera::ProcessImage_EqF(const sensor_msgs::ImageConstPtr& msg_left, const sensor_msgs::ImageConstPtr& msg_right)
{
    cv_bridge::CvImagePtr cv_ptr_left;
    cv_bridge::CvImagePtr cv_ptr_right;
    try
    {
        cv_ptr_left = cv_bridge::toCvCopy(msg_left, sensor_msgs::image_encodings::MONO8);
        cv_ptr_right = cv_bridge::toCvCopy(msg_right, sensor_msgs::image_encodings::MONO8);
    }
    catch(cv_bridge::Exception& e)
    {
        // throw(Exception("cv_bridge exception: %s", e.what()));
        return;
    }


    if (!flag)
    {
        Image_t1_L = cv_ptr_left->image.clone();
        Image_t1_R = cv_ptr_right->image.clone();
        vector<Point2f> newFeatures = this->detectNewFeatures(Image_t1_L);
        vector<Landmark> newLandmarks = this->createNewLandmarks(newFeatures);
        this->matchStereoFeatures(newLandmarks,Image_t1_L,Image_t1_R);
        this->init3DCoordinates(newLandmarks);
        this->addNewLandmarks(newLandmarks);

        this->update3DCoordinate(landmarks);

        flag = 1;
        Image_t0_L = cv_ptr_left->image.clone();
        Image_t0_R = cv_ptr_right->image.clone();
    }
    else
    {
        double t = cv_ptr_left->header.stamp.toSec();

        cout<<setprecision(14)<<t<<endl;

        Image_t1_L = cv_ptr_left->image.clone();
        Image_t1_R = cv_ptr_right->image.clone();

        vector<Point3f> pntset_0;
        vector<Point2f> lm_t1_image_left;
        
        
        this->TrackLandmarks(Image_t0_L,Image_t1_L);

        this->matchStereoFeatures(landmarks,Image_t1_L,Image_t1_R);

        for (const auto & lm : landmarks)
        {
            pntset_0.emplace_back(lm.p_t_bff);
            lm_t1_image_left.emplace_back(lm.camcoor_left);
        }

        vector<Point2f> newFeatures = this->detectNewFeatures(Image_t1_L);
        vector<Landmark> newLandmarks = this->createNewLandmarks(newFeatures);
        this->matchStereoFeatures(newLandmarks,Image_t1_L,Image_t1_R);
        this->init3DCoordinates(newLandmarks);
        this->addNewLandmarks(newLandmarks);
        this->update3DCoordinate(landmarks);

        Rotation << 1,0,0,0,1,0,0,0,1;
        Translation << 0,0,0;

        Mat rvec, tvec;

        solvePnPRansac(pntset_0, lm_t1_image_left, Camera_left, noArray(), rvec, tvec, false, 100, 4.0F, 0.98999, noArray(), cv::SOLVEPNP_EPNP);
        cv::Mat R_inbuilt;
        cv::Rodrigues(rvec, R_inbuilt);
        Eigen::Matrix3d r_mat;
        Eigen::MatrixXd t_mat;
        cv2eigen(tvec,t_mat);
        cv::cv2eigen(R_inbuilt, r_mat);
        Eigen::Vector3d t_1;
        t_1 = -r_mat.transpose()*t_mat;

        
        
        Rotation = r_mat.transpose();
        Translation = t_1;

        for (int iteration_1 = 0; iteration_1 < 6; iteration_1++)
        {
            int iter4 = reprojection_gauss_newton(lm_t1_image_left,pntset_0,Rotation,Translation);
        }

        Rotation << Rotation(0,0), -Rotation(0,1), -Rotation(0,2),-Rotation(1,0),Rotation(1,1), -Rotation(1,2),-Rotation(2,0),-Rotation(2,1),Rotation(2,2);
        Translation << -Translation;

        Eigen::Matrix4d tfmat = Eigen::Matrix4d::Identity();
        tfmat.block<3,3>(0,0) << Rotation;
        tfmat.block<3,1>(0,3) << Translation;

        Save_Matrix(tfmat, "/home/shawnge/euroc_test/trajec.txt");
        Save_t(t,"/home/shawnge/euroc_test/time.txt");




        // EqF
        if (Translation.norm()>0.5)
        {
            this->update_vel(Eigen::Matrix4d::Identity());

        }
        else
        {
            this->update_vel(tfmat);
        }
        


        Eigen::MatrixXd C;
        C = compute_c();
        Eigen::MatrixXd err;
        err = compute_error();
        
        Eigen::MatrixXd Sigma = this->build_Sigma();
        this->update_Sigma(C,Sigma);
        
        Innov innovation;
        innovation = Compute_innovation(C,err,Sigma);
        this->update_innovation(innovation);

        Eigen::Matrix4d pose;
        pose = P_init*X_rb;

        Save_Matrix(pose, "/home/shawnge/euroc_test/trajec_eqf.txt");
        Image_t0_L = cv_ptr_left->image.clone();
        Image_t0_R = cv_ptr_right->image.clone();

        
    }
    
}
