#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <Eigen/Eigen>

#include <unsupported/Eigen/MatrixFunctions>


#include <opencv2/video/tracking.hpp>
#include <opencv2/stereo/stereo.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <string>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <typeinfo>

using namespace std;
using namespace cv;

using colorVec = std::array<uchar, 3>;



typedef Eigen::Vector3d             vec3d;
typedef Eigen::Matrix3d             mat3d;
typedef Eigen::Matrix<double, 6, 1> vec6d;
typedef Eigen::Matrix<double, 6, 6> mat6d;
typedef Eigen::Matrix<double, 3, 6> mat3x6d;
typedef Eigen::Matrix<double, 2, 6> mat2x6d;

struct Innov
{
    
    Eigen::Matrix4d Del;
    
    Eigen::MatrixXd del;

};

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

    Eigen::Matrix3d x_skew(const Eigen::Vector3d x)
    {
        Eigen::Matrix3d x_s;
        x_s << 0.0,-x.x(),x.y(),
                x.z(),0,-x.x(),
                -x.y(),x.x(),0;
        
        return x_s;
    }


    // Velocity estimation functions


    void TrackLandmarks(const Mat &image_old, const Mat &image_new)
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



    vector<Point2f> removeDuplicateFeatures(const vector<Point2f> &proposedFeatures)
    {
        vector<Point2f> newfeatures;

        for (const auto & proposed : proposedFeatures)
        {
            bool useFlag = true;
            for ( const auto & feature : this->landmarks)
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

    vector<Point2f> detectNewFeatures(const Mat &image)
    {
        vector<Point2f> proposedfeatures;
        goodFeaturesToTrack(image,proposedfeatures,maxFeatures,minHarrisQuality,featureDist);

        vector<Point2f> newFeatures = this->removeDuplicateFeatures(proposedfeatures);

        return newFeatures;
    }

    vector<Landmark> createNewLandmarks(const vector<Point2f> &newFeatures)
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
        

    }

    void matchStereoFeatures(vector<Landmark> &proposedLandmarks, const Mat &image_left, const Mat &image_right)
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


    void init3DCoordinates (vector<Landmark> &newLandmarks)
    {
        
        Eigen::Matrix4d Phat;
        Phat = P_init*X_rb;

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
            Eigen::Vector4d pnt_eigen(x,y,z,1);
           

            Eigen::Vector4d pnt_global;
            pnt_global = Phat*XL*pnt_eigen;           
            newLandmarks[i].p_0 = pnt_global.head(3);
       
        }

    }

    void update3DCoordinate(vector<Landmark> &newLandmarks)
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

    void addNewLandmarks(vector<Landmark> newlandmarks)
    {
        for (auto & lm : newlandmarks)
        {
            if (landmarks.size() >= maxFeatures) break;
            landmarks.emplace_back(lm);
        }
    }



    // Old feature tracking functions

    vector<Point2f> DetectNewFeatures(const Mat &image)
    {
        vector<Point2f> InitFeatures;
        goodFeaturesToTrack(image, InitFeatures, maxFeatures, minHarrisQuality, featureDist);
        return InitFeatures;
    }

    vector<Point2f> TrackFeatures_Time(const Mat &image,const Mat &image_new, const vector<Point2f> &lm)
    {
        // if (landmarksLeft_t0.empty()) return;

        vector<Point2f> newPoints;
        vector<uchar> status;
        vector<float> error;

        calcOpticalFlowPyrLK(image, image_new, lm, newPoints,status, error);

        return newPoints;
    }

    vector<Point2f> TrackFeatures_LR(const Mat &image_left, const Mat &image_right, const vector<Point2f> &landmarksleft)
    {
        // if (landmarksleft.empty()) return;

        vector<Point2f> newPoints;
        vector<uchar> status;
        vector<float> error;

        calcOpticalFlowPyrLK(image_left, image_right, landmarksleft, newPoints, status, error);

        return newPoints;
    }

    void Triangulation_Euroc(const vector<Point2f> &pl, const vector<Point2f> &pr, vector<Point3f> &p_3d)
    {
        int N = pl.size();
        Eigen::Matrix4d R1;
        R1 << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
         0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
         0.0, 0.0, 0.0, 1.0;
        
        Eigen::Matrix4d R2;
        R2 << 0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
         0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
        -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
         0.0, 0.0, 0.0, 1.0;

        Eigen::Matrix4d R12 = R1.inverse()*R2;

        Eigen::Matrix3d Rot = R12.block<3,3>(0,0);
        Eigen::Vector3d Trans = R12.block<3,1>(0,3);


        for (int i = 0; i < N; i++)
        {
            
            Eigen::Matrix3d pnt_skew;
            pnt_skew << 0, -1, pr[i].y,
                        1, 0, -pr[i].x,
                        -pr[i].y, pr[i].x, 0;

            Eigen::Vector3d pnt_l;
            pnt_l << pl[i].x, pl[i].y, 1;
            
            Eigen::Vector3d nom = pnt_skew*Rot.transpose()*Trans;
            Eigen::Vector3d den = pnt_skew*Rot.transpose()*pnt_l;

            float z = nom.norm()/den.norm();
            
            float x = z*pl[i].x;
            float y = z*pl[i].y;
            Point3f pnt;
            pnt.x = x;
            pnt.y = y;
            pnt.z = z;
           
            p_3d.push_back(pnt);
            // if(z<7) p_3d.push_back(pnt);
            
        }
        
    }


    vector<int> GenerateDiffNumber(int min,int max,int num)
    {
        int rnd;
        vector<int> diff;
        vector<int> tmp;
        
        for(int i = min;i < max+1 ; i++ )
        {
            tmp.push_back(i);
        }
        srand((unsigned)time(0)); 
        for(int i = 0 ; i < num ; i++)
        {
            do{
                rnd = min+rand()%(max-min+1);
        
            }while(tmp.at(rnd-min)==-1);
            diff.push_back(rnd);
            tmp.at(rnd-min) = -1;
        }
        return diff;
    }

    void Save_Point(vector<Point3f>& vs, const string file)
    {
        vector<Point3f>::iterator it=vs.begin();
        
        ofstream saving;
        saving.open(file);
        for(;it!=vs.end();it++)
        {
            saving<<*it<<"\n";
        }
        saving.close();
    }   

    void Save_t(double t, const string file)
    {
        ofstream trajectory;
        trajectory.open(file,ios::app);
        trajectory<<setprecision(14)<<t<<"\n";
        trajectory.close();
    }   


    void Save_Matrix(Eigen::Matrix4d tfmat, const string file)
    {
        ofstream trajectory;
        trajectory.open(file,ios::app);
        trajectory<<tfmat<<"\n";
        trajectory.close();
    }   

    int reprojection_gauss_newton(
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

    void update_vel(const Eigen::Matrix4d vel)
    {
        X_rb = X_rb * vel;
    }

    Eigen::MatrixXd compute_c()
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

    Eigen::MatrixXd compute_error()
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

    Eigen::MatrixXd build_Sigma()
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

    void update_Sigma(Eigen::MatrixXd &C_mat, Eigen::MatrixXd &Sigma)
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


    Innov Compute_innovation(const Eigen::MatrixXd &C_mat, const Eigen::MatrixXd &err, const Eigen::MatrixXd &Sigma)
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
            q_a = x_skew(landmarks[i].X_lm+q_i.head(3));

            A.block<3,3>(3*(i),0) = q_a;
            A.block<3,3>(3*(i),3) = -1*Eigen::Matrix3d::Identity();

        }

        Eigen::VectorXd v;
        v = (A.transpose()*A).inverse()*A.transpose()*gamma;

        Eigen::Matrix3d v_skew;
        v_skew = x_skew(v.head(3));

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

    void update_innovation(const Innov &innovation)
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

    void ProcessImage_EqF(const sensor_msgs::ImageConstPtr& msg_left, const sensor_msgs::ImageConstPtr& msg_right)
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
            ROS_ERROR("cv_bridge exception: %s", e.what());
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

};

int main(int argc, char** argv)
{
    string rosbagFilename, rosbagTopic_1, rosbagTopic_2;

    rosbagFilename = argv[1];
    rosbagTopic_1 = argv[2];
    rosbagTopic_2 = argv[3];

    rosbag::Bag mybag;
    mybag.open(rosbagFilename);

    StereoCamera sc;

    int left_ready = 0;
    int right_ready = 0;
    sensor_msgs::Image::ConstPtr imgPtr_Left;
    sensor_msgs::Image::ConstPtr imgPtr_Right;
    for (rosbag::MessageInstance const m: rosbag::View(mybag))
    {
        if ((m.getTopic() != rosbagTopic_1) && (m.getTopic() != rosbagTopic_2)) continue;

        if (m.getTopic() == rosbagTopic_1)
        {
            imgPtr_Left = m.instantiate<sensor_msgs::Image>();
            left_ready = 1;
        }

        if (m.getTopic() == rosbagTopic_2)
        {
            imgPtr_Right = m.instantiate<sensor_msgs::Image>();
            right_ready = 1;
        }
        
        if (left_ready*right_ready) 
        {
            // sc.ProcessImage(imgPtr_Left, imgPtr_Right);
            sc.ProcessImage_EqF(imgPtr_Left, imgPtr_Right);
            left_ready = 0;
            right_ready = 0;
        }

    }

    mybag.close();

    return 0;
    
}
