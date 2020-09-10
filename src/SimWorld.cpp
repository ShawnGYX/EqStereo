#include "SimWorld.h"
#include "eigen3/unsupported/Eigen/MatrixFunctions"
#include "iostream"

Vector4d homog(const Vector3d& p) {
    Vector4d pbar;
    pbar << p.x(), p.y(), p.z(), 1.0;
    return pbar;
}
Vector3d unhomog(const Vector4d& pbar) {
    return pbar.block<3,1>(0,0);
}

vector<Landmark> SimWorld::generateRandomLandmarks(const int& number) {
    vector<Landmark> landmarks(number);
    worldPoints.clear();
    for (Landmark& lm : landmarks) {
        // Generate a landmark
        const Vector3d p = (2*Vector3d::Random() + Vector3d(0,0,10));
        const Vector3d pBody = Rotation.transpose() * (p - Translation);

        // Ignore the distorted coordinates lm.camcoor_left_distorted and lm.camcoor_right_distorted
        const Vector3d p_cam_left = unhomog(XL.inverse() * homog(pBody));
        const Vector3d p_cam_right = unhomog(XR.inverse() * homog(pBody));

        const Vector3d p_img_left = 1/p_cam_left.z() * KL * p_cam_left;
        const Vector3d p_img_right = 1/p_cam_right.z() * KL * p_cam_right;

        lm.camcoor_left_norm = Point2f(p_cam_left.x(), p_cam_left.y()) / p_cam_left.z();
        lm.camcoor_left = Point2f(p_img_left.x(), p_img_left.y()) / p_img_left.z();
        lm.camcoor_right_norm = Point2f(p_cam_right.x(), p_cam_right.y()) / p_cam_right.z();
        lm.camcoor_right = Point2f(p_img_right.x(), p_img_right.y()) / p_img_right.z();

        lm.X_lm = Vector3d::Zero();
        lm.sig = Matrix3d::Identity()*Sigma_coef;
        lm.isNew = true;
        lm.lifecycle = -1;

        // Set the 3d coordinates of the landmark
        lm.p_0 = p;
        const Vector3d p_cam_left_noisy = p_cam_left + Vector3d::Random();
        lm.p_t_bff = Point3f(p_cam_left_noisy.x(),p_cam_left_noisy.y(),p_cam_left_noisy.z());


        // Save the true point
        worldPoints.emplace_back(p);
    }

    return landmarks;
}

Matrix4d SimWorld::simulateMotion(vector<Landmark>& landmarks) {
    // Motion is circular on the xy plane
    double omU = 0.1;
    double vU = 0.1;
    double dt = 0.1;
    Matrix4d U = Matrix4d::Zero();
    U.block<3,3>(0,0) << 0.0,-omU, 0.0,
                         omU, 0.0, 0.0,
                         0.0, 0.0, 0.0;
    U.block<3,1>(0,3) <<  vU, 0.0, 0.0;

    Matrix4d tfMat = (dt*U).exp();

    // Update the position and attitude
    Translation = Translation + Rotation * tfMat.block<3,1>(0,3);
    Rotation = Rotation * tfMat.block<3,3>(0,0);

    updateLandmarkMeasurements(landmarks);

    return tfMat;

}

void SimWorld::updateLandmarkMeasurements(vector<Landmark>& landmarks) const {
    for (int i=0; i<landmarks.size(); ++i) {
        Landmark& lm = landmarks[i];
        const Vector3d& p = worldPoints[i];

        // Compute the left and right camera coordinates
        const Vector3d pBody = Rotation.transpose() * (p - Translation);
        const Vector3d p_cam_left = unhomog(XL.inverse() * homog(pBody));
        const Vector3d p_img_left = 1/p_cam_left.z() * KL * p_cam_left;
        lm.camcoor_left_norm = Point2f(p_cam_left.x(), p_cam_left.y()) / p_cam_left.z();
        lm.camcoor_left = Point2f(p_img_left.x(), p_img_left.y()) / p_img_left.z();

        const Vector3d p_cam_right = unhomog(XR.inverse() * homog(pBody));
        const Vector3d p_img_right = 1/p_cam_right.z() * KR * p_cam_right;
        lm.camcoor_right_norm = Point2f(p_cam_right.x(), p_cam_right.y()) / p_cam_right.z();
        lm.camcoor_right = Point2f(p_img_right.x(), p_img_right.y()) / p_img_right.z();

        // Update landmark
        ++lm.lifecycle;
        lm.p_t_bff = Point3f(p_cam_left.x(),p_cam_left.y(),p_cam_left.z());
    }
}