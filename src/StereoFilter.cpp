#include "StereoFilter.h"
#include "eigen3/unsupported/Eigen/MatrixFunctions"
#include "fstream"
#include <iostream>
#include <iomanip>

void Save_Matrix(Eigen::Matrix4d tfmat, const string file);

void StereoFilter::update_vel(const Eigen::Matrix4d vel)
{
    X_rb = X_rb * vel;
}

void StereoFilter::update_p0(vector<Landmark>& landmarks) const
{
    Eigen::Matrix4d Phat;
    Phat = P_init*X_rb;
    
    for (int i = 0; i < landmarks.size(); i++)
    {
        if (landmarks[i].isNew)
        {
            Eigen::Vector4d pf;
            pf << landmarks[i].p_t_bff.x,landmarks[i].p_t_bff.y,landmarks[i].p_t_bff.z,1;
            Eigen::Vector4d p0;
            p0 = Phat*XL*pf;
            landmarks[i].p_0 = p0.head(3);
            landmarks[i].isNew = false;
        }
    }
    
}

Eigen::MatrixXd StereoFilter::compute_c(vector<Landmark>& landmarks) const
{
    int lm_num = landmarks.size();
    Eigen::Matrix4d Phat;
    Phat = P_init*X_rb;

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(4*lm_num,3*lm_num);

    for (int i = 0; i < lm_num; i++)
    {
        Eigen::Vector3d pi_hat = landmarks[i].p_0 + P_init.block<3,3>(0,0)*landmarks[i].X_lm;

        Eigen::Vector4d pi_hat_homo;
        pi_hat_homo << pi_hat.x(),pi_hat.y(),pi_hat.z(),1;

        Vector4d qi_hat_homo = Phat.inverse()*pi_hat_homo;

        Eigen::Vector4d yi_hat_left = XL.inverse() * qi_hat_homo;
        Eigen::Vector4d yi_hat_right = XR.inverse() * qi_hat_homo;

        Eigen::Matrix<double,2,3> de_left;
        de_left << fx_left/yi_hat_left.z(),0,-fx_left*yi_hat_left.x()/(yi_hat_left.z()*yi_hat_left.z()),
                    0,fy_left/yi_hat_left.z(),-fy_left*yi_hat_left.y()/(yi_hat_left.z()*yi_hat_left.z());

        Eigen::Matrix<double,2,3> ci_left = de_left*XL.block<3,3>(0,0).inverse()*X_rb.block<3,3>(0,0).inverse();

        Eigen::Matrix<double,2,3> de_right;
        de_right << fx_right/yi_hat_right.z(),0,-fx_right*yi_hat_right.x()/(yi_hat_right.z()*yi_hat_right.z()),
                    0,fy_right/yi_hat_right.z(),-fy_right*yi_hat_right.y()/(yi_hat_right.z()*yi_hat_right.z());

        Eigen::Matrix<double,2,3> ci_right = de_right*XR.block<3,3>(0,0).inverse()*X_rb.block<3,3>(0,0).inverse();


        // cout<<"Ci_left: "<<ci_left<<endl;
        // cout<<"de_right: "<<de_right<<endl;


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

Eigen::MatrixXd StereoFilter::compute_error(vector<Landmark>& landmarks)
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

Eigen::MatrixXd StereoFilter::build_Sigma(const vector<Landmark>& landmarks)
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

void StereoFilter::update_Sigma(Eigen::MatrixXd &C_mat, Eigen::MatrixXd &Sigma, vector<Landmark>& landmarks)
{
    const int lm_num = landmarks.size();

    

    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(3*lm_num,3*lm_num)*P_coef;
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(4*lm_num,4*lm_num)*Q_coef;

    Eigen::MatrixXd s;
    s = C_mat*Sigma*C_mat.transpose()+Q;

    Sigma += dt*(P-Sigma*C_mat.transpose()*s.inverse()*C_mat*Sigma);
    MatrixXd eigs = Sigma.eigenvalues().real();

    // The eigenvalues of Sigma should always be positive!
    if (Sigma.hasNaN()) {
        cout << "Sigma has NaN values!" << endl;
        throw("bad");
    }
    cout << "Sigma min eigen: " << eigs.minCoeff() << endl;
    cout << "Sigma max eigen: " << eigs.maxCoeff() << endl;

    for (int i = 0; i < lm_num; i++)
    {
        landmarks[i].sig = Sigma.block<3,3>(3*(i),3*(i));
    }
}


Innov StereoFilter::Compute_innovation(const Eigen::MatrixXd &C_mat, const Eigen::MatrixXd &err,
                    const Eigen::MatrixXd &Sigma, const vector<Landmark>& landmarks, bool isMoving)
{
    
    // Compute the base innovation
    int lm_num = landmarks.size();
    const Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(4*lm_num,4*lm_num)*Q_coef;
    const Eigen::MatrixXd s = C_mat*Sigma*C_mat.transpose()+Q;
    const Eigen::MatrixXd gamma = Sigma*C_mat.transpose()*s.inverse()*err;

    // Lift the innovation
    // -------------------

    // Compute Pose innovation Delta
    Eigen::MatrixXd lsqmat = Eigen::MatrixXd::Zero(3*lm_num, 6);
    for (int i = 0; i < lm_num; i++)
    {   
        Eigen::Vector4d pi_homo;
        pi_homo << landmarks[i].p_0.x(),landmarks[i].p_0.y(),landmarks[i].p_0.z(),1;
        Eigen::Vector4d q_i = P_init.inverse()*pi_homo;

        lsqmat.block<3,3>(3*(i),0) = skew(landmarks[i].X_lm + q_i.head(3));
        lsqmat.block<3,3>(3*(i),3) = - Eigen::Matrix3d::Identity();
    }

    Matrix<double,6,6> hess = lsqmat.transpose() * Sigma.inverse() * lsqmat;
    Eigen::Matrix<double,6,1> DeltaVec = hess.ldlt().solve(lsqmat.transpose() * Sigma.inverse() * gamma);
    Eigen::Matrix4d Delta = Eigen::Matrix4d::Zero();
    Delta.block<3,3>(0,0) = skew(DeltaVec.head(3));
    Delta.block<3,1>(0,3) = DeltaVec.tail(3);

    // Compute point innovations delta
    Eigen::MatrixXd delta = Eigen::MatrixXd::Zero(3*lm_num,1);
    for (int i = 0; i < lm_num; i++)
    {
        Eigen::Vector4d pi_homo;
        pi_homo << landmarks[i].p_0.x(),landmarks[i].p_0.y(),landmarks[i].p_0.z(),1;
        Eigen::Vector4d q_i = P_init.inverse()*pi_homo;

        delta.block<3,1>(3*(i),0) = gamma.block<3,1>(3*(i),0) + Delta.block<3,3>(0,0)*q_i.head(3) + Delta.block<3,1>(0,3);
    }

    // Put into innovation struct
    Innov Inn;
    if (isMoving)
    {
        Inn.Del = Delta;
        Inn.del = delta;
    }else
    {
        Inn.Del = Eigen::Matrix4d::Zero();
        Inn.del = 0*gamma;
    }
    
    // Inn.Del = Eigen::Matrix4d::Zero();
    // Inn.del = gamma;

    return Inn;

}

void StereoFilter::update_innovation(const Innov &innovation, vector<Landmark>& landmarks)
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


    // if (Delta.block<3,1>(0,3).norm()*dt>0.4)
    // {
    //     return;
    // }
    // else
    // {
    //     X_rb = (dt*Delta).exp()*X_rb;
    //     for (int i = 0; i < lm_num; i++)
    //     {
    //         Eigen::Vector3d d;
    //         d << delta(3*i,0),delta(1+3*i,0),delta(2+3*i,0);
    //         landmarks[i].X_lm = landmarks[i].X_lm + dt * d + dt * Delta.block<3,3>(0,0) * landmarks[i].X_lm;
    //     }
    // }
    
}





// Processing image
void StereoFilter::integrateEquations(vector<Landmark>& landmarks, const Matrix4d& velocity)
{

    // EqF
    if (velocity.block<3,1>(0,3).norm()>0.5 || velocity.hasNaN())
    {
        this->update_vel(Eigen::Matrix4d::Identity());

    }
    else
    {
        this->update_vel(velocity);
    }
    
    this->update_p0(landmarks);

    Eigen::MatrixXd C = compute_c(landmarks);
    Eigen::MatrixXd err;
    err = compute_error(landmarks);
    
    Eigen::MatrixXd Sigma = this->build_Sigma(landmarks);
    this->update_Sigma(C, Sigma, landmarks);

    bool isMoving = true;
    if (velocity.block<3,1>(0,3).norm()<0.0004)
    {
        isMoving = false;
    }
    
    Innov innovation = Compute_innovation(C,err,Sigma, landmarks, isMoving);
    this->update_innovation(innovation, landmarks);

    Eigen::Matrix4d pose = P_init*X_rb;

    Save_Matrix(pose, "trajec_eqf.txt");
    
}



void StereoFilter::Save_trajec(const Eigen::Matrix4d tfmat, const string file, const double t)
{
    Quaterniond q(tfmat.block<3,3>(0,0));
    
    ofstream traj;
    traj.open(file,ios::app);
    traj<<setprecision(14)<<t<<',';
    traj<<tfmat(0,3)<<','<<tfmat(1,3)<<','<<tfmat(2,3)<<','<<q.w()<<','<<q.x()<<','<<q.y()<<','<<q.z()<<"\n";

    traj.close();
}

void Save_Matrix(Eigen::Matrix4d tfmat, const string file)
{
    ofstream trajectory;
    trajectory.open(file, ios::app);
    trajectory<<tfmat<<"\n";
    trajectory.close();
}   