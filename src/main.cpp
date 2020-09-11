#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <math.h>
#include <Eigen/Eigen>

#include "yaml-cpp/yaml.h"
#ifndef CONFIG_FILE
#define CONFIG_FILE "config.yaml"
#endif

#include "StereoCamera.h"
#include "Landmark.h"
#include "StereoFilter.h"

#include <cv_bridge/cv_bridge.h>


#include <opencv2/stereo/stereo.hpp>
#include <string>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <typeinfo>

using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
    
    fstream file("trajec_eqf.txt", ios::out);
    fstream file2("trajec.txt", ios::out);
    fstream file3("Formatted_traj.txt", ios::out);
    fstream file4("time.txt", ios::out);
    fstream filt5("err.txt", ios::out);

    Eigen::Matrix4d vel_pre;

    string rosbagFilename, rosbagTopic_1, rosbagTopic_2;

    rosbagFilename = argv[1];
    if (argc <= 2) {
        rosbagTopic_1 = "/cam0/image_raw";
        rosbagTopic_2 = "/cam1/image_raw";
    } else {
        rosbagTopic_1 = argv[2];
        rosbagTopic_2 = argv[3];
    }

    rosbag::Bag mybag;
    mybag.open(rosbagFilename);

    YAML::Node configNode = YAML::LoadFile(CONFIG_FILE);

    StereoCamera sc(configNode);
    StereoFilter sf(configNode);
    vector<Landmark> landmarks;
    

    bool left_ready = false;
    bool right_ready = false;
    sensor_msgs::Image::ConstPtr imgPtr_Left;
    sensor_msgs::Image::ConstPtr imgPtr_Right;
    for (rosbag::MessageInstance const m: rosbag::View(mybag))
    {
        if ((m.getTopic() != rosbagTopic_1) && (m.getTopic() != rosbagTopic_2)) continue;

        if (m.getTopic() == rosbagTopic_1)
        {
            imgPtr_Left = m.instantiate<sensor_msgs::Image>();
            left_ready = true;
        }

        if (m.getTopic() == rosbagTopic_2)
        {
            imgPtr_Right = m.instantiate<sensor_msgs::Image>();
            right_ready = true;
        }
        
        if (left_ready & right_ready) 
        {
            // sc.ProcessImage(imgPtr_Left, imgPtr_Right);
            cv_bridge::CvImagePtr cv_ptr_left;
            cv_bridge::CvImagePtr cv_ptr_right;
            try
            {
                cv_ptr_left = cv_bridge::toCvCopy(imgPtr_Left, sensor_msgs::image_encodings::MONO8);
                cv_ptr_right = cv_bridge::toCvCopy(imgPtr_Right, sensor_msgs::image_encodings::MONO8);
            }
            catch(cv_bridge::Exception& e)
            {
                // throw(Exception("cv_bridge exception: %s", e.what()));
                return 1;
            }

            double t = cv_ptr_left->header.stamp.toSec();
            if (t > configNode["start"].as<double>()  )
            {
                Eigen::Matrix4d velocity = sc.processImages(landmarks, sf.getPose(), cv_ptr_left->image.clone(), cv_ptr_right->image.clone(), t);
                
                if (velocity.block<3,1>(0,3).norm()>0.5 || velocity.hasNaN())
                {
                    velocity = vel_pre;
                }
                
                sf.integrateEquations(landmarks, velocity);
            
                sf.Save_trajec(sf.getPose(),"Formatted_traj.txt",t);

                vel_pre=velocity;
            
            }
            left_ready = false;
            right_ready = false;
            
        }

    }

    mybag.close();

    return 0;
    
}
