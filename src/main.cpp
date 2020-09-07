#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <math.h>
#include <Eigen/Eigen>

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
    string rosbagFilename, rosbagTopic_1, rosbagTopic_2;

    rosbagFilename = argv[1];
    rosbagTopic_1 = argv[2];
    rosbagTopic_2 = argv[3];

    rosbag::Bag mybag;
    mybag.open(rosbagFilename);

    StereoCamera sc;
    StereoFilter sf;
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
            // sc.ProcessImage_EqF(cv_ptr_left->image.clone(), cv_ptr_right->image.clone(), t);
            Eigen::Matrix4d velocity = sc.processImages(landmarks, sf.getPose(), cv_ptr_left->image.clone(), cv_ptr_right->image.clone(), t);
            sf.integrateEquations(landmarks, velocity);


            left_ready = false;
            right_ready = false;
        }

    }

    mybag.close();

    return 0;
    
}
