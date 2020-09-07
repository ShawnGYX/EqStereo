#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <math.h>
#include <Eigen/Eigen>

#include "StereoCamera.h"


#include <opencv2/stereo/stereo.hpp>
#include <string>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <typeinfo>

#include "Landmark.h"

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
