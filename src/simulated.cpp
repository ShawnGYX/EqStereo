// Simulate data for the stereo filter

#include "Landmark.h"
#include <vector>
#include "StereoFilter.h"
#include "SimWorld.h"
#include "yaml-cpp/yaml.h"
#include "iostream"

using namespace std;

int main(int argc, char const *argv[])
{
    YAML::Node config = YAML::LoadFile(SIM_CONFIG_FILE);
    StereoFilter sf(config);
    SimWorld sw;
    vector<Landmark> landmarks = sw.generateRandomLandmarks(100);
    sf.integrateEquations(landmarks, Matrix4d::Identity());

    for (int step=0; step<100; ++step) {
        Matrix4d vel = sw.simulateMotion(landmarks);
        sf.integrateEquations(landmarks, vel);
    }



    return 0;
}
