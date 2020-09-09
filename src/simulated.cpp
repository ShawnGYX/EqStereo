// Simulate data for the stereo filter

#include "Landmark.h"
#include <vector>
#include "StereoFilter.h"
#include "SimWorld.h"

using namespace std;

int main(int argc, char const *argv[])
{
    StereoFilter sf;
    SimWorld sw;
    vector<Landmark> landmarks = sw.generateRandomLandmarks(100);

    for (int step=0; step<100; ++step) {
        Matrix4d vel = sw.simulateMotion(landmarks);
        sf.integrateEquations(landmarks, vel);
    }



    return 0;
}
