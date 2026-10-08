#include "OptimizableTypes.h"
#include "G2oTypes.h"

#include <iostream>

int main()
{
    double max_err = 0.0;
    const bool ok_vis = ORB_SLAM3::TestTagCornerEdgeJacobians(&max_err);
    if(!ok_vis)
    {
        std::cerr << "Tag corner edge Jacobian check FAILED, max_abs_err="
                  << max_err << std::endl;
        return 1;
    }
    std::cout << "Tag corner edge Jacobian check OK, max_abs_err=" << max_err
              << std::endl;

    double max_err_inertial = 0.0;
    const bool ok_imu = ORB_SLAM3::TestTagCornerInertialEdgeJacobians(&max_err_inertial);
    if(!ok_imu)
    {
        std::cerr << "Tag inertial corner edge Jacobian check FAILED, max_abs_err="
                  << max_err_inertial << std::endl;
        return 1;
    }
    std::cout << "Tag inertial corner edge Jacobian check OK, max_abs_err="
              << max_err_inertial << std::endl;
    return 0;
}
