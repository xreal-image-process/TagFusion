/**
* Stereo-Inertial TagFusion runner for Aqua/Tank sequences
* (left/right dehazed images + imu0.txt).
*
* Expected layout (any of):
*   <seq>/left + <seq>/right + <seq>/imu0.txt
*   <seq>/cam0/data + <seq>/cam1/data + <seq>/imu0.txt
*   <seq>/mav0/cam0/data + <seq>/mav0/cam1/data + <seq>/mav0/imu0.txt
*
* Image filenames must contain timestamps, e.g. frame_1652274610.785289764.png
* Left/right pairs share the same filename.
*
* imu0.txt lines:
*   <t> gx gy gz ax ay az            (rad/s, m/s^2)
*   <t> <temp> gx gy gz ax ay az     (also accepted)
*
* Usage:
*   ./stereo_inertial_test_tag path_to_vocabulary path_to_settings
*                              path_to_sequence [trajectory_file_name]
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>

#include "ImuTypes.h"
#include "System.h"

using namespace std;

namespace {

bool FileExists(const string &path)
{
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool DirExists(const string &path)
{
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

string StripTrailingSlash(string path)
{
    while(!path.empty() && (path.back() == '/' || path.back() == '\\'))
        path.pop_back();
    return path;
}

string ParentDir(const string &path)
{
    const string p = StripTrailingSlash(path);
    const size_t slash = p.find_last_of("/\\");
    if(slash == string::npos)
        return ".";
    if(slash == 0)
        return "/";
    return p.substr(0, slash);
}

bool ParseTimestampFromFilename(const string &filename, double &timestamp)
{
    const string prefix = "frame_";
    const string suffix = ".png";

    if(filename.size() <= prefix.size() + suffix.size())
        return false;
    if(filename.compare(0, prefix.size(), prefix) != 0)
        return false;
    if(filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) != 0)
        return false;

    string tsStr = filename.substr(prefix.size(),
                                   filename.size() - prefix.size() - suffix.size());
    if(tsStr.empty())
        return false;

    try
    {
        timestamp = stod(tsStr);
    }
    catch(...)
    {
        return false;
    }

    return true;
}

bool ResolveStereoDirs(const string &inputPath,
                       string &leftPath, string &rightPath)
{
    leftPath.clear();
    rightPath.clear();

    const string path = StripTrailingSlash(inputPath);

    const pair<string, string> candidates[] = {
        {path + "/left", path + "/right"},
        {path + "/cam0/data", path + "/cam1/data"},
        {path + "/mav0/cam0/data", path + "/mav0/cam1/data"},
        {path + "/cam0/images", path + "/cam1/images"},
    };

    for(const auto &c : candidates)
    {
        if(DirExists(c.first) && DirExists(c.second))
        {
            leftPath = c.first;
            rightPath = c.second;
            return true;
        }
    }

    return false;
}

bool ResolveImuPath(const string &inputPath, const string &leftPath,
                    string &imuPath)
{
    imuPath.clear();
    const string path = StripTrailingSlash(inputPath);

    const string candidates[] = {
        path + "/imu0.txt",
        path + "/mav0/imu0.txt",
        ParentDir(leftPath) + "/imu0.txt",
        ParentDir(ParentDir(leftPath)) + "/imu0.txt",
        ParentDir(ParentDir(ParentDir(leftPath))) + "/imu0.txt",
    };

    for(const string &c : candidates)
    {
        if(FileExists(c))
        {
            imuPath = c;
            return true;
        }
    }
    return false;
}

void LoadImages(const string &seqPath,
                vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight,
                vector<double> &vTimeStamps)
{
    vstrImageLeft.clear();
    vstrImageRight.clear();
    vTimeStamps.clear();

    string leftPath, rightPath;
    if(!ResolveStereoDirs(seqPath, leftPath, rightPath))
    {
        cerr << "ERROR: Cannot find stereo left/right directories under "
             << seqPath << endl;
        cerr << "Expect left/right, cam0/data+cam1/data, or mav0/cam0|cam1/data"
             << endl;
        return;
    }

    vector<pair<double, string> > vAllImages;
    DIR *dir = opendir(leftPath.c_str());
    if(!dir)
    {
        cerr << "ERROR: Cannot open left image directory: " << leftPath << endl;
        return;
    }

    struct dirent *entry;
    while((entry = readdir(dir)) != NULL)
    {
        string name = entry->d_name;
        if(name == "." || name == "..")
            continue;

        double t;
        if(!ParseTimestampFromFilename(name, t))
            continue;

        const string rightFile = rightPath + "/" + name;
        if(!FileExists(rightFile))
            continue;

        vAllImages.push_back(make_pair(t, name));
    }
    closedir(dir);

    sort(vAllImages.begin(), vAllImages.end());

    vstrImageLeft.reserve(vAllImages.size());
    vstrImageRight.reserve(vAllImages.size());
    vTimeStamps.reserve(vAllImages.size());
    for(size_t i = 0; i < vAllImages.size(); i++)
    {
        const string &name = vAllImages[i].second;
        vTimeStamps.push_back(vAllImages[i].first);
        vstrImageLeft.push_back(leftPath + "/" + name);
        vstrImageRight.push_back(rightPath + "/" + name);
    }

    cout << "Left:  " << leftPath << endl;
    cout << "Right: " << rightPath << endl;
}

void LoadIMU(const string &imuPath,
             vector<double> &vTimeStamps,
             vector<cv::Point3f> &vAcc,
             vector<cv::Point3f> &vGyro)
{
    vTimeStamps.clear();
    vAcc.clear();
    vGyro.clear();

    ifstream fImu(imuPath.c_str());
    if(!fImu.is_open())
    {
        cerr << "ERROR: Cannot open IMU file: " << imuPath << endl;
        return;
    }

    string line;
    while(getline(fImu, line))
    {
        if(line.empty() || line[0] == '#')
            continue;

        stringstream ss(line);
        vector<double> vals;
        double x = 0.0;
        while(ss >> x)
            vals.push_back(x);

        double t = 0.0, gx = 0.0, gy = 0.0, gz = 0.0, ax = 0.0, ay = 0.0, az = 0.0;
        if(vals.size() >= 8)
        {
            t = vals[0];
            gx = vals[2];
            gy = vals[3];
            gz = vals[4];
            ax = vals[5];
            ay = vals[6];
            az = vals[7];
        }
        else if(vals.size() >= 7)
        {
            t = vals[0];
            gx = vals[1];
            gy = vals[2];
            gz = vals[3];
            ax = vals[4];
            ay = vals[5];
            az = vals[6];
        }
        else
            continue;

        vTimeStamps.push_back(t);
        vGyro.push_back(cv::Point3f(static_cast<float>(gx),
                                    static_cast<float>(gy),
                                    static_cast<float>(gz)));
        vAcc.push_back(cv::Point3f(static_cast<float>(ax),
                                   static_cast<float>(ay),
                                   static_cast<float>(az)));
    }
}

}  // namespace

int main(int argc, char **argv)
{
    if(argc < 4)
    {
        cerr << endl
             << "Usage: ./stereo_inertial_test_tag path_to_vocabulary "
                "path_to_settings path_to_sequence [trajectory_file_name]"
             << endl;
        cerr << "Sequence layout: left/ + right/ + imu0.txt "
                "(or cam0/data + cam1/data)"
             << endl;
        cerr << "Image filenames must contain timestamps, e.g. "
                "frame_1652274610.785289764.png"
             << endl;
        return 1;
    }

    const bool bFileName = (argc >= 5);
    if(bFileName)
        cout << "file name: " << argv[4] << endl;

    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestampsCam;

    cout << "Loading stereo sequence from " << argv[3] << " ..." << endl;
    LoadImages(string(argv[3]), vstrImageLeft, vstrImageRight, vTimestampsCam);
    cout << "LOADED " << vstrImageLeft.size() << " stereo pairs!" << endl;

    if(vstrImageLeft.empty() || vstrImageLeft.size() != vstrImageRight.size())
    {
        cerr << "ERROR: No valid stereo pairs loaded from " << argv[3] << endl;
        return 1;
    }

    string leftPath, rightPath, imuPath;
    ResolveStereoDirs(string(argv[3]), leftPath, rightPath);
    if(!ResolveImuPath(string(argv[3]), leftPath, imuPath))
    {
        cerr << "ERROR: Cannot find imu0.txt under " << argv[3] << endl;
        cerr << "Extract with: python3 extract_imu_from_bag.py <bag>" << endl;
        return 1;
    }

    vector<double> vTimestampsImu;
    vector<cv::Point3f> vAcc, vGyro;
    cout << "Loading IMU from " << imuPath << " ..." << endl;
    LoadIMU(imuPath, vTimestampsImu, vAcc, vGyro);
    cout << "LOADED " << vTimestampsImu.size() << " IMU samples!" << endl;

    if(vTimestampsImu.empty())
    {
        cerr << "ERROR: No IMU samples loaded from " << imuPath << endl;
        return 1;
    }

    if(vTimestampsImu.size() >= 2)
    {
        const double dt = (vTimestampsImu.back() - vTimestampsImu.front()) /
                          static_cast<double>(vTimestampsImu.size() - 1);
        const double freq = (dt > 0.0) ? (1.0 / dt) : 0.0;
        cout << "IMU mean dt=" << dt << " s  (~" << freq << " Hz)" << endl;
        if(std::abs(freq - 333.0) > 50.0)
            cout << "NOTE: this sequence is not 333 Hz; yaml IMU.Frequency "
                    "should match ~"
                 << static_cast<int>(freq + 0.5) << " Hz." << endl;
    }

    int first_imu = 0;
    if(vTimestampsImu.front() <= vTimestampsCam.front())
    {
        while(first_imu < static_cast<int>(vTimestampsImu.size()) &&
              vTimestampsImu[first_imu] <= vTimestampsCam.front())
            ++first_imu;
        if(first_imu > 0)
            --first_imu;
    }

    vector<float> vTimesTrack(vstrImageLeft.size());

    cout << endl << "-------" << endl;
    cout.precision(17);

    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_STEREO, true);
    float imageScale = SLAM.GetImageScale();

    double t_resize = 0.f;
    double t_track = 0.f;

    cv::Mat imLeft, imRight;
    for(size_t ni = 0; ni < vstrImageLeft.size(); ni++)
    {
        imLeft = cv::imread(vstrImageLeft[ni], cv::IMREAD_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni], cv::IMREAD_UNCHANGED);
        double tframe = vTimestampsCam[ni];

        if(imLeft.empty())
        {
            cerr << endl << "Failed to load left image at: "
                 << vstrImageLeft[ni] << endl;
            return 1;
        }
        if(imRight.empty())
        {
            cerr << endl << "Failed to load right image at: "
                 << vstrImageRight[ni] << endl;
            return 1;
        }

        if(imageScale != 1.f)
        {
#ifdef REGISTER_TIMES
#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t_Start_Resize =
                std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t_Start_Resize =
                std::chrono::monotonic_clock::now();
#endif
#endif
            int width = imLeft.cols * imageScale;
            int height = imLeft.rows * imageScale;
            cv::resize(imLeft, imLeft, cv::Size(width, height));
            cv::resize(imRight, imRight, cv::Size(width, height));
#ifdef REGISTER_TIMES
#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t_End_Resize =
                std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t_End_Resize =
                std::chrono::monotonic_clock::now();
#endif
            t_resize = std::chrono::duration_cast<
                std::chrono::duration<double, std::milli> >(t_End_Resize -
                                                            t_Start_Resize)
                           .count();
            SLAM.InsertResizeTime(t_resize);
#endif
        }

        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        if(ni > 0)
        {
            while(first_imu < static_cast<int>(vTimestampsImu.size()) &&
                  vTimestampsImu[first_imu] <= vTimestampsCam[ni])
            {
                vImuMeas.emplace_back(vAcc[first_imu].x, vAcc[first_imu].y,
                                      vAcc[first_imu].z, vGyro[first_imu].x,
                                      vGyro[first_imu].y, vGyro[first_imu].z,
                                      vTimestampsImu[first_imu]);
                ++first_imu;
            }
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 =
            std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 =
            std::chrono::monotonic_clock::now();
#endif

        SLAM.TrackStereo(imLeft, imRight, tframe, vImuMeas);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 =
            std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 =
            std::chrono::monotonic_clock::now();
#endif

#ifdef REGISTER_TIMES
        t_track =
            t_resize +
            std::chrono::duration_cast<
                std::chrono::duration<double, std::milli> >(t2 - t1)
                .count();
        SLAM.InsertTrackTime(t_track);
#else
        (void)t_resize;
        (void)t_track;
#endif

        double ttrack =
            std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1)
                .count();
        vTimesTrack[ni] = ttrack;

        double T = 0;
        if(ni + 1 < vTimestampsCam.size())
            T = vTimestampsCam[ni + 1] - tframe;
        else if(ni > 0)
            T = tframe - vTimestampsCam[ni - 1];

        if(ttrack < T)
            usleep((T - ttrack) * 1e6);
    }

    SLAM.Shutdown();

    if(bFileName)
    {
        const string kf_file = "kf_" + string(argv[4]) + ".txt";
        const string f_file = "f_" + string(argv[4]) + ".txt";
        SLAM.SaveTrajectoryTUM(f_file);
        SLAM.SaveKeyFrameTrajectoryTUM(kf_file);
    }
    else
    {
        SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    }

    return 0;
}
