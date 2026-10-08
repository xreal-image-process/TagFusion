/**
* Stereo-Inertial TagFusion runner for XREAL Glass cam0/cam1 + imu0 sequences.
*
* Expected layout (any of):
*   .../glass/                 (contains cam0/images + cam1/images + imu0.txt)
*   .../glass/cam0             (sibling cam1/ and imu0.txt)
*   .../glass/cam0/images      (sibling ../cam1/images, ../../imu0.txt)
*
* timestamps.txt lines (left preferred; same names on both sides):
*   m0000000.pgm 4471.736307964
*
* imu0.txt lines:
*   <t> <temp> gx gy gz ax ay az     (rad/s, m/s^2)
*
* Usage:
*   ./stereo_inertial_xreal_glass_tag vocabulary settings sequence_or_glass_path [trajectory_file_name]
*/

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
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

bool ResolveStereoDirs(const string &inputPath,
                       string &leftPath, string &rightPath, string &timesPath)
{
    leftPath.clear();
    rightPath.clear();
    timesPath.clear();

    const string path = StripTrailingSlash(inputPath);

    struct Candidate
    {
        string left;
        string right;
    };

    const Candidate candidates[] = {
        {path + "/cam0/images", path + "/cam1/images"},
        {path + "/images", StripTrailingSlash(path + "/../cam1/images")},
        {path, StripTrailingSlash(path + "/../cam1/images")},
        {path + "/raw_data/glass/cam0/images",
         path + "/raw_data/glass/cam1/images"},
    };

    for(const Candidate &c : candidates)
    {
        if(!DirExists(c.left) || !DirExists(c.right))
            continue;

        const string ts = c.left + "/timestamps.txt";
        if(FileExists(ts))
        {
            leftPath = c.left;
            rightPath = c.right;
            timesPath = ts;
            return true;
        }

        if(leftPath.empty())
        {
            leftPath = c.left;
            rightPath = c.right;
            timesPath = c.left + "/timestamps.txt";
        }
    }

    return !leftPath.empty() && !rightPath.empty();
}

bool ResolveImuPath(const string &inputPath, const string &leftPath,
                    string &imuPath)
{
    imuPath.clear();
    const string path = StripTrailingSlash(inputPath);

    const string candidates[] = {
        path + "/imu0.txt",
        path + "/raw_data/glass/imu0.txt",
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

    string leftPath, rightPath, timesPath;
    if(!ResolveStereoDirs(seqPath, leftPath, rightPath, timesPath))
    {
        cerr << "ERROR: Cannot find Glass cam0/cam1 image directories under "
             << seqPath << endl;
        cerr << "Expect .../cam0/images and .../cam1/images with mXXXXXXX.pgm"
             << endl;
        return;
    }
    if(!FileExists(timesPath))
    {
        cerr << "ERROR: Cannot find timestamps.txt near " << leftPath << endl;
        return;
    }

    ifstream fTimes(timesPath.c_str());
    if(!fTimes.is_open())
    {
        cerr << "ERROR: Cannot open timestamps file: " << timesPath << endl;
        return;
    }

    string line;
    while(getline(fTimes, line))
    {
        if(line.empty())
            continue;

        stringstream ss(line);
        string name;
        double t = 0.0;
        if(!(ss >> name >> t))
            continue;

        if(name.find(".pgm") == string::npos && name.find(".png") == string::npos)
        {
            string maybeName;
            if(ss >> maybeName)
            {
                t = stod(name);
                name = maybeName;
            }
            else
                continue;
        }

        const size_t slash = name.find_last_of("/\\");
        if(slash != string::npos)
            name = name.substr(slash + 1);

        const string leftFile = leftPath + "/" + name;
        const string rightFile = rightPath + "/" + name;
        if(!FileExists(leftFile) || !FileExists(rightFile))
            continue;

        vTimeStamps.push_back(t);
        vstrImageLeft.push_back(leftFile);
        vstrImageRight.push_back(rightFile);
    }
    fTimes.close();

    cout << "Timestamps: " << timesPath << " (" << vTimeStamps.size() << ")"
         << endl;
    cout << "Left:       " << leftPath << endl;
    cout << "Right:      " << rightPath << endl;
}

// imu0.txt: t temp gx gy gz ax ay az  (or t gx gy gz ax ay az)
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
            // t temp gx gy gz ax ay az
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
            // t gx gy gz ax ay az
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
             << "Usage: ./stereo_inertial_xreal_glass_tag path_to_vocabulary "
                "path_to_settings path_to_glass_or_cam0_or_images "
                "[trajectory_file_name]"
             << endl;
        cerr << "Example:" << endl;
        cerr << "  ./stereo_inertial_xreal_glass_tag Vocabulary/ORBvoc.txt "
                "Examples/TagFusion/XREAL_Glass/q3hy_stereo+imu.yaml "
                "/home/geneta/dataset/01-1/raw_data/glass"
             << endl;
        return 1;
    }

    const bool bFileName = (argc >= 5);
    if(bFileName)
        cout << "file name: " << argv[4] << endl;

    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestampsCam;
    cout << "Loading XREAL Glass stereo sequence from " << argv[3] << " ..."
         << endl;
    LoadImages(string(argv[3]), vstrImageLeft, vstrImageRight, vTimestampsCam);
    cout << "LOADED " << vstrImageLeft.size() << " stereo pairs!" << endl;

    {
        cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
        int maxFrames = 0;
        if(fsSettings.isOpened())
        {
            const cv::FileNode n = fsSettings["System.maxFrames"];
            if(!n.empty())
                maxFrames = static_cast<int>(n);
        }
        if(maxFrames > 0 && maxFrames < static_cast<int>(vstrImageLeft.size()))
        {
            cout << "System.maxFrames=" << maxFrames
                 << "  (using first " << maxFrames << " / "
                 << vstrImageLeft.size() << " pairs)" << endl;
            vstrImageLeft.resize(static_cast<size_t>(maxFrames));
            vstrImageRight.resize(static_cast<size_t>(maxFrames));
            vTimestampsCam.resize(static_cast<size_t>(maxFrames));
        }
    }

    if(vstrImageLeft.empty() || vstrImageLeft.size() != vstrImageRight.size())
    {
        cerr << "ERROR: No valid stereo pairs loaded from " << argv[3] << endl;
        return 1;
    }

    string leftPath, rightPath, timesPath, imuPath;
    ResolveStereoDirs(string(argv[3]), leftPath, rightPath, timesPath);
    if(!ResolveImuPath(string(argv[3]), leftPath, imuPath))
    {
        cerr << "ERROR: Cannot find imu0.txt under " << argv[3] << endl;
        cerr << "Expect .../raw_data/glass/imu0.txt" << endl;
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
        if(std::abs(freq - 200.0) > 50.0)
            cout << "NOTE: this sequence is not 200 Hz; yaml IMU.Frequency "
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
            int width = imLeft.cols * imageScale;
            int height = imLeft.rows * imageScale;
            cv::resize(imLeft, imLeft, cv::Size(width, height));
            cv::resize(imRight, imRight, cv::Size(width, height));
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
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        SLAM.TrackStereo(imLeft, imRight, tframe, vImuMeas);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

#ifdef REGISTER_TIMES
        t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
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
