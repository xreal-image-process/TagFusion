/**
* Stereo TagFusion runner for XREAL Glass cam0/cam1 sequences.
*
* Expected layout (any of):
*   .../glass/                 (contains cam0/images + cam1/images)
*   .../glass/cam0             (sibling cam1/ next to it)
*   .../glass/cam0/images      (sibling ../cam1/images)
*
* timestamps.txt lines (left preferred; same names on both sides):
*   m0000000.pgm 3853.54170192
*
* Usage:
*   ./stereo_xreal_glass_tag vocabulary settings sequence_or_glass_path [trajectory_file_name]
*/

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include <opencv2/core/core.hpp>

#include <System.h>

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

// Resolve left/right image dirs + left timestamps.txt
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

        // strip directory if present
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

}  // namespace

int main(int argc, char **argv)
{
    if(argc < 4)
    {
        cerr << endl
             << "Usage: ./stereo_xreal_glass_tag path_to_vocabulary "
                "path_to_settings path_to_glass_or_cam0_or_images "
                "[trajectory_file_name]"
             << endl;
        cerr << "Example:" << endl;
        cerr << "  ./stereo_xreal_glass_tag Vocabulary/ORBvoc.txt "
                "Examples/TagFusion/XREAL_Glass/XREAL_Glass_stereo.yaml "
                "/home/geneta/dataset/20260227-room-patrol-11-1/raw_data/glass"
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

    if(vstrImageLeft.empty() || vstrImageLeft.size() != vstrImageRight.size())
    {
        cerr << "ERROR: No valid stereo pairs loaded from " << argv[3] << endl;
        return 1;
    }

    vector<float> vTimesTrack(vstrImageLeft.size());

    cout << endl << "-------" << endl;
    cout.precision(17);

    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::STEREO, true);
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

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        SLAM.TrackStereo(imLeft, imRight, tframe);

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
