/**
 * Stereo TagFusion runner for PennCOSYVIO VI-Sensor sequences (e.g. visensor/af).
 *
 * Expected sequence layout:
 *   left_cam_frames/frame_0001.png ...
 *   right_cam_frames/frame_0001.png ...
 *   timestamps_cameras.txt          (preferred, sequence root)
 *   left_cam_frames/timestamps.txt  (fallback)
 *
 * Usage:
 *   ./stereo_penncosyvio_tag vocabulary settings sequence_path [trajectory_file_name]
 *
 * Example:
 *   ./stereo_penncosyvio_tag Vocabulary/ORBvoc.txt \
 *       Examples/TagFusion/PennCOSYVIO/PennCOSYVIO_VI_Stereo.yaml \
 *       /home/geneta/下载/PennCOSYVIO/visensor/af af_stereo
 */

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
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

// Accept:
//   .../visensor/af
//   .../visensor/af/left_cam_frames  (sibling right_cam_frames/)
bool ResolveStereoPaths(const string &inputPath,
                        string &leftPath, string &rightPath, string &timesPath)
{
    leftPath.clear();
    rightPath.clear();
    timesPath.clear();

    const string path = StripTrailingSlash(inputPath);
    const string nestedLeft = path + "/left_cam_frames";
    const string nestedRight = path + "/right_cam_frames";

    string seqRoot;
    if(DirExists(nestedLeft) && DirExists(nestedRight))
    {
        leftPath = nestedLeft;
        rightPath = nestedRight;
        seqRoot = path;
    }
    else if(DirExists(path))
    {
        // path may already be left_cam_frames
        const size_t slash = path.find_last_of("/\\");
        const string parent =
            (slash == string::npos) ? string(".") : path.substr(0, slash);
        const string siblingRight = parent + "/right_cam_frames";
        if(DirExists(siblingRight))
        {
            leftPath = path;
            rightPath = siblingRight;
            seqRoot = parent;
        }
    }

    if(leftPath.empty() || rightPath.empty())
        return false;

    const string candidates[] = {
        seqRoot + "/timestamps_cameras.txt",
        leftPath + "/timestamps.txt",
        seqRoot + "/timestamps.txt",
        rightPath + "/timestamps.txt",
    };
    for(const string &c : candidates)
    {
        if(FileExists(c))
        {
            timesPath = c;
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
    if(!ResolveStereoPaths(seqPath, leftPath, rightPath, timesPath))
    {
        cerr << "ERROR: Cannot find PennCOSYVIO stereo dirs under " << seqPath
             << endl;
        cerr << "Expect .../visensor/af with left_cam_frames + right_cam_frames "
                "and timestamps_cameras.txt"
             << endl;
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
        double t = 0.0;
        if(!(ss >> t))
            continue;
        vTimeStamps.push_back(t);
    }
    fTimes.close();

    vstrImageLeft.reserve(vTimeStamps.size());
    vstrImageRight.reserve(vTimeStamps.size());

    size_t nValid = 0;
    for(size_t i = 0; i < vTimeStamps.size(); ++i)
    {
        stringstream ssL, ssR;
        ssL << leftPath << "/frame_" << setfill('0') << setw(4) << (i + 1)
            << ".png";
        ssR << rightPath << "/frame_" << setfill('0') << setw(4) << (i + 1)
            << ".png";
        if(!FileExists(ssL.str()) || !FileExists(ssR.str()))
            break;
        vstrImageLeft.push_back(ssL.str());
        vstrImageRight.push_back(ssR.str());
        ++nValid;
    }

    if(nValid < vTimeStamps.size())
        vTimeStamps.resize(nValid);

    cout << "Timestamps: " << timesPath << " (" << vTimeStamps.size() << ")"
         << endl;
    cout << "Left:       " << leftPath << "/frame_XXXX.png" << endl;
    cout << "Right:      " << rightPath << "/frame_XXXX.png" << endl;
}

// SaveTrajectoryEuRoC writes timestamp_ns = t_sec * 1e9. Convert first column
// back to seconds so the file matches TUM / PennCOSYVIO pose_tum.txt time base.
void ConvertEuRoCTimestampsToSeconds(const string &filename)
{
    ifstream fin(filename.c_str());
    if(!fin.is_open())
    {
        cerr << "WARNING: Cannot reopen " << filename
             << " to convert EuRoC timestamps to seconds" << endl;
        return;
    }

    vector<string> lines;
    string line;
    while(getline(fin, line))
    {
        if(line.empty())
            continue;

        stringstream ss(line);
        double t_ns = 0.0;
        if(!(ss >> t_ns))
            continue;

        string rest;
        getline(ss, rest);
        stringstream out;
        out.setf(ios::fixed);
        out << setprecision(6) << (t_ns * 1e-9) << rest;
        lines.push_back(out.str());
    }
    fin.close();

    ofstream fout(filename.c_str());
    if(!fout.is_open())
    {
        cerr << "WARNING: Cannot rewrite " << filename << endl;
        return;
    }
    for(size_t i = 0; i < lines.size(); ++i)
        fout << lines[i] << '\n';
    fout.close();
    cout << "Converted EuRoC timestamps to seconds in " << filename << endl;
}

}  // namespace

int main(int argc, char **argv)
{
    if(argc < 4)
    {
        cerr << endl
             << "Usage: ./stereo_penncosyvio_tag path_to_vocabulary "
                "path_to_settings path_to_sequence (trajectory_file_name)"
             << endl;
        cerr << "Example:" << endl;
        cerr << "  ./stereo_penncosyvio_tag Vocabulary/ORBvoc.txt "
                "Examples/TagFusion/PennCOSYVIO/PennCOSYVIO_VI_Stereo.yaml "
                "/home/geneta/下载/PennCOSYVIO/visensor/af af_stereo"
             << endl;
        return 1;
    }

    const bool bFileName = (argc >= 5);
    if(bFileName)
        cout << "file name: " << argv[4] << endl;

    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestampsCam;
    cout << "Loading PennCOSYVIO stereo sequence from " << argv[3] << " ..."
         << endl;
    LoadImages(string(argv[3]), vstrImageLeft, vstrImageRight, vTimestampsCam);
    cout << "LOADED " << vstrImageLeft.size() << " stereo pairs!" << endl;

    if(vstrImageLeft.empty() ||
       vstrImageLeft.size() != vstrImageRight.size() ||
       vstrImageLeft.size() != vTimestampsCam.size())
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
        std::chrono::monotonic_clock::time_point t1 =
            std::chrono::monotonic_clock::now();
#endif

        SLAM.TrackStereo(imLeft, imRight, tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 =
            std::chrono::monotonic_clock::now();
#endif

#ifdef REGISTER_TIMES
        t_track =
            t_resize +
            std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(
                t2 - t1)
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
        SLAM.SaveTrajectoryEuRoC(f_file);
        SLAM.SaveKeyFrameTrajectoryTUM(kf_file);
        ConvertEuRoCTimestampsToSeconds(f_file);
    }
    else
    {
        SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
        ConvertEuRoCTimestampsToSeconds("CameraTrajectory.txt");
    }

    return 0;
}
