/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * Monocular TagFusion runner for PennCOSYVIO VI-Sensor sequences.
 *
 * Expected sequence layout (e.g. .../visensor/af):
 *   left_cam_frames/frame_0001.png ...
 *   timestamps_cameras.txt          (preferred)
 *   left_cam_frames/timestamps.txt  (fallback)
 *
 * Usage:
 *   ./mono_penncosyvio_tag vocabulary settings sequence_path [trajectory_file_name]
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

// Accept either:
//   .../visensor/af                    (sequence root)
//   .../visensor/af/left_cam_frames    (image folder)
void ResolveSequencePaths(const string &inputPath,
                          string &imagePath, string &timesPath)
{
    imagePath.clear();
    timesPath.clear();

    const string path = StripTrailingSlash(inputPath);
    const string nestedImages = path + "/left_cam_frames";

    // Case 1: sequence root containing left_cam_frames/
    if(DirExists(nestedImages))
    {
        imagePath = nestedImages;
        const string candidates[] = {
            path + "/timestamps_cameras.txt",
            nestedImages + "/timestamps.txt",
            path + "/timestamps.txt"
        };
        for(const string &c : candidates)
        {
            if(FileExists(c))
            {
                timesPath = c;
                return;
            }
        }
        return;
    }

    // Case 2: path is already left_cam_frames/
    imagePath = path;
    const size_t slash = path.find_last_of("/\\");
    const string parent = (slash == string::npos) ? string(".") : path.substr(0, slash);
    const string candidates[] = {
        parent + "/timestamps_cameras.txt",
        path + "/timestamps.txt",
        parent + "/timestamps.txt",
        path + "/timestamps_cameras.txt"
    };
    for(const string &c : candidates)
    {
        if(FileExists(c))
        {
            timesPath = c;
            return;
        }
    }
}

void LoadImages(const string &seqPath,
                vector<string> &vstrImages, vector<double> &vTimeStamps)
{
    vstrImages.clear();
    vTimeStamps.clear();

    string imagePath;
    string timesPath;
    ResolveSequencePaths(seqPath, imagePath, timesPath);

    if(imagePath.empty() || !DirExists(imagePath))
    {
        cerr << "ERROR: Cannot find image directory under " << seqPath << endl;
        cerr << "Expect .../visensor/af or .../visensor/af/left_cam_frames" << endl;
        return;
    }
    if(timesPath.empty())
    {
        cerr << "ERROR: Cannot find timestamps near " << seqPath << endl;
        cerr << "Tried timestamps_cameras.txt (sequence root) and timestamps.txt (image folder)" << endl;
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

    vstrImages.reserve(vTimeStamps.size());
    for(size_t i = 0; i < vTimeStamps.size(); ++i)
    {
        stringstream ss;
        ss << imagePath << "/frame_" << setfill('0') << setw(4) << (i + 1) << ".png";
        vstrImages.push_back(ss.str());
    }

    cout << "Timestamps: " << timesPath << " (" << vTimeStamps.size() << ")" << endl;
    cout << "Images:     " << imagePath << "/frame_XXXX.png" << endl;
}

// SaveTrajectoryEuRoC writes timestamp_ns = t_sec * 1e9. Convert first column back to seconds
// so the file matches TUM / PennCOSYVIO pose_tum.txt time base.
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
             << "Usage: ./mono_penncosyvio_tag path_to_vocabulary path_to_settings "
                "path_to_sequence_or_left_cam_frames (trajectory_file_name)" << endl;
        cerr << "Example:" << endl;
        cerr << "  ./mono_penncosyvio_tag Vocabulary/ORBvoc.txt "
                "Examples/TagFusion/PennCOSYVIO/PennCOSYVIO_VI_Left.yaml "
                "/path/to/PennCOSYVIO/visensor/af" << endl;
        cerr << "  ./mono_penncosyvio_tag Vocabulary/ORBvoc.txt "
                "Examples/TagFusion/PennCOSYVIO/PennCOSYVIO_VI_Left.yaml "
                "/path/to/PennCOSYVIO/visensor/af/left_cam_frames" << endl;
        return 1;
    }

    const bool bFileName = (argc >= 5);
    if(bFileName)
        cout << "file name: " << argv[4] << endl;

    vector<string> vstrImageFilenames;
    vector<double> vTimestampsCam;
    cout << "Loading PennCOSYVIO sequence from " << argv[3] << " ..." << endl;
    LoadImages(string(argv[3]), vstrImageFilenames, vTimestampsCam);
    cout << "LOADED " << vstrImageFilenames.size() << " images!" << endl;

    if(vstrImageFilenames.empty())
    {
        cerr << "ERROR: No images loaded from " << argv[3] << endl;
        return 1;
    }

    vector<float> vTimesTrack(vstrImageFilenames.size());

    cout << endl << "-------" << endl;
    cout.precision(17);

    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);
    float imageScale = SLAM.GetImageScale();

    double t_resize = 0.f;
    double t_track = 0.f;

    cv::Mat im;
    for(size_t ni = 0; ni < vstrImageFilenames.size(); ni++)
    {
        im = cv::imread(vstrImageFilenames[ni], cv::IMREAD_UNCHANGED);
        double tframe = vTimestampsCam[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << vstrImageFilenames[ni] << endl;
            return 1;
        }

        if(imageScale != 1.f)
        {
#ifdef REGISTER_TIMES
#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t_Start_Resize = std::chrono::monotonic_clock::now();
#endif
#endif
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
#ifdef REGISTER_TIMES
#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t_End_Resize = std::chrono::monotonic_clock::now();
#endif
            t_resize = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Resize - t_Start_Resize).count();
            SLAM.InsertResizeTime(t_resize);
#endif
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        SLAM.TrackMonocular(im, tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

#ifdef REGISTER_TIMES
        t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
        SLAM.InsertTrackTime(t_track);
#endif

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
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

    // SaveTrajectoryEuRoC 会把时间戳写成纳秒(t*1e9)，与 PennCOSYVIO pose_tum.txt(秒)对不上。
    // 关键帧轨迹用 TUM(秒)；全帧轨迹 EuRoC 写出后再把时间戳除回秒。
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
