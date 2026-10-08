/**
* Monocular TagFusion runner for XREAL Glass cam0 sequences.
*
* Expected layout (any of):
*   .../glass/cam0/images/m0000000.pgm ... + timestamps.txt
*   .../glass/cam0/          (contains images/)
*   .../glass/               (contains cam0/images/)
*
* timestamps.txt lines:
*   m0000000.pgm 3853.54170192
*
* Usage:
*   ./mono_xreal_glass_tag vocabulary settings sequence_or_image_path [trajectory_file_name]
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

// Resolve to image directory containing mXXXXXXX.pgm + timestamps.txt
void ResolveImageDir(const string &inputPath, string &imagePath, string &timesPath)
{
    imagePath.clear();
    timesPath.clear();

    const string path = StripTrailingSlash(inputPath);
    const string candidates[] = {
        path,
        path + "/images",
        path + "/cam0/images",
        path + "/raw_data/glass/cam0/images"
    };

    for(const string &c : candidates)
    {
        if(!DirExists(c))
            continue;
        const string ts = c + "/timestamps.txt";
        if(FileExists(ts))
        {
            imagePath = c;
            timesPath = ts;
            return;
        }
        // allow directory even if timestamps missing; LoadImages will error clearly
        if(imagePath.empty())
            imagePath = c;
    }

    if(!imagePath.empty() && timesPath.empty())
        timesPath = imagePath + "/timestamps.txt";
}

void LoadImages(const string &seqPath,
                vector<string> &vstrImages, vector<double> &vTimeStamps)
{
    vstrImages.clear();
    vTimeStamps.clear();

    string imagePath;
    string timesPath;
    ResolveImageDir(seqPath, imagePath, timesPath);

    if(imagePath.empty() || !DirExists(imagePath))
    {
        cerr << "ERROR: Cannot find Glass cam0 image directory under " << seqPath << endl;
        cerr << "Expect .../cam0/images with mXXXXXXX.pgm" << endl;
        return;
    }
    if(timesPath.empty() || !FileExists(timesPath))
    {
        cerr << "ERROR: Cannot find timestamps.txt near " << imagePath << endl;
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

        // allow "timestamp filename" as fallback
        if(name.find(".pgm") == string::npos && name.find(".png") == string::npos)
        {
            // e.g. "3853.54 m0000000.pgm"
            string maybeName;
            if(ss >> maybeName)
            {
                t = stod(name);
                name = maybeName;
            }
            else
                continue;
        }

        vTimeStamps.push_back(t);
        if(!name.empty() && name[0] == '/')
            vstrImages.push_back(name);
        else
            vstrImages.push_back(imagePath + "/" + name);
    }
    fTimes.close();

    cout << "Timestamps: " << timesPath << " (" << vTimeStamps.size() << ")" << endl;
    cout << "Images:     " << imagePath << "/mXXXXXXX.pgm" << endl;
}

}  // namespace

int main(int argc, char **argv)
{
    if(argc < 4)
    {
        cerr << endl
             << "Usage: ./mono_xreal_glass_tag path_to_vocabulary path_to_settings "
                "path_to_glass_or_cam0_or_images [trajectory_file_name]" << endl;
        cerr << "Example:" << endl;
        cerr << "  ./mono_xreal_glass_tag Vocabulary/ORBvoc.txt "
                "Examples/TagFusion/XREAL_Glass/XREAL_Glass_cam0.yaml "
                "/home/geneta/dataset/20260227-room-patrol-11-1/raw_data/glass/cam0/images"
             << endl;
        return 1;
    }

    const bool bFileName = (argc >= 5);
    if(bFileName)
        cout << "file name: " << argv[4] << endl;

    vector<string> vstrImageFilenames;
    vector<double> vTimestampsCam;
    cout << "Loading XREAL Glass sequence from " << argv[3] << " ..." << endl;
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

    // Viewer on：配合 Tag.viewer；ORB Viewer 默认开启便于调试
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

    // 与 cam0_pose.txt / timestamps.txt 一致：TUM 格式，时间戳单位为秒。
    // 勿用 SaveTrajectoryEuRoC（会写成 t*1e9 纳秒），否则无法与真值对齐。
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
