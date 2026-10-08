/**
 * CLI wrapper around Thirdparty/apriltag (ethz_apriltag2 / basalt stack).
 * Used by scripts/tag_detect.py so Python detection shares the same backend
 * as ORB-SLAM3 ua_tag::AprilTagDetector.
 *
 * Usage:
 *   ethz_tag_detect [--family tag36h11] [--hamming 2] [--black-border 1] image.png
 *
 * Prints one JSON object per line to stdout:
 *   {"id":0,"hamming":0,"perimeter":123.4,"center":[cx,cy],"corners":[[x,y],...]}
 */

#include <apriltags/TagDetector.h>
#include <apriltags/TagFamily.h>
#include <apriltags/Tag16h5.h>
#include <apriltags/Tag25h7.h>
#include <apriltags/Tag25h9.h>
#include <apriltags/Tag36h11.h>
#include <apriltags/Tag36h9.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

namespace {

const AprilTags::TagCodes *GetTagCodes(const std::string &family)
{
    if(family == "tag36h11") return &AprilTags::tagCodes36h11;
    if(family == "tag36h9")  return &AprilTags::tagCodes36h9;
    if(family == "tag25h9")  return &AprilTags::tagCodes25h9;
    if(family == "tag25h7")  return &AprilTags::tagCodes25h7;
    if(family == "tag16h5")  return &AprilTags::tagCodes16h5;
    return nullptr;
}

void PrintUsage(const char *argv0)
{
    std::cerr
        << "Usage: " << argv0
        << " [--family tag36h11] [--hamming 2] [--black-border 1] <image>\n";
}

}  // namespace

int main(int argc, char **argv)
{
    std::string family = "tag36h11";
    int hamming = 2;
    int black_border = 1;
    std::string image_path;

    for(int i = 1; i < argc; ++i)
    {
        if(std::strcmp(argv[i], "--family") == 0 && i + 1 < argc)
            family = argv[++i];
        else if(std::strcmp(argv[i], "--hamming") == 0 && i + 1 < argc)
            hamming = std::atoi(argv[++i]);
        else if(std::strcmp(argv[i], "--black-border") == 0 && i + 1 < argc)
            black_border = std::atoi(argv[++i]);
        else if(std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0)
        {
            PrintUsage(argv[0]);
            return 0;
        }
        else if(argv[i][0] == '-')
        {
            std::cerr << "unknown arg: " << argv[i] << "\n";
            PrintUsage(argv[0]);
            return 2;
        }
        else
            image_path = argv[i];
    }

    if(image_path.empty())
    {
        PrintUsage(argv[0]);
        return 2;
    }

    const AprilTags::TagCodes *codes = GetTagCodes(family);
    if(!codes)
    {
        std::cerr << "unsupported family: " << family << "\n";
        return 2;
    }

    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if(image.empty())
    {
        std::cerr << "failed to read image: " << image_path << "\n";
        return 1;
    }

    AprilTags::TagDetector detector(*codes, static_cast<size_t>(std::max(1, black_border)));
    std::vector<AprilTags::TagDetection> dets = detector.extractTags(image);

    for(const AprilTags::TagDetection &det : dets)
    {
        if(!det.good)
            continue;
        if(det.hammingDistance > hamming)
            continue;

        std::printf(
            "{\"id\":%d,\"hamming\":%d,\"perimeter\":%.4f,"
            "\"center\":[%.4f,%.4f],\"corners\":[[%.4f,%.4f],[%.4f,%.4f],"
            "[%.4f,%.4f],[%.4f,%.4f]]}\n",
            det.id, det.hammingDistance, det.observedPerimeter,
            det.cxy.first, det.cxy.second,
            det.p[0].first, det.p[0].second,
            det.p[1].first, det.p[1].second,
            det.p[2].first, det.p[2].second,
            det.p[3].first, det.p[3].second);
    }

    return 0;
}
