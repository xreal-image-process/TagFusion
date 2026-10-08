#include "ua_tag/TagPoseLogger.h"

#include <iomanip>

namespace ORB_SLAM3
{
namespace ua_tag
{

TagPoseLogger::TagPoseLogger(const std::string &csvPath)
    : mFile(csvPath.c_str(), std::ios::out | std::ios::trunc)
{
    if(mFile.is_open())
    {
        mFile << std::setprecision(17);
        WriteHeader();
    }
}

TagPoseLogger::~TagPoseLogger()
{
    if(mFile.is_open())
        mFile.close();
}

void TagPoseLogger::WriteHeader()
{
    mFile
        << "frame_id,timestamp,tag_id,"
        << "has_pose_estimate,selected_candidate,ambiguity_ratio,"
        // candidate 0
        << "c0_valid,c0_err,"
        << "c0_tx,c0_ty,c0_tz,"
        << "c0_r00,c0_r01,c0_r02,"
        << "c0_r10,c0_r11,c0_r12,"
        << "c0_r20,c0_r21,c0_r22,"
        // candidate 1
        << "c1_valid,c1_err,"
        << "c1_tx,c1_ty,c1_tz,"
        << "c1_r00,c1_r01,c1_r02,"
        << "c1_r10,c1_r11,c1_r12,"
        << "c1_r20,c1_r21,c1_r22\n";
}

void TagPoseLogger::WriteCandidate(const tag::TagPoseCandidate &cand, bool valid)
{
    if(!valid)
    {
        // 14 fields: valid + err + t(3) + R(9)
        mFile << "0,,,,,,,,,,,,,";
        return;
    }

    const Eigen::Matrix3f R = cand.T_ct.rotationMatrix();
    const Eigen::Vector3f t = cand.T_ct.translation();

    mFile << "1,"
          << cand.reprojection_error << ","
          << t[0] << "," << t[1] << "," << t[2] << ","
          << R(0, 0) << "," << R(0, 1) << "," << R(0, 2) << ","
          << R(1, 0) << "," << R(1, 1) << "," << R(1, 2) << ","
          << R(2, 0) << "," << R(2, 1) << "," << R(2, 2);
}

void TagPoseLogger::LogFrame(size_t frame_id,
                             double timestamp,
                             const std::vector<tag::TagObservation> &tags)
{
    if(!mFile.is_open())
        return;

    for(size_t i = 0; i < tags.size(); ++i)
    {
        const tag::TagObservation &tag = tags[i];
        const bool has_pose = tag.pose_estimate.has_value();
        const int selected = has_pose ? tag.pose_estimate->selected_candidate : -1;
        const float ambiguity_ratio = has_pose ? tag.pose_estimate->ambiguity_ratio : -1.0f;

        mFile << frame_id << ","
              << timestamp << ","
              << tag.tag_id << ","
              << (has_pose ? 1 : 0) << ","
              << selected << ","
              << ambiguity_ratio << ",";

        if(has_pose)
        {
            WriteCandidate(tag.pose_estimate->candidates[0],
                           tag.pose_estimate->candidates[0].valid);
            mFile << ",";
            WriteCandidate(tag.pose_estimate->candidates[1],
                           tag.pose_estimate->candidates[1].valid);
        }
        else
        {
            WriteCandidate(tag::TagPoseCandidate{}, false);
            mFile << ",";
            WriteCandidate(tag::TagPoseCandidate{}, false);
        }
        mFile << "\n";
    }
}

}  // namespace ua_tag
}  // namespace ORB_SLAM3
