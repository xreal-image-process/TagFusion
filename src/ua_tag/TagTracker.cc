#include "ua_tag/TagTracker.h"

#include "Frame.h"
#include "KeyFrame.h"
#include "Map.h"
#include "ua_tag/MapTagData.h"
#include "ua_tag/TagMapExporter.h"
#include "ua_tag/TagPoseConstraints.h"

#include <iostream>

#include <opencv2/core.hpp>

namespace ORB_SLAM3 {
namespace tag {

TagTracker::TagTracker() = default;

TagTracker::~TagTracker() = default;

TagTracker::TagTracker(const std::string &settingsFile)
{
    cv::FileStorage fs(settingsFile, cv::FileStorage::READ);
    if(!fs.isOpened())
        return;

    cv::FileNode node = fs["Tag.verbose"];
    if(!node.empty())
        mbVerbose = static_cast<int>(node) != 0;

    // Tag.export=1 时启用导出；目录默认 tag_export，可用 Tag.export_dir 覆盖
    bool enable_export = false;
    node = fs["Tag.export"];
    if(!node.empty())
        enable_export = static_cast<int>(node) != 0;

    std::string export_dir = "tag_export";
    node = fs["Tag.export_dir"];
    if(!node.empty())
        export_dir = static_cast<std::string>(node);

    if(enable_export)
    {
        mpExporter = std::make_unique<TagMapExporter>(
            export_dir, TagFactorWeightParams::FromSettings(settingsFile));
    }
}

void TagTracker::LogCameraPose(const Frame &frame)
{
    if(!mpExporter || !mpExporter->IsEnabled() || !frame.HasPose())
        return;
    mpExporter->AppendCameraPose(frame.mnId, frame.mTimeStamp, frame.GetPose());
}

void TagTracker::LogTagDetections(const Frame &frame)
{
    if(!mpExporter || !mpExporter->IsEnabled())
        return;
    mpExporter->AppendTagDetections(frame.mnId, frame.mTimeStamp, frame.mTagFrameData);
}

void TagTracker::LogMapTagFirstRegistration(KeyFrame *pKF, int tag_id,
                                            int left_idx, int right_idx,
                                            MapTagData *pTag)
{
    if(!mpExporter || !mpExporter->IsEnabled() || !pKF)
        return;
    mpExporter->AppendMapTagFirstRegistration(pKF, tag_id, left_idx, right_idx,
                                              pTag);
}

void TagTracker::LogKeyFrameMapTagFirstRegistrations(KeyFrame *pKF)
{
    if(!mpExporter || !mpExporter->IsEnabled() || !pKF)
        return;
    mpExporter->LogKeyFrameMapTagFirstRegistrations(pKF);
}

void TagTracker::SaveExports(Map &map)
{
    if(!mpExporter || !mpExporter->IsEnabled())
        return;
    mpExporter->SaveFinalMap(map, true);
    mpExporter->CloseTrajectory();
}

void TagTracker::SaveTagMapOnline(Map &map)
{
    if(!mpExporter || !mpExporter->IsEnabled())
        return;
    mpExporter->SaveTagMapCorners(map, false);
}

bool TagTracker::SaveInitMaps(
    Map &map,
    const std::vector<Eigen::Vector3f> &orb_points,
    const std::vector<std::pair<double, Sophus::SE3f>> &orb_kf_tcw)
{
    if(!mpExporter || !mpExporter->IsEnabled())
        return false;

    bool ok = true;
    ok = mpExporter->SaveTagMapCorners(map) && ok;
    ok = mpExporter->SaveTagInitKeyFrames(map) && ok;
    ok = mpExporter->SaveOrbInitMap(orb_points, orb_kf_tcw) && ok;
    return ok;
}

void TagTracker::SeedLastPose(const Sophus::SE3f &Tcw)
{
    mLastTcw = Tcw;
    mbHasLastTcw = true;
    mbHasVelocity = false;
}

void TagTracker::ClearMotionCache()
{
    mbHasLastTcw = false;
    mbHasVelocity = false;
    mnReferenceKFId = 0;
    mbHasReferenceKF = false;
}

void TagTracker::ResetExports()
{
    if(mpExporter && mpExporter->IsEnabled())
        mpExporter->ResetSession();
}

std::optional<Sophus::SE3f> TagTracker::PredictPoseWithMotionModel() const
{
    if(!mbHasVelocity || !mbHasLastTcw)
        return std::nullopt;

    return mVelocity * mLastTcw;
}

void TagTracker::SelectReferenceKeyFrame(const Frame &frame, Map &map)
{
    (void)frame;
    (void)map;
    // 休眠：完整共视参考 KF 选择尚未接入真实 Map/KF
    mbHasReferenceKF = false;
    mnReferenceKFId = 0;
}

bool TagTracker::NeedNewKeyFrame(const Frame &frame, const Map &map) const
{
    (void)frame;
    (void)map;
    return false;
}

void TagTracker::UpdateMotionCache(const Sophus::SE3f &Tcw_current)
{
    if(mbHasLastTcw)
    {
        mVelocity = Tcw_current * mLastTcw.inverse();
        mbHasVelocity = true;
    }

    mLastTcw = Tcw_current;
    mbHasLastTcw = true;
}

bool TagTracker::Track(Frame &frame, Map &map)
{
    (void)frame;
    (void)map;
    // 休眠：Tag 跟踪尚未接入 Tracking 主循环
    return false;
}

bool TagTracker::Relocalize(Frame &frame, Map &map)
{
    (void)frame;
    (void)map;
    // TODO: Tag 重定位
    return false;
}

}  // namespace tag
}  // namespace ORB_SLAM3
