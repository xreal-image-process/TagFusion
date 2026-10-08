#include "ua_tag/TagFrameData.h"

#include <utility>

namespace ORB_SLAM3 {
namespace tag {

void TagFrameData::Add(TagObservation observation)
{
    if(observation.camera_id == CameraId::RIGHT)
        right.push_back(std::move(observation));
    else
        left.push_back(std::move(observation));
}

const TagObservation *TagFrameData::Find(int tag_id, CameraId camera_id) const noexcept
{
    const Observations &obs =
        (camera_id == CameraId::RIGHT) ? right : left;

    for(const TagObservation &o : obs)
    {
        if(o.tag_id == tag_id)
            return &o;
    }
    return nullptr;
}

TagObservation *TagFrameData::Find(int tag_id, CameraId camera_id) noexcept
{
    Observations &obs = (camera_id == CameraId::RIGHT) ? right : left;

    for(TagObservation &o : obs)
    {
        if(o.tag_id == tag_id)
            return &o;
    }
    return nullptr;
}

bool TagFrameData::WriteBackSelectedCandidate(int tag_id, CameraId camera_id,
                                              int selected_candidate,
                                              const Sophus::SE3f *T_ct)
{
    if(selected_candidate < 0 || selected_candidate >= 2)
        return false;

    TagObservation *obs = Find(tag_id, camera_id);
    if(!obs || !obs->pose_estimate.has_value())
        return false;

    TagPoseEstimate &est = *obs->pose_estimate;
    if(!est.candidates[selected_candidate].valid)
        return false;

    est.selected_candidate = selected_candidate;
    if(T_ct)
        est.candidates[selected_candidate].T_ct = *T_ct;
    return true;
}

void TagFrameData::Clear()
{
    left.clear();
    right.clear();
}

void TagFrameData::ResetOptOutliers()
{
    for(TagObservation &obs : left)
        obs.ResetOptOutliers();
    for(TagObservation &obs : right)
        obs.ResetOptOutliers();
}

bool TagFrameData::Empty() const noexcept
{
    return left.empty() && right.empty();
}

std::size_t TagFrameData::Size() const noexcept
{
    return left.size() + right.size();
}

}  // namespace tag
}  // namespace ORB_SLAM3
