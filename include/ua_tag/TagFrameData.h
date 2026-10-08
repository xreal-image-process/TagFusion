#ifndef UA_TAG_TAG_FRAME_DATA_H
#define UA_TAG_TAG_FRAME_DATA_H

#include "ua_tag/TagObservation.h"

#include <cstddef>
#include <vector>

namespace ORB_SLAM3 {
namespace tag {

// 一帧内左右目 Tag 观测集合（供 Frame / Tracking 挂载）
class TagFrameData
{
public:
    using Observations = std::vector<TagObservation>;

    Observations left;
    Observations right;

    void Add(TagObservation observation);

    const TagObservation *Find(int tag_id, CameraId camera_id) const noexcept;
    TagObservation *Find(int tag_id, CameraId camera_id) noexcept;

    // 回写消歧结果：仅更新 selected_candidate 与所选候选 T_ct（T_ct 可选）
    bool WriteBackSelectedCandidate(int tag_id, CameraId camera_id,
                                    int selected_candidate,
                                    const Sophus::SE3f *T_ct = nullptr);

    void Clear();

    // 重置本帧所有 Tag 观测的优化外点（同帧多次 PoseOptimization 前调用）
    void ResetOptOutliers();

    bool Empty() const noexcept;
    std::size_t Size() const noexcept;
};

}  // namespace tag
}  // namespace ORB_SLAM3

#endif
