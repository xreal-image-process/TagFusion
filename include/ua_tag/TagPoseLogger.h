#ifndef UA_TAG_TAG_POSE_LOGGER_H
#define UA_TAG_TAG_POSE_LOGGER_H

#include "ua_tag/TagObservation.h"

#include <fstream>
#include <string>
#include <vector>

namespace ORB_SLAM3
{
namespace ua_tag
{

// 将每帧各 Tag 的位姿字段写入 CSV（一行一个 tag）
class TagPoseLogger
{
public:
    explicit TagPoseLogger(const std::string &csvPath);
    ~TagPoseLogger();

    TagPoseLogger(const TagPoseLogger &) = delete;
    TagPoseLogger &operator=(const TagPoseLogger &) = delete;

    bool IsOpen() const { return mFile.is_open(); }

    // 记录一帧中所有 Tag 的 pose；无检测也可调用（不写行）
    void LogFrame(size_t frame_id,
                  double timestamp,
                  const std::vector<tag::TagObservation> &tags);

private:
    void WriteHeader();
    void WriteCandidate(const tag::TagPoseCandidate &cand, bool valid);

    std::ofstream mFile;
};

}  // namespace ua_tag
}  // namespace ORB_SLAM3

#endif
