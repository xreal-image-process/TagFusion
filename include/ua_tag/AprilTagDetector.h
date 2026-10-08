#ifndef UA_TAG_APRILTAG_DETECTOR_H
#define UA_TAG_APRILTAG_DETECTOR_H

#include "ua_tag/TagObservation.h"

#include <opencv2/core/core.hpp>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace ORB_SLAM3
{

class Settings;
class GeometricCamera;

namespace ua_tag
{

// 相机模型：角点去畸变 + IPPE PnP
// 检测在原图上进行；corners_raw 为原图像素；
// corners_undistorted 为与 GeometricCamera::unproject 一致的理想针孔像素（供 IPPE 初值）
struct CameraModel
{
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    cv::Mat dist_coeffs;  // PinHole: k1,k2,p1,p2[,k3]；Fisheye(KB8): k1..k4
    bool is_fisheye = false;

    cv::Matx33d K() const
    {
        return cv::Matx33d(fx, 0.0, cx,
                           0.0, fy, cy,
                           0.0, 0.0, 1.0);
    }

    // 复用 ORB-SLAM3 已加载的 Settings（含可能的 newWidth/newHeight 缩放）
    static CameraModel FromSettings(Settings *settings);
};

// 可选时序先验：某 tag 的上一帧 T_ct，用于二义性消解
struct PosePrediction
{
    int tag_id = -1;
    cv::Matx33d R = cv::Matx33d::eye();
    cv::Vec3d t = cv::Vec3d(0, 0, 0);
    bool valid = false;
};

struct AprilTagDetectorConfig
{
    // ethz_apriltag2 (Thirdparty/apriltag)
    std::string family = "tag36h11";
    int hamming = 2;              // 接受的最大 ethz hammingDistance
    // ethz TagDetector blackBorder；可配置多个，DetectCorners 会对每个 border 各扫一遍再按 id 合并
    // yaml: Tag.black_borders: [1, 2]  或兼容旧项 Tag.black_border: 1
    std::vector<int> black_borders = {1};
    double tag_size = 0.16;       // 默认 Tag 边长（米）；未在 size_by_id 中列出的 id 使用此值
    // 按 tag_id 覆盖边长，例如 yaml:
    //   Tag.size_by_id: [ {id: 0, size: 0.06}, {id: 10, size: 0.144} ]
    std::map<int, double> tag_size_by_id;

    // 屏蔽指定 id：DetectCorners 丢弃这些检测结果
    // yaml: Tag.ignore_ids: [1, 2, 99]
    std::set<int> ignore_ids;

    // 送入 AprilTag 前的 gamma：out = 255*(in/255)^gamma；<1 提亮暗部；1=关闭
    double gamma = 1.0;

    // tag_id 在 tag_size_by_id 中则用覆盖值，否则用 tag_size
    double GetTagSize(int tag_id) const;
    bool IsIgnored(int tag_id) const;

    // 从 ORB-SLAM3 settings yaml 读取 Tag.*（缺省保留默认值）
    static AprilTagDetectorConfig FromYaml(const std::string &settingsFile);
};

// AprilTag 角点顺序与 OpenCV IPPE_SQUARE 物体点一致：
// (-s/2,s/2,0), (s/2,s/2,0), (s/2,-s/2,0), (-s/2,-s/2,0)
inline void BuildSquareObjectPoints(double tag_size, std::vector<cv::Point3f> &object_pts)
{
    const float h = static_cast<float>(tag_size) * 0.5f;
    object_pts.clear();
    object_pts.reserve(4);
    object_pts.emplace_back(-h,  h, 0.f);
    object_pts.emplace_back( h,  h, 0.f);
    object_pts.emplace_back( h, -h, 0.f);
    object_pts.emplace_back(-h, -h, 0.f);
}

// 交叉重投影消歧（约定 P_left = T_lr * P_right）：
//   1) L→R：左目未消歧时，两候选经 T_rl 投到右目，写回 left
//   2) 若左目仍失败，则 R→L：右目两候选经 T_lr 投到左目，写回 right
// 任一侧成功即返回 true。
// log_stereo=false：不打印 [TagStereo] ok/fail（例如 Tag 已在地图中时）
bool DisambiguateWithStereo(tag::TagObservation &left_obs,
                            tag::TagObservation &right_obs,
                            const Sophus::SE3f &T_lr,
                            GeometricCamera *cam_left,
                            GeometricCamera *cam_right,
                            double tag_size,
                            double tau_rho = 3.0,
                            bool log_stereo = true);

class AprilTagDetector
{
public:
    explicit AprilTagDetector(const AprilTagDetectorConfig &config = AprilTagDetectorConfig());
    explicit AprilTagDetector(const std::string &settingsFile);
    ~AprilTagDetector();

    // 禁止拷贝：Impl 持有 apriltag 裸指针资源，拷贝会导致双重释放
    AprilTagDetector(const AprilTagDetector &) = delete;
    AprilTagDetector &operator=(const AprilTagDetector &) = delete;

    // 设置角点去畸变所用相机（含真实畸变；鱼眼设 is_fisheye=true）
    void SetCameraModel(const CameraModel &camera);

    // 注入 ORB GeometricCamera：鱼眼去畸变/重投影误差与 BA 共用 project/unproject
    void SetGeometricCamera(GeometricCamera *camera);

    // 在原图（可选 gamma）上检测；填充 corners_raw 与 corners_undistorted（不做 IPPE）
    // camera_id：写入观测所属相机；geometric_camera 非空时覆盖默认模型做去畸变（右目用）
    bool DetectCorners(const cv::Mat &image,
                       std::vector<tag::TagObservation> &observations,
                       tag::CameraId camera_id = tag::CameraId::LEFT_OR_MONO,
                       GeometricCamera *geometric_camera = nullptr);

    // 上一帧实际送入检测器的图像（原图灰度 + 可选 gamma），供可视化
    const cv::Mat &GetLastPreprocessedImage() const;

    // IPPE_SQUARE PnP（在 GeometricCamera 一致的针孔像素上）；prediction 可选
    bool EstimatePose(tag::TagObservation &observation,
                      const CameraModel &camera,
                      const PosePrediction *prediction = nullptr);

    const AprilTagDetectorConfig &GetConfig() const;

    // 按 tag_id 查询物理边长（米）；见 AprilTagDetectorConfig::GetTagSize
    double GetTagSize(int tag_id) const;

private:
    struct Impl;
    Impl *mpImpl;
};

}  // namespace ua_tag
}  // namespace ORB_SLAM3

#endif
