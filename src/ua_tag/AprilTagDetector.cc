#ifdef HAS_APRILTAG

#include "ua_tag/AprilTagDetector.h"

#include "Settings.h"
#include "CameraModels/GeometricCamera.h"

#include <apriltags/TagDetector.h>
#include <apriltags/TagFamily.h>
#include <apriltags/Tag16h5.h>
#include <apriltags/Tag25h7.h>
#include <apriltags/Tag25h9.h>
#include <apriltags/Tag36h11.h>
#include <apriltags/Tag36h9.h>

#include <Eigen/Core>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace ORB_SLAM3
{
namespace ua_tag
{

namespace
{

const AprilTags::TagCodes *GetTagCodes(const std::string &family)
{
    if(family == "tag36h11")
        return &AprilTags::tagCodes36h11;
    if(family == "tag36h9")
        return &AprilTags::tagCodes36h9;
    if(family == "tag25h9")
        return &AprilTags::tagCodes25h9;
    if(family == "tag25h7")
        return &AprilTags::tagCodes25h7;
    if(family == "tag16h5")
        return &AprilTags::tagCodes16h5;
    return nullptr;
}

// 将图像转换为灰度图像
cv::Mat ToGray(const cv::Mat &image)
{
    if(image.empty())
        throw std::invalid_argument("AprilTagDetector: empty image");

    cv::Mat gray;
    if(image.channels() == 1)
        gray = image;
    else if(image.channels() == 3)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else if(image.channels() == 4)
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    else
        throw std::invalid_argument("AprilTagDetector: unsupported channel count");

    if(!gray.isContinuous())
        gray = gray.clone();
    return gray;
}

Sophus::SE3f MakeTct(const cv::Matx33d &R, const cv::Vec3d &t)
{
    Eigen::Matrix3f Rf;
    for(int r = 0; r < 3; ++r)
        for(int c = 0; c < 3; ++c)
            Rf(r, c) = static_cast<float>(R(r, c));
    return Sophus::SE3f(Rf, Eigen::Vector3f(static_cast<float>(t[0]),
                                            static_cast<float>(t[1]),
                                            static_cast<float>(t[2])));
}

bool HasSignificantDistortion(const cv::Mat &dist)
{
    if(dist.empty())
        return false;
    cv::Mat d64;
    dist.convertTo(d64, CV_64F);
    for(int i = 0; i < d64.rows * d64.cols; ++i)
    {
        if(std::abs(d64.at<double>(i)) > 1e-12)
            return true;
    }
    return false;
}

// 用 GeometricCamera::unproject 把原图角点收到与 ORB 一致的理想针孔像素（K 同相机）
void UndistortCornersWithGeometricCamera(const tag::TagObservation &src,
                                         GeometricCamera *geom,
                                         std::array<cv::Point2f, 4> &out)
{
    const float fx = geom->getParameter(0);
    const float fy = geom->getParameter(1);
    const float cx = geom->getParameter(2);
    const float cy = geom->getParameter(3);
    for(int k = 0; k < 4; ++k)
    {
        const cv::Point3f ray = geom->unproject(src.corners_raw[k]);
        const float invz = (std::abs(ray.z) > 1e-12f) ? (1.f / ray.z) : 1.f;
        out[k] = cv::Point2f(fx * ray.x * invz + cx, fy * ray.y * invz + cy);
    }
}

// 角点去畸变：优先 GeometricCamera（与 ORB BA 同源）；否则针孔用 cv::undistortPoints
void UndistortCorners(const tag::TagObservation &src, const CameraModel &camera,
                      GeometricCamera *geom, std::array<cv::Point2f, 4> &out)
{
    if(geom)
    {
        UndistortCornersWithGeometricCamera(src, geom, out);
        return;
    }

    if(!HasSignificantDistortion(camera.dist_coeffs))
    {
        out = src.corners_raw;
        return;
    }

    // 无 GeometricCamera 时：针孔走 OpenCV；鱼眼不再用 cv::fisheye（与 KB 不一致）
    if(camera.is_fisheye)
    {
        out = src.corners_raw;
        return;
    }

    std::vector<cv::Point2f> pts(src.corners_raw.begin(), src.corners_raw.end());
    std::vector<cv::Point2f> undist;
    const cv::Mat K = cv::Mat(camera.K());
    cv::undistortPoints(pts, undist, K, camera.dist_coeffs, cv::noArray(), K);
    for(int k = 0; k < 4; ++k)
        out[k] = undist[k];
}

// 针孔零畸变重投影（仅作无 GeometricCamera 时的回退）
double ComputeReprojectionErrorPinhole(const std::vector<cv::Point3f> &object_pts,
                                       const std::vector<cv::Point2f> &image_pts,
                                       const cv::Mat &rvec,
                                       const cv::Mat &tvec,
                                       const cv::Mat &K)
{
    if(object_pts.empty() || object_pts.size() != image_pts.size())
        return std::numeric_limits<double>::infinity();

    std::vector<cv::Point2f> projected;
    cv::projectPoints(object_pts, rvec, tvec, K, cv::Mat(), projected);
    if(projected.size() != image_pts.size())
        return std::numeric_limits<double>::infinity();

    double sum_sq = 0.0;
    for(size_t i = 0; i < image_pts.size(); ++i)
    {
        const double dx = projected[i].x - image_pts[i].x;
        const double dy = projected[i].y - image_pts[i].y;
        sum_sq += dx * dx + dy * dy;
    }
    return std::sqrt(sum_sq / static_cast<double>(image_pts.size()));
}

// 与 ORB BA 一致：GeometricCamera::project 到原图像素，对比 corners_raw
double ComputeReprojectionErrorGeometric(const std::vector<cv::Point3f> &object_pts,
                                         const std::array<cv::Point2f, 4> &corners_raw,
                                         const Sophus::SE3f &T_ct,
                                         GeometricCamera *geom)
{
    if(!geom || object_pts.size() != 4)
        return std::numeric_limits<double>::infinity();

    double sum_sq = 0.0;
    for(int i = 0; i < 4; ++i)
    {
        const Eigen::Vector3f P_t(object_pts[i].x, object_pts[i].y, object_pts[i].z);
        const Eigen::Vector3f P_c = T_ct * P_t;
        if(P_c.z() <= 0.0f)
            return std::numeric_limits<double>::infinity();

        const cv::Point2f uv = geom->project(cv::Point3f(P_c.x(), P_c.y(), P_c.z()));
        const double dx = static_cast<double>(uv.x - corners_raw[i].x);
        const double dy = static_cast<double>(uv.y - corners_raw[i].y);
        sum_sq += dx * dx + dy * dy;
    }
    return std::sqrt(sum_sq / 4.0);
}

// 相对预测的旋转角偏差（弧度）：R_err = R_pred^T * R_cand
double RotationDeltaToPrediction(const tag::TagPoseCandidate &cand, const PosePrediction &pred)
{
    const Eigen::Matrix3f Rf = cand.T_ct.rotationMatrix();
    cv::Matx33d R_cand;
    for(int r = 0; r < 3; ++r)
        for(int c = 0; c < 3; ++c)
            R_cand(r, c) = static_cast<double>(Rf(r, c));

    const cv::Matx33d R_err = pred.R.t() * R_cand;
    const double trace = R_err(0, 0) + R_err(1, 1) + R_err(2, 2);
    const double cos_angle = std::max(-1.0, std::min(1.0, 0.5 * (trace - 1.0)));
    return std::acos(cos_angle);
}

// 相对预测的平移欧氏偏差（米）
double TranslationDeltaToPrediction(const tag::TagPoseCandidate &cand, const PosePrediction &pred)
{
    const Eigen::Vector3f tf = cand.T_ct.translation();
    const cv::Vec3d dt = cv::Vec3d(tf[0], tf[1], tf[2]) - pred.t;
    return std::sqrt(dt.dot(dt));
}

// 第 2/3 层歧义消解。
// 输入 candidates 必须已经通过：正深度检查（候选生成阶段）。
// 本函数内再做：数值有效性、绝对重投影 RMSE、误差比、时序门控/代价。
// IPPE_SQUARE 最多 2 个候选；num_candidates > 2 直接失败。
// 返回 selected_candidate；不可信 / 未消歧时返回 -1（正式返回值不做“临时最优”）。
// 若双解均有效，写入 ambiguity_ratio = e_worse / e_better。
int ResolvePoseAmbiguity(const std::array<tag::TagPoseCandidate, 2> &candidates,
                         int num_candidates,
                         const PosePrediction *prediction,
                         int tag_id,
                         float &ambiguity_ratio)
{
    ambiguity_ratio = -1.0f;

    if(num_candidates <= 0 || num_candidates > 2)
        return -1;

    constexpr double kMaxReprojectionRmse = 3.0;  // 需标定
    constexpr double kTauRhoStrong = 3.0;
    constexpr double kErrorEps = 1e-9; // 避免除零

    int i_better = 0;
    int i_worse = -1;
    if(num_candidates == 2)
    {
        i_worse = 1;
        if(candidates[1].reprojection_error < candidates[0].reprojection_error)
        {
            i_better = 1;
            i_worse = 0;
        }
    }

    const double e1 = static_cast<double>(candidates[i_better].reprojection_error);
    if(!std::isfinite(e1) || e1 > kMaxReprojectionRmse)
        return -1;

    // 只有一个有效候选
    if(num_candidates == 1)
        return i_better;

    const double e2 = static_cast<double>(candidates[i_worse].reprojection_error);
    if(!std::isfinite(e2))
        return -1;

    // ρ = e2/e1（加 eps 避免 e1=e2=0 被当成强支持）
    const double rho = (e2 + kErrorEps) / (e1 + kErrorEps);
    ambiguity_ratio = static_cast<float>(rho);
    const bool strong_reproj = (rho > kTauRhoStrong);

    const bool has_pred = prediction && prediction->valid &&
                          (prediction->tag_id < 0 || prediction->tag_id == tag_id);

    // -----------------------------------------------------
    // 强重投影证据：优先最小误差解；有预测时仅做宽松异常检查
    // -----------------------------------------------------
    if(strong_reproj)
    {
        if(has_pred)
        {
            constexpr double kLooseTauR = 1.0;   // rad，需标定
            constexpr double kLooseTauT = 0.30;  // m，需标定

            const double dtheta =
                RotationDeltaToPrediction(candidates[i_better], *prediction);
            const double dtrans =
                TranslationDeltaToPrediction(candidates[i_better], *prediction);

            // 图像与可靠预测严重冲突
            if(dtheta >= kLooseTauR || dtrans >= kLooseTauT)
                return -1;
        }

        return i_better;
    }

    // -----------------------------------------------------
    // 重投影无法消歧，又没有时序预测 → 不可信
    // -----------------------------------------------------
    if(!has_pred)
        return -1;

    // -----------------------------------------------------
    // 弱重投影证据：严格时序门控 → 门内综合代价（要求足够代价差）
    // -----------------------------------------------------
    constexpr double kTauR = 0.5;   // rad
    constexpr double kTauT = 0.15;  // m

    double dtheta[2];
    double dtrans[2];
    bool pass[2];
    int n_pass = 0;
    int sole_pass = -1;

    for(int i = 0; i < 2; ++i)
    {
        dtheta[i] = RotationDeltaToPrediction(candidates[i], *prediction);
        dtrans[i] = TranslationDeltaToPrediction(candidates[i], *prediction);
        pass[i] = (dtheta[i] < kTauR) && (dtrans[i] < kTauT);
        if(pass[i])
        {
            ++n_pass;
            sole_pass = i;
        }
    }

    if(n_pass == 0)
        return -1;

    if(n_pass == 1)
        return sole_pass;

    // 两个均通过时序门控，计算综合代价
    constexpr double kWe = 1.0;
    constexpr double kWR = 1.0;
    constexpr double kWt = 1.0;
    constexpr double kSigmaE = 1.0;   // px
    constexpr double kSigmaR = 0.2;   // rad
    constexpr double kSigmaT = 0.05;  // m
    constexpr double kMinJGap = 0.2;  // 需标定

    double J[2];
    for(int i = 0; i < 2; ++i)
    {
        J[i] = kWe * static_cast<double>(candidates[i].reprojection_error) / kSigmaE +
               kWR * dtheta[i] / kSigmaR +
               kWt * dtrans[i] / kSigmaT;
    }

    const int best = (J[0] <= J[1]) ? 0 : 1;
    const int second = 1 - best;

    // 综合代价过于接近，不声称已消歧
    if(J[second] - J[best] < kMinJGap)
        return -1;

    return best;
}

}  // namespace

namespace {

cv::Mat BuildGammaLut(double gamma)
{
    cv::Mat lut(1, 256, CV_8U);
    uchar *p = lut.ptr<uchar>(0);
    const double inv = 1.0 / 255.0;
    for(int i = 0; i < 256; ++i)
        p[i] = cv::saturate_cast<uchar>(std::pow(static_cast<double>(i) * inv, gamma) * 255.0);
    return lut;
}

cv::Mat ApplyGamma(const cv::Mat &im, const cv::Mat &lut)
{
    if(im.empty() || lut.empty())
        return im;
    cv::Mat out;
    cv::LUT(im, lut, out);
    return out;
}

}  // namespace

struct AprilTagDetector::Impl
{
    // ethz_apriltag2：每个 black_border 一个检测器，DetectCorners 合并结果
    std::vector<std::unique_ptr<AprilTags::TagDetector>> detectors;
    std::string family_name;
    double tag_size;
    AprilTagDetectorConfig config;

    // 角点去畸变用相机（检测仍在原图上）
    CameraModel camera_model;
    bool has_camera_model = false;
    // 与 ORB Tracking/BA 共用；鱼眼路径优先用其 project/unproject
    GeometricCamera *geometric_camera = nullptr;

    // 最近一次送入检测器的图像（原图灰度 + 可选 gamma）
    cv::Mat last_preprocessed;
    cv::Mat gamma_lut;

    Impl(const AprilTagDetectorConfig &cfg)
        : family_name(cfg.family), tag_size(cfg.tag_size), config(cfg)
    {
        const AprilTags::TagCodes *codes = GetTagCodes(cfg.family);
        if(codes == nullptr)
            throw std::invalid_argument("AprilTagDetector: unsupported tag family: " + cfg.family);

        std::vector<int> borders = cfg.black_borders;
        if(borders.empty())
            borders.push_back(1);

        // 去重、保证 >=1，保持配置顺序
        std::vector<int> unique_borders;
        unique_borders.reserve(borders.size());
        for(int b : borders)
        {
            const int bb = std::max(1, b);
            if(std::find(unique_borders.begin(), unique_borders.end(), bb) == unique_borders.end())
                unique_borders.push_back(bb);
        }
        config.black_borders = unique_borders;

        detectors.reserve(unique_borders.size());
        // 每个 black_border 一个检测器，DetectCorners 合并结果
        for(int bb : unique_borders)
            detectors.emplace_back(new AprilTags::TagDetector(*codes, static_cast<size_t>(bb)));

        if(cfg.gamma > 0.0 && std::abs(cfg.gamma - 1.0) > 1e-6)
            gamma_lut = BuildGammaLut(cfg.gamma);
    }

    ~Impl() = default;
};

// 复用ORB-SLAM3的Settings，获取相机模型
CameraModel CameraModel::FromSettings(Settings *settings)
{
    if(!settings)
        throw std::invalid_argument("CameraModel::FromSettings: settings is null");

    GeometricCamera *geom = settings->camera1();
    if(!geom)
        throw std::runtime_error("CameraModel::FromSettings: camera1 is null");

    CameraModel cam;
    cam.fx = static_cast<double>(geom->getParameter(0));
    cam.fy = static_cast<double>(geom->getParameter(1));
    cam.cx = static_cast<double>(geom->getParameter(2));
    cam.cy = static_cast<double>(geom->getParameter(3));

    if(settings->cameraType() == Settings::KannalaBrandt)
    {
        cam.is_fisheye = true;
        cam.dist_coeffs = (cv::Mat_<double>(4, 1) <<
            geom->getParameter(4), geom->getParameter(5),
            geom->getParameter(6), geom->getParameter(7));
        return cam;
    }

    cv::Mat dist = settings->camera1DistortionCoef();
    if(!dist.empty())
        dist.convertTo(cam.dist_coeffs, CV_64F);

    return cam;
}

// 从yaml文件中读取AprilTagDetector的配置
AprilTagDetectorConfig AprilTagDetectorConfig::FromYaml(const std::string &settingsFile)
{
    AprilTagDetectorConfig config;

    cv::FileStorage fs(settingsFile, cv::FileStorage::READ);
    if(!fs.isOpened())
        throw std::runtime_error("AprilTagDetectorConfig: failed to open settings file: " + settingsFile);

    cv::FileNode node;
    node = fs["Tag.family"];
    if(!node.empty() && node.isString())
        config.family = static_cast<std::string>(node);

    node = fs["Tag.hamming"];
    if(!node.empty())
        config.hamming = static_cast<int>(node);

    // 优先 Tag.black_borders: [1, 2] 或标量 2；否则兼容 Tag.black_border: 1
    node = fs["Tag.black_borders"];
    if(!node.empty() && node.isSeq())
    {
        config.black_borders.clear();
        for(cv::FileNodeIterator it = node.begin(); it != node.end(); ++it)
        {
            const int b = static_cast<int>(*it);
            if(b >= 1)
                config.black_borders.push_back(b);
        }
        if(config.black_borders.empty())
            config.black_borders.push_back(1);
    }
    else if(!node.empty())
    {
        config.black_borders = {std::max(1, static_cast<int>(node))};
    }
    else
    {
        node = fs["Tag.black_border"];
        if(!node.empty())
            config.black_borders = {std::max(1, static_cast<int>(node))};
    }

    node = fs["Tag.size"];
    if(!node.empty())
        config.tag_size = static_cast<double>(node);

    // 按 id 覆盖边长：Tag.size_by_id: [ {id: 0, size: 0.06}, {id: 10, size: 0.144} ]
    node = fs["Tag.size_by_id"];
    if(!node.empty() && node.isSeq())
    {
        for(cv::FileNodeIterator it = node.begin(); it != node.end(); ++it)
        {
            const cv::FileNode item = *it;
            const cv::FileNode id_node = item["id"];
            const cv::FileNode size_node = item["size"];
            if(id_node.empty() || size_node.empty())
                continue;
            const int id = static_cast<int>(id_node);
            const double sz = static_cast<double>(size_node);
            if(id < 0 || sz <= 0.0)
                continue;
            config.tag_size_by_id[id] = sz;
        }
    }

    // 屏蔽指定 id：Tag.ignore_ids: [1, 2, 99]
    node = fs["Tag.ignore_ids"];
    if(!node.empty() && node.isSeq())
    {
        for(cv::FileNodeIterator it = node.begin(); it != node.end(); ++it)
        {
            const int id = static_cast<int>(*it);
            if(id >= 0)
                config.ignore_ids.insert(id);
        }
    }

    node = fs["Tag.gamma"];
    if(!node.empty())
        config.gamma = static_cast<double>(node);

    return config;
}

double AprilTagDetectorConfig::GetTagSize(int tag_id) const
{
    const auto it = tag_size_by_id.find(tag_id);
    if(it != tag_size_by_id.end() && it->second > 0.0)
        return it->second;
    return tag_size;
}

bool AprilTagDetectorConfig::IsIgnored(int tag_id) const
{
    return ignore_ids.find(tag_id) != ignore_ids.end();
}

const AprilTagDetectorConfig &AprilTagDetector::GetConfig() const
{
    return mpImpl->config;
}

double AprilTagDetector::GetTagSize(int tag_id) const
{
    return mpImpl->config.GetTagSize(tag_id);
}

void AprilTagDetector::SetCameraModel(const CameraModel &camera)
{
    mpImpl->camera_model = camera;
    mpImpl->has_camera_model = (camera.fx > 0.0 && camera.fy > 0.0);
}

void AprilTagDetector::SetGeometricCamera(GeometricCamera *camera)
{
    mpImpl->geometric_camera = camera;
}

// 返回最近一次 DetectCorners 的预处理图（引用内部缓冲，下一帧会被覆盖）
const cv::Mat &AprilTagDetector::GetLastPreprocessedImage() const
{
    return mpImpl->last_preprocessed;
}

// 默认初始化AprilTagDetector
AprilTagDetector::AprilTagDetector(const AprilTagDetectorConfig &config)
    : mpImpl(new Impl(config))
{
}

// 从yaml文件中初始化AprilTagDetector
AprilTagDetector::AprilTagDetector(const std::string &settingsFile)
    : mpImpl(new Impl(AprilTagDetectorConfig::FromYaml(settingsFile)))
{
}

// 销毁AprilTagDetector
AprilTagDetector::~AprilTagDetector()
{
    delete mpImpl;
}

// 检测 AprilTags 角点：原图（可选 gamma）上检测，再对角点做去畸变
// 后端：Thirdparty/apriltag ethz_apriltag2；对每个 black_border 各扫一遍，按 id 合并
bool AprilTagDetector::DetectCorners(const cv::Mat &image,
                                     std::vector<tag::TagObservation> &observations,
                                     tag::CameraId camera_id,
                                     GeometricCamera *geometric_camera)
{
    observations.clear();
    mpImpl->last_preprocessed.release();

    cv::Mat gray = ToGray(image);
    cv::Mat processed = ApplyGamma(gray, mpImpl->gamma_lut);
    if(!processed.isContinuous())
        processed = processed.clone();

    mpImpl->last_preprocessed = processed;

    if(mpImpl->detectors.empty())
        return false;

    // 右目必须显式传入相机模型，禁止回退默认/左目
    GeometricCamera *geom = geometric_camera;
    if(!geom)
    {
        if(camera_id == tag::CameraId::RIGHT)
        {
            std::cerr << "ERROR: DetectCorners(RIGHT) requires GeometricCamera "
                         "(no left/default fallback)"
                      << std::endl;
            return false;
        }
        geom = mpImpl->geometric_camera;
    }

    // tag_id -> 最优观测（更低 hamming，其次更大 observed_perimeter）
    std::unordered_map<int, tag::TagObservation> best_by_id;

    for(const auto &det_ptr : mpImpl->detectors)
    {
        if(!det_ptr)
            continue;

        const std::vector<AprilTags::TagDetection> dets = det_ptr->extractTags(processed);
        for(const AprilTags::TagDetection &det : dets)
        {
            if(!det.good)
                continue;
            if(det.hammingDistance > mpImpl->config.hamming)
                continue;
            if(mpImpl->config.IsIgnored(det.id))
                continue;

            tag::TagObservation obs;
            obs.tag_id = det.id;
            obs.camera_id = camera_id;
            obs.hamming = det.hammingDistance;
            obs.observed_perimeter = det.observedPerimeter;
            obs.decision_margin = obs.observed_perimeter;
            for(int k = 0; k < 4; ++k)
                obs.corners_raw[k] = cv::Point2f(det.p[k].first, det.p[k].second);
            obs.observed_area = tag::TagObservation::ComputeQuadArea(obs.corners_raw);

            if(mpImpl->has_camera_model || geom)
                UndistortCorners(obs, mpImpl->camera_model, geom, obs.corners_undistorted);
            else
                obs.corners_undistorted = obs.corners_raw;

            auto it = best_by_id.find(obs.tag_id);
            if(it == best_by_id.end())
            {
                best_by_id.emplace(obs.tag_id, std::move(obs));
                continue;
            }

            const tag::TagObservation &prev = it->second;
            const bool better =
                (obs.hamming < prev.hamming) ||
                (obs.hamming == prev.hamming &&
                 obs.observed_perimeter > prev.observed_perimeter);
            if(better)
                it->second = std::move(obs);
        }
    }

    observations.reserve(best_by_id.size());
    for(auto &kv : best_by_id)
        observations.push_back(std::move(kv.second));

    return true;
}

// 估计AprilTags的位姿
bool AprilTagDetector::EstimatePose(tag::TagObservation &observation,
                                    const CameraModel &camera,
                                    const PosePrediction *prediction)
{
    observation.pose_estimate = std::nullopt;

    const double tag_size = mpImpl->config.GetTagSize(observation.tag_id);
    if(tag_size <= 0.0 || camera.fx <= 0.0 || camera.fy <= 0.0)
        return false;

    // 构建AprilTag的物体点，Tag自身坐标系下的4个角点3D坐标
    std::vector<cv::Point3f> object_pts;
    BuildSquareObjectPoints(tag_size, object_pts);

    // IPPE 初值：GeometricCamera 一致的针孔像素 + 零畸变 K
    // 候选误差：优先用 GeometricCamera::project vs corners_raw（与 ORB BA 同域）
    const cv::Mat K = cv::Mat(camera.K());
    const cv::Mat dist_empty;
    std::vector<cv::Point2f> image_pts(observation.corners_undistorted.begin(),
                                       observation.corners_undistorted.end());

    std::vector<cv::Mat> rvecs, tvecs;
    int nsol = cv::solvePnPGeneric(object_pts, image_pts, K, dist_empty,
                                   rvecs, tvecs,
                                   false, cv::SOLVEPNP_IPPE_SQUARE);
    if(nsol <= 0 || rvecs.empty() || tvecs.empty())
        return false;

    GeometricCamera *geom = mpImpl->geometric_camera;
    tag::TagPoseEstimate estimate;

    // 仅正深度解进入候选（只保留有效 IPPE 解）
    int num_candidates = 0;
    for(int i = 0; i < nsol && num_candidates < 2; ++i)
    {
        const cv::Vec3d t(tvecs[i].at<double>(0),
                          tvecs[i].at<double>(1),
                          tvecs[i].at<double>(2));
        if(t[2] <= 0.0)
            continue;

        cv::Matx33d R;
        cv::Rodrigues(rvecs[i], R);

        tag::TagPoseCandidate &cand = estimate.candidates[num_candidates++];
        cand.T_ct = MakeTct(R, t);
        if(geom)
        {
            cand.reprojection_error = static_cast<float>(ComputeReprojectionErrorGeometric(
                object_pts, observation.corners_raw, cand.T_ct, geom));
        }
        else
        {
            cand.reprojection_error = static_cast<float>(ComputeReprojectionErrorPinhole(
                object_pts, image_pts, rvecs[i], tvecs[i], K));
        }
        cand.valid = std::isfinite(cand.reprojection_error);
    }

    if(num_candidates <= 0)
        return false;

    // IPPE_SQUARE 期望 2 个有效解；不足则不写入 pose_estimate
    if(num_candidates < 2)
        return false;

    float ambiguity_ratio = -1.0f;
    const int best = ResolvePoseAmbiguity(estimate.candidates,
                                          num_candidates,
                                          prediction,
                                          observation.tag_id,
                                          ambiguity_ratio);
    if(best < 0)
    {
        // 已有候选但未消歧：仍写入 estimate，selected_candidate 保持 -1
        estimate.ambiguity_ratio = ambiguity_ratio;
        observation.pose_estimate = estimate;
        return false;
    }

    estimate.selected_candidate = best;
    estimate.ambiguity_ratio = ambiguity_ratio;
    observation.pose_estimate = estimate;
    return true;
}

namespace {

// 将 source 的两候选经 T_target_source 变换到目标相机系，对 target_corners 算 RMSE。
// T_target_source：P_target = T_target_source * P_source
bool CrossReprojDisambiguate(tag::TagObservation &source_obs,
                             const std::array<cv::Point2f, 4> &target_corners,
                             const Sophus::SE3f &T_target_source,
                             GeometricCamera *cam_target,
                             double tag_size,
                             double tau_rho,
                             const char *direction_label,
                             bool log_fail)
{
    if(!cam_target || tag_size <= 0.0)
        return false;
    if(!source_obs.IsDetectValid())
        return false;
    if(!source_obs.pose_estimate.has_value())
        return false;

    tag::TagPoseEstimate &est = *source_obs.pose_estimate;
    // 已消歧：直接返回，不重复打 fail 日志
    if(est.selected_candidate >= 0)
        return false;
    if(!est.candidates[0].valid || !est.candidates[1].valid)
        return false;

    std::vector<cv::Point3f> object_pts;
    BuildSquareObjectPoints(tag_size, object_pts);

    double e[2] = {std::numeric_limits<double>::infinity(),
                   std::numeric_limits<double>::infinity()};

    for(int i = 0; i < 2; ++i)
    {
        const Sophus::SE3f &T_cs_t = est.candidates[i].T_ct;
        double sum_sq = 0.0;
        bool ok = true;
        for(int k = 0; k < 4; ++k)
        {
            const Eigen::Vector3f P_t(object_pts[k].x, object_pts[k].y,
                                     object_pts[k].z);
            // 源相机系：P_cs = T_cs_t * P_t
            const Eigen::Vector3f P_cs = T_cs_t * P_t;
            if(P_cs.z() <= 0.0f)
            {
                ok = false;
                break;
            }
            // 目标相机系：P_ct = T_target_source * P_cs
            const Eigen::Vector3f P_ct = T_target_source * P_cs;
            if(P_ct.z() <= 0.0f)
            {
                ok = false;
                break;
            }
            const cv::Point2f uv =
                cam_target->project(cv::Point3f(P_ct.x(), P_ct.y(), P_ct.z()));
            const double dx =
                static_cast<double>(uv.x - target_corners[k].x);
            const double dy =
                static_cast<double>(uv.y - target_corners[k].y);
            sum_sq += dx * dx + dy * dy;
        }
        if(ok)
            e[i] = std::sqrt(sum_sq / 4.0);
    }

    if(!std::isfinite(e[0]) && !std::isfinite(e[1]))
        return false;

    int i_better = 0;
    int i_worse = 1;
    if(e[1] < e[0])
    {
        i_better = 1;
        i_worse = 0;
    }

    if(!std::isfinite(e[i_better]))
        return false;

    constexpr double kErrorEps = 1e-9;
    const double e_better = e[i_better];
    const double e_worse =
        std::isfinite(e[i_worse]) ? e[i_worse]
                                  : std::numeric_limits<double>::infinity();
    const double rho = (e_worse + kErrorEps) / (e_better + kErrorEps);

    if(!(rho > tau_rho) || !std::isfinite(rho))
    {
        // 仅对尚未消歧的观测打印；同 ID 另一目已消歧时由调用方关闭 log_fail
        if(log_fail && est.selected_candidate < 0)
        {
            std::cout << "[TagStereo] fail tag_id=" << source_obs.tag_id
                      << " dir=" << direction_label
                      << " reason=stereo_ratio_weak"
                      << " e0=" << e[0] << " e1=" << e[1]
                      << " rho=" << rho << " tau=" << tau_rho
                      << " mono_ratio=" << est.ambiguity_ratio
                      << std::endl;
        }
        return false;
    }

    est.selected_candidate = i_better;
    est.ambiguity_ratio = static_cast<float>(rho);

    if(log_fail)  // CrossReproj 用 log_fail 兼控 ok；调用方已合入“是否值得打印”
    {
        std::cout << "[TagStereo] ok tag_id=" << source_obs.tag_id
                  << " dir=" << direction_label
                  << " selected=" << i_better
                  << " e_better=" << e_better
                  << " e_worse=" << e_worse
                  << " stereo_ratio=" << rho
                  << std::endl;
    }
    return true;
}

}  // namespace

bool DisambiguateWithStereo(tag::TagObservation &left_obs,
                            tag::TagObservation &right_obs,
                            const Sophus::SE3f &T_lr,
                            GeometricCamera *cam_left,
                            GeometricCamera *cam_right,
                            double tag_size,
                            double tau_rho,
                            bool log_stereo)
{
    if(!cam_left || !cam_right || tag_size <= 0.0)
        return false;
    if(!left_obs.IsDetectValid() || !right_obs.IsDetectValid())
        return false;
    if(left_obs.tag_id < 0 || left_obs.tag_id != right_obs.tag_id)
        return false;

    // 任一相机已消歧，或调用方关闭日志（如 Tag 已入图）：不刷 stereo fail/ok
    const bool any_resolved =
        left_obs.IsAmbiguityResolved() || right_obs.IsAmbiguityResolved();
    const bool log_fail = log_stereo && !any_resolved;

    // 1) L→R：消歧左目（P_right = T_rl * P_left）
    if(!left_obs.IsAmbiguityResolved() && left_obs.pose_estimate.has_value())
    {
        if(CrossReprojDisambiguate(left_obs, right_obs.corners_raw,
                                   T_lr.inverse(), cam_right, tag_size, tau_rho,
                                   "L2R", log_fail))
            return true;
    }

    // 2) 左目失败 → R→L：消歧右目（P_left = T_lr * P_right）
    if(!right_obs.IsAmbiguityResolved() && right_obs.pose_estimate.has_value())
    {
        if(CrossReprojDisambiguate(right_obs, left_obs.corners_raw, T_lr,
                                   cam_left, tag_size, tau_rho, "R2L",
                                   log_fail))
            return true;
    }

    return false;
}

}  // namespace ua_tag
}  // namespace ORB_SLAM3

#endif  // HAS_APRILTAG
