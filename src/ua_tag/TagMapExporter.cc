#include "ua_tag/TagMapExporter.h"

#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"
#include "ua_tag/MapTagData.h"
#include "ua_tag/TagFrameData.h"
#include "ua_tag/TagObservation.h"
#include "ua_tag/TagPoseConstraints.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sys/stat.h>
#include <sys/types.h>
#include <utility>
#include <vector>

namespace ORB_SLAM3 {
namespace tag {

namespace {

bool EnsureDir(const std::string &dirPath)
{
    if(dirPath.empty())
        return false;

    struct stat st;
    if(stat(dirPath.c_str(), &st) == 0)
        return S_ISDIR(st.st_mode);

    return mkdir(dirPath.c_str(), 0755) == 0;
}

MapTagData::FactorVizWeight ToVizWeight(const TagFactorWeightSummary &w)
{
    MapTagData::FactorVizWeight viz;
    viz.valid = w.valid;
    viz.n_obs = w.n_obs;
    viz.mean_area = w.mean_area;
    viz.mean_w_s = w.mean_w_s;
    viz.mean_w_theta = w.mean_w_theta;
    viz.mean_w_amb = w.mean_w_amb;
    viz.mean_w_obs = w.mean_w_obs;
    viz.mean_w_bar = w.mean_w_bar;
    viz.min_w_bar = w.min_w_bar;
    viz.max_w_bar = w.max_w_bar;
    viz.mean_alpha_w = w.mean_alpha_w;
    return viz;
}

TagFactorWeightSummary FromVizWeight(const MapTagData::FactorVizWeight &viz)
{
    TagFactorWeightSummary w;
    if(!viz.valid)
        return w;
    w.valid = true;
    w.n_obs = viz.n_obs;
    w.mean_area = viz.mean_area;
    w.mean_w_s = viz.mean_w_s;
    w.mean_w_theta = viz.mean_w_theta;
    w.mean_w_amb = viz.mean_w_amb;
    w.mean_w_obs = viz.mean_w_obs;
    w.mean_w_bar = viz.mean_w_bar;
    w.min_w_bar = viz.min_w_bar;
    w.max_w_bar = viz.max_w_bar;
    w.mean_alpha_w = viz.mean_alpha_w;
    return w;
}

void WeightRgb(float w_bar, bool valid, int &r, int &g, int &b)
{
    if(!valid)
    {
        r = 127;
        g = 127;
        b = 127;
        return;
    }
    const float w = std::min(1.0f, std::max(0.0f, w_bar));
    if(w < 0.5f)
    {
        const float t = w * 2.0f;
        r = 242;
        g = static_cast<int>(51.0f + t * (191.0f - 51.0f));
        b = 26;
        return;
    }
    const float t = (w - 0.5f) * 2.0f;
    r = static_cast<int>(242.0f - t * (242.0f - 38.0f));
    g = static_cast<int>(191.0f + t * (139.0f - 191.0f));
    b = static_cast<int>(26.0f + t * (38.0f - 26.0f));
}

}  // namespace

TagMapExporter::TagMapExporter(const std::string &output_dir,
                               TagFactorWeightParams factor_weight)
    : mOutputDir(output_dir), mFactorWeight(std::move(factor_weight))
{
    if(mOutputDir.empty())
        return;

    if(!EnsureDir(mOutputDir))
    {
        std::cerr << "[TagMapExporter] failed to create dir: " << mOutputDir
                  << std::endl;
        return;
    }

    const bool traj_ok = OpenTrajectoryFile();
    const bool det_ok = OpenDetectionLogFile();
    const bool first_ok = OpenFirstRegistrationLogFile();
    mbEnabled = traj_ok || det_ok || first_ok;
    if(mbEnabled)
        std::cout << "[TagMapExporter] export dir=" << mOutputDir << std::endl;
}

TagMapExporter::~TagMapExporter()
{
    CloseTrajectory();
}

bool TagMapExporter::OpenTrajectoryFile()
{
    // TUM 轨迹：timestamp tx ty tz qx qy qz qw（Twc）
    const std::string path = mOutputDir + "/tag_camera_trajectory.txt";
    mTrajFile.open(path.c_str(), std::ios::out | std::ios::trunc);
    if(!mTrajFile.is_open())
    {
        std::cerr << "[TagMapExporter] failed to open: " << path << std::endl;
        return false;
    }
    mTrajFile << std::fixed;
    return true;
}

bool TagMapExporter::OpenDetectionLogFile()
{
    const std::string path = mOutputDir + "/tag_detections.csv";
    mDetFile.open(path.c_str(), std::ios::out | std::ios::trunc);
    if(!mDetFile.is_open())
    {
        std::cerr << "[TagMapExporter] failed to open: " << path << std::endl;
        return false;
    }
    mDetFile << std::setprecision(9);
    mDetFile << "frame_id,timestamp,camera,tag_id,"
             << "observed_area,observed_perimeter,hamming,"
             << "cx,cy,"
             << "x0,y0,x1,y1,x2,y2,x3,y3\n";
    return true;
}

bool TagMapExporter::OpenFirstRegistrationLogFile()
{
    const std::string path = mOutputDir + "/tag_first_registration.csv";
    mFirstRegFile.open(path.c_str(), std::ios::out | std::ios::trunc);
    if(!mFirstRegFile.is_open())
    {
        std::cerr << "[TagMapExporter] failed to open: " << path << std::endl;
        return false;
    }
    mFirstRegFile << "kf_id,frame_id,timestamp,tag_id,camera,"
                  << "map_has_pose,pose_source,observed_area,distance_m,"
                  << "axis_plane_angle_deg,axis_normal_angle_deg,"
                  << "nx,ny,nz,tx,ty,tz\n";
    return true;
}

void TagMapExporter::AppendCameraPose(unsigned long frame_id,
                                      double timestamp,
                                      const Sophus::SE3f &Tcw)
{
    (void)frame_id;
    if(!mbEnabled || !mTrajFile.is_open())
        return;

    const Sophus::SE3f Twc = Tcw.inverse();
    const Eigen::Quaternionf q = Twc.unit_quaternion();
    const Eigen::Vector3f t = Twc.translation();

    mTrajFile << std::setprecision(6) << timestamp << " "
              << std::setprecision(9) << t.x() << " " << t.y() << " " << t.z() << " "
              << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
}

void TagMapExporter::AppendTagDetections(unsigned long frame_id,
                                         double timestamp,
                                         const TagFrameData &frame_data)
{
    if(!mbEnabled || !mDetFile.is_open() || frame_data.Empty())
        return;

    auto write_obs = [&](const TagObservation &obs, const char *cam) {
        if(!obs.IsDetectValid())
            return;

        float cx = 0.0f;
        float cy = 0.0f;
        for(const cv::Point2f &p : obs.corners_raw)
        {
            cx += p.x;
            cy += p.y;
        }
        cx *= 0.25f;
        cy *= 0.25f;

        mDetFile << frame_id << ","
                 << std::setprecision(6) << timestamp << ","
                 << cam << ","
                 << obs.tag_id << ","
                 << std::setprecision(3) << obs.observed_area << ","
                 << obs.observed_perimeter << ","
                 << obs.hamming << ","
                 << cx << "," << cy;
        for(const cv::Point2f &p : obs.corners_raw)
            mDetFile << "," << p.x << "," << p.y;
        mDetFile << "\n";
    };

    for(const TagObservation &obs : frame_data.left)
        write_obs(obs, "left");
    for(const TagObservation &obs : frame_data.right)
        write_obs(obs, "right");
}

void TagMapExporter::AppendMapTagFirstRegistration(KeyFrame *pKF,
                                                   int tag_id,
                                                   int left_idx,
                                                   int right_idx,
                                                   MapTagData *pTag)
{
    if(!mbEnabled || !mFirstRegFile.is_open() || !pKF || tag_id < 0)
        return;

    const Sophus::SE3f Tcw = pKF->GetPose();
    const Sophus::SE3f Tlr = pKF->GetRelativePoseTlr();
    constexpr float kRad2Deg = 180.0f / 3.14159265358979323846f;
    const bool map_has_pose = pTag && pTag->HasPose();

    auto resolve_Tct = [&](const TagObservation &obs, bool is_right,
                           Sophus::SE3f &T_ct, const char *&pose_source) -> bool {
        if(obs.pose_estimate.has_value() && obs.pose_estimate->Selected())
        {
            T_ct = obs.pose_estimate->Selected()->T_ct;
            pose_source = "selected";
            return true;
        }
        if(obs.pose_estimate.has_value())
        {
            int best = -1;
            float best_err = std::numeric_limits<float>::infinity();
            for(int i = 0; i < 2; ++i)
            {
                const TagPoseCandidate &c = obs.pose_estimate->candidates[i];
                if(!c.valid)
                    continue;
                if(c.reprojection_error >= 0.0f &&
                   c.reprojection_error < best_err)
                {
                    best_err = c.reprojection_error;
                    best = i;
                }
            }
            if(best >= 0)
            {
                T_ct = obs.pose_estimate->candidates[best].T_ct;
                pose_source = "best_ippe";
                return true;
            }
        }
        if(map_has_pose)
        {
            if(is_right)
                T_ct = Tlr.inverse() * Tcw * pTag->GetPose();
            else
                T_ct = Tcw * pTag->GetPose();
            pose_source = "map_twt";
            return true;
        }
        pose_source = "none";
        return false;
    };

    auto write_one = [&](const TagObservation &obs, const char *cam,
                         bool is_right) {
        if(!obs.IsDetectValid())
            return;

        Sophus::SE3f T_ct;
        const char *pose_source = "none";
        const bool have_T = resolve_Tct(obs, is_right, T_ct, pose_source);

        mFirstRegFile << pKF->mnId << ","
                      << pKF->mnFrameId << ","
                      << std::setprecision(6) << pKF->mTimeStamp << ","
                      << tag_id << ","
                      << cam << ","
                      << (map_has_pose ? 1 : 0) << ","
                      << pose_source << ","
                      << std::setprecision(3) << obs.observed_area << ",";

        if(have_T)
        {
            const Eigen::Matrix3f R = T_ct.rotationMatrix();
            const Eigen::Vector3f t = T_ct.translation();
            Eigen::Vector3f n = R.col(2);
            const float n_norm = n.norm();
            if(n_norm < 1e-8f)
            {
                mFirstRegFile << ",,,,,,,,\n";
                return;
            }
            n /= n_norm;
            const float abs_cos = std::min(1.0f, std::abs(n.z()));
            const float axis_plane_deg = std::asin(abs_cos) * kRad2Deg;
            const float axis_normal_deg = std::acos(abs_cos) * kRad2Deg;
            const float distance_m = t.norm();

            mFirstRegFile << std::setprecision(6) << distance_m << ","
                          << axis_plane_deg << ","
                          << axis_normal_deg << ","
                          << n.x() << "," << n.y() << "," << n.z() << ","
                          << t.x() << "," << t.y() << "," << t.z() << "\n";

            std::cout << "[TagFirstReg] tag_id=" << tag_id
                      << " cam=" << cam
                      << " kf_id=" << pKF->mnId
                      << " area=" << obs.observed_area
                      << " dist_m=" << distance_m
                      << " axis_plane_deg=" << axis_plane_deg
                      << " axis_normal_deg=" << axis_normal_deg
                      << " pose_source=" << pose_source
                      << std::endl;
        }
        else
        {
            mFirstRegFile << ",,,,,,,,\n";
            std::cout << "[TagFirstReg] tag_id=" << tag_id
                      << " cam=" << cam
                      << " kf_id=" << pKF->mnId
                      << " area=" << obs.observed_area
                      << " pose_source=none"
                      << std::endl;
        }
    };

    if(left_idx >= 0 &&
       left_idx < static_cast<int>(pKF->mTagFrameData.left.size()))
        write_one(pKF->mTagFrameData.left[left_idx], "left", false);
    if(right_idx >= 0 &&
       right_idx < static_cast<int>(pKF->mTagFrameData.right.size()))
        write_one(pKF->mTagFrameData.right[right_idx], "right", true);

    mFirstRegFile.flush();
}

void TagMapExporter::LogKeyFrameMapTagFirstRegistrations(KeyFrame *pKF)
{
    if(!mbEnabled || !mFirstRegFile.is_open() || !pKF)
        return;

    const auto associations = pKF->GetMapTagAssociations();
    for(const auto &kv : associations)
    {
        const KeyFrame::MapTagAssociation &assoc = kv.second;
        AppendMapTagFirstRegistration(pKF, kv.first, assoc.leftObservationIndex,
                                      assoc.rightObservationIndex, assoc.pMapTag);
    }
}

bool TagMapExporter::SaveTagMapCorners(Map &map, bool verbose) const
{
    const bool sel_ok =
        WriteTagCornersCsv(map, "tag_map_corners.csv", false, verbose);
    const bool mir_ok =
        WriteTagCornersCsv(map, "tag_map_mirror_corners.csv", true, verbose);
    return sel_ok && mir_ok;
}

bool TagMapExporter::WriteTagCornersCsv(Map &map, const std::string &filename,
                                        bool mirror, bool verbose) const
{
    if(mOutputDir.empty())
        return false;

    if(!EnsureDir(mOutputDir))
        return false;

    const std::string path = mOutputDir + "/" + filename;
    std::ofstream ofs(path.c_str(), std::ios::out | std::ios::trunc);
    if(!ofs.is_open())
    {
        std::cerr << "[TagMapExporter] failed to open: " << path << std::endl;
        return false;
    }

    ofs << std::setprecision(17);
    ofs << "tag_id,is_fixed,has_corners,"
        << "c0_x,c0_y,c0_z,"
        << "c1_x,c1_y,c1_z,"
        << "c2_x,c2_y,c2_z,"
        << "c3_x,c3_y,c3_z,"
        << "n_obs,mean_area,mean_w_s,mean_w_theta,mean_w_amb,mean_w_obs,"
        << "mean_w_bar,min_w_bar,max_w_bar,mean_alpha_w\n";

    std::vector<Map::MapTagPtr> tags = map.GetAllMapTags();
    // 按 tag_id 排序，便于对比
    std::sort(tags.begin(), tags.end(),
              [](const Map::MapTagPtr &a, const Map::MapTagPtr &b) {
                  if(!a)
                      return true;
                  if(!b)
                      return false;
                  return a->Id() < b->Id();
              });

    auto write_weight_fields = [&](const TagFactorWeightSummary &w) {
        ofs << "," << w.n_obs;
        if(!w.valid)
        {
            ofs << ",,,,,,,,,";
            return;
        }
        ofs << "," << w.mean_area
            << "," << w.mean_w_s
            << "," << w.mean_w_theta
            << "," << w.mean_w_amb
            << "," << w.mean_w_obs
            << "," << w.mean_w_bar
            << "," << w.min_w_bar
            << "," << w.max_w_bar
            << "," << w.mean_alpha_w;
    };

    std::size_t n_with_corners = 0;
    std::vector<std::pair<int, TagFactorWeightSummary>> weight_rows;
    for(const auto &map_tag : tags)
    {
        if(!map_tag)
            continue;

        const bool has_corners = mirror ? map_tag->HasMirrorWorldCorners()
                                        : map_tag->HasWorldCorners();
        ofs << map_tag->Id() << ","
            << (map_tag->IsFixed() ? 1 : 0) << ","
            << (has_corners ? 1 : 0);

        if(has_corners)
        {
            const auto c = mirror ? map_tag->GetMirrorWorldCorners()
                                  : map_tag->GetWorldCorners();
            for(int i = 0; i < 4; ++i)
                ofs << "," << c[i].x() << "," << c[i].y() << "," << c[i].z();
            ++n_with_corners;
        }
        else
        {
            ofs << ",,,,,,,,,,,,";
        }

        TagFactorWeightSummary w =
            SummarizeMapTagFactorWeight(*map_tag, mFactorWeight);
        // KF 剔除会清掉 MapTag 观测，但位姿仍在。保留上次有效汇总，避免最终 CSV 变成 n/a。
        if(w.valid)
        {
            map_tag->SetFactorVizWeight(ToVizWeight(w));
        }
        else
        {
            const TagFactorWeightSummary cached =
                FromVizWeight(map_tag->GetFactorVizWeight());
            if(cached.valid)
                w = cached;
        }
        write_weight_fields(w);
        ofs << "\n";

        if(!mirror && w.valid)
            weight_rows.emplace_back(map_tag->Id(), w);
    }

    ofs.close();
    if(verbose)
    {
        std::cout << "[TagMapExporter] saved "
                  << (mirror ? "tag map mirror corners" : "tag map corners")
                  << " -> " << path
                  << " num_tags=" << tags.size()
                  << " with_corners=" << n_with_corners << std::endl;
        if(!mirror && !weight_rows.empty())
        {
            std::sort(weight_rows.begin(), weight_rows.end(),
                      [](const auto &a, const auto &b) {
                          return a.second.mean_w_bar < b.second.mean_w_bar;
                      });
            std::cout << "[TagMapExporter] tag factor weights "
                      << "(same formula as [TagW], mean over detect-valid KF obs; "
                      << "sorted by mean_w_bar, low=suppressed; "
                      << "applied_in_ba="
                      << (mFactorWeight.enable ? 1 : 0) << "):\n";
            for(const auto &row : weight_rows)
            {
                const TagFactorWeightSummary &w = row.second;
                std::cout << "  tag=" << row.first
                          << " n_obs=" << w.n_obs
                          << " mean_wBar=" << w.mean_w_bar
                          << " min_wBar=" << w.min_w_bar
                          << " max_wBar=" << w.max_w_bar
                          << " mean_wS=" << w.mean_w_s
                          << " mean_wTh=" << w.mean_w_theta
                          << " mean_wAmb=" << w.mean_w_amb
                          << " mean_alpha_w=" << w.mean_alpha_w
                          << std::endl;
            }
        }
    }
    return true;
}

bool TagMapExporter::SaveTagInitKeyFrames(Map &map) const
{
    return WriteKeyFramesTum(map, "tag_init_keyframes_tum.txt", true, true);
}

bool TagMapExporter::WriteKeyFramesTum(Map &map, const std::string &filename,
                                       bool tag_associated_only, bool verbose) const
{
    if(mOutputDir.empty())
        return false;

    if(!EnsureDir(mOutputDir))
        return false;

    const std::string path = mOutputDir + "/" + filename;
    std::ofstream ofs(path.c_str(), std::ios::out | std::ios::trunc);
    if(!ofs.is_open())
    {
        std::cerr << "[TagMapExporter] failed to open: " << path << std::endl;
        return false;
    }

    ofs << std::fixed;
    std::vector<KeyFrame *> kfs = map.GetAllKeyFrames();
    std::sort(kfs.begin(), kfs.end(),
              [](KeyFrame *a, KeyFrame *b) {
                  if(!a)
                      return true;
                  if(!b)
                      return false;
                  return a->mTimeStamp < b->mTimeStamp;
              });

    std::size_t n = 0;
    for(KeyFrame *pKF : kfs)
    {
        if(!pKF || pKF->isBad())
            continue;
        if(tag_associated_only && pKF->GetMapTagMatches().empty())
            continue;

        const Sophus::SE3f Twc = pKF->GetPoseInverse();
        const Eigen::Quaternionf q = Twc.unit_quaternion();
        const Eigen::Vector3f t = Twc.translation();
        ofs << std::setprecision(6) << pKF->mTimeStamp << " "
            << std::setprecision(9) << t.x() << " " << t.y() << " " << t.z() << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        ++n;
    }

    ofs.close();
    if(verbose)
        std::cout << "[TagMapExporter] saved keyframes -> " << path
                  << " num_kf=" << n << std::endl;
    return true;
}

bool TagMapExporter::SaveOrbInitMap(
    const std::vector<Eigen::Vector3f> &points,
    const std::vector<std::pair<double, Sophus::SE3f>> &orb_kf_tcw) const
{
    if(mOutputDir.empty())
        return false;

    if(!EnsureDir(mOutputDir))
        return false;

    const std::string pts_path = mOutputDir + "/orb_init_map_points.csv";
    {
        std::ofstream ofs(pts_path.c_str(), std::ios::out | std::ios::trunc);
        if(!ofs.is_open())
        {
            std::cerr << "[TagMapExporter] failed to open: " << pts_path << std::endl;
            return false;
        }
        ofs << std::setprecision(17);
        ofs << "x,y,z\n";
        for(const Eigen::Vector3f &p : points)
            ofs << p.x() << "," << p.y() << "," << p.z() << "\n";
        ofs.close();
        std::cout << "[TagMapExporter] saved ORB init map points -> " << pts_path
                  << " num_pts=" << points.size() << std::endl;
    }

    const std::string kf_path = mOutputDir + "/orb_init_keyframes_tum.txt";
    {
        std::ofstream ofs(kf_path.c_str(), std::ios::out | std::ios::trunc);
        if(!ofs.is_open())
        {
            std::cerr << "[TagMapExporter] failed to open: " << kf_path << std::endl;
            return false;
        }
        ofs << std::fixed;
        for(const auto &item : orb_kf_tcw)
        {
            const Sophus::SE3f Twc = item.second.inverse();
            const Eigen::Quaternionf q = Twc.unit_quaternion();
            const Eigen::Vector3f t = Twc.translation();
            ofs << std::setprecision(6) << item.first << " "
                << std::setprecision(9) << t.x() << " " << t.y() << " " << t.z()
                << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
                << "\n";
        }
        ofs.close();
        std::cout << "[TagMapExporter] saved ORB init keyframes -> " << kf_path
                  << " num_kf=" << orb_kf_tcw.size() << std::endl;
    }

    return true;
}

bool TagMapExporter::WriteOrbPointsCsv(const std::vector<Eigen::Vector3f> &points,
                                       const std::string &filename, bool verbose) const
{
    if(mOutputDir.empty())
        return false;
    if(!EnsureDir(mOutputDir))
        return false;

    const std::string path = mOutputDir + "/" + filename;
    std::ofstream ofs(path.c_str(), std::ios::out | std::ios::trunc);
    if(!ofs.is_open())
    {
        std::cerr << "[TagMapExporter] failed to open: " << path << std::endl;
        return false;
    }
    ofs << std::setprecision(17);
    ofs << "x,y,z\n";
    for(const Eigen::Vector3f &p : points)
        ofs << p.x() << "," << p.y() << "," << p.z() << "\n";
    ofs.close();
    if(verbose)
        std::cout << "[TagMapExporter] saved ORB map points -> " << path
                  << " num_pts=" << points.size() << std::endl;
    return true;
}

bool TagMapExporter::WriteFinalMapPly(Map &map,
                                      const std::vector<Eigen::Vector3f> &orb_points,
                                      bool verbose) const
{
    if(mOutputDir.empty())
        return false;
    if(!EnsureDir(mOutputDir))
        return false;

    const std::string path = mOutputDir + "/final_map.ply";
    std::vector<std::array<float, 6>> verts;
    std::vector<std::array<int, 4>> faces;
    verts.reserve(orb_points.size() + 64);

    auto append_quad = [&](const std::array<Eigen::Vector3f, 4> &c,
                           float r, float g, float b) {
        const int base = static_cast<int>(verts.size());
        for(int i = 0; i < 4; ++i)
            verts.push_back({c[i].x(), c[i].y(), c[i].z(), r, g, b});
        // Tag 局部：0=(-h,h), 1=(h,h), 2=(h,-h), 3=(-h,-h)
        faces.push_back({base, base + 1, base + 2, base + 3});
    };

    for(const Eigen::Vector3f &p : orb_points)
        verts.push_back({p.x(), p.y(), p.z(), 76.0f, 114.0f, 176.0f});

    const auto tags = map.GetAllMapTags();
    for(const auto &map_tag : tags)
    {
        if(!map_tag)
            continue;
        int r = 242, g = 140, b = 26;
        const auto viz = map_tag->GetFactorVizWeight();
        WeightRgb(viz.mean_w_bar, viz.valid, r, g, b);
        if(map_tag->HasWorldCorners())
            append_quad(map_tag->GetWorldCorners(),
                        static_cast<float>(r), static_cast<float>(g),
                        static_cast<float>(b));
    }

    std::vector<KeyFrame *> kfs = map.GetAllKeyFrames();
    std::sort(kfs.begin(), kfs.end(),
              [](KeyFrame *a, KeyFrame *b) {
                  if(!a)
                      return true;
                  if(!b)
                      return false;
                  return a->mTimeStamp < b->mTimeStamp;
              });

    std::vector<std::array<int, 5>> edges;  // v0, v1, r, g, b
    int prev_idx = -1;
    for(KeyFrame *pKF : kfs)
    {
        if(!pKF || pKF->isBad())
            continue;
        const Eigen::Vector3f t = pKF->GetPoseInverse().translation();
        const bool has_tag = !pKF->GetMapTagMatches().empty();
        const int r = has_tag ? 44 : 214;
        const int g = has_tag ? 160 : 39;
        const int b = has_tag ? 44 : 40;
        const int idx = static_cast<int>(verts.size());
        verts.push_back({t.x(), t.y(), t.z(),
                         static_cast<float>(r), static_cast<float>(g),
                         static_cast<float>(b)});
        if(prev_idx >= 0)
            edges.push_back({prev_idx, idx, r, g, b});
        prev_idx = idx;
    }

    std::ofstream ofs(path.c_str(), std::ios::out | std::ios::trunc);
    if(!ofs.is_open())
    {
        std::cerr << "[TagMapExporter] failed to open: " << path << std::endl;
        return false;
    }
    ofs << "ply\nformat ascii 1.0\n";
    ofs << "element vertex " << verts.size() << "\n";
    ofs << "property float x\nproperty float y\nproperty float z\n";
    ofs << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    ofs << "element face " << faces.size() << "\n";
    ofs << "property list uchar int vertex_indices\n";
    ofs << "element edge " << edges.size() << "\n";
    ofs << "property int vertex1\nproperty int vertex2\n";
    ofs << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    ofs << "end_header\n";
    ofs << std::setprecision(9);
    for(const auto &v : verts)
        ofs << v[0] << " " << v[1] << " " << v[2] << " "
            << static_cast<int>(v[3]) << " " << static_cast<int>(v[4]) << " "
            << static_cast<int>(v[5]) << "\n";
    for(const auto &f : faces)
        ofs << "4 " << f[0] << " " << f[1] << " " << f[2] << " " << f[3] << "\n";
    for(const auto &e : edges)
        ofs << e[0] << " " << e[1] << " " << e[2] << " " << e[3] << " " << e[4]
            << "\n";
    ofs.close();
    if(verbose)
        std::cout << "[TagMapExporter] saved final map PLY -> " << path
                  << " vertices=" << verts.size()
                  << " tag_planes=" << faces.size()
                  << " traj_edges=" << edges.size() << std::endl;
    return true;
}

bool TagMapExporter::SaveFinalMap(Map &map, bool verbose) const
{
    const bool tags_ok = SaveTagMapCorners(map, verbose);

    std::vector<Eigen::Vector3f> orb_pts;
    const auto mps = map.GetAllMapPoints();
    orb_pts.reserve(mps.size());
    for(MapPoint *pMP : mps)
    {
        if(!pMP || pMP->isBad())
            continue;
        const Eigen::Vector3f p = pMP->GetWorldPos();
        if(!std::isfinite(p.x()) || !std::isfinite(p.y()) || !std::isfinite(p.z()))
            continue;
        orb_pts.push_back(p);
    }

    const bool orb_ok = WriteOrbPointsCsv(orb_pts, "orb_final_map_points.csv", verbose);
    const bool tag_kf_ok =
        WriteKeyFramesTum(map, "tag_final_keyframes_tum.txt", true, verbose);
    const bool orb_kf_ok =
        WriteKeyFramesTum(map, "orb_final_keyframes_tum.txt", false, verbose);
    const bool ply_ok = WriteFinalMapPly(map, orb_pts, verbose);

    if(verbose)
        std::cout << "[TagMapExporter] final map "
                  << (tags_ok && orb_ok && tag_kf_ok && orb_kf_ok && ply_ok
                          ? "ok"
                          : "partial")
                  << " orb_pts=" << orb_pts.size() << std::endl;

    return tags_ok && orb_ok && tag_kf_ok && orb_kf_ok && ply_ok;
}

void TagMapExporter::CloseTrajectory()
{
    if(mTrajFile.is_open())
    {
        mTrajFile.flush();
        mTrajFile.close();
    }
    if(mDetFile.is_open())
    {
        mDetFile.flush();
        mDetFile.close();
    }
    if(mFirstRegFile.is_open())
    {
        mFirstRegFile.flush();
        mFirstRegFile.close();
    }
}

void TagMapExporter::ResetSession()
{
    CloseTrajectory();
    const bool traj_ok = OpenTrajectoryFile();
    const bool det_ok = OpenDetectionLogFile();
    const bool first_ok = OpenFirstRegistrationLogFile();
    mbEnabled = traj_ok || det_ok || first_ok;
}

}  // namespace tag
}  // namespace ORB_SLAM3
