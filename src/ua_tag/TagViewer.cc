#include "ua_tag/TagViewer.h"

#include "KeyFrame.h"
#include "Map.h"
#include "ua_tag/AprilTagVisualizer.h"
#include "ua_tag/MapTagData.h"

#include <opencv2/imgproc.hpp>
#include <pangolin/gl/glfont.h>
#include <pangolin/pangolin.h>

#include <cstdio>
#include <iostream>
#include <utility>

namespace ORB_SLAM3 {
namespace tag {

namespace {

pangolin::OpenGlMatrix ToGlMatrix(const Sophus::SE3f &T)
{
    pangolin::OpenGlMatrix M;
    const Eigen::Matrix4f m = T.matrix();
    for(int i = 0; i < 4; ++i)
    {
        M.m[4 * i] = m(0, i);
        M.m[4 * i + 1] = m(1, i);
        M.m[4 * i + 2] = m(2, i);
        M.m[4 * i + 3] = m(3, i);
    }
    return M;
}

}  // namespace

TagViewer::TagViewer(const std::string &settingsFile)
{
    if(settingsFile.empty())
        return;

    cv::FileStorage fs(settingsFile, cv::FileStorage::READ);
    if(!fs.isOpened())
        return;

    cv::FileNode node = fs["Tag.viewer_camera_size"];
    if(!node.empty())
        mCameraSize = static_cast<float>(static_cast<double>(node));

    node = fs["Tag.viewer_viewpoint_x"];
    if(!node.empty())
        mViewpointX = static_cast<float>(static_cast<double>(node));
    node = fs["Tag.viewer_viewpoint_y"];
    if(!node.empty())
        mViewpointY = static_cast<float>(static_cast<double>(node));
    node = fs["Tag.viewer_viewpoint_z"];
    if(!node.empty())
        mViewpointZ = static_cast<float>(static_cast<double>(node));
    node = fs["Tag.viewer_viewpoint_f"];
    if(!node.empty())
        mViewpointF = static_cast<float>(static_cast<double>(node));
}

TagViewer::~TagViewer()
{
    RequestFinish();
    if(mThread.joinable())
        mThread.join();
}

void TagViewer::Start()
{
    if(mThread.joinable())
        return;

    mbFinished = false;
    mbFinishRequested = false;
    mThread = std::thread(&TagViewer::Run, this);
}

void TagViewer::RequestFinish()
{
    mbFinishRequested = true;
}

bool TagViewer::isFinished() const
{
    return mbFinished.load();
}

void TagViewer::Freeze()
{
    mbFrozen = true;
}

bool TagViewer::IsFrozen() const
{
    return mbFrozen.load();
}

bool TagViewer::CheckFinish()
{
    return mbFinishRequested.load();
}

void TagViewer::SetFinish()
{
    mbFinished = true;
}

void TagViewer::Update(const cv::Mat &image,
                       const TagFrameData &frame_data,
                       bool has_pose,
                       const Sophus::SE3f &Tcw,
                       unsigned long frame_id,
                       Map *pMap,
                       int tracking_state)
{
    if(mbFrozen.load())
        return;

    cv::Mat vis = ua_tag::DrawTags(image, frame_data);
    if(vis.empty() && !image.empty())
    {
        if(image.channels() == 1)
            cv::cvtColor(image, vis, cv::COLOR_GRAY2BGR);
        else
            vis = image.clone();
    }

    // Tracking::eTrackingState: NOT_INITIALIZED=1, OK=2, RECENTLY_LOST=3, LOST=4
    const char *state_str = "OTHER";
    if(tracking_state == 1)
        state_str = "NOT_INIT";
    else if(tracking_state == 2)
        state_str = "OK";
    else if(tracking_state == 3)
        state_str = "RECENTLY_LOST";
    else if(tracking_state == 4)
        state_str = "LOST";
    else if(tracking_state == 0)
        state_str = "NO_IMAGES";

    const std::size_t n_map_tags = pMap ? pMap->MapTagsInMap() : 0;
    std::size_t n_tag_kf = 0;
    if(pMap)
    {
        for(KeyFrame *pKF : pMap->GetAllKeyFrames())
        {
            if(pKF && !pKF->isBad() && !pKF->GetMapTagMatches().empty())
                ++n_tag_kf;
        }
    }
    char overlay[128];
    std::snprintf(overlay, sizeof(overlay),
                  "frame:%lu  det:%zu  map_tags:%zu  tag_kf:%zu  state:%s",
                  frame_id, frame_data.Size(), n_map_tags, n_tag_kf, state_str);
    if(!vis.empty())
    {
        cv::putText(vis, overlay, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX,
                    0.7, cv::Scalar(40, 220, 255), 2, cv::LINE_AA);
        if(has_pose)
        {
            const Eigen::Vector3f twc = Tcw.inverse().translation();
            char pose_txt[96];
            std::snprintf(pose_txt, sizeof(pose_txt), "cam: %.2f %.2f %.2f",
                          twc.x(), twc.y(), twc.z());
            cv::putText(vis, pose_txt, cv::Point(12, 56),
                        cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(40, 220, 255),
                        2, cv::LINE_AA);
        }
    }

    std::vector<TagVis> tags;
    std::vector<Sophus::SE3f> kf_twc;
    if(pMap)
    {
        const auto map_tags = pMap->GetAllMapTags();
        tags.reserve(map_tags.size());
        for(const auto &map_tag : map_tags)
        {
            if(!map_tag)
                continue;
            TagVis tv;
            tv.id = map_tag->Id();
            tv.fixed = map_tag->IsFixed();
            tv.has_pose = map_tag->HasPose();
            tv.has_corners = map_tag->HasWorldCorners();
            if(tv.has_pose)
                tv.T_wt = map_tag->GetPose();
            if(tv.has_corners)
                tv.corners = map_tag->GetWorldCorners();
            tags.push_back(tv);
        }

        const auto kfs = pMap->GetAllKeyFrames();
        kf_twc.reserve(kfs.size());
        for(KeyFrame *pKF : kfs)
        {
            if(!pKF || pKF->isBad())
                continue;
            if(pKF->GetMapTagMatches().empty())
                continue;
            kf_twc.push_back(pKF->GetPoseInverse());
        }
    }

    {
        std::unique_lock<std::mutex> lock(mutex_data_);
        mImage = std::move(vis);
        mbHasPose = has_pose;
        mTcw = Tcw;
        mnFrameId = frame_id;
        mTrackingState = tracking_state;
        mTags = std::move(tags);
        mKfTwc = std::move(kf_twc);
    }
}

void TagViewer::DrawCameraFrustum(const Sophus::SE3f &Twc, float r, float g,
                                  float b, float size) const
{
    const float w = size;
    const float h = w * 0.75f;
    const float z = w * 0.6f;

    glPushMatrix();
    glMultMatrixd(ToGlMatrix(Twc).m);

    glLineWidth(2.0f);
    glColor3f(r, g, b);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(w, h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, h, z);

    glVertex3f(w, h, z);
    glVertex3f(w, -h, z);
    glVertex3f(-w, h, z);
    glVertex3f(-w, -h, z);
    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);
    glVertex3f(-w, -h, z);
    glVertex3f(w, -h, z);
    glEnd();

    glPopMatrix();
}

void TagViewer::DrawAxes(const Sophus::SE3f &Tw, float len) const
{
    glPushMatrix();
    glMultMatrixd(ToGlMatrix(Tw).m);
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    glColor3f(1.0f, 0.2f, 0.2f);
    glVertex3f(0, 0, 0);
    glVertex3f(len, 0, 0);
    glColor3f(0.2f, 1.0f, 0.2f);
    glVertex3f(0, 0, 0);
    glVertex3f(0, len, 0);
    glColor3f(0.2f, 0.4f, 1.0f);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0, len);
    glEnd();
    glPopMatrix();
}

void TagViewer::DrawTagSquare(const TagVis &tag) const
{
    if(!tag.has_corners)
        return;

    // 与 AprilTagVisualizer 边颜色一致：绿 / 黄 / 橙 / 蓝
    const float edge_rgb[4][3] = {
        {0.0f, 1.0f, 0.0f},
        {1.0f, 1.0f, 0.0f},
        {1.0f, 0.5f, 0.0f},
        {0.0f, 0.4f, 1.0f},
    };

    glLineWidth(tag.fixed ? 3.0f : 1.5f);
    glBegin(GL_LINES);
    for(int k = 0; k < 4; ++k)
    {
        const Eigen::Vector3f &p0 = tag.corners[k];
        const Eigen::Vector3f &p1 = tag.corners[(k + 1) % 4];
        glColor3f(edge_rgb[k][0], edge_rgb[k][1], edge_rgb[k][2]);
        glVertex3f(p0.x(), p0.y(), p0.z());
        glVertex3f(p1.x(), p1.y(), p1.z());
    }
    glEnd();

    // 半透明填充区分 FIXED / UNFIXED
    if(tag.fixed)
        glColor4f(0.15f, 0.85f, 0.25f, 0.25f);
    else
        glColor4f(0.95f, 0.55f, 0.10f, 0.20f);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBegin(GL_QUADS);
    for(int k = 0; k < 4; ++k)
    {
        const Eigen::Vector3f &p = tag.corners[k];
        glVertex3f(p.x(), p.y(), p.z());
    }
    glEnd();
    // 保持 BLEND 开启：GlFont 使用 GL_ALPHA 纹理，关掉会变成实心方块

    if(tag.has_pose)
        DrawAxes(tag.T_wt, mCameraSize * 0.6f);
}

void TagViewer::DrawTagId(const TagVis &tag) const
{
    if(!tag.has_corners && !tag.has_pose)
        return;

    Eigen::Vector3f center = Eigen::Vector3f::Zero();
    if(tag.has_corners)
    {
        for(int k = 0; k < 4; ++k)
            center += tag.corners[k];
        center *= 0.25f;

        // 沿法向略抬起，避免与平面 z-fight
        const Eigen::Vector3f e0 = tag.corners[1] - tag.corners[0];
        const Eigen::Vector3f e1 = tag.corners[3] - tag.corners[0];
        Eigen::Vector3f n = e0.cross(e1);
        if(n.squaredNorm() > 1e-12f)
            center += n.normalized() * (mCameraSize * 0.35f);
    }
    else
    {
        center = tag.T_wt.translation();
    }

    // GlFont 依赖 alpha 混合；并清掉可能残留的 2D 纹理绑定
    GLboolean blend_was_on = GL_FALSE;
    glGetBooleanv(GL_BLEND, &blend_was_on);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_TEXTURE_2D);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

    pangolin::GlFont::I().Text("id:%d", tag.id).Draw(
        center.x(), center.y(), center.z());

    glEnable(GL_DEPTH_TEST);
    if(!blend_was_on)
        glDisable(GL_BLEND);
}

void TagViewer::Run()
{
    constexpr int kWinW = 1400;
    constexpr int kWinH = 720;
    constexpr int kMenuW = 160;
    // 左侧图像区：菜单右侧到窗口 52%；宽高比按真实图像动态设置，保证完整显示
    constexpr float kImageRight = 0.52f;

    pangolin::CreateWindowAndBind("ORB-SLAM3: Tag Viewer", kWinW, kWinH);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::CreatePanel("tag_menu").SetBounds(0.0, 1.0, 0.0,
                                                pangolin::Attach::Pix(kMenuW));
    pangolin::Var<bool> menu_follow("tag_menu.Follow Camera", true, true);
    pangolin::Var<bool> menu_show_tags("tag_menu.Show Tags", true, true);
    pangolin::Var<bool> menu_show_ids("tag_menu.Show Tag IDs", true, true);
    pangolin::Var<bool> menu_show_kf("tag_menu.Show KF Cams", true, true);
    pangolin::Var<bool> menu_show_traj("tag_menu.Show Trajectory", true, true);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(kWinW, kWinH, mViewpointF, mViewpointF,
                                   kWinW / 2.0, kWinH / 2.0, 0.05, 100),
        pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0,
                                  0.0, -1.0, 0.0));

    pangolin::View &d_img =
        pangolin::Display("tag_image")
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(kMenuW), kImageRight)
            .SetLock(pangolin::LockLeft, pangolin::LockCenter);

    pangolin::View &d_cam =
        pangolin::Display("tag_map")
            .SetBounds(0.0, 1.0, kImageRight, 1.0, -640.0f / 480.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::GlTexture tex(640, 480, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
    bool tex_inited = false;
    bool followed = true;

    std::cout << "[TagViewer] started" << std::endl;

    while(!CheckFinish())
    {
        glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        cv::Mat image;
        bool has_pose = false;
        Sophus::SE3f Tcw;
        std::vector<TagVis> tags;
        std::vector<Sophus::SE3f> kf_twc;

        {
            std::unique_lock<std::mutex> lock(mutex_data_);
            image = mImage.clone();
            has_pose = mbHasPose;
            Tcw = mTcw;
            tags = mTags;
            kf_twc = mKfTwc;
        }

        // 左侧检测图：按图像真实宽高比 letterbox，完整显示
        if(!image.empty())
        {
            d_img.SetAspect(static_cast<double>(image.cols) /
                            static_cast<double>(image.rows));

            cv::Mat rgb;
            cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
            if(!rgb.isContinuous())
                rgb = rgb.clone();

            if(!tex_inited || tex.width != rgb.cols || tex.height != rgb.rows)
            {
                tex.Reinitialise(rgb.cols, rgb.rows, GL_RGB, false, 0, GL_RGB,
                                 GL_UNSIGNED_BYTE);
                tex_inited = true;
            }

            glDisable(GL_DEPTH_TEST);
            d_img.Activate();
            glColor3f(1.0f, 1.0f, 1.0f);
            // 宽非 4 对齐（如 612）时默认 UNPACK_ALIGNMENT=4 会错位/裁切
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            tex.Upload(rgb.data, GL_RGB, GL_UNSIGNED_BYTE);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
            tex.RenderToViewportFlipY();
            glEnable(GL_DEPTH_TEST);
        }

        // 跟随当前相机
        if(menu_follow && has_pose)
        {
            const Sophus::SE3f Twc = Tcw.inverse();
            pangolin::OpenGlMatrix M = ToGlMatrix(Twc);
            s_cam.Follow(M);
            followed = true;
        }
        else if(!menu_follow && followed)
        {
            followed = false;
        }

        d_cam.Activate(s_cam);
        glEnable(GL_DEPTH_TEST);

        // 世界坐标轴
        glLineWidth(1.5f);
        glBegin(GL_LINES);
        glColor3f(0.6f, 0.1f, 0.1f);
        glVertex3f(0, 0, 0);
        glVertex3f(0.3f, 0, 0);
        glColor3f(0.1f, 0.6f, 0.1f);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 0.3f, 0);
        glColor3f(0.1f, 0.2f, 0.7f);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 0, 0.3f);
        glEnd();

        if(menu_show_tags)
        {
            for(const TagVis &tag : tags)
                DrawTagSquare(tag);
        }

        if(menu_show_kf)
        {
            for(const Sophus::SE3f &Twc : kf_twc)
                DrawCameraFrustum(Twc, 0.55f, 0.55f, 0.55f, mCameraSize * 0.7f);
        }

        if(menu_show_traj && kf_twc.size() >= 2)
        {
            glLineWidth(1.5f);
            glColor3f(0.85f, 0.85f, 0.2f);
            glBegin(GL_LINE_STRIP);
            for(const Sophus::SE3f &Twc : kf_twc)
            {
                const Eigen::Vector3f t = Twc.translation();
                glVertex3f(t.x(), t.y(), t.z());
            }
            if(has_pose)
            {
                const Eigen::Vector3f t = Tcw.inverse().translation();
                glVertex3f(t.x(), t.y(), t.z());
            }
            glEnd();
        }

        if(has_pose)
        {
            const Sophus::SE3f Twc = Tcw.inverse();
            DrawCameraFrustum(Twc, 0.1f, 1.0f, 0.2f, mCameraSize);
            DrawAxes(Twc, mCameraSize);
        }

        // ID 放在最后画：GlText::Draw 会临时切到 DisplayBase
        if(menu_show_tags && menu_show_ids)
        {
            for(const TagVis &tag : tags)
            {
                d_cam.Activate(s_cam);
                DrawTagId(tag);
            }
        }

        pangolin::FinishFrame();
    }

    SetFinish();
    std::cout << "[TagViewer] finished" << std::endl;
}

}  // namespace tag
}  // namespace ORB_SLAM3
