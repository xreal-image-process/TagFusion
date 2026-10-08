#ifndef UA_TAG_TAG_INITIALIZER_H
#define UA_TAG_TAG_INITIALIZER_H

#include "Frame.h"
#include "ua_tag/MapTagData.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

namespace ORB_SLAM3 {

class GeometricCamera;
class KeyFrame;
class Map;

namespace tag {

class TagInitializer
{
public:
    using MapTagPtr = std::shared_ptr<MapTagData>;
    using TagContainer = std::unordered_map<int, MapTagPtr>;

    // 初始化模式：由 Tracking 按 mSensor 显式传入
    enum class InitMode : std::uint8_t
    {
        Monocular = 0,     // 两帧路径（共视检查 / 单帧+第二帧 / 双帧联合）
        Stereo = 1         // 单帧即可（STEREO / IMU_STEREO）
    };

    // Tag 地图初始化结果
    struct Result
    {
        // 初始化完成的 Tag 地图实体
        TagContainer tags;

        // 各初始化帧的 Tag 位姿（Tag world -> Camera），与 keyframes 一一对应：
        //   Stereo：{I}；Monocular 单帧+第二帧：{I, Tcw_2}；双帧联合：{I, T_21}
        std::vector<Sophus::SE3f> Tcw_current;

        // 初始化涉及的 Frame（已 SetPose；历史字段，提交阶段以真实 KeyFrame 为准）：
        //   Stereo：仅 first 一帧；Monocular：first + second 两帧
        std::vector<Frame> keyframes;

        // true: 经单帧建图（Stereo，或 Monocular 单帧+第二帧 motion-only）；
        // false: 双帧联合消歧
        bool from_single_frame = false;
    };

    // 双视图 IPPE 联合消歧结果
    struct TwoViewAmbiguityResult
    {
        Sophus::SE3f T_21;  // 参考帧(1) -> 当前帧(2) 的相对位姿
        float mean_reproj_error = std::numeric_limits<float>::infinity();

        // 各共视 Tag 消歧后选中的 IPPE 候选下标
        std::unordered_map<int, int> selected_ref_candidate;
        std::unordered_map<int, int> selected_cur_candidate;
        // 消歧后的 Tag->Camera 位姿
        std::unordered_map<int, Sophus::SE3f> T_ct_ref;
        std::unordered_map<int, Sophus::SE3f> T_ct_cur;

        bool valid = false;
    };

    // 双帧初始化所需的最少共同 Tag 数
    static constexpr std::size_t kMinCommonTagsForTwoFrame = 2;
    // 双帧初始化所需的平均角点像素位移相对图像宽度的最小比例
    static constexpr float kMinMeanPixelDispRatio = 0.02f;
    // 双帧最优解平均重投影误差上限（像素）
    static constexpr float kMaxMeanReprojErrorPx = 2.5f;
    // 双帧最优解相对位姿平移范数下限（米）
    static constexpr float kMinTranslationNormM = 0.07f;

public:
    // 从 settings yaml 读取 Tag.size、Tag.verbose
    explicit TagInitializer(const std::string &settingsFile);

    // 仅计算：写 Result + 必要时回写 Frame 消歧/位姿；不写入 Map / 不关联 KeyFrame。
    // 必须在 new KeyFrame(Frame) 之前调用，以便 KF 拷贝到最终 TagFrameData。
    //   Stereo：忽略 second_frame，仅对 first_frame 做单帧建图
    //   Monocular：
    //     0) 先检查 first/second 共视 Tag 是否 >= kMinCommonTagsForTwoFrame，不足则失败
    //     1) 对 first_frame 单帧建图，成功后再用 motion-only BA 估计 second_frame 并注册两帧
    //     2) 上述失败则用已算好的共视做双帧联合初始化
    bool TryInitialize(Frame &first_frame, Frame &second_frame, Result &result,
                       InitMode mode);

    // 仅提交：将 Result 中的 MapTag 写入 ORB Map，并与真实 KeyFrame 建立双向关联。
    // 必须在 KeyFrame 已创建且 AddKeyFrame 之后、InsertKeyFrame 之前调用。
    // 成功则 SetTagInitialized(true)；中途失败会回滚已插入的 MapTag。
    bool CommitTagInitialization(const Result &result, Map *pMap,
                                 const std::vector<KeyFrame *> &vpKFs) const;

    // 将单帧 Tag 初始化结果从「当前左相机 = Tag 世界」对齐到已有 ORB 世界系。
    // 约定：result.from_single_frame==true，且各 MapTag::GetPose() 仍为 T_ct（Tag→当前左相机）。
    // 变换：T_wt_orb = Tcw_orb.inverse() * T_ct；并写回 result.Tcw_current = {Tcw_orb}。
    // 不修改传入的 Frame / ORB 相机位姿。
    bool AlignResultToOrbWorld(Result &result, const Sophus::SE3f &Tcw_orb) const;

    // 单帧：frame 中无歧义/有效 IPPE 的 Tag 数 >= 3 则成功。
    // Stereo 与 Monocular 单帧子路径内部调用。
    bool TryInitializeSingleFrame(Frame &frame, Result &result);

private:
    // tag_id -> 已有两个有效 IPPE 候选的观测（不要求已消歧）
    using TagObsByIdMap = std::unordered_map<int, const TagObservation *>;
    // tag_id -> 无歧义且有效 IPPE 的观测
    using UnambiguousObsMap = std::unordered_map<int, const TagObservation *>;

    // 同一 tag_id 在两帧中的观测对（cur = second，ref = first）
    struct CommonTagPair
    {
        const TagObservation *cur = nullptr;
        const TagObservation *ref = nullptr;
    };
    using CommonTagObsMap = std::unordered_map<int, CommonTagPair>;

    // 非离群且 pose_estimate 中含两个有效 IPPE 候选
    static bool HasTwoValidIppeCandidates(const TagObservation &obs) noexcept;

    // 按 tag_id 收集观测；同 id 时左目优先于右目（要求两个有效 IPPE 候选）
    static void CollectObservationById(TagObsByIdMap &by_id, const TagObservation &obs);
    static TagObsByIdMap CollectObservationsById(const Frame &frame);

    // 将无歧义有效观测写入集合；同 tag_id 时左目优先于右目
    static void CollectUnambiguousObservation(UnambiguousObsMap &unambiguous,
                                              const TagObservation &obs);

    static UnambiguousObsMap CollectUnambiguousObservations(const Frame &frame);

    // 查找 second 与 first 之间共同 id 的 Tag 观测对
    static CommonTagObsMap FindCommonTagObservations(const Frame &frame,
                                                     const Frame &ref_frame);

    // 共同观测 Tag 的去畸变角点在两帧之间的平均欧式像素位移
    static float ComputeMeanCornerPixelDisplacement(const CommonTagObsMap &common_obs);

    // GeometricCamera::project vs corners_raw（原图像素）的 RMSE
    static float ComputeTagCornerReprojRmse(const Sophus::SE3f &T_ct,
                                            const std::array<cv::Point2f, 4> &corners_obs,
                                            GeometricCamera *pCamera,
                                            const std::vector<cv::Point3f> &object_pts);

    // 双视图 IPPE 联合消歧（独立流程）：
    //   1) 枚举全部共视 Tag 的 2x2 组合，按全场平均双向重投影选全局最优 T_21*
    //   2) 固定 T_21*，对每个共视 Tag 再遍历 4 种 IPPE 组合，取误差最小者消歧并回写
    bool ResolveTwoViewIppeAmbiguity(const CommonTagObsMap &common_obs,
                                     GeometricCamera *pCamera,
                                     TwoViewAmbiguityResult &out) const;

    // 单帧建图成功后：motion-only BA 估计 second_frame 位姿，并像双帧一样注册两帧
    // （Tcw_current={I,Tcw_2}，keyframes 两帧，tags 上追加 second 观测）
    bool CompleteSingleFrameInitWithSecondFrame(Frame &first_frame,
                                                Frame &second_frame,
                                                Result &result);

    // 双帧：first_frame + second_frame 联合初始化（common_obs 由调用方预先计算）
    bool TryInitializeTwoFrames(Frame &frame, Frame &ref_frame,
                                const CommonTagObsMap &common_obs, Result &result);

    // 从 settings yaml 的 Tag.size / Tag.size_by_id 读取（米）
    double mTagSize = 0.16;
    std::map<int, double> mTagSizeById;

    double GetTagSize(int tag_id) const;

    // 从 settings yaml 的 Tag.verbose 读取；为 true 时打印初始化日志
    bool mbVerbose = false;
};

}  // namespace tag
}  // namespace ORB_SLAM3

#endif
