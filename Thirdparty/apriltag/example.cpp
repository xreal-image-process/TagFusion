void apriltag_detect::detect()
{
    basalt::ApriltagDetector detector{587}; // Accept all tags. For basalt::ApriltagDetectorData,
                                            // the number limits the id of tags
    Eigen::aligned_vector<Eigen::Vector2d> corners, corners_rejected;
    std::vector<int> ids, ids_rejected;
    std::vector<double> radii, radii_rejected;
    detector.detectTags(
        m_picture,
        corners, ids, radii,
        corners_rejected, ids_rejected, radii_rejected
    );

    auto combine_corners_map = [this] (
        const Eigen::aligned_vector<Eigen::Vector2d>& corners,
        const std::vector<int> ids
    ) {
        std::map<int, Eigen::Vector2d> result;
        if (corners.size() != ids.size()) {
            throw std::runtime_error{"ApriltagDetector::detectTags corner internal error"};
        }
        for (int i : range(corners.size())) {
            result.emplace(ids[i], corners[i]);
        }
        return result;
    };

    m_corners = combine_corners_map(corners, ids);
    m_corners_rejected = combine_corners_map(corners_rejected, ids_rejected);
}
