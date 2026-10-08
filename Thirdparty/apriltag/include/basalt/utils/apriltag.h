#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

namespace Eigen
{
  template <typename T>
  using aligned_vector = std::vector<T, Eigen::aligned_allocator<T>>;
}

namespace basalt {

struct ApriltagDetectorData;

class ApriltagDetector {
 public:
  ApriltagDetector(int numTags);

  ~ApriltagDetector();

  void detectTags(cv::Mat& img_raw,
                  Eigen::aligned_vector<Eigen::Vector2d>& corners,
                  std::vector<int>& ids, std::vector<double>& radii,
                  Eigen::aligned_vector<Eigen::Vector2d>& corners_rejected,
                  std::vector<int>& ids_rejected,
                  std::vector<double>& radii_rejected);

 private:
  ApriltagDetectorData* data;
};

}  // namespace basalt
