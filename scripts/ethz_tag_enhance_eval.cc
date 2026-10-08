/**
 * Batch AprilTag recall eval over a full image sequence.
 * Same backend / merge rules as ua_tag::AprilTagDetector::DetectCorners:
 *   family=tag36h11, hamming=2, black_borders={1,2}, merge by id.
 *
 * Enhancements:
 *   - none (baseline)
 *   - gamma 0.50 / 0.55 / 0.60 / 0.65
 *   - GaussianBlur(3x3, sigma=0) then CLAHE(clip, tile=8)
 *     (sigma=0 => OpenCV default from ksize: 0.3*((k-1)*0.5-1)+0.8 = 0.8)
 */

#include <apriltags/TagDetector.h>
#include <apriltags/TagFamily.h>
#include <apriltags/Tag36h11.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct DetObs
{
    int id = -1;
    int hamming = 99;
    float perimeter = 0.f;
};

struct MethodSpec
{
    std::string name;
    enum class Kind { None, Gamma, ClaheGauss } kind = Kind::None;
    double gamma = 1.0;
    double clahe_clip = 0.0;
    int clahe_tile = 8;
};

struct MethodStats
{
    int64_t frames = 0;
    int64_t hit_frames = 0;
    int64_t tags = 0;
    std::map<int, int64_t> by_id;
};

cv::Mat BuildGammaLut(double gamma)
{
    cv::Mat lut(1, 256, CV_8U);
    uchar *p = lut.ptr<uchar>(0);
    const double inv = 1.0 / 255.0;
    for(int i = 0; i < 256; ++i)
        p[i] = cv::saturate_cast<uchar>(std::pow(static_cast<double>(i) * inv, gamma) * 255.0);
    return lut;
}

bool IsImageExt(const std::string &ext)
{
    return ext == ".pgm" || ext == ".png" || ext == ".jpg" || ext == ".jpeg" ||
           ext == ".bmp" || ext == ".tif" || ext == ".tiff";
}

std::vector<fs::path> ListImages(const fs::path &dir)
{
    std::vector<fs::path> out;
    for(const auto &ent : fs::directory_iterator(dir))
    {
        if(!ent.is_regular_file())
            continue;
        std::string ext = ent.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if(IsImageExt(ext))
            out.push_back(ent.path());
    }
    std::sort(out.begin(), out.end(),
              [](const fs::path &a, const fs::path &b) { return a.filename() < b.filename(); });
    return out;
}

void MergeDets(std::unordered_map<int, DetObs> &best,
               const std::vector<AprilTags::TagDetection> &dets,
               int hamming_max,
               const std::set<int> &ignore)
{
    for(const AprilTags::TagDetection &det : dets)
    {
        if(!det.good)
            continue;
        if(det.hammingDistance > hamming_max)
            continue;
        if(ignore.count(det.id))
            continue;

        DetObs obs;
        obs.id = det.id;
        obs.hamming = det.hammingDistance;
        obs.perimeter = static_cast<float>(det.observedPerimeter);

        auto it = best.find(obs.id);
        if(it == best.end())
        {
            best.emplace(obs.id, obs);
            continue;
        }
        const DetObs &prev = it->second;
        if(obs.hamming < prev.hamming ||
           (obs.hamming == prev.hamming && obs.perimeter > prev.perimeter))
            it->second = obs;
    }
}

void PrintUsage(const char *argv0)
{
    std::cerr
        << "Usage: " << argv0 << " --image-dir DIR [options]\n"
        << "  --threads N          default: hardware_concurrency\n"
        << "  --family NAME        default: tag36h11\n"
        << "  --hamming N          default: 2\n"
        << "  --black-borders A,B  default: 1,2\n"
        << "  --ignore-ids A,B     default: 91,92,93\n"
        << "  --no-ignore          do not drop any id\n"
        << "  --no-baseline        skip the none enhancement\n"
        << "  --no-gamma           skip gamma methods\n"
        << "  --clahe-clips A,B    default: 0.5,1,1.5,2 ; empty = no CLAHE\n"
        << "  --clahe-tile N       default: 8\n"
        << "  --no-gauss           CLAHE only (skip 3x3 Gaussian)\n";
}

std::vector<std::string> SplitList(const std::string &s)
{
    std::vector<std::string> out;
    std::string cur;
    for(char c : s)
    {
        if(c == ',' || c == ' ')
        {
            if(!cur.empty())
            {
                out.push_back(cur);
                cur.clear();
            }
        }
        else
            cur.push_back(c);
    }
    if(!cur.empty())
        out.push_back(cur);
    return out;
}

std::vector<int> ParseIntList(const std::string &s)
{
    std::vector<int> out;
    for(const auto &t : SplitList(s))
        out.push_back(std::stoi(t));
    return out;
}

std::vector<double> ParseDoubleList(const std::string &s)
{
    std::vector<double> out;
    for(const auto &t : SplitList(s))
        out.push_back(std::stod(t));
    return out;
}

}  // namespace

int main(int argc, char **argv)
{
    fs::path image_dir;
    int nthreads = static_cast<int>(std::thread::hardware_concurrency());
    if(nthreads <= 0)
        nthreads = 8;
    std::string family = "tag36h11";
    int hamming = 2;
    std::vector<int> black_borders = {1, 2};
    std::set<int> ignore_ids = {91, 92, 93};
    bool include_baseline = true;
    bool include_gamma = true;
    bool clahe_gauss = true;
    int clahe_tile = 8;
    std::vector<double> clahe_clips = {0.50, 1.00, 1.50, 2.00};

    for(int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto need = [&](const char *flag) -> const char * {
            if(i + 1 >= argc)
            {
                std::cerr << "missing value for " << flag << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if(a == "--image-dir")
            image_dir = need("--image-dir");
        else if(a == "--threads")
            nthreads = std::max(1, std::atoi(need("--threads")));
        else if(a == "--family")
            family = need("--family");
        else if(a == "--hamming")
            hamming = std::atoi(need("--hamming"));
        else if(a == "--black-borders")
            black_borders = ParseIntList(need("--black-borders"));
        else if(a == "--ignore-ids")
        {
            ignore_ids.clear();
            for(int id : ParseIntList(need("--ignore-ids")))
                ignore_ids.insert(id);
        }
        else if(a == "--no-ignore")
            ignore_ids.clear();
        else if(a == "--no-baseline")
            include_baseline = false;
        else if(a == "--no-gamma")
            include_gamma = false;
        else if(a == "--clahe-clips")
            clahe_clips = ParseDoubleList(need("--clahe-clips"));
        else if(a == "--clahe-tile")
            clahe_tile = std::max(1, std::atoi(need("--clahe-tile")));
        else if(a == "--no-gauss")
            clahe_gauss = false;
        else if(a == "-h" || a == "--help")
        {
            PrintUsage(argv[0]);
            return 0;
        }
        else
        {
            std::cerr << "unknown arg: " << a << "\n";
            PrintUsage(argv[0]);
            return 2;
        }
    }

    if(image_dir.empty() || !fs::is_directory(image_dir))
    {
        std::cerr << "need a valid --image-dir\n";
        PrintUsage(argv[0]);
        return 2;
    }
    if(family != "tag36h11")
    {
        std::cerr << "this eval binary is compiled for tag36h11 only\n";
        return 2;
    }
    if(black_borders.empty())
        black_borders.push_back(1);

    std::vector<MethodSpec> methods;
    if(include_baseline)
        methods.push_back({"none", MethodSpec::Kind::None});
    if(include_gamma)
    {
        for(double g : {0.50, 0.55, 0.60, 0.65})
        {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "gamma_%.2f", g);
            methods.push_back({buf, MethodSpec::Kind::Gamma, g});
        }
    }
    for(double clip : clahe_clips)
    {
        char buf[48];
        if(clahe_gauss)
            std::snprintf(buf, sizeof(buf), "g3x3_clahe_%.1f", clip);
        else
            std::snprintf(buf, sizeof(buf), "clahe_%.1f", clip);
        methods.push_back({buf, MethodSpec::Kind::ClaheGauss, 1.0, clip, clahe_tile});
    }

    const std::vector<fs::path> images = ListImages(image_dir);
    if(images.empty())
    {
        std::cerr << "no images in " << image_dir << "\n";
        return 1;
    }

    std::vector<cv::Mat> gamma_luts(methods.size());
    for(size_t m = 0; m < methods.size(); ++m)
    {
        if(methods[m].kind == MethodSpec::Kind::Gamma)
            gamma_luts[m] = BuildGammaLut(methods[m].gamma);
    }

    std::cout << "images=" << images.size()
              << " dir=" << image_dir
              << " threads=" << nthreads
              << " family=" << family
              << " hamming=" << hamming
              << " black_borders=";
    for(size_t i = 0; i < black_borders.size(); ++i)
        std::cout << (i ? "," : "") << black_borders[i];
    std::cout << " ignore_ids=";
    if(ignore_ids.empty())
        std::cout << "(none)";
    else
    {
        bool first = true;
        for(int id : ignore_ids)
        {
            std::cout << (first ? "" : ",") << id;
            first = false;
        }
    }
    std::cout << "\nmethods:";
    for(const auto &m : methods)
        std::cout << " " << m.name;
    std::cout << std::endl;

    const size_t n_methods = methods.size();
    std::vector<MethodStats> stats(n_methods);
    std::mutex stats_mu;
    std::atomic<size_t> next_idx{0};
    std::atomic<size_t> done{0};
    std::atomic<size_t> read_fail{0};

    const auto t0 = std::chrono::steady_clock::now();

    auto worker = [&]() {
        std::vector<std::unique_ptr<AprilTags::TagDetector>> dets;
        dets.reserve(black_borders.size());
        for(int bb : black_borders)
            dets.emplace_back(new AprilTags::TagDetector(AprilTags::tagCodes36h11,
                                                         static_cast<size_t>(std::max(1, bb))));

        std::vector<cv::Ptr<cv::CLAHE>> clahes(n_methods);
        for(size_t m = 0; m < n_methods; ++m)
        {
            if(methods[m].kind == MethodSpec::Kind::ClaheGauss)
            {
                const int tile = std::max(1, methods[m].clahe_tile);
                clahes[m] = cv::createCLAHE(methods[m].clahe_clip, cv::Size(tile, tile));
            }
        }

        std::vector<MethodStats> local(n_methods);

        for(;;)
        {
            const size_t i = next_idx.fetch_add(1);
            if(i >= images.size())
                break;

            cv::Mat gray = cv::imread(images[i].string(), cv::IMREAD_GRAYSCALE);
            if(gray.empty())
            {
                read_fail.fetch_add(1);
                done.fetch_add(1);
                continue;
            }
            if(!gray.isContinuous())
                gray = gray.clone();

            for(size_t m = 0; m < n_methods; ++m)
            {
                cv::Mat processed;
                switch(methods[m].kind)
                {
                case MethodSpec::Kind::None:
                    processed = gray;
                    break;
                case MethodSpec::Kind::Gamma:
                    cv::LUT(gray, gamma_luts[m], processed);
                    break;
                case MethodSpec::Kind::ClaheGauss:
                    if(clahe_gauss)
                    {
                        cv::GaussianBlur(gray, processed, cv::Size(3, 3), 0);
                        clahes[m]->apply(processed, processed);
                    }
                    else
                    {
                        clahes[m]->apply(gray, processed);
                    }
                    break;
                }
                if(!processed.isContinuous())
                    processed = processed.clone();

                std::unordered_map<int, DetObs> best;
                for(const auto &det_ptr : dets)
                    MergeDets(best, det_ptr->extractTags(processed), hamming, ignore_ids);

                local[m].frames += 1;
                local[m].tags += static_cast<int64_t>(best.size());
                if(!best.empty())
                    local[m].hit_frames += 1;
                for(const auto &kv : best)
                    local[m].by_id[kv.first] += 1;
            }

            const size_t n = done.fetch_add(1) + 1;
            if(n == 1 || n % 500 == 0 || n == images.size())
            {
                const auto t = std::chrono::steady_clock::now();
                const double sec = std::chrono::duration<double>(t - t0).count();
                const double fps = sec > 0.0 ? static_cast<double>(n) / sec : 0.0;
                std::cerr << "[progress] " << n << "/" << images.size()
                          << "  " << fps << " img/s\n";
            }
        }

        std::lock_guard<std::mutex> lock(stats_mu);
        for(size_t m = 0; m < n_methods; ++m)
        {
            stats[m].frames += local[m].frames;
            stats[m].hit_frames += local[m].hit_frames;
            stats[m].tags += local[m].tags;
            for(const auto &kv : local[m].by_id)
                stats[m].by_id[kv.first] += kv.second;
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(nthreads));
    for(int t = 0; t < nthreads; ++t)
        pool.emplace_back(worker);
    for(auto &th : pool)
        th.join();

    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\nread_fail=" << read_fail.load()
              << " elapsed_s=" << sec
              << " img/s=" << (sec > 0.0 ? static_cast<double>(images.size()) / sec : 0.0)
              << "\n\n";

    std::printf("%-16s %8s %10s %10s %10s %10s %8s\n",
                "method", "frames", "hit_frames", "hit_rate", "tags", "tags/frame", "uniq_id");
    for(size_t m = 0; m < n_methods; ++m)
    {
        const MethodStats &s = stats[m];
        const double hit = s.frames > 0
                               ? 100.0 * static_cast<double>(s.hit_frames) / static_cast<double>(s.frames)
                               : 0.0;
        const double tpf = s.frames > 0
                               ? static_cast<double>(s.tags) / static_cast<double>(s.frames)
                               : 0.0;
        std::printf("%-16s %8lld %10lld %9.2f%% %10lld %10.4f %8zu\n",
                    methods[m].name.c_str(),
                    static_cast<long long>(s.frames),
                    static_cast<long long>(s.hit_frames),
                    hit,
                    static_cast<long long>(s.tags),
                    tpf,
                    s.by_id.size());
    }

    std::cout << "\n--- per-id detection counts ---\n";
    std::set<int> all_ids;
    for(const auto &s : stats)
        for(const auto &kv : s.by_id)
            all_ids.insert(kv.first);

    std::printf("%-6s", "id");
    for(const auto &m : methods)
        std::printf(" %12s", m.name.c_str());
    std::printf("\n");
    for(int id : all_ids)
    {
        std::printf("%-6d", id);
        for(size_t m = 0; m < n_methods; ++m)
        {
            const auto it = stats[m].by_id.find(id);
            const int64_t c = (it == stats[m].by_id.end()) ? 0 : it->second;
            std::printf(" %12lld", static_cast<long long>(c));
        }
        std::printf("\n");
    }

    return 0;
}
