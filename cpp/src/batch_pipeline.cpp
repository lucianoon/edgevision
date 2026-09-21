#include "edgevision/batch_pipeline.hpp"

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "edgevision/cuda_check.hpp"
#include "edgevision/detector.hpp"
#include "edgevision/gpu_frame.hpp"
#include "edgevision/letterbox.hpp"
#include "edgevision/trt_engine.hpp"
#include "edgevision/video_decoder.hpp"

namespace edgevision {

namespace {

using Clock = std::chrono::steady_clock;
double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Shared round bookkeeping between the N workers and the coordinator.
struct Sync {
    std::mutex m;
    std::condition_variable cv;
    std::vector<int> ready;  // per round: workers that filled their slot
    int completed = -1;      // last round whose inference is done (its buffer is free again)
    bool abort = false;
    std::string error;

    void fail(const std::string& what) {
        std::lock_guard<std::mutex> lock(m);
        if (!abort) {
            abort = true;
            error = what;
        }
        cv.notify_all();
    }
};

struct SharedBuffers {
    float* input[2] = {nullptr, nullptr};  // ping-pong, N x 3 x S x S floats each
    std::vector<LetterboxInfo> info[2];    // per slot, written by workers
    size_t slot_floats = 0;
    int size = 640;
};

// One worker: decodes its stream and letterboxes each frame into its slot of the
// current round's input buffer.
void worker(int index, const BatchOptions& opt, SharedBuffers& buffers, Sync& sync, int total_rounds,
            PerformanceMetrics& metrics) try {
    cudaStream_t stream = nullptr;
    EV_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // OpenCV path resources (lazy)
    std::uint8_t* device_frame = nullptr;
    size_t device_frame_bytes = 0;
    void* pinned = nullptr;
    size_t pinned_bytes = 0;

    NvVideoDecoder* decoder = nullptr;
    cv::VideoCapture cap;
    if (opt.nvdec) {
        decoder = new NvVideoDecoder(opt.sources[index]);
    } else if (!cap.open(opt.sources[index])) {
        throw std::runtime_error("cannot open source: " + opt.sources[index]);
    }

    GpuFrame gpu;
    cv::Mat frame;
    for (int round = 0; round < total_rounds; ++round) {
        const int buf = round & 1;
        {  // the buffer is free once round-2 (same buffer) finished inference
            std::unique_lock<std::mutex> lock(sync.m);
            sync.cv.wait(lock, [&] { return sync.abort || sync.completed >= round - 2; });
            if (sync.abort) break;
        }
        if (round == opt.warmup) metrics.reset();

        // decode (loop the clip)
        bool ok = false;
        {
            ScopedTimer t(metrics, "decode");
            while (!ok) {
                if (opt.nvdec) {
                    ok = decoder->next(gpu);
                    if (!ok) decoder->reopen();
                } else {
                    ok = cap.read(frame);
                    if (!ok) {
                        cap.release();
                        if (!cap.open(opt.sources[index]))
                            throw std::runtime_error("cannot reopen " + opt.sources[index]);
                    }
                }
            }
        }

        {
            ScopedTimer t(metrics, "preprocess");
            float* dst = buffers.input[buf] + static_cast<size_t>(index) * buffers.slot_floats;
            if (opt.nvdec) {
                const LetterboxInfo info = compute_letterbox(gpu.width, gpu.height, buffers.size);
                letterbox_nv12_cuda(gpu, info, dst, stream);
                buffers.info[buf][index] = info;
            } else {
                const size_t bytes = frame.step[0] * static_cast<size_t>(frame.rows);
                if (bytes > pinned_bytes) {
                    if (pinned) cudaFreeHost(pinned);
                    EV_CUDA_CHECK(cudaHostAlloc(&pinned, bytes, cudaHostAllocDefault));
                    pinned_bytes = bytes;
                }
                if (bytes > device_frame_bytes) {
                    if (device_frame) cudaFree(device_frame);
                    EV_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_frame), bytes));
                    device_frame_bytes = bytes;
                }
                std::memcpy(pinned, frame.data, bytes);
                EV_CUDA_CHECK(cudaMemcpyAsync(device_frame, pinned, bytes, cudaMemcpyHostToDevice, stream));
                const LetterboxInfo info = compute_letterbox(frame.cols, frame.rows, buffers.size);
                letterbox_cuda(device_frame, static_cast<int>(frame.step[0]), info, dst, stream);
                buffers.info[buf][index] = info;
            }
            EV_CUDA_CHECK(cudaStreamSynchronize(stream));  // slot written; decoder frame may be recycled
        }
        {
            std::lock_guard<std::mutex> lock(sync.m);
            ++sync.ready[round];
        }
        sync.cv.notify_all();
    }

    delete decoder;
    if (device_frame) cudaFree(device_frame);
    if (pinned) cudaFreeHost(pinned);
    cudaStreamDestroy(stream);
} catch (const std::exception& e) {
    sync.fail("stream " + std::to_string(index) + " (" + opt.sources[index] + "): " + e.what());
}

}  // namespace

BatchResult run_batched(const BatchOptions& opt, const std::map<int, std::string>& names) {
    const int n = static_cast<int>(opt.sources.size());
    if (n < 1) throw std::runtime_error("run_batched: no sources");

    TrtEngine engine(opt.engine_path);
    if (engine.dynamic_batch()) engine.set_batch(n);
    if (engine.batch() != n)
        throw std::runtime_error("engine batch is " + std::to_string(engine.batch()) + " but " + std::to_string(n) +
                                 " streams were given (export the ONNX with batch=" + std::to_string(n) + ")");
    const auto in = engine.input_dims();
    if (in.nbDims != 4 || in.d[1] != 3 || in.d[2] != in.d[3] || engine.input_type() != nvinfer1::DataType::kFLOAT)
        throw std::runtime_error("engine input must be Nx3xSxS FP32");
    const auto out = engine.output_dims();
    if (out.nbDims != 3 || out.d[2] != 6) throw std::runtime_error("engine output must be Nxmax_detx6 (NMS in graph)");
    const int size = static_cast<int>(in.d[2]);
    const int max_det = static_cast<int>(out.d[1]);

    SharedBuffers buffers;
    buffers.size = size;
    buffers.slot_floats = 3ull * size * size;
    for (int b = 0; b < 2; ++b) {
        EV_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers.input[b]), engine.input_bytes()));
        buffers.info[b].resize(n);
    }
    float* host_out = nullptr;
    EV_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&host_out), engine.output_bytes(), cudaHostAllocDefault));
    cudaStream_t infer_stream = nullptr;
    EV_CUDA_CHECK(cudaStreamCreateWithFlags(&infer_stream, cudaStreamNonBlocking));

    const int total_rounds = opt.warmup + opt.frames;
    Sync sync;
    sync.ready.assign(total_rounds, 0);

    BatchResult result;
    result.batch = n;
    result.workers.resize(n);
    result.first_measured.resize(n);

    std::vector<std::thread> threads;
    for (int i = 0; i < n; ++i)
        threads.emplace_back(worker, i, std::cref(opt), std::ref(buffers), std::ref(sync), total_rounds,
                             std::ref(result.workers[i]));

    Clock::time_point measure_start;
    try {
        for (int round = 0; round < total_rounds; ++round) {
            const int buf = round & 1;
            const auto t0 = Clock::now();
            {
                std::unique_lock<std::mutex> lock(sync.m);
                sync.cv.wait(lock, [&] { return sync.abort || sync.ready[round] == n; });
                if (sync.abort) throw std::runtime_error(sync.error);
            }
            const auto t1 = Clock::now();

            engine.bind_input(buffers.input[buf]);
            engine.enqueue(infer_stream);
            EV_CUDA_CHECK(cudaMemcpyAsync(host_out, engine.output_device(), engine.output_bytes(),
                                          cudaMemcpyDeviceToHost, infer_stream));
            EV_CUDA_CHECK(cudaStreamSynchronize(infer_stream));
            const std::vector<LetterboxInfo> infos = buffers.info[buf];  // copy before releasing the buffer
            const auto t2 = Clock::now();
            {
                std::lock_guard<std::mutex> lock(sync.m);
                sync.completed = round;  // buffer `buf` may be reused for round+2
            }
            sync.cv.notify_all();

            std::vector<std::vector<Detection>> dets(n);
            for (int i = 0; i < n; ++i)
                dets[i] = collect_detections(host_out + static_cast<size_t>(i) * max_det * 6, max_det, infos[i],
                                             opt.confidence, names);
            const auto t3 = Clock::now();

            if (round == opt.warmup) {
                result.rounds.reset();
                measure_start = t0;
            }
            if (round == opt.warmup) result.first_measured = dets;
            result.rounds.record("gather", ms_between(t0, t1));
            result.rounds.record("inference", ms_between(t1, t2));
            result.rounds.record("postprocess", ms_between(t2, t3));
            result.rounds.record(PerformanceMetrics::kEndToEnd, ms_between(t0, t3));
            result.rounds.count_frame();
        }
    } catch (...) {
        sync.fail("coordinator failed");
        for (auto& t : threads) t.join();
        throw;
    }
    for (auto& t : threads) t.join();
    if (sync.abort) throw std::runtime_error(sync.error);

    result.rounds_measured = opt.frames;
    result.measured_seconds = std::chrono::duration<double>(Clock::now() - measure_start).count();

    cudaStreamDestroy(infer_stream);
    cudaFreeHost(host_out);
    for (int b = 0; b < 2; ++b) cudaFree(buffers.input[b]);
    return result;
}

}  // namespace edgevision
