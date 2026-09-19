#pragma once

#include <string>

#include "edgevision/gpu_frame.hpp"

struct AVFormatContext;
struct AVCodecContext;
struct AVBufferRef;
struct AVPacket;
struct AVFrame;

namespace edgevision {

// Hardware video decoder built on FFmpeg's libavformat/libavcodec with the CUDA
// hwaccel (NVDEC). Frames are returned as NV12 device pointers; nothing is copied to
// the host. Works for files and for RTSP/RTMP/HTTP URLs (demuxing is libavformat's).
class NvVideoDecoder {
public:
    explicit NvVideoDecoder(const std::string& url);
    ~NvVideoDecoder();
    NvVideoDecoder(const NvVideoDecoder&) = delete;
    NvVideoDecoder& operator=(const NvVideoDecoder&) = delete;

    // Next decoded frame. Returns false at end of stream. The returned pointers stay
    // valid until the next call to next() or reopen().
    bool next(GpuFrame& out);

    // Close and reopen the same URL (loop a clip).
    void reopen();

    int width() const;
    int height() const;
    std::string codec_name() const;
    std::string hw_pixel_format() const;

private:
    void open();
    void close();
    bool read_and_send_packet();  // false when the demuxer hit EOF (flush sent)

    std::string url_;
    AVFormatContext* format_ = nullptr;
    AVCodecContext* codec_ = nullptr;
    AVBufferRef* hw_device_ = nullptr;
    AVPacket* packet_ = nullptr;
    AVFrame* frame_ = nullptr;
    int stream_index_ = -1;
    bool flushed_ = false;
};

}  // namespace edgevision
