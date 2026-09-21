#include "edgevision/video_decoder.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

#include <stdexcept>
#include <string>

namespace edgevision {

namespace {

std::string av_err(int code) {
    char buf[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(code, buf, sizeof(buf));
    return buf;
}

// Tell libavcodec to pick the CUDA (NVDEC) surface format among the offered ones.
enum AVPixelFormat pick_cuda_format(AVCodecContext*, const enum AVPixelFormat* formats) {
    for (const enum AVPixelFormat* p = formats; *p != AV_PIX_FMT_NONE; ++p)
        if (*p == AV_PIX_FMT_CUDA) return *p;
    return AV_PIX_FMT_NONE;  // decoder will fail: we do not want a silent CPU fallback
}

}  // namespace

NvVideoDecoder::NvVideoDecoder(const std::string& url) : url_(url) {
    open();
}

NvVideoDecoder::~NvVideoDecoder() {
    close();
}

void NvVideoDecoder::open() {
    AVDictionary* options = nullptr;
    av_dict_set(&options, "rtsp_transport", "tcp", 0);  // robust RTSP; ignored for files
    av_dict_set(&options, "stimeout", "5000000", 0);    // 5 s socket timeout (us), RTSP only
    int rc = avformat_open_input(&format_, url_.c_str(), nullptr, &options);
    av_dict_free(&options);
    if (rc < 0) throw std::runtime_error("avformat_open_input(" + url_ + "): " + av_err(rc));
    if ((rc = avformat_find_stream_info(format_, nullptr)) < 0)
        throw std::runtime_error("avformat_find_stream_info: " + av_err(rc));

    const AVCodec* decoder = nullptr;
    stream_index_ = av_find_best_stream(format_, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
    if (stream_index_ < 0 || !decoder) throw std::runtime_error("no decodable video stream in " + url_);

    codec_ = avcodec_alloc_context3(decoder);
    if (!codec_) throw std::runtime_error("avcodec_alloc_context3 failed");
    if ((rc = avcodec_parameters_to_context(codec_, format_->streams[stream_index_]->codecpar)) < 0)
        throw std::runtime_error("avcodec_parameters_to_context: " + av_err(rc));

    if ((rc = av_hwdevice_ctx_create(&hw_device_, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0)) < 0)
        throw std::runtime_error("av_hwdevice_ctx_create(CUDA): " + av_err(rc) +
                                 " (is libnvcuvid visible? docker needs NVIDIA_DRIVER_CAPABILITIES=...,video)");
    codec_->hw_device_ctx = av_buffer_ref(hw_device_);
    codec_->get_format = pick_cuda_format;
    codec_->pkt_timebase = format_->streams[stream_index_]->time_base;

    if ((rc = avcodec_open2(codec_, decoder, nullptr)) < 0)
        throw std::runtime_error(std::string("avcodec_open2(") + decoder->name + "): " + av_err(rc));

    packet_ = av_packet_alloc();
    frame_ = av_frame_alloc();
    if (!packet_ || !frame_) throw std::runtime_error("av_packet_alloc/av_frame_alloc failed");
    flushed_ = false;
}

void NvVideoDecoder::close() {
    if (frame_) av_frame_free(&frame_);
    if (packet_) av_packet_free(&packet_);
    if (codec_) avcodec_free_context(&codec_);
    if (hw_device_) av_buffer_unref(&hw_device_);
    if (format_) avformat_close_input(&format_);
    stream_index_ = -1;
}

void NvVideoDecoder::reopen() {
    close();
    open();
}

bool NvVideoDecoder::read_and_send_packet() {
    while (true) {
        int rc = av_read_frame(format_, packet_);
        if (rc < 0) {  // EOF or error: flush the decoder
            avcodec_send_packet(codec_, nullptr);
            flushed_ = true;
            return false;
        }
        if (packet_->stream_index != stream_index_) {
            av_packet_unref(packet_);
            continue;
        }
        rc = avcodec_send_packet(codec_, packet_);
        av_packet_unref(packet_);
        if (rc < 0 && rc != AVERROR(EAGAIN)) throw std::runtime_error("avcodec_send_packet: " + av_err(rc));
        return true;
    }
}

bool NvVideoDecoder::next(GpuFrame& out) {
    while (true) {
        const int rc = avcodec_receive_frame(codec_, frame_);
        if (rc == 0) {
            if (frame_->format != AV_PIX_FMT_CUDA)
                throw std::runtime_error(std::string("decoder produced ") +
                                         av_get_pix_fmt_name((AVPixelFormat)frame_->format) +
                                         " instead of CUDA surfaces (NVDEC unavailable for this codec?)");
            const auto* frames_ctx = reinterpret_cast<AVHWFramesContext*>(frame_->hw_frames_ctx->data);
            if (frames_ctx->sw_format != AV_PIX_FMT_NV12)
                throw std::runtime_error(std::string("unsupported surface format ") +
                                         av_get_pix_fmt_name(frames_ctx->sw_format) + " (only NV12 / 8-bit)");
            out.y = frame_->data[0];
            out.uv = frame_->data[1];
            out.pitch_y = frame_->linesize[0];
            out.pitch_uv = frame_->linesize[1];
            out.width = frame_->width;
            out.height = frame_->height;
            return true;
        }
        if (rc == AVERROR_EOF) return false;
        if (rc != AVERROR(EAGAIN)) throw std::runtime_error("avcodec_receive_frame: " + av_err(rc));
        if (flushed_) return false;
        read_and_send_packet();
    }
}

int NvVideoDecoder::width() const {
    return codec_ ? codec_->width : 0;
}
int NvVideoDecoder::height() const {
    return codec_ ? codec_->height : 0;
}
std::string NvVideoDecoder::codec_name() const {
    return codec_ && codec_->codec ? codec_->codec->name : "";
}
std::string NvVideoDecoder::hw_pixel_format() const {
    return "cuda/nv12";
}

}  // namespace edgevision
