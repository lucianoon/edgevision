import pytest

from edgevision.video import VideoSource


def test_video_source_reads_every_frame_then_ends(clip):
    source = VideoSource(str(clip))

    frames = 0
    while True:
        ok, frame = source.read()
        if not ok:
            break
        assert frame.shape == (48, 64, 3)
        frames += 1
    source.release()

    assert frames == 5


def test_video_source_rejects_a_missing_file(tmp_path):
    with pytest.raises(RuntimeError, match="Could not open video source"):
        VideoSource(str(tmp_path / "missing.mp4"))
