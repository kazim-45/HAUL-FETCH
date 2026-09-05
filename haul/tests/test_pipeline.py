"""Unit tests for haul.core.pipeline using fake extractor/downloader
doubles — no network or yt-dlp involved, only the orchestration
described in spec §29 (validate -> detect -> extract -> select ->
download -> merge -> metadata -> save)."""

from __future__ import annotations

import pytest

from haul.core import errors
from haul.core.extractor import Extractor, MediaFormat, MediaInfo, MediaType
from haul.core.pipeline import DownloadRequest, run
from haul.core.registry import Registry


class FakeExtractor(Extractor):
    name = "fake"

    def __init__(self, info):
        self._info = info

    def supports(self, url):
        return "fake.example" in url

    def extract(self, url):
        return self._info

    def extract_profile_picture(self, url):
        return self._info


class FakeDownloader:
    """Stands in for haul.core.downloader.Downloader: writes small
    marker files instead of touching the network, so pipeline wiring
    can be tested without yt-dlp or requests involved."""

    def __init__(self):
        self.calls = []

    def resolve_destination(self, destination, *, policy, prompt=None):
        if not destination.exists():
            return destination
        if policy == "force":
            return destination
        if policy == "skip":
            return None
        if prompt:
            choice = prompt(destination)
            if choice == "overwrite":
                return destination
            if choice == "rename":
                return destination.with_name(destination.stem + "_1" + destination.suffix)
        return None

    def download(self, url, destination, *, on_progress=None):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"content from {url}")
        self.calls.append(("download", url, destination))
        return destination

    def merge_audio_video(self, video_path, audio_path, output_path):
        output_path.write_text("merged")
        video_path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)
        self.calls.append(("merge", output_path))
        return output_path

    def extract_audio(self, input_path, output_path):
        output_path.write_text("audio")
        self.calls.append(("extract_audio", output_path))
        return output_path


def make_info(**overrides):
    defaults = dict(
        platform="fake",
        media_type=MediaType.VIDEO,
        id="abc123",
        source_url="https://fake.example/post/1",
        author="creator",
        title="A Test Video",
        formats=[
            MediaFormat(
                url="https://cdn.example/video.mp4",
                extension="mp4",
                width=1920,
                height=1080,
                has_audio=True,
            )
        ],
    )
    defaults.update(overrides)
    return MediaInfo(**defaults)


def build_registry(info):
    registry = Registry()
    registry.register(FakeExtractor(info))
    return registry


def test_run_downloads_to_expected_path(tmp_path):
    info = make_info()
    registry = build_registry(info)
    downloader = FakeDownloader()
    req = DownloadRequest(url="https://fake.example/post/1", output_dir=tmp_path)

    [result] = run(req, registry, downloader)

    assert result.path.exists()
    assert result.path.parent == tmp_path / "fake" / "creator"
    assert result.path.name == "fake_creator_abc123.mp4"


def test_run_writes_metadata_when_requested(tmp_path):
    info = make_info()
    registry = build_registry(info)
    downloader = FakeDownloader()
    req = DownloadRequest(url="https://fake.example/post/1", output_dir=tmp_path, want_metadata=True)

    [result] = run(req, registry, downloader)

    assert result.metadata_path is not None
    assert result.metadata_path.exists()
    assert result.metadata_path.suffix == ".json"


def test_run_skips_existing_file_with_skip_policy(tmp_path):
    info = make_info()
    registry = build_registry(info)
    downloader = FakeDownloader()
    req = DownloadRequest(url="https://fake.example/post/1", output_dir=tmp_path, duplicate_policy="skip")

    existing = tmp_path / "fake" / "creator" / "fake_creator_abc123.mp4"
    existing.parent.mkdir(parents=True)
    existing.write_text("already here")

    [result] = run(req, registry, downloader)

    assert result.skipped is True
    assert existing.read_text() == "already here"
    assert downloader.calls == []


def test_run_overwrites_with_force_policy(tmp_path):
    info = make_info()
    registry = build_registry(info)
    downloader = FakeDownloader()
    req = DownloadRequest(url="https://fake.example/post/1", output_dir=tmp_path, duplicate_policy="force")

    existing = tmp_path / "fake" / "creator" / "fake_creator_abc123.mp4"
    existing.parent.mkdir(parents=True)
    existing.write_text("stale")

    [result] = run(req, registry, downloader)

    assert result.skipped is False
    assert existing.read_text() != "stale"


def test_run_merges_separate_audio_and_video_streams(tmp_path, monkeypatch):
    import haul.core.pipeline as pipeline

    # Deterministic regardless of whether this machine actually has
    # FFmpeg installed — the fallback behavior itself is covered
    # separately below.
    monkeypatch.setattr(pipeline, "ffmpeg_available", lambda: True)

    fmt = MediaFormat(
        url="https://cdn.example/video-only.mp4",
        extension="mp4",
        width=3840,
        height=2160,
        has_audio=False,
        audio_url="https://cdn.example/audio-only.m4a",
        audio_extension="m4a",
    )
    info = make_info(formats=[fmt])
    registry = build_registry(info)
    downloader = FakeDownloader()
    req = DownloadRequest(url="https://fake.example/post/1", output_dir=tmp_path)

    [result] = run(req, registry, downloader)

    assert result.path.exists()
    assert result.path.read_text() == "merged"
    assert any(call[0] == "merge" for call in downloader.calls)
    downloaded_urls = [call[1] for call in downloader.calls if call[0] == "download"]
    assert "https://cdn.example/video-only.mp4" in downloaded_urls
    assert "https://cdn.example/audio-only.m4a" in downloaded_urls


def test_run_picks_best_standalone_audio_stream_for_audio_only(tmp_path):
    info = make_info(
        audio_formats=[
            MediaFormat(url="https://cdn.example/low.m4a", extension="m4a", bitrate=64, has_audio=True),
            MediaFormat(url="https://cdn.example/high.m4a", extension="m4a", bitrate=192, has_audio=True),
        ],
    )
    registry = build_registry(info)
    downloader = FakeDownloader()
    req = DownloadRequest(url="https://fake.example/post/1", output_dir=tmp_path, want_audio_only=True)

    [result] = run(req, registry, downloader)

    assert result.path.suffix == ".m4a"
    downloaded_url = downloader.calls[0][1]
    assert "high.m4a" in downloaded_url
    assert not any(call[0] == "extract_audio" for call in downloader.calls)


def test_run_falls_back_to_video_extraction_when_no_standalone_audio(tmp_path):
    info = make_info()  # only a video format, no audio_formats at all
    registry = build_registry(info)
    downloader = FakeDownloader()
    req = DownloadRequest(url="https://fake.example/post/1", output_dir=tmp_path, want_audio_only=True)

    [result] = run(req, registry, downloader)

    assert result.path.suffix == ".m4a"
    assert any(call[0] == "extract_audio" for call in downloader.calls)


def test_run_expands_gallery_into_multiple_results(tmp_path):
    info_list = [make_info(id=f"img{i}") for i in range(3)]

    class GalleryExtractor(FakeExtractor):
        def extract(self, url):
            return info_list

    registry = Registry()
    registry.register(GalleryExtractor(info_list))
    downloader = FakeDownloader()
    req = DownloadRequest(url="https://fake.example/gallery/1", output_dir=tmp_path)

    results = run(req, registry, downloader)

    assert len(results) == 3
    assert {r.path.name for r in results} == {
        "fake_creator_img0.mp4",
        "fake_creator_img1.mp4",
        "fake_creator_img2.mp4",
    }


def test_run_uses_custom_filename_template(tmp_path):
    info = make_info(title="My Great Video")
    registry = build_registry(info)
    downloader = FakeDownloader()
    req = DownloadRequest(
        url="https://fake.example/post/1",
        output_dir=tmp_path,
        filename_template="{author}_{title}.{ext}",
    )

    [result] = run(req, registry, downloader)

    assert result.path.name == "creator_my_great_video.mp4"


# -- FFmpeg-missing fallback behavior -----------------------------------
#
# Regression coverage for the exact bug a user hit in production: a
# 1080p YouTube video needed a separate audio+video merge, HAUL
# downloaded both streams in full, and only then failed because
# FFmpeg wasn't installed — after already spending the bandwidth and
# leaving two orphaned temp files behind. The fix: check FFmpeg
# availability *before* downloading, and prefer a quality tier that
# doesn't need merging when one exists.


def _formats_needing_and_not_needing_merge():
    needs_merge = MediaFormat(
        url="https://cdn.example/1080p-video-only.mp4",
        extension="mp4",
        height=1080,
        has_audio=False,
        audio_url="https://cdn.example/1080p-audio-only.m4a",
        audio_extension="m4a",
    )
    merge_free = MediaFormat(
        url="https://cdn.example/720p-with-audio.mp4",
        extension="mp4",
        height=720,
        has_audio=True,
    )
    return [merge_free, needs_merge]


def test_falls_back_to_merge_free_quality_when_ffmpeg_missing(tmp_path, monkeypatch):
    import haul.core.pipeline as pipeline

    monkeypatch.setattr(pipeline, "ffmpeg_available", lambda: False)

    info = make_info(formats=_formats_needing_and_not_needing_merge())
    registry = build_registry(info)
    downloader = FakeDownloader()
    warnings = []
    req = DownloadRequest(url="https://fake.example/post/1", output_dir=tmp_path, quality="1080p")

    [result] = run(req, registry, downloader, on_warning=warnings.append)

    # Fell back to the 720p file that already has audio, not the
    # 1080p file that would need a merge.
    assert result.chosen.height == 720
    assert not any(call[0] == "merge" for call in downloader.calls)
    assert len(warnings) == 1
    assert "720p" in warnings[0] and "1080p" in warnings[0]


def test_uses_full_quality_when_ffmpeg_available(tmp_path, monkeypatch):
    import haul.core.pipeline as pipeline

    monkeypatch.setattr(pipeline, "ffmpeg_available", lambda: True)

    info = make_info(formats=_formats_needing_and_not_needing_merge())
    registry = build_registry(info)
    downloader = FakeDownloader()
    warnings = []
    req = DownloadRequest(url="https://fake.example/post/1", output_dir=tmp_path, quality="1080p")

    [result] = run(req, registry, downloader, on_warning=warnings.append)

    assert result.chosen.height == 1080
    assert any(call[0] == "merge" for call in downloader.calls)
    assert warnings == []


def test_raises_ffmpeg_missing_when_no_merge_free_alternative_exists(tmp_path, monkeypatch):
    import haul.core.pipeline as pipeline

    monkeypatch.setattr(pipeline, "ffmpeg_available", lambda: False)

    only_format = MediaFormat(
        url="https://cdn.example/video-only.mp4",
        extension="mp4",
        height=1080,
        has_audio=False,
        audio_url="https://cdn.example/audio-only.m4a",
        audio_extension="m4a",
    )
    info = make_info(formats=[only_format])
    registry = build_registry(info)
    downloader = FakeDownloader()
    req = DownloadRequest(url="https://fake.example/post/1", output_dir=tmp_path)

    with pytest.raises(errors.FFmpegMissing):
        run(req, registry, downloader)

    # Must fail before downloading anything — no wasted bandwidth.
    assert downloader.calls == []


def test_cleans_up_temp_files_when_merge_fails(tmp_path, monkeypatch):
    import haul.core.pipeline as pipeline

    monkeypatch.setattr(pipeline, "ffmpeg_available", lambda: True)

    class FailingMergeDownloader(FakeDownloader):
        def merge_audio_video(self, video_path, audio_path, output_path):
            # Simulate FFmpeg itself failing for some other reason,
            # after both streams were already downloaded to disk.
            assert video_path.exists()
            assert audio_path.exists()
            raise errors.DownloadError(detail="ffmpeg exited with a non-zero status")

    info = make_info(formats=[
        MediaFormat(
            url="https://cdn.example/video-only.mp4",
            extension="mp4",
            height=2160,
            has_audio=False,
            audio_url="https://cdn.example/audio-only.m4a",
            audio_extension="m4a",
        )
    ])
    registry = build_registry(info)
    downloader = FailingMergeDownloader()
    req = DownloadRequest(url="https://fake.example/post/1", output_dir=tmp_path)

    with pytest.raises(errors.DownloadError):
        run(req, registry, downloader)

    leftovers = list((tmp_path / "fake" / "creator").glob("*.video.*")) + list((tmp_path / "fake" / "creator").glob("*.audio.*"))
    assert leftovers == []
