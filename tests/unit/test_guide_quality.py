"""T006: measure codec drift on a fixed fixture, with an explicit edge exclusion."""

from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from features.asset.ingest import load_asset
from features.guidance.video import compile_guidance
from features.media.ingest import normalize_video
from features.placement.math import Placement


class GuideQualityTests(unittest.TestCase):
    def command(self, arguments):
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", *arguments],
                       capture_output=True, check=True, timeout=90)

    def test_rgb_psnr_and_ssim_on_all_frames_outside_edge_band(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.mp4"
            self.command(["-f", "lavfi", "-i", "color=0x204060:s=128x72:r=24:d=5",
                          "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", str(source)])
            video = normalize_video(source)
            normalized = root / "normalized.mp4"
            normalized.write_bytes(video.video_bytes)
            frame_path = root / "source-frame.png"
            self.command(["-i", str(normalized), "-frames:v", "1", "-threads", "2", str(frame_path)])
            with Image.open(frame_path) as frame:
                reference = frame.convert("RGB")

            sprite = Image.new("RGBA", (128, 128), (240, 120, 20, 128))
            sprite.putpixel((0, 0), (0, 0, 0, 0))
            asset_path = root / "asset.png"
            sprite.save(asset_path)
            asset = load_asset(asset_path, non_human=True)
            guidance = compile_guidance(video, asset, Placement(0.5, 0.5, 128 / 720))
            guide = root / "guide.mp4"
            guide.write_bytes(guidance.guide_mp4)

            # Independent pre-encode reference; sprite bounds are [576,704) x [296,424).
            for y in range(128):
                for x in range(128):
                    position = (576 + x, 296 + y)
                    alpha = sprite.getpixel((x, y))[3]
                    background = reference.getpixel(position)
                    pixel = tuple((alpha * color + (255 - alpha) * source + 127) // 255
                                  for color, source in zip((240, 120, 20), background))
                    reference.putpixel(position, pixel)
            with Image.open(BytesIO(guidance.keyframe_png)) as keyframe:
                self.assertEqual(reference.tobytes(), keyframe.tobytes())
            raw = root / "reference.rgb"
            with raw.open("wb") as file:
                for _ in range(120):
                    file.write(reference.tobytes())

            # These disjoint rectangles exclude a two-pixel band on each side of
            # the sprite's four edges. All retained pixels have 8-bit RGB channels.
            regions = [(574, 720, 0, 0), (574, 720, 706, 0),
                       (132, 294, 574, 0), (132, 294, 574, 426),
                       (124, 124, 578, 298)]
            report = {"frames": 120, "color": "8-bit RGB", "edgeExclusionPixels": 2,
                      "pixelsPerFrame": sum(width * height for width, height, _, _ in regions),
                      "regions": []}
            self.assertEqual(report["pixelsPerFrame"], 919_552)
            for index, (width, height, x, y) in enumerate(regions):
                psnr_path = root / f"psnr-{index}.txt"
                ssim_path = root / f"ssim-{index}.txt"
                crop = f"format=gbrp,crop={width}:{height}:{x}:{y},split=2"
                filters = (f"[0:v]{crop}[ref0][ref1];[1:v]{crop}[out0][out1];"
                           f"[out0][ref0]psnr=stats_file={psnr_path}[psnr];"
                           f"[out1][ref1]ssim=stats_file={ssim_path}[ssim]")
                self.command(["-threads", "2", "-f", "rawvideo", "-pixel_format", "rgb24",
                              "-video_size", "1280x720", "-framerate", "24", "-i", str(raw),
                              "-threads", "2", "-i", str(guide), "-filter_complex_threads", "2",
                              "-filter_complex", filters, "-map", "[psnr]", "-map", "[ssim]",
                              "-f", "null", "-"])
                psnr = [float(value) for value in re.findall(r"psnr_avg:([\d.]+|inf)", psnr_path.read_text())]
                ssim = [float(value) for value in re.findall(r"All:([\d.]+)", ssim_path.read_text())]
                self.assertEqual(len(psnr), 120)
                self.assertEqual(len(ssim), 120)
                self.assertGreaterEqual(min(psnr), 30)
                self.assertGreaterEqual(min(ssim), 0.95)
                report["regions"].append({"crop": [width, height, x, y],
                                          "minimumPsnrDb": min(psnr), "minimumSsim": min(ssim)})
            print("T006 codec fixture: " + json.dumps(report))
