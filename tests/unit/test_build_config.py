"""Check the canonical settings before application work starts."""

import json
from pathlib import Path
import unittest


CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "build_config.json"


class BuildConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_fixture_has_the_required_media_format(self):
        self.assertEqual(self.config["fixture"], {
            "clipSeconds": 5,
            "inputSecondsRange": [5, 20],
            "trim": "first_five_seconds",
            "width": 1280,
            "height": 720,
            "fps": 24,
            "frames": 120,
            "codec": "h264",
            "pixFmt": "yuv420p",
            "audio": False,
            "orientation": "landscape_letterbox_portrait",
        })

    def test_scope_requires_static_non_human_placement(self):
        self.assertEqual(self.config["schemaVersion"], "1.4")
        self.assertEqual(self.config["date"], "2026-11-01")
        self.assertEqual(self.config["scope"], "fixed_camera_feasibility_non_human")
        self.assertEqual(self.config["asset"], {
            "mime": "image/png", "channels": 4, "nonHuman": True,
            "alphaRequired": True, "automaticMatte": False,
        })
        self.assertEqual(self.config["placement"], {
            "coordinates": ["u", "v", "heightFrac"],
            "rotationDeg": 0, "motionKeyframes": False,
        })

    def test_provider_settings_keep_submission_unverified(self):
        sdk = self.config["sdk"]
        self.assertEqual(sdk["language"], "python")
        self.assertEqual(sdk["package"], "runwayml")
        self.assertEqual(sdk["version"], "PIN_EXACT_VERIFIED_RELEASE_BEFORE_PAID_POST")
        self.assertEqual(sdk["createMethod"], "video_to_video.create")
        self.assertEqual(sdk["uploadMethod"], "uploads.create_ephemeral")
        self.assertEqual(sdk["model"], "aleph2")
        self.assertEqual(sdk["route"], "POST /v1/video_to_video")
        self.assertEqual(sdk["createMaxRetries"], 0)
        self.assertIs(sdk["pollRetriesAllowed"], True)
        self.assertIs(sdk["maskParameter"], False)
        self.assertEqual(sdk["modeA"], {
            "keyframes": [{"uri": "<keyframe_runway_uri>", "seconds": 0}],
        })
        self.assertEqual(sdk["modeB"], {"keyframesField": "OMITTED"})

    def test_budget_requires_server_approval_and_price_check(self):
        self.assertEqual(self.config["paidBudget"], {
            "currency": "USD", "maxUsd": "10.00", "maxAttempts": 6,
            "maxElapsedMinutes": 180,
            "illustrativeAlephRateUsdPerOutputSecond": "0.28",
            "sourcePricing": "https://docs.dev.runwayml.com/guides/pricing/",
            "priceMustBeReverified": True,
            "authorization": "server-side approved run budget, not client boolean",
        })

    def test_review_requires_human_signoff_and_separate_outcomes(self):
        self.assertEqual(self.config["review"], {
            "p0Views": ["source", "matching_guide", "raw_generation"],
            "p1OptionalFourthView": "preserved_composite",
            "requiredDimensions": ["identity", "motion", "scene", "placement"],
            "minimumPerDimension": 3, "normalSpeedPlays": 2,
            "matchedTimestampsSec": [0, 1, 2, 3, 4],
            "humanReviewerRequired": True,
        })
        self.assertEqual(self.config["outcomes"], [
            "visualEffectDemonstrated", "automatedWorkflowComplete",
        ])
        self.assertEqual(self.config["verdicts"], ["PASS", "FAIL", "BLOCKED", "NOT_RUN"])
