"""Fleet tab: running per-camera counts, the /api/hub/fleet endpoint and the three-tab page."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

from server.collection_dashboard import collection_page
from server.fleet_stats import BUCKET_SECONDS, FleetStats, best_label, visit_totals
from server.raspberry_pi_server import create_app

BIRD = {"status": "complete", "is_bird": True, "timing_ms": {"total": 3000},
        "species": {"boxes": [{"top_k": [{"label": "House Sparrow", "probability": 0.9}]}]}}
EMPTY = {"status": "complete", "is_bird": False, "timing_ms": {"total": 1000}}
JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"


class Clock:
    def __init__(self, now: float = 1_800_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class FleetStatsTests(unittest.TestCase):
    def test_a_frame_is_counted_once_and_finalised_once(self):
        stats = FleetStats(None, clock=Clock())
        stats.observe("cam", "c1", {"status": "pending"})
        stats.observe("cam", "c1", BIRD)
        stats.observe("cam", "c1", BIRD)  # a repeated write must not double count
        stats.observe("cam", "c2", EMPTY)
        stats.observe("cam", "c3", {"status": "error", "error": "boom"})
        totals = stats.totals(0)["cam"]
        self.assertEqual((totals["frames"], totals["analysed"], totals["bird_frames"], totals["errors"]), (3, 2, 1, 1))
        self.assertEqual(totals["labels"], [("House Sparrow", 1)])
        self.assertEqual(totals["mean_processing_ms"], 2000)

    def test_since_selects_buckets_and_counts_survive_a_restart(self):
        clock = Clock()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fleet_stats.json"
            stats = FleetStats(path, clock=clock)
            stats.observe("cam", "yesterday", BIRD)
            clock.now += 86400
            stats.observe("cam", "today", EMPTY)
            stats.save_if_due(force=True)
            reloaded = FleetStats(path, clock=clock)
            self.assertEqual(reloaded.totals(clock.now - 3600)["cam"]["frames"], 1)
            self.assertEqual(reloaded.totals(0)["cam"]["frames"], 2)
            self.assertEqual(reloaded.earliest(), int((clock.now - 86400) // BUCKET_SECONDS) * BUCKET_SECONDS)

    def test_old_buckets_are_dropped_after_a_week(self):
        clock = Clock()
        stats = FleetStats(None, clock=clock)
        stats.observe("cam", "old", EMPTY)
        clock.now += 9 * 86400
        stats.observe("cam", "new", EMPTY)
        self.assertEqual(stats.totals(0)["cam"]["frames"], 1)

    def test_best_label_and_visit_totals(self):
        self.assertEqual(best_label(BIRD), "House Sparrow")
        self.assertIsNone(best_label(EMPTY))
        self.assertEqual(best_label({"is_bird": True, "species_top_k": [{"label": "Blue Jay"}]}), "Blue Jay")
        visits = [
            {"visit_id": "esp-a/c1", "end": "2026-11-22T15:00:00+00:00", "result": {"decision": "species"}},
            {"visit_id": "esp-a/c2", "end": "2026-11-22T16:00:00+00:00", "result": {"decision": "review"}},
            {"visit_id": "esp-a/c0", "end": "2026-11-21T16:00:00+00:00", "result": {"decision": "species"}},
            {"visit_id": "esp-b/c3", "end": "not a time"},
        ]
        since = datetime(2026, 11, 22, tzinfo=timezone.utc).timestamp()
        self.assertEqual(visit_totals(visits, since, lambda hardware: {"esp-a": "reg-a"}.get(hardware)),
                         {"reg-a": {"visits": 2, "auto": 1, "review": 1}})


class FleetEndpointTests(unittest.TestCase):
    def test_uploads_are_counted_and_unregistered_cameras_still_listed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            client = create_app(root).test_client()
            self.assertEqual(client.post("/upload", data=JPEG).status_code, 200)
            body = client.get("/api/hub/fleet").get_json()
            row = next(r for r in body["cameras"] if r["id"] == "unregistered")
            self.assertEqual(row["frames"], 1)
            self.assertFalse(row["registered"])
            for key in ("storage_free_bytes", "uptime_seconds", "queue", "models", "clock"):
                self.assertIn(key, body["hub"])
            self.assertIsNotNone(body["counted_since"])
            # The heartbeat summary keeps its own /api/fleet route.
            self.assertIn("cameras", client.get("/api/fleet").get_json())

    def test_since_is_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            client = create_app(Path(temporary)).test_client()
            for value in ("nonsense", "nan", "1e30"):
                with self.subTest(value=value):
                    response = client.get(f"/api/hub/fleet?since={value}")
                    self.assertEqual(response.status_code, 200)
                    self.assertLessEqual(response.get_json()["since"], response.get_json()["generated_at"])


class Ids(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.tabs: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if attrs.get("role") == "tab":
            self.tabs.append(attrs.get("data-tab"))


class PageTests(unittest.TestCase):
    def test_three_tabs_and_the_requested_opening_tab(self):
        parser = Ids()
        parser.feed(collection_page(remote=True))
        self.assertEqual(parser.tabs, ["cameras", "fleet", "settings"])
        self.assertEqual(len(parser.ids), len(set(parser.ids)))
        for removed in ("pane-live", "pane-review", 'id="metrics"', "__REMOTE__", "__INITIAL_TAB__"):
            self.assertNotIn(removed, collection_page())
        self.assertIn("INITIAL_TAB='settings'", collection_page(initial_tab="settings"))
        self.assertIn("INITIAL_TAB='cameras'", collection_page(initial_tab="live"))  # older callers

    def test_preset_lives_in_settings_and_fleet_uses_the_hub_endpoint(self):
        html = collection_page()
        settings = html[html.index('id="pane-settings"'):html.index("</section>", html.index('id="pane-settings"'))]
        self.assertIn('id="preset"', settings)
        self.assertIn("/api/hub/fleet?since=", html)
        self.assertNotIn("'/api/fleet", html)


if __name__ == "__main__":
    unittest.main()
