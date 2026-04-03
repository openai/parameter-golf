from __future__ import annotations

import argparse
import unittest
from unittest.mock import patch

from tools.run_deepfloor_runpod import (
    append_lease_to_state,
    arm_lease,
    build_watch_command,
    cmd_delete,
    cmd_stop,
    cmd_watch,
    extract_pod_id,
    new_lease_record,
    summarize_lease_state,
)


class RunDeepFloorRunPodTests(unittest.TestCase):
    def test_new_lease_record_uses_requested_window(self) -> None:
        lease = new_lease_record(
            lease_minutes=15,
            owner="agent-a",
            reason="smoke",
            now_epoch=1_700_000_000,
            lease_id="lease-1",
        )
        self.assertEqual(lease["lease_id"], "lease-1")
        self.assertEqual(lease["owner"], "agent-a")
        self.assertEqual(lease["reason"], "smoke")
        self.assertEqual(lease["expires_at"], 1_700_000_900)

    def test_append_lease_to_state_keeps_overlapping_leases(self) -> None:
        first = new_lease_record(lease_minutes=30, owner="agent-a", reason="create", now_epoch=1_000, lease_id="a")
        second = new_lease_record(lease_minutes=45, owner="agent-b", reason="extend", now_epoch=1_100, lease_id="b")
        state = append_lease_to_state(
            None,
            pod_id="pod-123",
            pod_name="deepfloor",
            lease_record=first,
            watch_token="watch-a",
            now_epoch=1_000,
        )
        state = append_lease_to_state(
            state,
            pod_id="pod-123",
            pod_name="deepfloor",
            lease_record=second,
            watch_token="watch-b",
            now_epoch=1_100,
        )
        self.assertEqual(len(state["leases"]), 2)
        self.assertEqual(state["watch_token"], "watch-b")
        self.assertEqual(state["created_at"], 1_000)

    def test_summarize_lease_state_reports_active_count_and_latest_expiry(self) -> None:
        state = {
            "pod_id": "pod-123",
            "pod_name": "deepfloor",
            "leases": [
                {"lease_id": "expired", "owner": "agent-a", "expires_at": 900},
                {"lease_id": "active-a", "owner": "agent-a", "expires_at": 1_600},
                {"lease_id": "active-b", "owner": "agent-b", "expires_at": 1_900},
            ],
        }
        summary = summarize_lease_state(state, now_epoch=1_000)
        self.assertEqual(summary["active_lease_count"], 2)
        self.assertEqual(summary["total_lease_count"], 3)
        self.assertEqual(summary["next_expiry_epoch"], 1_900)
        self.assertEqual(summary["owners"], ["agent-a", "agent-b"])

    def test_build_watch_command_points_back_to_helper(self) -> None:
        cmd = build_watch_command("pod-123", "watch-token")
        self.assertEqual(cmd[-4:], ["watch", "pod-123", "--watch-token", "watch-token"])

    def test_extract_pod_id_handles_nested_runpod_payloads(self) -> None:
        payload = {"data": {"pod": {"id": "pod-abc", "name": "deepfloor-smallbox"}}}
        self.assertEqual(extract_pod_id(payload), "pod-abc")

    def test_arm_lease_saves_state_and_spawns_watchdog(self) -> None:
        with (
            patch("tools.run_deepfloor_runpod.epoch_now", return_value=1_000),
            patch("tools.run_deepfloor_runpod.load_lease_state", return_value=None),
            patch("tools.run_deepfloor_runpod.save_lease_state") as save_lease_state,
            patch("tools.run_deepfloor_runpod.spawn_watchdog") as spawn_watchdog,
            patch("tools.run_deepfloor_runpod.print"),
            patch("tools.run_deepfloor_runpod.uuid.uuid4") as uuid4,
        ):
            uuid4.return_value.hex = "watch-token"
            state = arm_lease(
                "pod-123",
                pod_name="deepfloor",
                lease_minutes=30,
                owner="agent-a",
                reason="create",
            )

        self.assertEqual(state["pod_id"], "pod-123")
        self.assertEqual(state["watch_token"], "watch-token")
        self.assertEqual(len(state["leases"]), 1)
        self.assertEqual(state["leases"][0]["owner"], "agent-a")
        self.assertEqual(state["leases"][0]["expires_at"], 2_800)
        save_lease_state.assert_called_once_with(state)
        spawn_watchdog.assert_called_once_with("pod-123", "watch-token")

    def test_cmd_watch_stops_and_clears_state_when_leases_expire(self) -> None:
        expired_state = {
            "pod_id": "pod-123",
            "pod_name": "deepfloor",
            "watch_token": "watch-token",
            "leases": [{"lease_id": "old", "owner": "agent-a", "expires_at": 900}],
        }
        args = argparse.Namespace(pod_id="pod-123", watch_token="watch-token")

        with (
            patch("tools.run_deepfloor_runpod.load_lease_state", return_value=expired_state),
            patch("tools.run_deepfloor_runpod.epoch_now", return_value=1_000),
            patch("tools.run_deepfloor_runpod.maybe_stop_expired_pod") as stop_pod,
            patch("tools.run_deepfloor_runpod.clear_lease_state") as clear_lease_state,
        ):
            cmd_watch(args)

        stop_pod.assert_called_once_with("pod-123")
        clear_lease_state.assert_called_once_with("pod-123")

    def test_cmd_stop_and_delete_clear_state_after_successful_lifecycle(self) -> None:
        for command, fn in (("stop", cmd_stop), ("delete", cmd_delete)):
            with self.subTest(command=command):
                events: list[str] = []
                args = argparse.Namespace(pod_id="pod-123")

                def record_passthrough(cmd: list[str]) -> None:
                    self.assertEqual(cmd, ["runpodctl", "pod", command, "pod-123"])
                    events.append(command)

                def record_clear(pod_id: str) -> None:
                    self.assertEqual(pod_id, "pod-123")
                    events.append("clear")

                with (
                    patch("tools.run_deepfloor_runpod.run_passthrough", side_effect=record_passthrough),
                    patch("tools.run_deepfloor_runpod.clear_lease_state", side_effect=record_clear),
                ):
                    fn(args)

                self.assertEqual(events, [command, "clear"])


if __name__ == "__main__":
    unittest.main()
