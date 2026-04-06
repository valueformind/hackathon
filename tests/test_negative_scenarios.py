"""Negative scenario tests for RL-style task/environment behavior."""

import unittest

from env.openenv_env import OpenEnv
from tasks.task import YourTask


class TestNegativeScenarios(unittest.TestCase):
    def setUp(self) -> None:
        self.task = YourTask()
        self.env = OpenEnv(task=self.task)
        self.env.reset()

    def test_invalid_action_format(self) -> None:
        step = self.env.step("run_tests")
        self.assertEqual(step["info"]["status"], "invalid_action_format")
        self.assertLess(step["reward"], 0)
        self.assertFalse(step["done"])

    def test_invalid_action_type(self) -> None:
        step = self.env.step({"action_type": "delete_repo", "payload": {}})
        self.assertEqual(step["info"]["status"], "invalid_action_type")
        self.assertLess(step["reward"], 0)
        self.assertFalse(step["done"])

    def test_invalid_generate_tests_payload_type(self) -> None:
        step = self.env.step({"action_type": "generate_tests", "payload": {"tests": "not-a-list"}})
        self.assertEqual(step["info"]["status"], "invalid_payload")
        self.assertLess(step["reward"], 0)
        self.assertFalse(step["done"])

    def test_invalid_modify_code_payload_type(self) -> None:
        step = self.env.step({"action_type": "modify_code", "payload": {"code": 123}})
        self.assertEqual(step["info"]["status"], "invalid_payload")
        self.assertLess(step["reward"], 0)
        self.assertFalse(step["done"])

    def test_premature_finish_is_incomplete(self) -> None:
        step = self.env.step({"action_type": "finish", "payload": {}})
        self.assertEqual(step["info"]["status"], "incomplete")
        self.assertTrue(step["done"])
        self.assertEqual(step["reward"], -1.0)

    def test_deadlock_signal_is_recorded(self) -> None:
        step = self.env.step(
            {
                "action_type": "run_tests",
                "payload": {
                    "passed": 0,
                    "failed": 1,
                    "coverage": 0.3,
                    "deadlock_detected": True,
                    "timeout_count": 1,
                    "flaky_rate": 0.0,
                },
            }
        )
        self.assertEqual(step["info"]["status"], "tests_ran")
        self.assertFalse(step["done"])

        last = step["state"]["last_test_result"]
        self.assertTrue(last["deadlock_detected"])
        self.assertTrue(last["found_bug"])

    def test_env_timeout_hard_fails_with_reason(self) -> None:
        step = self.env.step(
            {
                "action_type": "run_tests",
                "payload": {
                    "timeout_count": 1,
                    "timeout_source": "env",
                },
            }
        )
        self.assertEqual(step["info"]["status"], "env_setup_issue")
        self.assertTrue(step["done"])
        self.assertEqual(step["info"]["details"]["reason"], "Env setup issues")
        self.assertEqual(step["state"]["last_test_result"]["failure_reason"], "Env setup issues")


if __name__ == "__main__":
    unittest.main()
