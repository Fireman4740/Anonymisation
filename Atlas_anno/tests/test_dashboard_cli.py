from __future__ import annotations

import unittest
from unittest.mock import patch

from atlas_anno.cli import main


class DashboardCliTest(unittest.TestCase):
    def test_dashboard_command_delegates_to_streamlit_runner(self) -> None:
        with patch("atlas_anno.cli.run_dashboard_command") as run_dashboard:
            exit_code = main(["dashboard", "--batch", "pilot_100", "--strategy", "masking"])

        self.assertEqual(exit_code, 0)
        run_dashboard.assert_called_once_with(batch="pilot_100", strategy="masking")


if __name__ == "__main__":
    unittest.main()
