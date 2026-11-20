"""CLI smoke tests: run subcommands as subprocesses."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_cli(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "edge_detect_lab", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(PROJECT_ROOT),
        timeout=120,
    )


class TestCLI(unittest.TestCase):
    def test_canny_subcommand_writes_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "canny.png"
            proc = run_cli("canny", "--image", "square", "--size", "64",
                           "--out", str(out))
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)
            self.assertIn("canny", proc.stdout.lower())

    def test_blur_subcommand(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "blur.png"
            proc = run_cli("blur", "--image", "circle", "--size", "48",
                           "--sigma", "1.5", "--out", str(out))
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(out.exists())

    def test_compare_subcommand(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "compare.png"
            proc = run_cli("compare", "--image", "checkerboard", "--size", "64",
                           "--out", str(out))
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
