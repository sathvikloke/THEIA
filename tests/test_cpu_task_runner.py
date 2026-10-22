"""Small real subprocess cases; these are execution plumbing, not study evidence."""
import datetime as dt
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_cpu_task.py"


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.session = self.root / "session.json"
        self.write_window(20, 30)
        self.source = self.root / "job.py"

    def write_window(self, cutoff, deadline):
        now = dt.datetime.now(dt.timezone.utc)
        self.session.write_text(json.dumps({
            "admission_cutoff_utc": (now + dt.timedelta(seconds=cutoff)).isoformat(),
            "deadline_utc": (now + dt.timedelta(seconds=deadline)).isoformat(),
        }))

    def run_job(self, extra=()):
        return subprocess.run([
            sys.executable, str(SCRIPT), "--session", str(self.session),
            "--attempt-root", str(self.root), "--cwd", str(self.root),
            "--task-id", "E05", "--job-key", "fixture", "--source", str(self.source),
            *extra, "--", sys.executable, str(self.source),
        ], capture_output=True, text=True, timeout=10)

    def receipts(self):
        return [json.loads(p.read_text()) for p in sorted(self.root.glob("E05_*/EXIT.json"))]

    def test_failure_is_retained(self):
        self.source.write_text("print('retained failure')\nraise SystemExit(7)\n")
        result = self.run_job()
        self.assertEqual(result.returncode, 7, result.stderr)
        receipt = self.receipts()[0]
        self.assertEqual(receipt["exit_code"], 7)
        self.assertEqual(receipt["status"], "incomplete_or_failed")
        self.assertIn("retained failure", next(self.root.glob("E05_*/stdout.log")).read_text())

    def test_resume_same_source_keeps_attempts_and_refuses_mutation(self):
        self.source.write_text("from pathlib import Path\np=Path('counter');p.write_text(str(int(p.read_text())+1) if p.exists() else '1')\n")
        self.assertEqual(self.run_job().returncode, 0)
        prior = next(self.root.glob("E05_*/START.json"))
        self.assertEqual(self.run_job(("--resume-receipt", str(prior))).returncode, 0)
        self.assertEqual((self.root / "counter").read_text(), "2")
        self.source.write_text("raise SystemExit(0)\n")
        refusal = self.run_job(("--resume-receipt", str(prior)))
        self.assertEqual(refusal.returncode, 2)
        self.assertIn("fingerprint changed", refusal.stderr)
        self.assertEqual(len(self.receipts()), 2)

    def test_deadline_interrupt_and_resume_checkpoint(self):
        self.source.write_text(
            "import signal,time\nfrom pathlib import Path\n"
            "p=Path('checkpoint')\n"
            "def stop(*_):\n p.write_text('saved');raise SystemExit(0)\n"
            "signal.signal(signal.SIGINT,stop)\n"
            "if not p.exists():\n while True: time.sleep(.05)\n"
            "else: Path('resumed').write_text(p.read_text())\n"
        )
        self.write_window(1.2, 1.8)
        interrupted = self.run_job(("--stop-grace-seconds", "0.5"))
        self.assertEqual(interrupted.returncode, 1, interrupted.stderr)
        self.assertEqual(self.receipts()[0]["stop_reason"], "fixed_deadline")
        self.assertEqual((self.root / "checkpoint").read_text(), "saved")
        prior = next(self.root.glob("E05_*/START.json"))
        self.write_window(20, 30)
        resumed = self.run_job(("--resume-receipt", str(prior)))
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual((self.root / "resumed").read_text(), "saved")
        self.assertEqual(len(self.receipts()), 2)

    def test_expired_window_does_not_launch(self):
        self.source.write_text("raise RuntimeError('must not run')\n")
        self.write_window(-2, -1)
        self.assertEqual(self.run_job().returncode, 2)
        self.assertEqual(self.receipts(), [])

    def test_missing_declared_output_is_failure(self):
        self.source.write_text("pass\n")
        result = self.run_job(("--output", str(self.root / "absent")))
        self.assertEqual(result.returncode, 1)
        self.assertTrue(self.receipts()[0]["output_errors"])

    def test_background_descendant_is_stopped(self):
        self.source.write_text(
            "import subprocess,sys\n"
            "subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'])\n"
        )
        result = self.run_job()
        self.assertEqual(result.returncode, 1, result.stderr)
        receipt = self.receipts()[0]
        self.assertEqual(receipt["stop_reason"], "remaining_descendants")
        self.assertTrue(receipt["signals"])

    def test_foreign_job_resume_is_refused(self):
        self.source.write_text("pass\n")
        self.assertEqual(self.run_job().returncode, 0)
        prior = next(self.root.glob("E05_*/START.json"))
        result = self.run_job(("--resume-receipt", str(prior), "--job-key", "other-job"))
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(len(self.receipts()), 1)

    def test_slow_receipt_write_cannot_launch_after_deadline(self):
        spec = importlib.util.spec_from_file_location("bounded_runner_fixture", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.source.write_text("from pathlib import Path\nPath('should-not-exist').touch()\n")
        self.write_window(0.2, 0.4)
        original_save = module.save_new

        def slow_save(path, value):
            original_save(path, value)
            if path.name == "START.json":
                time.sleep(0.5)

        args = SimpleNamespace(
            command=[sys.executable, str(self.source)], task_id="E05", job_key="fixture",
            stop_grace_seconds=0.1, session=self.session, cwd=self.root,
            attempt_root=self.root, source=[self.source], input=[], output=[],
            resume_receipt=None,
        )
        with mock.patch.object(module, "save_new", side_effect=slow_save):
            self.assertEqual(module.main(args), 1)
        self.assertFalse((self.root / "should-not-exist").exists())
        self.assertIsNone(self.receipts()[0]["exit_code"])
        self.assertIn("immediately before launch", self.receipts()[0]["runner_error"])


if __name__ == "__main__":
    unittest.main()
