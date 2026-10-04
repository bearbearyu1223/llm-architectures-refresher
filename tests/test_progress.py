"""Progress must stay on stderr: stdout is the receipt document."""

from llmrefresher import progress


def test_training_reports_on_stderr_only(capsys, monkeypatch):
    monkeypatch.delenv("LLMREFRESHER_QUIET", raising=False)
    on_step = progress.training("unit test run", 8)
    for step in range(8):
        on_step(step, 1.0 / (step + 1))
    out, err = capsys.readouterr()
    assert out == ""
    assert "unit test run, 8 steps" in err
    # Not a terminal under pytest, so one line per quarter of the run.
    assert [line.split()[1] for line in err.splitlines() if line.startswith("    step")] == [
        "2/8", "4/8", "6/8", "8/8"]


def test_quiet_silences_everything(capsys, monkeypatch):
    monkeypatch.setenv("LLMREFRESHER_QUIET", "1")
    on_step = progress.training("silent", 4)
    for step in range(4):
        on_step(step, 0.5)
    progress.status("nothing")
    assert capsys.readouterr() == ("", "")
