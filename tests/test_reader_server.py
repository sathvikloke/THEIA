"""Tests for the blinded scoring server.

Everything here protects data the reader cohort can only give once. A server
that drops a score, accepts a bad one, or lets a restart skip a case costs
clinician time that cannot be recovered by rerunning anything.
"""
import json
import os

import pytest

flask = pytest.importorskip("flask")


def _cases(tmp_path, n=3):
    pres = []
    for i in range(n):
        pres.append({"token": f"P{i:04d}",
                     "arm_payload": {"status": {"egfr": i % 2},
                                     "questions": ["useful"]}})
    d = tmp_path / "cases"
    d.mkdir(exist_ok=True)   # a restart reuses the same directory
    (d / "presentations.json").write_text(json.dumps(pres))
    (d / "genes.json").write_text(json.dumps(["EGFR"]))
    (d / "img").mkdir(exist_ok=True)
    return str(d), pres


def _client(tmp_path, reader="r1"):
    from theia.reader_study import serve

    cases, pres = _cases(tmp_path)
    out = str(tmp_path / f"{reader}.jsonl")
    serve.STATE.update(presentations=pres, out=out, reader=reader,
                       pos=serve._resume_position(out, pres), cases=cases)
    serve.app.config.update(TESTING=True)
    return serve.app.test_client(), pres, out, serve


def test_scores_are_persisted_and_advance_exactly_one_case(tmp_path):
    c, pres, out, serve = _client(tmp_path)
    assert c.get("/next").get_json()["token"] == "P0000"

    r = c.post("/score", json={"token": "P0000", "scores": {"useful": 4}})
    assert r.status_code == 200
    assert c.get("/next").get_json()["token"] == "P0001", "did not advance"

    lines = [json.loads(l) for l in open(out)]
    assert len(lines) == 1 and lines[0]["scores"] == {"useful": 4}
    assert lines[0]["reader"] == "r1", "reader identity not recorded"


def test_a_stale_token_is_rejected_rather_than_misfiled(tmp_path):
    """Two tabs, or a back button, must not write a score against the wrong case.

    Accepting it would silently attach a reader's judgement of case A to case B,
    which is unrecoverable and invisible in the output.
    """
    c, pres, out, serve = _client(tmp_path)
    r = c.post("/score", json={"token": "P0002", "scores": {"useful": 5}})
    assert r.status_code == 409
    assert not os.path.exists(out) or open(out).read() == ""
    assert c.get("/next").get_json()["token"] == "P0000", "position moved on a bad token"


def test_incomplete_or_out_of_range_scores_are_refused(tmp_path):
    c, pres, out, serve = _client(tmp_path)
    assert c.post("/score", json={"token": "P0000", "scores": {}}).status_code == 400
    assert c.post("/score", json={"token": "P0000",
                                  "scores": {"useful": 9}}).status_code == 400
    assert c.post("/score", json={"token": "P0000",
                                  "scores": {"useful": "4"}}).status_code == 400
    assert c.get("/next").get_json()["token"] == "P0000"


def test_a_restart_resumes_where_the_reader_stopped(tmp_path):
    """A reader who closes the tab must not be shown cases twice, or skipped past.

    Re-showing costs their time; skipping loses a case with no error.
    """
    c, pres, out, serve = _client(tmp_path)
    c.post("/score", json={"token": "P0000", "scores": {"useful": 3}})
    c.post("/score", json={"token": "P0001", "scores": {"useful": 2}})

    assert serve._resume_position(out, pres) == 2
    c2, _, _, serve2 = _client(tmp_path)          # same reader, fresh process
    assert c2.get("/next").get_json()["token"] == "P0002"


def test_the_run_ends_cleanly_and_refuses_extra_scores(tmp_path):
    c, pres, out, serve = _client(tmp_path)
    for p in pres:
        assert c.post("/score", json={"token": p["token"],
                                      "scores": {"useful": 4}}).status_code == 200
    assert c.get("/next").get_json().get("done") is True
    r = c.post("/score", json={"token": "P0000", "scores": {"useful": 4}})
    assert r.status_code == 400, "accepted a score after the study finished"
    assert len([l for l in open(out)]) == len(pres), "extra row written"


def test_served_payload_never_reveals_the_arm_or_the_patient(tmp_path):
    """Blinding has to hold at the wire, not just in the file on disk."""
    c, pres, out, serve = _client(tmp_path)
    body = c.get("/next").get_data(as_text=True)
    # "arm_payload" is a container name and reveals nothing; the arm's IDENTITY
    # and the patient are what must not cross the wire.
    for leak in ("theia_full", "label_only", "ground_truth", "patient_id"):
        assert leak not in body, f"{leak!r} leaked to the reader"
    assert '"arm"' not in body, "the arm field itself was served"

