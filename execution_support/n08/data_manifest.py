"""T02/T03: the data identity every N08 consumer verifies (runner, acceptance, maps).

Contract: BW/input_repair/M1_SCHEMA.md (BW = revamp/campaign/nine_hour_corrected_inputs_2026-10-02).
No torch import: the runner uses this before any model exists.

    verify_manifest(manifest_path, rows_path, expected_version, root, acceptance_path=None)
        -> DataIdentity dict, or raises DataRefused naming every problem.

A consumer is configured with an explicit manifest + rows path + dataset version. It
refuses when the rows file it is about to read is not the manifest's (sha256), when
any referenced npz differs in bytes from the manifest entry (full re-hash), when rows
and entries are not 1:1 by patient_id AND npz path, when the version differs from the
configured one, or (non-fixture runs) when the supervisor's acceptance record is not
`accepted` for the same manifest_body_sha256. Matching patient ids never suffice.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_KEYS = ("dataset_version", "parent_version", "rows_path", "rows_sha256", "n_rows", "entries",
               "preprocessing", "manifest_body_sha256")
ENTRY_KEYS = ("patient_id", "npz_path", "npz_sha256", "npz_bytes", "has_mask")


class DataRefused(RuntimeError):
    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = list(problems)


def canon_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(p: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest(), p.stat().st_size


def norm_rel(p: str) -> str:
    """'./data/processed/X.npz' and 'data/processed/X.npz' name the same repo-relative file."""
    return os.path.normpath(p)


def rel_or_abs(p, root) -> str:
    """Repo-relative when under root, else absolute (fixtures live in $TMPDIR)."""
    try:
        return norm_rel(str(Path(p).resolve().relative_to(Path(root).resolve())))
    except ValueError:
        return str(Path(p).resolve())


def body_sha(manifest: dict) -> str:
    return canon_sha({k: v for k, v in manifest.items() if k != "manifest_body_sha256"})


def annotation_preprocessing_sha(manifest: dict) -> str:
    """What the grounding targets and images are derived from, beyond the npz bytes."""
    return canon_sha({"preprocessing": manifest.get("preprocessing"),
                      "entries": [[e.get("patient_id"), e.get("has_mask"), e.get("m1_status"), e.get("source")]
                                  for e in manifest.get("entries", [])]})


def is_fixture(version: str) -> bool:
    return version.endswith("-FIXTURE")


def verify_manifest(manifest_path, rows_path, expected_version: str, root, acceptance_path=None,
                    rehash: bool = True) -> dict:
    root = Path(root)
    mp, rp = Path(manifest_path), Path(rows_path)
    probs: list[str] = []
    if not mp.exists():
        raise DataRefused([f"manifest missing: {mp}"])
    if not rp.exists():
        raise DataRefused([f"rows file missing: {rp}"])
    m = json.loads(mp.read_text())
    missing = [k for k in SCHEMA_KEYS if k not in m]
    if missing:
        raise DataRefused([f"manifest lacks keys {missing}"])
    if body_sha(m) != m["manifest_body_sha256"]:
        probs.append("manifest_body_sha256 does not match the manifest body")
    if m["dataset_version"] != expected_version:
        probs.append(f"dataset_version {m['dataset_version']!r} != configured {expected_version!r}")
    rows_bytes = rp.read_bytes()
    rows_sha = hashlib.sha256(rows_bytes).hexdigest()
    if rows_sha != m["rows_sha256"]:
        probs.append(f"rows file {rp} sha256 {rows_sha[:12]} != manifest rows_sha256 {m['rows_sha256'][:12]}")
    rel = rel_or_abs(rp, root)
    if (root / m["rows_path"]).resolve() != rp.resolve():
        probs.append(f"rows file read ({rel}) is not the manifest rows_path ({m['rows_path']})")
    rows = [json.loads(l) for l in rows_bytes.decode().splitlines() if l.strip()]
    ents = m["entries"]
    if not (len(rows) == len(ents) == m["n_rows"]):
        probs.append(f"counts differ: rows {len(rows)}, entries {len(ents)}, n_rows {m['n_rows']}")
    by_pid = {}
    for e in ents:
        if any(k not in e for k in ENTRY_KEYS):
            probs.append(f"entry lacks keys: {e.get('patient_id')}")
            continue
        if e["patient_id"] in by_pid:
            probs.append(f"duplicate manifest entry {e['patient_id']}")
        by_pid[e["patient_id"]] = e
        if str(e.get("m1_status", "")).startswith("unresolved_"):
            probs.append(f"{e['patient_id']}: blocking m1_status {e['m1_status']}")
    seen = set()
    triples = []
    for r in rows:
        pid = r.get("patient_id")
        if pid in seen:
            probs.append(f"duplicate row {pid}")
        seen.add(pid)
        e = by_pid.get(pid)
        if e is None:
            probs.append(f"row {pid} has no manifest entry")
            continue
        if norm_rel(r["npz"]) != norm_rel(e["npz_path"]):
            probs.append(f"{pid}: row npz {r['npz']} != manifest npz_path {e['npz_path']}")
        if bool(r.get("has_mask", True)) != bool(e["has_mask"]):
            probs.append(f"{pid}: has_mask row {r.get('has_mask')} != manifest {e['has_mask']}")
        triples.append([pid, norm_rel(e["npz_path"]), e["npz_sha256"]])
        if rehash:
            p = root / norm_rel(r["npz"])
            if not p.exists():
                probs.append(f"{pid}: npz missing {p}")
                continue
            s, n = file_sha(p)
            if s != e["npz_sha256"] or n != e["npz_bytes"]:
                probs.append(f"{pid}: npz bytes differ from manifest ({s[:12]}/{n} vs "
                             f"{e['npz_sha256'][:12]}/{e['npz_bytes']})")
    extra = set(by_pid) - seen
    if extra:
        probs.append(f"manifest entries without rows: {sorted(extra)[:5]}")
    acc = None
    if not is_fixture(expected_version) or acceptance_path is not None:
        if acceptance_path is None or not Path(acceptance_path).exists():
            probs.append(f"acceptance record missing ({acceptance_path}); {expected_version} not released")
        else:
            acc = json.loads(Path(acceptance_path).read_text())
            if acc.get("decision") != "accepted":
                probs.append(f"acceptance decision is {acc.get('decision')!r}, not 'accepted'")
            if acc.get("manifest_body_sha256") != m["manifest_body_sha256"]:
                probs.append("acceptance record names a different manifest_body_sha256")
    if probs:
        raise DataRefused(probs)
    return {"dataset_version": m["dataset_version"], "manifest_path": rel_or_abs(mp, root),
            "manifest_body_sha256": m["manifest_body_sha256"], "rows_path": rel, "rows_sha256": rows_sha,
            "n_entries": len(ents), "npz_all_sha256": canon_sha(triples),
            "annotation_preprocessing_sha256": annotation_preprocessing_sha(m),
            "acceptance": None if acc is None else {"decision": acc.get("decision"),
                                                    "sha256": file_sha(Path(acceptance_path))[0]},
            "npz_rehashed": bool(rehash),
            "verified_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def rows_labels(rows_path) -> dict:
    out = {}
    for line in Path(rows_path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            lab = {str(k).lower(): v for k, v in r["labels"].items()}
            out[r["patient_id"]] = {"egfr": int(lab.get("egfr", -1)), "kras": int(lab.get("kras", -1)),
                                    "has_mask": bool(r.get("has_mask", True))}
    return out


def partition_invariance(rows_path, splits: dict) -> list[str]:
    """A15 precondition, torch-free: the split function's inputs (row order, patient ids,
    stratification labels) are the ones SPLITS.json was frozen on, so the trainer's own
    nested_kfold_indices returns the frozen memberships (acceptance then checks the
    trainer's split component per fit). Returns problems."""
    probs = []
    rows = [json.loads(l) for l in Path(rows_path).read_text().splitlines() if l.strip()]
    ids = [r["patient_id"] for r in rows]
    if len(rows) != splits["rows_file"]["n_rows"]:
        probs.append(f"n_rows {len(rows)} != frozen {splits['rows_file']['n_rows']}")
    for f in splits["folds"]:
        for part in ("inner_train", "inner_val", "outer_test"):
            p = f[part]
            got = [ids[i] if i < len(ids) else None for i in p["indices"]]
            if sorted(got, key=str) != sorted(p["patient_ids"]):
                probs.append(f"fold {f['fold']} {part}: index->patient mapping differs")
                continue
            lab = [int({k.lower(): v for k, v in rows[i]["labels"].items()}.get("egfr", -1)) for i in p["indices"]]
            if sum(1 for x in lab if x in (0, 1)) != p["egfr_n"] or sum(1 for x in lab if x == 1) != p["egfr_n_pos"]:
                probs.append(f"fold {f['fold']} {part}: EGFR label counts differ from the frozen split")
    return probs


# ------------------------------------------------------------------ fixtures (tests only)
def build_from_ledger(ledger_path, out_dir, version: str, rows_src, npz_override: dict | None = None,
                      root=None, write_acceptance: bool = True, decision: str = "accepted") -> dict:
    """Disposable manifest in the M1 schema built from BW/supervision/M0_LEDGER.json.
    Tests only: never writes into BW/input_repair. `npz_override` {pid: repo-rel path} points
    rows/entries at substituted files (hashes taken from those files)."""
    root = Path(root) if root else Path(ledger_path).resolve().parents[4]
    led = json.loads(Path(ledger_path).read_text())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_rows = [json.loads(l) for l in Path(rows_src).read_text().splitlines() if l.strip()]
    ov = npz_override or {}
    rows_out, ents = [], []
    by = {e["patient_id"]: e for e in led["entries"]}
    for r in src_rows:
        e = by[r["patient_id"]]
        path = ov.get(r["patient_id"], e["npz_path"])
        if r["patient_id"] in ov:
            s, n = file_sha(root / path)
        else:
            s, n = e["npz_sha256"], e["npz_bytes"]
        rows_out.append({**r, "npz": path})
        ents.append({"patient_id": r["patient_id"], "npz_path": path, "npz_sha256": s, "npz_bytes": n,
                     "has_mask": bool(r.get("has_mask", True)),
                     "m1_status": "fixture_changed" if r["patient_id"] in ov else "fixture_unchanged",
                     "changed_from_m0": r["patient_id"] in ov,
                     "source": {"study_uid": None, "series_uid": None, "sop_uid": None, "aim_sha256": None}})
    rows_p = out_dir / "rows.jsonl"
    rows_p.write_text("".join(json.dumps(r) + "\n" for r in rows_out))
    rows_rel = rel_or_abs(rows_p, root)
    m = {"dataset_version": version, "parent_version": "M0", "rows_path": rows_rel,
         "rows_sha256": file_sha(rows_p)[0], "n_rows": len(rows_out), "entries": ents,
         "preprocessing": {"config_path": "configs/default.yaml", "config_sha256": led["files_sha256"].get("configs/default.yaml"),
                           "code_sha256": {k: v for k, v in led["files_sha256"].items() if k.startswith("theia/")}},
         "fixture_note": "disposable test fixture built from M0_LEDGER.json; not M1"}
    m["manifest_body_sha256"] = body_sha(m)
    (out_dir / "CONTENT_MANIFEST.json").write_text(json.dumps(m, indent=1) + "\n")
    if write_acceptance:
        (out_dir / "ACCEPTANCE.json").write_text(json.dumps({"decision": decision,
                                                             "manifest_body_sha256": m["manifest_body_sha256"],
                                                             "note": "fixture"}) + "\n")
    return m
