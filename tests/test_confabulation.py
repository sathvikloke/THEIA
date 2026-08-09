"""Tests for the reader-study gate.

The gate exists to catch the one failure a reader cannot: a fluent, confident
clinical sentence is indistinguishable from a correct one unless you hold the
source annotation. So the tests are mostly about what must NOT pass.
"""
import pytest

from theia.analysis.confabulation import audit, check_one, gate

SRC = "A spiculated round solid lesion, peripheral, with attachment to pleura."
# The actual output that motivated this module.
REAL_BAD = ("A 55-year-old woman was referred to our hospital for a solid nodule "
            "in the right lower lobe of the lung.")


def test_the_observed_confabulation_is_caught():
    r = check_one(REAL_BAD, SRC)
    assert not r["clean"]
    assert "age" in r["ungroundable"], "invented age not caught"
    assert "sex" in r["ungroundable"], "invented sex not caught"
    assert "history" in r["ungroundable"], "invented referral not caught"
    assert "laterality" in r["unsupported"], "invented laterality not caught"
    assert "lobe" in r["unsupported"], "invented lobe not caught"


def test_a_faithful_rationale_passes():
    good = ("A spiculated round solid lesion, peripheral, with attachment to "
            "the pleura.")
    r = check_one(good, SRC)
    assert r["clean"], r


def test_laterality_is_allowed_when_the_annotation_states_it():
    """The check must be against THIS patient's source, not a global ban.

    A blanket ban on 'right' would make the gate unusable on the many patients
    whose annotation does name a side, and a gate people switch off protects
    nothing.
    """
    src = "A lobulated solid lesion in the right upper lobe, peripheral."
    r = check_one("A lobulated solid lesion in the right upper lobe.", src)
    assert r["clean"], r
    # ...but the OTHER side is still invented.
    r2 = check_one("A lobulated solid lesion in the left upper lobe.", src)
    assert "laterality" in r2["unsupported"]


def test_fabricated_measurements_are_caught():
    """"5 cm" is not supported by "larger than 5 mm" — it is invented precision."""
    src = "A solid lesion, larger than 5 mm, peripheral."
    r = check_one("A large lesion, 5 cm in diameter, peripheral.", src)
    assert r["n_numeric"] == 1 and "5cm" in r["numeric"], r
    # The measurement the source actually states is fine.
    assert check_one("A solid lesion, larger than 5 mm.", src)["n_numeric"] == 0


def test_gate_defaults_to_zero_tolerance_for_ungroundable_claims():
    """There is no acceptable rate at which a model invents a patient's age.

    A small rate is worse than a large one: it survives spot-checking and reaches
    a reader.
    """
    one_bad = audit([(REAL_BAD, SRC)] + [("A solid lesion, peripheral.", SRC)] * 19)
    assert one_bad["ungroundable_rate"] == pytest.approx(0.05)
    ok, msg = gate(one_bad)
    assert not ok, "a 5% confabulation rate passed the gate"
    assert "reader study" in msg

    clean = audit([("A solid lesion, peripheral.", SRC)] * 20)
    ok2, _ = gate(clean)
    assert ok2


def test_empty_rationale_is_not_mistaken_for_a_clean_one():
    """An empty string trivially contains no invented claims.

    The generation head emitted exactly that for every patient before the
    sequence-start fix, so 'clean' must not be how the gate reports it. Callers
    check length separately; this pins that audit() does not crash and that the
    caller is not handed a misleading pass on zero cases.
    """
    r = check_one("", SRC)
    assert r["clean"], "empty text has no claims, so it is vacuously clean"
    assert audit([])["n"] == 0, "an empty audit must not divide by zero"


def test_specificity_is_stable_across_seeds():
    """A single random pairing is not enough to build a gate on.

    Measured on the real checkpoint, one draw gave a gap of +0.003 and another
    +0.032 — a gate on one draw would flip between PASS and FAIL by seed. The
    null is averaged over many pairings, so the answer must not move materially
    with the seed.
    """
    from theia.analysis.confabulation import specificity

    pairs = [(f"a solid lesion type{i%3}", f"a solid lesion type{i%3} peripheral")
             for i in range(30)]
    gaps = [specificity(pairs, n_perm=200, seed=s)["specificity_gap"] for s in (1, 2, 3)]
    assert max(gaps) - min(gaps) < 0.02, f"specificity gap moves with the seed: {gaps}"


def test_a_template_that_ignores_the_patient_fails_the_gate():
    """Honest but generic must still fail: the reader would score the template."""
    from theia.analysis.confabulation import audit, gate, specificity

    same = "a solid peripheral lesion with attachment to pleura"
    pairs = [(same, f"a {w} lesion, peripheral, with attachment to pleura")
             for w in ("spiculated", "lobulated", "round", "irregular") * 8]
    res, spec = audit(pairs), specificity(pairs, n_perm=200)
    assert res["ungroundable_rate"] == 0.0, "the text invents nothing"
    ok, msg = gate(res, spec=spec)
    assert not ok, "a patient-independent template passed the gate"
    assert "patient-specific" in msg


def test_a_genuinely_patient_specific_rationale_passes():
    from theia.analysis.confabulation import audit, gate, specificity

    kinds = ["spiculated round solid", "lobulated complex solid",
             "poorly defined ground glass", "irregular cavitary"]
    pairs = [(f"a {k} lesion, peripheral", f"a {k} lesion, peripheral, attached to pleura")
             for k in kinds * 8]
    ok, msg = gate(audit(pairs), spec=specificity(pairs, n_perm=200))
    assert ok, msg
