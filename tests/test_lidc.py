"""Tests for the LIDC-IDRI annotation reader.

LIDC ships four independent radiologists per scan as XML polygon contours, not
DICOM SEG. The consensus rule is a scientific choice, not a formatting one: a
union mask trains attention onto whatever a single reader over-called, and
requiring all four discards the ambiguous periphery that defines a lesion. These
tests pin the rule so it cannot drift silently the way defaults in this project
have before.

Synthetic XML throughout — no LIDC download required.
"""
import numpy as np
import pytest

TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<LidcReadMessage xmlns="http://www.nih.gov">
{sessions}
</LidcReadMessage>"""

SESSION = """  <readingSession>
    <unblindedReadNodule>
      <roi>
        <imageSOP_UID>{uid}</imageSOP_UID>
        <inclusion>{inclusion}</inclusion>
{edges}
      </roi>
    </unblindedReadNodule>
  </readingSession>"""


def _square(x0, y0, side):
    pts = [(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side)]
    return "\n".join(
        f"        <edgeMap><xCoord>{x}</xCoord><yCoord>{y}</yCoord></edgeMap>"
        for x, y in pts)


def _xml(tmp_path, readers, uid="1.2.3", inclusion="TRUE"):
    """readers: list of (x0, y0, side) — one square per radiologist."""
    sessions = "\n".join(
        SESSION.format(uid=uid, inclusion=inclusion, edges=_square(*r)) for r in readers)
    p = tmp_path / "ann.xml"
    p.write_text(TEMPLATE.format(sessions=sessions))
    return str(p)


def test_parses_one_contour_per_reader(tmp_path):
    from theia.data.lidc import parse_xml

    path = _xml(tmp_path, [(2, 2, 6), (3, 3, 6), (2, 2, 6), (10, 10, 4)])
    per_slice = parse_xml(path)
    assert set(per_slice) == {"1.2.3"}
    readers = {r for r, _ in per_slice["1.2.3"]}
    assert readers == {0, 1, 2, 3}, "not every readingSession was read"


def test_consensus_threshold_controls_what_survives(tmp_path):
    """The same annotations must give a larger mask at a looser threshold.

    Three readers outline the same square; a fourth outlines somewhere else. At
    >=3 only the agreed square survives; at >=1 the outlier is included too.
    """
    from theia.data.lidc import consensus_mask, parse_xml

    path = _xml(tmp_path, [(2, 2, 6), (2, 2, 6), (2, 2, 6), (14, 14, 4)])
    per_slice = parse_xml(path)
    idx, shape = {"1.2.3": 0}, (24, 24)

    strict = consensus_mask(per_slice, idx, shape, depth=1, min_agreement=3)
    loose = consensus_mask(per_slice, idx, shape, depth=1, min_agreement=1)
    assert strict.sum() > 0, "the agreed region vanished at >=3"
    assert loose.sum() > strict.sum(), "a looser threshold did not include more"
    # The outlier region must be absent under the strict rule.
    assert strict[0, 15, 15] == 0
    assert loose[0, 15, 15] == 1


def test_all_four_agreement_is_stricter_still(tmp_path):
    from theia.data.lidc import consensus_mask, parse_xml

    per_slice = parse_xml(_xml(tmp_path, [(2, 2, 6), (2, 2, 6), (2, 2, 6), (9, 9, 3)]))
    idx, shape = {"1.2.3": 0}, (24, 24)
    three = consensus_mask(per_slice, idx, shape, 1, min_agreement=3).sum()
    four = consensus_mask(per_slice, idx, shape, 1, min_agreement=4).sum()
    assert four < three, "requiring all four readers did not shrink the mask"


def test_one_reader_drawing_twice_does_not_count_as_two(tmp_path):
    """Agreement counts READERS, not contours.

    A radiologist who outlines two adjacent nodules on one slice must not push a
    voxel over the consensus threshold alone -- that would manufacture agreement
    where there is none.
    """
    from theia.data.lidc import consensus_mask, parse_xml

    # One reader, two overlapping contours in the same readingSession.
    sessions = """  <readingSession>
    <unblindedReadNodule>
      <roi><imageSOP_UID>1.2.3</imageSOP_UID><inclusion>TRUE</inclusion>
""" + _square(2, 2, 6) + """
      </roi>
    </unblindedReadNodule>
    <unblindedReadNodule>
      <roi><imageSOP_UID>1.2.3</imageSOP_UID><inclusion>TRUE</inclusion>
""" + _square(3, 3, 6) + """
      </roi>
    </unblindedReadNodule>
  </readingSession>"""
    p = tmp_path / "one.xml"
    p.write_text(TEMPLATE.format(sessions=sessions))

    per_slice = parse_xml(str(p))
    m = consensus_mask(per_slice, {"1.2.3": 0}, (24, 24), 1, min_agreement=2)
    assert m.sum() == 0, "one reader's two contours were counted as two readers"


def test_excluded_regions_are_not_treated_as_tumour(tmp_path):
    """inclusion=FALSE marks tissue the reader carved OUT, e.g. a vessel."""
    from theia.data.lidc import consensus_mask, parse_xml

    path = _xml(tmp_path, [(2, 2, 6)] * 3, inclusion="FALSE")
    per_slice = parse_xml(path)
    m = consensus_mask(per_slice, {"1.2.3": 0}, (24, 24), 1, min_agreement=1)
    assert m.sum() == 0, "an excluded region was rasterised as tumour"


def test_unmatched_sop_uid_is_skipped_not_misplaced(tmp_path):
    """A contour whose slice is not in this volume must be dropped.

    Falling back to index 0 would stamp the mask onto an unrelated slice, which
    is the segmentation equivalent of pairing a mask with the wrong series.
    """
    from theia.data.lidc import consensus_mask, parse_xml

    per_slice = parse_xml(_xml(tmp_path, [(2, 2, 6)] * 3, uid="NOT-IN-VOLUME"))
    m = consensus_mask(per_slice, {"1.2.3": 0}, (24, 24), depth=3, min_agreement=1)
    assert m.sum() == 0, "a contour was placed on a slice it does not belong to"


def test_degenerate_contours_are_ignored(tmp_path):
    """Fewer than three points is not a polygon."""
    from theia.data.lidc import parse_xml

    sessions = """  <readingSession><unblindedReadNodule><roi>
      <imageSOP_UID>1.2.3</imageSOP_UID><inclusion>TRUE</inclusion>
      <edgeMap><xCoord>5</xCoord><yCoord>5</yCoord></edgeMap>
      <edgeMap><xCoord>6</xCoord><yCoord>6</yCoord></edgeMap>
    </roi></unblindedReadNodule></readingSession>"""
    p = tmp_path / "deg.xml"
    p.write_text(TEMPLATE.format(sessions=sessions))
    assert parse_xml(str(p)) == {}, "a two-point 'polygon' was accepted"


def test_summarise_reports_what_each_threshold_would_keep(tmp_path):
    from theia.data.lidc import summarise

    a = _xml(tmp_path, [(2, 2, 6)] * 4)
    b = (tmp_path / "b.xml")
    b.write_text(TEMPLATE.format(sessions="\n".join(
        SESSION.format(uid="9.9.9", inclusion="TRUE", edges=_square(2, 2, 6))
        for _ in range(2))))

    s = summarise([a, str(b)], min_agreement=3)
    assert s["scans"] == 2
    assert s["annotated_slices"] == 2
    assert s["by_readers"] == {2: 1, 4: 1}
    assert s["kept_at_threshold"] == 1, "threshold accounting is wrong"
    assert s["kept_frac"] == pytest.approx(0.5)
