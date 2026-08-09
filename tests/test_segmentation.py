"""Tests for tumor extraction from multi-segment DICOM SEG.

This is the highest-consequence untested code in the project. NSCLC-Radiomics
ships radiotherapy-planning segmentations that bundle 4-6 structures — Neoplasm
Primary, both lungs, spinal cord, esophagus, heart — as frames of ONE
multi-frame object, so a 134-slice CT gets a 536-frame SEG. Reading it the
obvious way (sitk.ReadImage, threshold >0) makes "tumor" mean the entire thorax,
and nothing raises. A model pretrained on that learns to attend everywhere, and
every grounding number downstream is meaningless while looking perfectly normal.

These tests use duck-typed stand-ins for the pydicom dataset and the SimpleITK
image, so they exercise the segment-selection and frame-matching logic — the
part that decides what counts as tumor — without needing either library or a
real DICOM file.
"""
import numpy as np
import pytest


class _Seq(list):
    """A DICOM sequence is just an indexable list of datasets."""


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeCT:
    """Only the two calls segment_mask actually makes."""

    def __init__(self, depth, z0=0.0, dz=2.5):
        self._depth, self._z0, self._dz = depth, z0, dz

    def GetSize(self):
        return (64, 64, self._depth)

    def TransformIndexToPhysicalPoint(self, idx):
        return (0.0, 0.0, self._z0 + self._dz * idx[2])


def _frame(seg_no, z):
    return _Obj(
        SegmentIdentificationSequence=_Seq([_Obj(ReferencedSegmentNumber=seg_no)]),
        PlanePositionSequence=_Seq([_Obj(ImagePositionPatient=[0.0, 0.0, z])]),
    )


def _seg_dataset(segments, frames, arr):
    """segments: {number: label}; frames: [(seg_no, z)]; arr: [F,H,W] uint8."""
    return _Obj(
        SegmentSequence=_Seq([_Obj(SegmentNumber=n, SegmentLabel=l)
                              for n, l in segments.items()]),
        PerFrameFunctionalGroupsSequence=_Seq([_frame(s, z) for s, z in frames]),
        pixel_array=arr,
    )


@pytest.fixture
def patch_dcmread(monkeypatch):
    def _install(ds):
        import pydicom
        monkeypatch.setattr(pydicom, "dcmread", lambda *a, **k: ds)
    return _install


def test_only_the_tumor_survives_a_multi_segment_planning_seg(patch_dcmread):
    """The exact NSCLC-Radiomics shape: tumor + lung + heart in one object.

    If this regresses, "tumor" silently becomes "thorax" — the failure that
    motivated the whole function.
    """
    from theia.data.preprocess import segment_mask

    depth, h, w = 4, 8, 8
    segments = {1: "Neoplasm, Primary", 2: "Lung-Left", 3: "Heart"}
    frames, planes = [], []
    for seg_no in (1, 2, 3):
        for k in range(depth):
            a = np.zeros((h, w), dtype=np.uint8)
            if seg_no == 1:
                a[2:4, 2:4] = 1          # small central tumor
            elif seg_no == 2:
                a[0:8, 0:2] = 1          # a whole lung field
            else:
                a[6:8, 6:8] = 1          # heart
            frames.append((seg_no, 2.5 * k))
            planes.append(a)
    ds = _seg_dataset(segments, frames, np.stack(planes))
    patch_dcmread(ds)

    mask = segment_mask("fake.dcm", _FakeCT(depth))

    assert mask.shape == (depth, h, w)
    expected = np.zeros((h, w), dtype=np.uint8)
    expected[2:4, 2:4] = 1
    for k in range(depth):
        assert np.array_equal(mask[k], expected), (
            f"slice {k} is not the tumor alone — non-tumor segments leaked in")
    # The whole point, stated as a number: 4 tumor voxels/slice, not 4+16+4=24.
    assert mask.sum() == 4 * depth


def test_single_segment_is_taken_as_tumor_whatever_it_is_called(patch_dcmread):
    """NSCLC-RADIOGENOMICS labels its only segment "3D Slicer segmentation
    result", which matches no anatomical keyword. Requiring a tumor keyword
    would reject the entire labelled cohort."""
    from theia.data.preprocess import segment_mask

    depth = 3
    a = np.zeros((3, 8, 8), dtype=np.uint8)
    a[:, 1:5, 1:5] = 1
    ds = _seg_dataset({1: "3D Slicer segmentation result"},
                      [(1, 2.5 * k) for k in range(depth)], a)
    patch_dcmread(ds)

    mask = segment_mask("fake.dcm", _FakeCT(depth))
    assert mask.sum() == 16 * depth


def test_frames_are_matched_by_position_not_by_index(patch_dcmread):
    """The SEG may be ordered per-segment and need not span every CT slice.

    Frames are given in reverse z order here; index-based matching would put
    every one of them on the wrong slice while raising nothing.
    """
    from theia.data.preprocess import segment_mask

    depth = 5
    zs = [2.5 * k for k in range(depth)]
    # Only slices 1 and 3 carry tumor, supplied in reverse order.
    frames = [(1, zs[3]), (1, zs[1])]
    planes = []
    for tag in (3, 1):
        a = np.zeros((8, 8), dtype=np.uint8)
        a[tag, tag] = 1            # a marker unique to its true slice
        planes.append(a)
    ds = _seg_dataset({1: "Neoplasm, Primary"}, frames, np.stack(planes))
    patch_dcmread(ds)

    mask = segment_mask("fake.dcm", _FakeCT(depth))
    assert mask[3][3, 3] == 1, "frame did not land on the slice its position names"
    assert mask[1][1, 1] == 1
    assert mask[0].sum() == 0 and mask[2].sum() == 0 and mask[4].sum() == 0


def test_a_seg_with_no_tumor_segment_raises_rather_than_guessing(patch_dcmread):
    """Better to skip the patient than to invent a tumor from a lung contour."""
    from theia.data.preprocess import segment_mask

    a = np.zeros((2, 8, 8), dtype=np.uint8)
    a[:, 0:4, 0:4] = 1
    ds = _seg_dataset({1: "Lung-Left", 2: "Esophagus"}, [(1, 0.0), (2, 2.5)], a)
    patch_dcmread(ds)

    with pytest.raises(ValueError, match="no tumor segment"):
        segment_mask("fake.dcm", _FakeCT(2))


def test_a_tumor_label_that_is_also_a_non_tumor_label_is_rejected(patch_dcmread):
    """"Tumor bed - lung" contains both a tumor keyword and a non-tumor one.

    The exclusion list must win, or a lung contour enters through its name.
    """
    from theia.data.preprocess import segment_mask

    a = np.zeros((1, 8, 8), dtype=np.uint8)
    a[:, 0:4, 0:4] = 1
    ds = _seg_dataset({1: "Tumor bed - lung", 2: "Spinal cord"},
                      [(1, 0.0)], a)
    patch_dcmread(ds)

    with pytest.raises(ValueError, match="no tumor segment"):
        segment_mask("fake.dcm", _FakeCT(1))


def test_frame_count_mismatch_raises(patch_dcmread):
    """Frames and functional groups must correspond one-to-one.

    If they do not, every position lookup is off by an unknown amount, so the
    mask would be wrong in a way no downstream check could detect.
    """
    from theia.data.preprocess import segment_mask

    a = np.zeros((3, 8, 8), dtype=np.uint8)
    ds = _seg_dataset({1: "Neoplasm, Primary"}, [(1, 0.0)], a)   # 3 frames, 1 group
    patch_dcmread(ds)

    with pytest.raises(ValueError, match="functional groups"):
        segment_mask("fake.dcm", _FakeCT(3))


def test_tumor_matching_no_ct_slice_raises(patch_dcmread):
    """A tumor segment whose frames land nowhere near the CT is a resolution
    failure, not an empty mask to be passed downstream."""
    from theia.data.preprocess import segment_mask

    a = np.zeros((1, 8, 8), dtype=np.uint8)
    a[0, 1:3, 1:3] = 1
    # Frame has no ReferencedSegmentNumber at all -> unmatchable.
    ds = _Obj(
        SegmentSequence=_Seq([_Obj(SegmentNumber=1, SegmentLabel="Neoplasm, Primary")]),
        PerFrameFunctionalGroupsSequence=_Seq([_Obj()]),
        pixel_array=a,
    )
    patch_dcmread(ds)

    with pytest.raises(ValueError, match="matched no CT slice"):
        segment_mask("fake.dcm", _FakeCT(3))
