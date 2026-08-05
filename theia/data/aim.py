"""AIM (Annotation and Image Markup) parsing for NSCLC-RADIOGENOMICS.

The collection ships `AIM_files_updated-11-10-2020.zip`: one AIMv4 XML per
patient, covering **190 patients** — 46 more than have a pixel-level SEG. Each
file carries two things THEIA needs and was previously throwing away.

1. The controlled-vocabulary semantic annotation
   ---------------------------------------------
   `build_pseudo_report` used to read `Surface`, `Density`, `Location`,
   `SizeCategory`, `PleuralAttachment` and `VascularConvergence` off the clinical
   spreadsheet. **None of those columns exist in it.** The sheet is demographics,
   staging, treatment and outcome — the semantic annotations live here, in the
   AIM XML. So every patient fell through to the same hard-coded defaults and the
   generation head was trained on ~190 copies of one identical sentence. Its loss
   would drop to near zero and look excellent while the head learned nothing but
   a constant. Nothing errored.

   The real vocabulary is far richer than the six fields that were being faked:
   anatomic location, attenuation, margin pattern (primary and secondary), shape,
   calcification, associated findings, emphysema, and more, coded against RadLex.

2. Lesion markup
   -------------
   A `TwoDimensionCircle` with a centre coordinate, the SOP instance UID of the
   slice it sits on, and a frame number. That is a *location*, not an extent —
   the circles are only a couple of pixels across, so they mark where the lesion
   is rather than tracing it. Enough to centre a crop, not enough to supervise
   grounding. Patients with AIM but no SEG can therefore train the classifier and
   the generator, with the grounding term masked off for them.
"""
from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

# AIM v4 default namespace.
NS = "{gme://caCORE.caCORE/4.4/edu.northwestern.radiology.AIM}"


def _tag(el) -> str:
    return el.tag.split("}")[-1]


def _val(el, attr: str = "value"):
    return el.get(attr) if el is not None else None


@dataclass
class Markup:
    image_uid: str | None = None
    frame: int | None = None
    cx: float | None = None
    cy: float | None = None
    radius: float = 0.0


@dataclass
class AimAnnotation:
    patient_id: str
    semantics: dict[str, str] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    markups: list[Markup] = field(default_factory=list)
    study_uid: str | None = None

    @property
    def has_location(self) -> bool:
        return any(m.cx is not None and m.frame is not None for m in self.markups)


# Characteristic labels that can legitimately repeat (a lesion has several).
MULTI = {"Nodule Associated Findings", "Lung Parencyma Features"}


def parse_aim(path: str) -> AimAnnotation:
    """Parse one AIM XML into semantics + markups."""
    root = ET.parse(path).getroot()
    ann = AimAnnotation(patient_id=os.path.splitext(os.path.basename(path))[0])

    for el in root.iter():
        name = _tag(el)

        if name == "ImagingObservationCharacteristic":
            label = None
            value = None
            for child in el:
                if _tag(child) == "label":
                    label = _val(child)
                elif _tag(child) == "typeCode":
                    # The human-readable term sits in codeSystem, oddly, with the
                    # RadLex id in code. Prefer codeSystem, fall back to code.
                    value = (child.get("codeSystem") or child.get("code") or "").strip()
            if label and value:
                if label in MULTI:
                    ann.findings.append(value)
                ann.semantics.setdefault(label, value)

        elif name == "MarkupEntity":
            m = Markup()
            coords: list[tuple[int, float, float]] = []
            for child in el.iter():
                c = _tag(child)
                if c == "imageReferenceUid":
                    m.image_uid = child.get("root")
                elif c == "referencedFrameNumber":
                    try:
                        m.frame = int(float(_val(child)))
                    except (TypeError, ValueError):
                        pass
                elif c == "TwoDimensionSpatialCoordinate":
                    idx = x = y = None
                    for cc in child:
                        t = _tag(cc)
                        if t == "coordinateIndex":
                            idx = int(float(_val(cc) or 0))
                        elif t == "x":
                            x = float(_val(cc))
                        elif t == "y":
                            y = float(_val(cc))
                    if x is not None and y is not None:
                        coords.append((idx or 0, x, y))
            if coords:
                coords.sort()
                m.cx, m.cy = coords[0][1], coords[0][2]
                if len(coords) > 1:
                    m.radius = math.dist((coords[0][1], coords[0][2]),
                                         (coords[1][1], coords[1][2]))
            if m.cx is not None:
                ann.markups.append(m)

    return ann


def load_all(aim_dir: str) -> dict[str, AimAnnotation]:
    """Parse every *.xml in a directory, keyed by patient id."""
    out: dict[str, AimAnnotation] = {}
    for fn in sorted(os.listdir(aim_dir)):
        if not fn.lower().endswith(".xml"):
            continue
        try:
            a = parse_aim(os.path.join(aim_dir, fn))
            out[a.patient_id] = a
        except ET.ParseError as exc:
            print(f"[aim] skip {fn}: {exc}")
    return out


# Fields used to compose the rationale, in reading order.
REPORT_FIELDS = [
    ("Lung Nodule", "lesion"),
    ("Nodule Attenuation", None),
    ("Nodule Margins-Primary Pattern", None),
    ("Nodule Shape", None),
    ("Anatomic Location", None),
    ("Axial Location", None),
]


import re

# Size qualifiers arrive fused onto other fields, e.g. "Solid lessOrEqual 5mm".
# They read as a clause about the lesion, not an adjective in front of it, so
# build_report strips them out here and re-attaches them after the noun.
_SIZE_RE = [
    (re.compile(r"\bless\s*or\s*equal\s*(\d+)\s*(mm|cm)\b", re.I), r"\1 \2 or smaller"),
    (re.compile(r"\bgreater\s*than\s*(\d+)\s*(mm|cm)\b", re.I), r"larger than \1 \2"),
]


def split_size(term: str) -> tuple[str, str]:
    """Return (descriptor, size_clause); size_clause is '' when absent."""
    for rx, rep in _SIZE_RE:
        m = rx.search(term)
        if m:
            return rx.sub("", term).strip(), rx.sub(rep, m.group(0)).strip()
    return term, ""


def normalize_term(raw: str) -> str:
    """RadLex codeSystem strings arrive raw: mixed case, camelCase, stray spaces.

    Left alone they leak into the supervision signal as 'complex Solid
    lessOrEqual 5mm', which is not language and would teach the LM to emit
    tokenizer noise.
    """
    t = " ".join((raw or "").split()).strip()
    if not t:
        return ""
    # split camelCase that carries no capitalised proper noun
    out, prev = [], ""
    for ch in t:
        if ch.isupper() and prev and prev.islower():
            out.append(" ")
        out.append(ch)
        prev = ch
    return " ".join("".join(out).lower().split())


def _article(word: str) -> str:
    return "An" if word[:1].lower() in "aeiou" else "A"


def build_report(ann: AimAnnotation) -> str:
    """Compose a one-sentence rationale from the real semantic annotation.

    Only states what the annotation actually records. A missing field is omitted
    rather than defaulted, so the sentence never asserts a finding nobody made —
    which is exactly how the previous version produced one identical string for
    every patient.
    """
    s = {k: normalize_term(v) for k, v in ann.semantics.items()}
    kind, kind_size = split_size(s.get("Lung Nodule") or "lesion")
    parts, sizes = [], [kind_size]
    for key in ("Nodule Margins-Primary Pattern", "Nodule Shape", "Nodule Attenuation"):
        base, size = split_size(s.get(key, ""))
        if base:
            parts.append(base)
        if size:
            sizes.append(size)
    desc = " ".join(parts)

    head = f"{desc} {kind or 'lesion'}".strip()
    head = f"{_article(head)} {head}"
    size_clause = next((x for x in sizes if x), "")
    if size_clause:
        head += f", {size_clause}"
    if s.get("Anatomic Location"):
        head += f", in the {s['Anatomic Location']}"
    if s.get("Axial Location"):
        head += f", {s['Axial Location']}"

    extras = list(dict.fromkeys(normalize_term(f) for f in ann.findings if f))
    if extras:
        head += ", with " + ", ".join(extras)
    return head.rstrip(".") + "."


def coverage(anns: dict[str, AimAnnotation]) -> dict:
    """Summary stats, so the parse can be audited rather than trusted."""
    n = len(anns)
    with_loc = sum(1 for a in anns.values() if a.has_location)
    reports = {p: build_report(a) for p, a in anns.items()}
    return dict(
        patients=n,
        with_location=with_loc,
        distinct_reports=len(set(reports.values())),
        mean_fields=round(sum(len(a.semantics) for a in anns.values()) / max(n, 1), 1),
    )
