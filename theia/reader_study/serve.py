"""Minimal Flask app to run the blinded reader study.

Serves one presentation at a time, collects Likert scores, appends to a per-reader
JSONL. No patient identity or arm label is ever sent to the browser — only the
opaque token. Analysis joins scores to KEY_do_not_open.json afterward.

Run: python -m theia.reader_study.serve --cases theia/reader_study/cases --reader R1

Fixed here:
  * The gene row was hard-coded to EGFR and KRAS, so any other gene in the panel
    was invisible to the reader.
  * Every arm was asked all three questions, including "the rationale is
    clinically plausible" and "the highlighted region is anatomically sensible"
    for the label_only and ground_truth arms, which show neither a rationale nor
    an image. Two of the three arms were collecting noise. Each presentation now
    carries the questions it can support.
  * Submissions were accepted with no answers at all, silently advancing.
  * Restarting the server reset the position to 0 while the scores file stayed in
    append mode, duplicating every earlier case. Progress is now recovered from
    the scores file.
  * Model-generated text went through innerHTML.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)
STATE: dict = {}

QUESTION_TEXT = {
    "plausible": "Rationale is clinically plausible",
    "grounded": "Highlighted region is anatomically sensible",
    "useful": "I would find this useful in practice",
}


PAGE = """<!doctype html><meta charset=utf-8>
<title>THEIA reader study</title>
<style>
 body{font-family:system-ui;background:#0a0e14;color:#eef;max-width:720px;margin:40px auto}
 img{width:100%;border-radius:10px;border:1px solid #234}
 .card{background:#121821;border:1px solid #223;border-radius:14px;padding:20px}
 .q{margin:14px 0 6px;color:#9fb0c3;font-size:14px}
 button{background:#1a2230;color:#eef;border:1px solid #2fd4c6;border-radius:8px;
   padding:8px 12px;margin:3px;cursor:pointer}
 button:hover{background:#22303f} .status{font-size:20px;font-weight:700;color:#2fd4c6}
 .rat{color:#cdd;line-height:1.7;margin:10px 0}
 .err{color:#ff9b9b;font-size:13px;margin-top:10px;min-height:18px}
 .prog{color:#6b7889;font-size:12px;margin-bottom:10px}
</style>
<div class=card id=app>loading…</div>
<script>
let cur=null, ans={};
const QT=%QUESTION_TEXT%;
function label(v){return v===1?'Mutant':(v===0?'Wild-type':'—');}

async function load(){
 ans={};
 const r=await fetch('/next'); cur=await r.json();
 const app=document.getElementById('app');
 app.textContent='';
 if(cur.done){app.innerHTML='<h3>Done. Thank you.</h3>';return;}
 const p=cur.arm_payload;

 const prog=document.createElement('div'); prog.className='prog';
 prog.textContent=`Case ${cur.index+1} of ${cur.total}`; app.appendChild(prog);
 const hdr=document.createElement('div'); hdr.textContent='Case '+cur.token; app.appendChild(hdr);

 if(p.image){const im=document.createElement('img'); im.src='/img/'+encodeURIComponent(p.image); app.appendChild(im);}

 const st=document.createElement('p'); st.className='status';
 st.textContent=Object.keys(p.status||{}).map(g=>g.toUpperCase()+': '+label(p.status[g])).join(' · ');
 app.appendChild(st);

 if(p.rationale){const d=document.createElement('div'); d.className='rat';
  d.textContent=p.rationale;            // model output is data, never markup
  app.appendChild(d);}

 (p.questions||[]).forEach(k=>{
  const q=document.createElement('div'); q.className='q'; q.textContent=QT[k]||k; app.appendChild(q);
  const row=document.createElement('div');
  for(let i=1;i<=5;i++){const b=document.createElement('button');
   b.textContent=i; b.onclick=()=>pick(k,i,b,row); row.appendChild(b);}
  app.appendChild(row);
 });

 const err=document.createElement('div'); err.className='err'; err.id='err'; app.appendChild(err);
 const wrap=document.createElement('div'); wrap.style.marginTop='16px';
 const sub=document.createElement('button'); sub.textContent='Submit →'; sub.onclick=submit;
 wrap.appendChild(sub); app.appendChild(wrap);
}
function pick(k,v,el,row){ans[k]=v;
 [...row.children].forEach(b=>b.style.background='#1a2230');
 el.style.background='#22303f'; document.getElementById('err').textContent='';}
async function submit(){
 const need=cur.arm_payload.questions||[];
 const missing=need.filter(k=>!(k in ans));
 if(missing.length){document.getElementById('err').textContent=
   'Please answer all '+need.length+' question(s) before submitting.'; return;}
 const r=await fetch('/score',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({token:cur.token,scores:ans})});
 if(!r.ok){document.getElementById('err').textContent='Server rejected the submission.'; return;}
 load();
}
load();
</script>""".replace("%QUESTION_TEXT%", json.dumps(QUESTION_TEXT))


@app.route("/")
def index():
    return PAGE


@app.route("/img/<name>")
def img(name):
    return send_from_directory(os.path.join(STATE["cases"], "img"), name)


@app.route("/next")
def nxt():
    i = STATE["pos"]
    pres = STATE["presentations"]
    if i >= len(pres):
        return jsonify(done=True)
    return jsonify(dict(pres[i], index=i, total=len(pres)))


@app.route("/score", methods=["POST"])
def score():
    data = request.get_json(silent=True) or {}
    token, scores = data.get("token"), data.get("scores") or {}
    pres = STATE["presentations"]
    if STATE["pos"] >= len(pres):
        return jsonify(error="all cases already scored"), 400
    expected = pres[STATE["pos"]]
    if token != expected["token"]:
        return jsonify(error="token mismatch; reload the page"), 409
    required = set(expected["arm_payload"].get("questions", []))
    if not required.issubset(scores):
        return jsonify(error=f"missing answers: {sorted(required - set(scores))}"), 400
    if not all(isinstance(v, int) and 1 <= v <= 5 for v in scores.values()):
        return jsonify(error="scores must be integers 1-5"), 400

    with open(STATE["out"], "a") as fh:
        fh.write(json.dumps(dict(reader=STATE["reader"], token=token, scores=scores)) + "\n")
    STATE["pos"] += 1
    return jsonify(ok=True)


def _resume_position(out_path: str, presentations: list[dict]) -> int:
    """Skip past tokens this reader already scored, so a restart is safe."""
    if not os.path.exists(out_path):
        return 0
    done = set()
    with open(out_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line).get("token"))
            except json.JSONDecodeError:
                continue
    pos = 0
    while pos < len(presentations) and presentations[pos]["token"] in done:
        pos += 1
    return pos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="theia/reader_study/cases")
    ap.add_argument("--reader", required=True)
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    with open(Path(a.cases) / "presentations.json") as fh:
        presentations = json.load(fh)
    out = str(Path(a.cases) / f"scores_{a.reader}.jsonl")
    pos = _resume_position(out, presentations)
    STATE.update(cases=a.cases, presentations=presentations, pos=pos, reader=a.reader, out=out)
    resumed = f" (resuming at {pos})" if pos else ""
    print(f"[reader] serving {len(presentations)} cases for {a.reader}{resumed} "
          f"at http://127.0.0.1:{a.port}")
    app.run(port=a.port)


if __name__ == "__main__":
    main()
