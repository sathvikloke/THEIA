"""Minimal Flask app to run the blinded reader study.

Serves one presentation at a time, collects Likert scores, appends to a per-reader
JSONL. No patient identity or arm label is ever sent to the browser — only the
opaque token. Analysis joins scores to KEY_do_not_open.json afterward.

Run: python -m theia.reader_study.serve --cases theia/reader_study/cases --reader R1
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)
STATE: dict = {}


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
</style>
<div class=card id=app>loading…</div>
<script>
let cur=null;
async function load(){
 const r=await fetch('/next'); cur=await r.json();
 if(cur.done){document.getElementById('app').innerHTML='<h3>Done. Thank you.</h3>';return;}
 let h=`<div>Case <b>${cur.token}</b></div>`;
 const p=cur.arm_payload;
 if(p.image) h+=`<img src="/img/${p.image}">`;
 h+=`<p class=status>EGFR: ${label(p.status.egfr)} · KRAS: ${label(p.status.kras)}</p>`;
 if(p.rationale) h+=`<div class=rat>${p.rationale}</div>`;
 h+=scale('plausible','Rationale is clinically plausible');
 h+=scale('grounded','Highlighted region is anatomically sensible');
 h+=scale('useful','I would find this useful in practice');
 h+=`<div style=margin-top:16px><button onclick=submit()>Submit &rarr;</button></div>`;
 document.getElementById('app').innerHTML=h;
}
function label(v){return v===1?'Mutant':(v===0?'Wild-type':'—');}
function scale(k,q){let s=`<div class=q>${q}</div><div>`;
 for(let i=1;i<=5;i++) s+=`<button onclick="pick('${k}',${i},this)">${i}</button>`;
 return s+'</div>';}
let ans={};
function pick(k,v,el){ans[k]=v;
 [...el.parentNode.children].forEach(b=>b.style.background='#1a2230');
 el.style.background='#22303f';}
async function submit(){
 await fetch('/score',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({token:cur.token,scores:ans})});
 ans={}; load();
}
load();
</script>"""


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
    return jsonify(pres[i])


@app.route("/score", methods=["POST"])
def score():
    data = request.get_json()
    with open(STATE["out"], "a") as fh:
        fh.write(json.dumps(dict(reader=STATE["reader"], **data)) + "\n")
    STATE["pos"] += 1
    return jsonify(ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="theia/reader_study/cases")
    ap.add_argument("--reader", required=True)
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    with open(Path(a.cases) / "presentations.json") as fh:
        STATE.update(cases=a.cases, presentations=json.load(fh), pos=0,
                     reader=a.reader, out=str(Path(a.cases) / f"scores_{a.reader}.jsonl"))
    print(f"[reader] serving {len(STATE['presentations'])} cases for {a.reader} "
          f"at http://127.0.0.1:{a.port}")
    app.run(port=a.port)


if __name__ == "__main__":
    main()
