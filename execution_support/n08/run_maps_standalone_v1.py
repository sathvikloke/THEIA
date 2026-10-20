"""Native five-fold maps execution; retained methods, explicit E05 authority.

No final readout is enabled by this component. Each real execution needs five
currently full-verified native fits and the unchanged operational resource gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types

HERE = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
SCHEMA = 'theia.corrected_cpu.map_receipt.v1'
BINDING_SCHEMA = 'theia.corrected_cpu.map_binding.v1'
NATIVE_SHA = '013e165ee000bd7876f89c25eabc6911c98359fed148d12fb729034144319b42'
MAP_REL = Path('revamp/campaign/nine_hour_corrected_inputs_2026-10-02/n08/src/n08/readout/n08_maps_param.py')
MAP_SHA = '0688823dd61fc4f0bd3f8a9ed8850251a8a19c29e80565a674734b7e4c3aa7ed'
MAP_ONLY = {
    'theia/analysis/spatial_comparators.py': 'd97024e79ac4fc056375067c0ae4cd36771157381f22977d99662024f81bbbfb',
    'theia/analysis/spatial_projection.py': 'e74748ab60f77c3aa90c4cc84635c28fa46c87fdb80f20dd738d5ee33607348b',
}
DATA_KEYS = ('dataset_version', 'manifest_body_sha256', 'rows_sha256', 'npz_all_sha256')
COMMON_KEYS = ('root', 'output_root', 'support_dir', 'python', 'runtime', 'source_sha256',
               'environment', 'encoder_sha256', 'scientific_acceptance', 'rows', 'manifest',
               'splits', 'data_acceptance')
READOUT_MAP = {'A': 'mean', 'B': 'mean', 'C': 'query0', 'D': 'query0', 'E': 'query0'}
RESOURCE_FLOORS = {'internal_reserve_bytes': 1000000000, 'external_reserve_bytes': 2000000000,
                   'theia_reserve_bytes': 50000000000, 'backing_reserve_bytes': 100000000000,
                   'checkpoint_forecast_bytes': 1100000000}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_native():
    path = HERE / 'standalone_receipts.py'
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == NATIVE_SHA, 'Native verifier source changed')
    module = types.ModuleType('_native_maps_receipts_v1')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def five_context(root, output_root, cell, run, which, *, full=True):
    """Fresh five-unit verification; preparation may request zero-credit structural mode."""
    native = load_native()
    root, output = Path(root).resolve(strict=True), Path(output_root).resolve(strict=True)
    require(output.is_relative_to(root / 'revamp/cpu_runs'), 'Output root outside research CPU runs')
    require(which in ('best', 'last') and cell in READOUT_MAP, 'Invalid cell/checkpoint selector')
    evidence = native.Evidence()
    driver = evidence.module('driver')
    _, matrix = native._matrix(driver, evidence)
    rows = sorted([r for r in matrix if r['run_id'] == run], key=lambda r: int(r['fold']))
    require(len(rows) == 5 and [r['fold'] for r in rows] == list(map(str, range(5))),
            'Missing or duplicate frozen folds 0–4')
    require({r['cell'] for r in rows} == {cell} and len({r['init_seed'] for r in rows}) == 1,
            'Requested cell/seed differs from frozen run')
    paths = [output / 'units' / r['fit_id'] / 'RESULT.json' for r in rows]
    require(all(p.is_file() for p in paths), 'Missing native fold RESULT')
    mode = 'full' if full else 'structural'
    report = native.verify_results(paths, root=root, output_root=output, mode=mode)
    status = 'verified_current' if full else 'structural_only'
    require(report.get('schema') == native.SCHEMA and report.get('mode') == mode
            and report.get('status') == status and report.get('authority') == 'e05_native_execution'
            and type(report.get('scientific_fit_count')) is int
            and report['scientific_fit_count'] == (5 if full else 0)
            and report.get('full_matrix_verified') is False, 'Five-fold full/structural verification mismatch')
    units = report.get('units', [])
    require(len(units) == 5 and {u.get('fit_id') for u in units} == {r['fit_id'] for r in rows},
            'Native report has missing/duplicate/extra fold units')
    by_id = {u['fit_id']: u for u in units}
    common, data, checkpoints, inputs = None, None, [], {}
    for row, path in zip(rows, paths):
        unit = by_id[row['fit_id']]
        require(unit.get('kind') == 'fit' and unit.get('status') == status
                and type(unit.get('scientific_fit_count')) is int
                and unit['scientific_fit_count'] == int(full), 'Native unit status/credit mismatch')
        entries = unit.get('inputs', [])
        require(entries and len({r['path'] for r in entries}) == len(entries)
                and unit.get('native_bundle_sha256') == native.digest(sorted(entries, key=lambda r: r['path'])),
                'Native bundle digest mismatch')
        for item in entries:
            require(inputs.setdefault(item['path'], item) == item, 'Native input identity conflict')
            evidence.read(Path(item['path']), binary=True)
            require(evidence.files[item['path']] == item, 'Native input identity changed')
        unit_paths = [path, path.with_name('EXECUTION_BINDING.json'), path.with_name('WORKER_RESULT.json')]
        require(all(str(p) in {e['path'] for e in entries} for p in unit_paths), 'Native bundle omits own unit')
        binding, worker = (evidence.read(p) for p in unit_paths[1:])
        require(binding['row'] == row, 'Frozen row substitution')
        fold = output / 'checkpoints' / run / ('fold' + row['fold'])
        require(binding['fold_dir'] == str(fold), 'Bound checkpoint fold path changed')
        this_common = {k: binding[k] for k in COMMON_KEYS}
        require(common is None or common == this_common, 'Common runtime/source/data paths differ across folds')
        common = this_common
        this_data = {k: worker['data'][k] for k in native.DATA_KEYS}
        require(data is None or data == this_data, 'Common data identity differs across folds')
        data = this_data
        fi_sha = native.digest(worker['fit_identity'])
        require(fi_sha == worker['fit_identity_sha256'] == unit['fit_identity_sha256'], 'Fit identity changed')
        require(not full or fi_sha == unit.get('current_fit_identity_sha256'), 'Fresh fit identity differs')
        selected = worker['acceptance']['identities'][which + '_pt']
        require(type(selected['bytes']) is int and selected['bytes'] > 0, 'Missing selected checkpoint identity')
        checkpoints.append({'fold': int(row['fold']), 'fit_id': row['fit_id'],
            'path': str(fold / (which + '.pt')), 'bytes': selected['bytes'], 'sha256': selected['sha256'],
            'native_result_path': str(path), 'native_bundle_sha256': unit['native_bundle_sha256'],
            'execution_set_sha256': unit['execution_set_sha256'], 'fit_identity_sha256': fi_sha,
            'fold_summary': worker['acceptance']['identities']['fold_summary']})
    require(len({c['execution_set_sha256'] for c in checkpoints}) == 1, 'Conflicting execution sets')
    require(len({c['sha256'] for c in checkpoints}) == 5, 'Duplicate selected checkpoint evidence')
    evidence.unchanged()
    return {'cell': cell, 'run_id': run, 'which': which, 'rows': rows, 'common': common,
            'data': data, 'checkpoints': checkpoints,
            'execution_set_sha256': checkpoints[0]['execution_set_sha256'],
            'native_inputs': sorted(inputs.values(), key=lambda r: r['path'])}


def map_sources(context, evidence, native):
    root = Path(context['common']['root'])
    map_path = root / MAP_REL
    require(next((p for p in map_path.parents if (p / '.git').exists()), None) == root,
            'Retained maps program would discover a different research root')
    sources = dict(context['common']['source_sha256'])
    additions = {str(root / MAP_REL): MAP_SHA, str(HERE / 'standalone_receipts.py'): NATIVE_SHA,
                 str(root / MAP_REL.parent.parent / 'data_manifest.py'): native.PINS['data_manifest'],
                 **{str(root / name): sha for name, sha in MAP_ONLY.items()}}
    additions[str(SELF)] = evidence.sha(SELF)
    for name, sha in (sources | additions).items():
        require(evidence.sha(Path(name)) == sha, 'Map source changed: ' + name)
    return dict(sorted((sources | additions).items()))


def map_command(context, maps_dir):
    c = context['common']
    return [c['python'], str(Path(c['root']) / MAP_REL), '--cell', context['cell'],
            '--run', context['run_id'], '--which', context['which'], '--ckpt-root',
            str(Path(c['output_root']) / 'checkpoints'), '--out-dir', str(maps_dir),
            '--splits', c['splits'], '--rows', c['rows'], '--data-manifest', c['manifest'],
            '--dataset-version', 'M1', '--data-acceptance', c['data_acceptance']]


def new_target(output, target):
    output = Path(output).resolve(strict=True)
    target = Path(target)
    require(target.is_absolute() and target == target.resolve() and target.is_relative_to(output / 'maps'),
            'Map target must be a canonical private path under output_root/maps')
    require(not target.exists(), 'Map target already exists; preserve stale/failed outputs')
    return target


def job_key(context):
    return context['run_id'] + '-' + context['which']


def worker_command(binding, path):
    return [binding['context']['common']['python'], str(SELF), '_worker', '--binding', str(path)]


def minimum_resource_policy(policies):
    """Current distinct floors, retaining every higher applicable inherited floor."""
    result = dict(RESOURCE_FLOORS)
    for policy in policies:
        for key in RESOURCE_FLOORS:
            value = policy.get(key, 0)
            require(type(value) is int and value >= 0, 'Invalid inherited resource reserve: ' + key)
            result[key] = max(result[key], value)
    for key in ('theia_reserve_bytes', 'backing_reserve_bytes'):
        result[key] = max(result[key], result['external_reserve_bytes'])
    return {**result, 'ac_required': True}


def bound_resource_policy(context, evidence):
    inputs = {item['path']: item for item in context['native_inputs']}
    policies = []
    for checkpoint in context['checkpoints']:
        path = Path(checkpoint['native_result_path']).with_name('EXECUTION_BINDING.json')
        source = evidence.read(path)
        require(evidence.files[str(path)] == inputs.get(str(path)), 'Source resource binding changed')
        policies.append(source['resource_policy'])
    return minimum_resource_policy(policies)


def validate_resource_policy(policy):
    require(all(type(policy.get(key)) is int and policy[key] >= floor for key, floor in RESOURCE_FLOORS.items())
            and policy.get('ac_required') is True, 'Map resource floors weakened')
    require(all(policy[key] >= policy['external_reserve_bytes']
                for key in ('theia_reserve_bytes', 'backing_reserve_bytes')),
            'Inherited external reserve weakened')


def validate_binding(path, native, evidence):
    binding = evidence.read(path)
    require(binding.get('schema') == BINDING_SCHEMA, 'Wrong native maps binding schema')
    context = binding['context']
    c = context['common']
    root, output = Path(c['root']), Path(c['output_root'])
    target = native.path_under(binding['maps_dir'], output / 'maps')
    require(path == target / 'MAP_BINDING.json', 'Map binding target mismatch')
    require(binding['argv'] == map_command(context, target), 'Map command substitution')
    require(binding['source_sha256'] == map_sources(context, evidence, native), 'Map source closure changed')
    driver = evidence.module('driver')
    require(binding['session'] == {'admission_cutoff_utc': driver.FIXED_CUTOFF,
                                  'deadline_utc': driver.FIXED_DEADLINE}, 'Fixed map session changed')
    require(evidence.read(target / 'SESSION.json') == binding['session'], 'Session file changed')
    validate_resource_policy(binding['resource_policy'])
    require(binding['resource_policy'] == bound_resource_policy(context, evidence),
            'Map resource policy differs from current/inherited floors')
    require(Path(binding['volumes']['internal']['path']) == Path('/'), 'Internal reserve probe redirected')
    require(Path(binding['volumes']['theia']['path']) == Path('/Volumes') / root.parts[2],
            'External mount probe redirected')
    require(native.runtime_identity(Path(c['python']), root) == c['runtime'], 'Map runtime changed')
    return binding, driver


def resource(binding, driver, *, admission=False):
    require(driver.now() < driver.utc(driver.FIXED_STOP), 'Map safe-stop time reached')
    policy = binding['resource_policy']
    validate_resource_policy(policy)
    # Reuse the pinned driver's mount, AC, capacity and time probes. Its shared
    # external floor is only the common baseline; enforce each distinct floor
    # against the same measured snapshot below, without a second blocking probe.
    baseline = min(policy['theia_reserve_bytes'], policy['backing_reserve_bytes'])
    result = driver.resource_check(
        {**binding, 'resource_policy': {**policy, 'external_reserve_bytes': baseline}}, admission=admission)
    for name, key in (('theia', 'theia_reserve_bytes'), ('backing', 'backing_reserve_bytes')):
        needed = policy[key] + policy['checkpoint_forecast_bytes']
        sample = result['volumes'][name]
        require(sample['free_bytes'] >= needed, name + ' distinct capacity reserve unavailable')
        sample['required_bytes'] = needed
    require(driver.now() < driver.utc(driver.FIXED_STOP), 'Map safe-stop time reached during probe')
    return result


def verify_start(binding, binding_path, attempt, native, evidence, driver, *, live=False):
    context = binding['context']
    output = Path(context['common']['output_root'])
    attempt = native.path_under(attempt, output / 'map_attempts')
    require(attempt.parent == output / 'map_attempts'
            and attempt.name.startswith('M06_' + job_key(context) + '_'), 'Wrong M06 attempt path')
    start = evidence.read(attempt / 'START.json')
    process = evidence.read(attempt / 'PROCESS.json')
    launch = start['binding']
    command = worker_command(binding, binding_path)
    require(start.get('schema') == 'theia.cpu_attempt.v1' and start.get('task_id') == 'M06'
            and start.get('job_key') == job_key(context) and start.get('resume_receipt') is None,
            'Invalid M06 START authority')
    require(launch.get('command') == command and launch.get('cwd') == context['common']['root']
            and launch.get('task_id') == 'M06' and launch.get('job_key') == job_key(context),
            'E05 command/cwd/task changed')
    require(start['fingerprint'] == native.digest(launch), 'E05 fingerprint changed')
    require(start.get('runner') == evidence_item(evidence, native.FILES['wrapper']), 'E05 wrapper changed')
    require(launch.get('sources') == [evidence_item(evidence, Path(p)) for p in binding['source_sha256']],
            'E05 map sources changed')
    require(launch.get('inputs') == [evidence_item(evidence, binding_path)], 'E05 map input changed')
    require(start.get('session_sha256') == evidence.sha(binding_path.with_name('SESSION.json'))
            and native.utc(start['cutoff_utc']) == native.utc(driver.FIXED_CUTOFF)
            and native.utc(start['deadline_utc']) == native.utc(driver.FIXED_DEADLINE), 'E05 session mismatch')
    require(process.get('command') == command and type(process.get('pid')) is int and process['pid'] > 0
            and type(process.get('process_group')) is int and process['process_group'] == process['pid']
            and isinstance(process.get('identity'), str)
            and process['identity'].strip(), 'E05 process identity missing')
    require(native.utc(binding['created_utc']) <= native.utc(start['started_utc'])
            <= native.utc(process['launched_utc']) < native.utc(driver.FIXED_CUTOFF), 'E05 launch chronology changed')
    if live:
        require(process['pid'] == os.getpid() and process['process_group'] == os.getpgrp(),
                'Worker is not the live E05-owned process')
        require(os.getcwd() == context['common']['root'], 'Worker cwd changed')
        require(all(os.environ.get(k) == v for k, v in context['common']['environment'].items()),
                'Bound CPU/offline environment changed')
    return start, process


def evidence_item(evidence, path):
    evidence.read(path, binary=True)
    return evidence.files[str(path)]


def expected_membership(context, evidence):
    """Read only fresh bound split/OOF metadata; no new eligibility rule."""
    common = context['common']
    splits = evidence.read(Path(common['splits']))
    membership = {}
    for checkpoint in context['checkpoints']:
        fold = checkpoint['fold']
        summary_path = Path(checkpoint['path']).with_name('fold_summary.json')
        summary = evidence.read(summary_path)
        item = evidence.files[str(summary_path)]
        require(all(item[k] == checkpoint['fold_summary'][k] for k in ('bytes', 'sha256')),
                'Accepted OOF summary changed')
        ids = [r['patient_id'] for r in summary['summary']['oof']]
        outer = splits['folds'][fold]['outer_test']
        wanted = outer['patient_ids'] if isinstance(outer, dict) else outer
        require(sorted(ids) == sorted(wanted) and len(ids) == len(set(ids)), 'OOF membership differs from frozen fold')
        for pid in ids:
            require(pid not in membership, 'Patient appears in two OOF folds')
            membership[pid] = fold
    return membership


# check_output is copied byte-for-byte from the pinned retained producer below.
def check_output(o: dict, fits: list[dict], data: dict, cell: str, which: str) -> list[str]:
    probs = []
    if o.get("which", which) != which or o.get("cell") != cell:
        probs.append("output cell/which differ")
    if {k: (o.get("data") or {}).get(k) for k in ("dataset_version", "manifest_body_sha256", "rows_sha256",
                                                    "npz_all_sha256")} != \
            {k: data[k] for k in ("dataset_version", "manifest_body_sha256", "rows_sha256", "npz_all_sha256")}:
        probs.append("output data identity != configured data")
    got = {c["fold"]: c["sha256"] for c in o.get("checkpoints") or []}
    want = {f["fold"]: f["sha256"] for f in fits}
    if got != want:
        probs.append(f"scored checkpoints != accepted fits' {which}.pt ({sorted(k for k in want if got.get(k) != want[k])})")
    ctl = o.get("control") or {}
    if ctl.get("passed") is not True or o.get("control_passed") is not True:
        probs.append(f"reload control did not pass (control.passed={ctl.get('passed')!r})")
    ps = o.get("patient_set") or {}
    pids = sorted(o.get("per_patient") or {})
    if ps.get("n") != len(pids) or ps.get("patient_ids_sha256") != hashlib.sha256(
            json.dumps(pids, separators=(",", ":")).encode()).hexdigest():
        probs.append("patient_set does not describe per_patient")
    return probs

MAP_DEFINITION = {"maps": ["union=amax over queries", "mean=mean over queries", "query0", "per_query"],
                  "scorer": "theia.analysis.spatial_comparators.score_production",
                  "metrics": ["grounding_pointing", "grounding_mass", "grounding_iou"],
                  "target": "theia.analysis.spatial_projection.project(roi) of the configured rows' npz",
                  "oof_tolerance": 1e-4, "inner_tolerance": 1e-5}

CODE_FILES = ("theia/analysis/spatial_comparators.py", "theia/analysis/spatial_projection.py",
              "theia/models/theia_model.py", "theia/models/heads.py", "theia/engine/losses.py")


def check_native_output(doc, context, sources, membership):
    problems = check_output(doc, context['checkpoints'], context['data'], context['cell'], context['which'])
    require(not problems, '; '.join(problems))
    require(doc.get('schema') == 'n08.maps.v2' and doc.get('run') == context['run_id']
            and doc.get('which') == context['which']
            and doc.get('readout_map') == READOUT_MAP[context['cell']]
            and doc.get('status') == 'control_passed', 'Map output schema/run/status mismatch')
    expected_definition = {**MAP_DEFINITION, 'readout_map': READOUT_MAP[context['cell']],
        'code_sha256': {name: sources[str(Path(context['common']['root']) / name)] for name in CODE_FILES}
                       | {'maps_script': MAP_SHA}}
    require(doc.get('map_definition') == expected_definition, 'Map definition/source identity changed')
    expected = {c['fold']: {k: c[k] for k in ('fold', 'path', 'sha256', 'bytes')} for c in context['checkpoints']}
    got = doc.get('checkpoints', [])
    require(len(got) == 5 and {c['fold']: c for c in got} == expected, 'Selected checkpoint evidence changed')
    pp, ps = doc.get('per_patient', {}), doc.get('patient_set', {})
    require(set(pp) == set(membership), 'Map patients differ from accepted OOF set')
    counts = {}
    for pid, record in pp.items():
        require(record.get('fold') == membership[pid], 'Map patient fold differs from frozen OOF fold')
        counts[str(record['fold'])] = counts.get(str(record['fold']), 0) + 1
        if context['which'] == 'last':
            probability = record.get('p')
            require(type(probability) in (int, float) and math.isfinite(probability) and 0 <= probability <= 1,
                    'Invalid last checkpoint probability')
    require(ps.get('per_fold_n') == counts and ps.get('n_spatial') == sum('union' in r for r in pp.values()),
            'Map patient fold/spatial counts changed')
    require(set(doc.get('folds', {})) == set(map(str, range(5))), 'Map fold output incomplete')
    for row in context['rows']:
        fold = doc['folds'][row['fold']]
        require(fold.get('n') == counts.get(row['fold'], 0) and fold.get('region_pooling') == row['region_pooling'],
                'Map fold count/pooling differs')
    return True


def receipt_paths(binding):
    target, c = Path(binding['maps_dir']), binding['context']
    stem = c['run_id'] + '_' + c['which']
    return target / ('MAP_RECEIPT_' + stem + '.json'), target / ('N08_MAPS_' + stem + '.json'), \
        target / ('N08_MAPS_' + stem + '.control_failed.json'), target / ('maps_' + stem + '.log')


def worker(binding_path):
    native, evidence = load_native(), None
    evidence = native.Evidence()
    binding, driver = validate_binding(binding_path, native, evidence)
    receipt_path, good, bad, log = receipt_paths(binding)
    require(not any(p.exists() for p in (receipt_path, good, bad, log)), 'Stale map output/log/receipt')
    context = binding['context']
    receipt = {'schema': SCHEMA, 'status': 'failed', 'failed_reasons': [], 'exit_code': None,
               'binding': evidence_item(evidence, binding_path), 'context': context,
               'argv': binding['argv'], 'cwd': context['common']['root'],
               'runtime': context['common']['runtime'], 'source_sha256': binding['source_sha256'],
               'canonical_adoption': False, 'final_readout_complete': False,
               'authority': 'native_e05_pending_terminal_verification', 'started_utc': driver.now().isoformat()}
    try:
        attempt = Path(os.environ['THEIA_ATTEMPT_DIR'])
        until = time.monotonic() + 5
        while not (attempt / 'PROCESS.json').exists() and time.monotonic() < until:
            time.sleep(.05)
        verify_start(binding, binding_path, attempt, native, evidence, driver, live=True)
        resource(binding, driver, admission=True)
        current = five_context(context['common']['root'], context['common']['output_root'],
                               context['cell'], context['run_id'], context['which'])
        require(current == context, 'Native five-fold context changed after preparation')
        membership = expected_membership(context, evidence)
        require(map_sources(context, evidence, native) == binding['source_sha256'], 'Map source changed before launch')
        evidence.unchanged()
        with log.open('xb') as stream:
            # Metadata rechecks and external log creation can block. Admission
            # must be re-established after both, at the actual map-child boundary.
            resource(binding, driver, admission=True)
            # Same E05 process group; no detached model process or command override.
            result = subprocess.run(binding['argv'], cwd=context['common']['root'],
                                    env={**os.environ, **context['common']['environment']},
                                    stdout=stream, stderr=subprocess.STDOUT)
        receipt['exit_code'] = result.returncode
        resource(binding, driver)
        receipt['log'] = evidence_item(evidence, log)
        output = good if good.exists() else bad if bad.exists() else None
        if output:
            doc = evidence.read(output)
            receipt.update(output=evidence_item(evidence, output), control=doc.get('control'),
                           patient_set=doc.get('patient_set'), map_definition=doc.get('map_definition'))
        require(result.returncode == 0, 'Map process nonzero or signalled: ' + str(result.returncode))
        require(output == good and not bad.exists(), 'Missing output or control_failed artifact')
        check_native_output(doc, context, binding['source_sha256'], membership)
        evidence.unchanged()
        receipt['status'] = 'accepted'
    except (Exception, KeyboardInterrupt) as exc:
        receipt['failed_reasons'].append(type(exc).__name__ + ': ' + str(exc))
    receipt['ended_utc'] = driver.now().isoformat()
    driver.save_new(receipt_path, receipt)
    return 0 if receipt['status'] == 'accepted' else 1


def prepare(args):
    native = load_native()
    evidence = native.Evidence()
    driver = evidence.module('driver')
    root, output = args.root.resolve(strict=True), args.output_root.resolve(strict=True)
    target = new_target(output, args.maps_dir)
    require(driver.now() < driver.utc(driver.FIXED_CUTOFF), 'Fixed admission cutoff reached')
    context = five_context(root, output, args.cell, args.run, args.which, full=False)
    common = context['common']
    require(Path(sys.executable) == Path(common['python']), 'Use the bound research interpreter')
    original = evidence.read(output / 'units' / context['rows'][0]['fit_id'] / 'EXECUTION_BINDING.json')
    binding = {'schema': BINDING_SCHEMA, 'context': context, 'maps_dir': str(target),
        'created_utc': driver.now().isoformat(), 'argv': map_command(context, target),
        'source_sha256': map_sources(context, evidence, native),
        'session': {'admission_cutoff_utc': driver.FIXED_CUTOFF, 'deadline_utc': driver.FIXED_DEADLINE},
        'resource_policy': bound_resource_policy(context, evidence),
        'volumes': driver.initial_volumes(root, output, Path(original['volumes']['backing']['path']))}
    binding['preflight'] = resource(binding, driver, admission=True)
    evidence.unchanged()
    target.mkdir(parents=True, exist_ok=False)
    (output / 'map_attempts').mkdir(exist_ok=True)
    require((output / 'tmp').is_dir(), 'Bound external temporary directory missing')
    driver.save_new(target / 'SESSION.json', binding['session'])
    driver.save_new(target / 'MAP_BINDING.json', binding)
    return target / 'MAP_BINDING.json'


def execute(binding_path):
    native = load_native()
    evidence = native.Evidence()
    binding, driver = validate_binding(binding_path, native, evidence)
    context, target = binding['context'], binding_path.parent
    receipt_path, good, bad, log = receipt_paths(binding)
    require(not any(p.exists() for p in (receipt_path, good, bad, log, target / 'EXECUTION_RESULT.json')),
            'Existing terminal map output; no retry/overwrite')
    driver.save_new(target / 'LAUNCH_RESOURCE.json', resource(binding, driver, admission=True))
    output = Path(context['common']['output_root'])
    attempts = native.path_under(output / 'map_attempts', output)
    require(attempts.is_dir(), 'Create the owned map attempt root before E05 execution')
    command = [context['common']['python'], str(native.FILES['wrapper']), '--session', str(target / 'SESSION.json'),
               '--attempt-root', str(attempts), '--cwd', context['common']['root'], '--task-id', 'M06',
               '--job-key', job_key(context), '--stop-grace-seconds', '60']
    for path in binding['source_sha256']:
        command += ['--source', path]
    command += ['--input', str(binding_path), '--output', str(receipt_path), '--', *worker_command(binding, binding_path)]
    pattern = 'M06_' + job_key(context) + '_*/EXIT.json'
    prior = set(attempts.glob(pattern))
    stops = []
    with (target / 'WRAPPER.stdout').open('x') as out, (target / 'WRAPPER.stderr').open('x') as err:
        child = subprocess.Popen(command, cwd=context['common']['root'],
                                 env={**os.environ, **context['common']['environment']}, stdout=out, stderr=err)
        driver.save_new(target / 'OWNED_WRAPPER.json', {'pid': child.pid, 'command': command})
        try:
            while child.poll() is None:
                try:
                    sample = resource(binding, driver)
                    with (target / 'RESOURCE_SAMPLES.jsonl').open('a') as stream:
                        stream.write(json.dumps(sample) + '\n')
                except Exception as exc:
                    if not stops:
                        stops.append(str(exc))
                        child.send_signal(signal.SIGINT)
                time.sleep(1 if stops else 15)
        except KeyboardInterrupt:
            stops.append('coordinator_interrupted')
            child.send_signal(signal.SIGINT)
        code = child.wait()
    matches = sorted(set(attempts.glob(pattern)) - prior)
    result = {'schema': 'theia.corrected_cpu.map_execution_result.v1',
              'wrapper_exit': code, 'resource_stops': stops, 'e05_exit': str(matches[0]) if len(matches) == 1 else None,
              'binding': evidence_item(evidence, binding_path), 'ended_utc': driver.now().isoformat(),
              'canonical_adoption': False, 'final_readout_complete': False}
    driver.save_new(target / 'EXECUTION_RESULT.json', result)
    require(code == 0 and not stops and len(matches) == 1, 'Map execution failed or incomplete; artifacts retained')
    # Terminal validation is a separate call: do not redundantly run full science here.
    print(json.dumps({'status': 'terminal_verification_required', 'receipt': str(receipt_path),
                      'e05_exit': str(matches[0]), 'final_readout_complete': False}))
    return 0


def verify_map_receipt(receipt_path, exit_path):
    """Require the real terminal E05 chain plus fresh five-fit and map checks."""
    native = load_native()
    evidence = native.Evidence()
    receipt_path, exit_path = Path(receipt_path), Path(exit_path)
    receipt = evidence.read(receipt_path)
    binding_path = receipt_path.parent / 'MAP_BINDING.json'
    binding, driver = validate_binding(binding_path, native, evidence)
    expected_receipt, good, bad, log = receipt_paths(binding)
    require(receipt_path == expected_receipt and not bad.exists(), 'Wrong receipt or control_failed output')
    start, process = verify_start(binding, binding_path, exit_path.parent, native, evidence, driver)
    require(exit_path.name == 'EXIT.json', 'Explicit E05 EXIT required')
    terminal = evidence.read(exit_path)
    require(terminal.get('schema') == 'theia.cpu_attempt.v1' and terminal.get('task_id') == 'M06'
            and terminal.get('job_key') == job_key(binding['context'])
            and terminal.get('fingerprint') == start['fingerprint'], 'E05 terminal identity mismatch')
    require(terminal.get('status') == 'finished' and type(terminal.get('exit_code')) is int
            and terminal['exit_code'] == 0 and terminal.get('signals') == []
            and terminal.get('runner_error') is None and terminal.get('output_errors') == []
            and terminal.get('stop_reason') == 'process_exit', 'E05 failed, signalled or incomplete')
    require(terminal.get('outputs') == [evidence_item(evidence, receipt_path)], 'E05 receipt output hash changed')
    result = evidence.read(receipt_path.parent / 'EXECUTION_RESULT.json')
    require(result.get('schema') == 'theia.corrected_cpu.map_execution_result.v1'
            and type(result.get('wrapper_exit')) is int and result['wrapper_exit'] == 0
            and result.get('resource_stops') == [] and result.get('e05_exit') == str(exit_path)
            and result.get('binding') == evidence_item(evidence, binding_path), 'Coordinator/resource result failed')
    require(receipt.get('schema') == SCHEMA and receipt.get('status') == 'accepted'
            and receipt.get('failed_reasons') == [] and type(receipt.get('exit_code')) is int
            and receipt['exit_code'] == 0 and receipt.get('binding') == evidence_item(evidence, binding_path)
            and receipt.get('authority') == 'native_e05_pending_terminal_verification'
            and receipt.get('canonical_adoption') is False and receipt.get('final_readout_complete') is False,
            'Map receipt is failed or changed')
    context = binding['context']
    require(receipt.get('context') == context and receipt.get('argv') == binding['argv']
            and receipt.get('cwd') == context['common']['root'] and receipt.get('runtime') == context['common']['runtime']
            and receipt.get('source_sha256') == binding['source_sha256'], 'Native receipt/context equality failed')
    require(native.utc(process['launched_utc']) <= native.utc(receipt['started_utc']) <= native.utc(receipt['ended_utc'])
            <= native.utc(terminal['ended_utc']) <= native.utc(result['ended_utc'])
            and native.utc(terminal['ended_utc']) <= native.utc(driver.FIXED_DEADLINE), 'Map terminal chronology changed')
    fresh = five_context(context['common']['root'], context['common']['output_root'],
                         context['cell'], context['run_id'], context['which'])
    require(fresh == context, 'Current native five-fold context differs')
    require(receipt.get('output') == evidence_item(evidence, good)
            and receipt.get('log') == evidence_item(evidence, log), 'Map output/log identity changed')
    doc = evidence.read(good)
    require(all(receipt.get(k) == doc.get(k) for k in ('control', 'patient_set', 'map_definition')),
            'Receipt/output map metadata differs')
    check_native_output(doc, context, binding['source_sha256'], expected_membership(context, evidence))
    evidence.unchanged()
    return {'schema': 'theia.corrected_cpu.map_consumption.v1', 'status': 'verified_current',
            'receipt': evidence_item(evidence, receipt_path), 'e05_exit': evidence_item(evidence, exit_path),
            'native_context_sha256': native.digest(context), 'canonical_adoption': False,
            'final_readout_complete': False}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['_worker']:
        parser = argparse.ArgumentParser()
        parser.add_argument('--binding', required=True, type=Path)
        return worker(parser.parse_args(argv[1:]).binding)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--cell', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--which', choices=['best', 'last'], required=True)
    parser.add_argument('--maps-dir', type=Path, required=True)
    return execute(prepare(parser.parse_args(argv)))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        print('REFUSED: ' + str(exc), file=sys.stderr)
        raise SystemExit(2)
