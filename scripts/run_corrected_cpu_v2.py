#!/usr/bin/env python3
"""Prepare and execute one source-bound corrected CPU unit through the E05 wrapper (v2).

Version 2 differs from the reviewed v1 driver only operationally: the admission cutoff,
cooperative stop and deadline come from an explicitly recorded session-window file bound
into each new execution binding (instead of compiled 2026-10-20 constants), and THEIA and
backing-volume reserves are distinct. Matrix, split, data, initialization, loss, epochs,
CPU/offline policy, source closure and acceptance replay are unchanged. v1 bindings,
receipts and windows are never re-admitted by this driver.
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import plistlib
import shutil
import signal
import subprocess
import sys
import time

PORTABLE = Path(__file__).resolve().parents[1]
BW = Path('revamp/campaign/nine_hour_corrected_inputs_2026-10-02')
PF = Path('revamp/campaign/paper_first_activation_2026-10-07_224630-a6ba26')
SPLITS = Path('revamp/campaign/parallel_expansion_2026-10-02/n08/SPLITS.json')
MATRIX_SHA = 'fe0a0f2ffff724a184f18731bab5e7c8924eb6c94baf1409e8fc24280654aa34'
SPLITS_SHA = '48d85225b672cd52f49395c3846582c5fc836de31d71a69554f2179b80670d90'
ENCODER_SHA = '24064fdbf02384327d1af5e3e1e7fb25106fd60b9f9812c03140a46077f7fbb3'
# Scientific/checkpoint values from the retained runner DEFAULTS; no controller budgets.
SCIENCE = dict(threads=6, ckpt_bytes=351162341, ckpt_bytes_tol=65536,
               min_epochs=10, patience=8, max_epochs=40, partition_seed=1337)
BINDING_SCHEMA = 'theia.corrected_cpu.execution_binding.v2'
DRIVER_VERSION = 'corrected_cpu_v2'
WINDOW_KEYS = ('start_utc', 'admission_cutoff_utc', 'cooperative_stop_utc', 'deadline_utc')


def utc(value):
    result = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None or result.utcoffset() != dt.timedelta(0):
        raise ValueError('An explicit UTC timestamp is required')
    return result


def now():
    return dt.datetime.now(dt.timezone.utc)


def load_window(path):
    """Read one recorded session window; refuse malformed, misordered or already-closed windows."""
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    value = json.loads(raw)
    value = value.get('window', value)
    missing = [k for k in WINDOW_KEYS if not isinstance(value.get(k), str)]
    if missing:
        raise ValueError('Session window lacks explicit UTC values: ' + ', '.join(missing))
    times = [utc(value[k]) for k in WINDOW_KEYS]
    if not times[0] <= times[1] <= times[2] <= times[3]:
        raise ValueError('Session window must satisfy start <= cutoff <= stop <= deadline')
    if now() >= times[1]:
        raise ValueError('Session admission cutoff has already passed')
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
            **{k: value[k] for k in WINDOW_KEYS}}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def load_helper(directory, name):
    path = Path(directory) / (name + '.py')
    spec = importlib.util.spec_from_file_location('corrected_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def training_code(root):
    tree = ast.parse((root / 'theia/engine/train.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'TRAINING_CODE' for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise ValueError('Trainer has no static TRAINING_CODE closure')


def validate_row(row):
    argv = json.loads(row['argv_json'])
    if argv[:3] != ['.venv/bin/python', '-m', 'theia.engine.train']:
        raise ValueError('Unexpected trainer entry point')
    def flag(name):
        if argv.count(name) != 1:
            raise ValueError('Missing or repeated trainer flag: ' + name)
        return argv[argv.index(name) + 1]
    for name, expected in [('--device', 'cpu'), ('--max-retries', '0'),
                           ('--run_id', row['run_id']), ('--folds', row['fold'])]:
        if flag(name) != str(expected):
            raise ValueError('Matrix/argv mismatch: ' + name)
    sets = [argv[i+1].split('=', 1) for i, value in enumerate(argv) if value == '--set']
    values = dict(sets)
    if len(values) != len(sets):
        raise ValueError('Repeated config override')
    for key, wanted in [('model.region_pooling', row['region_pooling']),
                        ('train.ground_target', row['ground_target']),
                        ('seed', row['init_seed']), ('split.seed', row['partition_seed'])]:
        if values.get(key) != str(wanted):
            raise ValueError('Matrix/override mismatch: ' + key)
    if float(values['train.loss_weights.ground']) != float(row['ground_weight']):
        raise ValueError('Matrix/ground weight mismatch')
    if values.get('train.monitor') != '[[egfr_logloss,-1.0]]':
        raise ValueError('Selection monitor changed')
    env = json.loads(row['env_json'])
    if env.get('OMP_NUM_THREADS') != '6' or env.get('HF_HUB_OFFLINE') != '1' or env.get('TRANSFORMERS_OFFLINE') != '1':
        raise ValueError('Frozen CPU/offline environment changed')
    return argv


def select_row(matrix, kind, order_index):
    with Path(matrix).open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 50 or len({r['fit_id'] for r in rows}) != 50:
        raise ValueError('Expected exactly 50 unique fixed matrix units')
    for row in rows:
        validate_row(row)
    if kind == 'smoke':
        row = dict(next(r for r in rows if r['cell'] == 'B' and r['fold'] == '0' and r['init_seed'] == '42'))
        old = row['run_id']
        new = 'n08m1-smoke-cellB-init42-part1337-2ep'
        row = {k: v.replace(old, new) for k, v in row.items()}
        argv = [v for v in json.loads(row['argv_json']) if v != '--resume']
        row['argv_json'] = json.dumps(argv + ['--set', 'train.stop_after_epochs=2'])
        row['order_index'] = '0'
        return row
    return dict(next(r for r in rows if int(r['order_index']) == order_index))


def build_argv(row, root, output, control, stop_utc):
    argv = json.loads(row['argv_json'])
    replacements = {'{CKPT_ROOT}': str(output/'checkpoints'), '{OUT}': str(output),
                    '{TRAINER_DEADLINE}': stop_utc}
    for token, value in replacements.items():
        argv = [s.replace(token, value) for s in argv]
    if any('{' in value or '}' in value for value in argv):
        raise ValueError('Unresolved command placeholder')
    argv[0] = str(root/'.venv/bin/python')
    argv += ['--config', str(root/'configs/default.yaml'), '--coop-control', str(control),
             '--set', 'paths.processed_dir=' + str(root/PF/'source/manifest_S1_v1'),
             '--set', 'data.content_manifest=' + str(root/PF/'source/manifest_S1_v1/CONTENT_MANIFEST.json'),
             '--set', 'data.dataset_version=M1']
    return argv


def runtime_identity(python, root):
    code = "import importlib.metadata as m,json,platform,sys; names=['torch','numpy','scikit-learn','torchvision','transformers','timm','PyYAML']; print(json.dumps({'python':platform.python_version(),'executable':sys.executable,'packages':{n:m.version(n) for n in names}}))"
    value = json.loads(subprocess.check_output([str(python), '-c', code], cwd=root, text=True, timeout=30))
    value['python_binary_sha256'] = sha(python)
    return value


def initial_volumes(root, output, backing):
    if not str(root).startswith('/Volumes/') or not output.is_relative_to(root/'revamp/cpu_runs'):
        raise ValueError('Outputs must be inside the mounted research root revamp/cpu_runs')
    mount = Path('/Volumes') / root.parts[2]
    volumes = {}
    for name, path in [('theia', mount), ('backing', backing), ('internal', Path('/'))]:
        path = path.resolve(strict=True)
        if name != 'internal' and (not os.path.ismount(path) or path.stat().st_dev == Path('/').stat().st_dev):
            raise ValueError('Expected a distinct mounted external volume: ' + str(path))
        info = {} if name == 'internal' else plistlib.loads(subprocess.check_output(
            ['/usr/sbin/diskutil', 'info', '-plist', str(path)], timeout=30))
        volumes[name] = {'path': str(path), 'device': path.stat().st_dev,
                         'volume_uuid': info.get('VolumeUUID'), 'free_bytes': shutil.disk_usage(path).free}
    images = plistlib.loads(subprocess.check_output(['/usr/bin/hdiutil', 'info', '-plist'], timeout=10))
    matches = [image for image in images.get('images', []) if any(
        e.get('mount-point') == str(mount) for e in image.get('system-entities', []))]
    if len(matches) != 1:
        raise ValueError('THEIA mount lacks one unambiguous backing-image association')
    image = Path(matches[0]['image-path']).resolve(strict=True)
    if not image.is_relative_to(backing) or image.stat().st_dev != volumes['backing']['device']:
        raise ValueError('THEIA backing image is outside the measured backing volume')
    volumes['backing_image'] = str(image)
    return volumes


def resource_check(binding, admission=False):
    if binding.get('schema') != BINDING_SCHEMA:
        raise ValueError('Only v2 execution bindings are admitted by this driver')
    deadline = utc(binding['admission_cutoff_utc'] if admission else binding['deadline_utc'])
    if now() >= deadline:
        raise ValueError('Session time reached')
    snapshot = {'utc': now().isoformat(), 'volumes': {}}
    policy = binding['resource_policy']
    for name in ('theia', 'backing', 'internal'):
        volume = binding['volumes'][name]
        path = Path(volume['path'])
        if path.stat().st_dev != volume['device']:
            raise ValueError(name + ' mount/device changed')
        free = shutil.disk_usage(path).free
        needed = policy['internal_reserve_bytes'] if name == 'internal' else (
            policy[name + '_reserve_bytes'] + policy['checkpoint_forecast_bytes'])
        if free < needed:
            raise ValueError(name + ' capacity reserve unavailable')
        snapshot['volumes'][name] = {'device': volume['device'], 'free_bytes': free, 'required_bytes': needed}
    remaining = (deadline-now()).total_seconds()
    if remaining <= 0:
        raise ValueError('Session time reached during probes')
    power = subprocess.check_output(['/usr/bin/pmset', '-g', 'batt'], text=True, timeout=min(3, remaining))
    if "Now drawing from 'AC Power'" not in power:
        raise ValueError('Known AC power is required for this CPU job')
    if now() >= deadline:
        raise ValueError('Session time reached during power probe')
    snapshot['power'] = power.strip()
    return snapshot


def verify_binding(binding):
    for group in ('source_sha256', 'input_sha256'):
        for name, expected in binding[group].items():
            if sha(name) != expected:
                raise ValueError('Bound file changed: ' + name)
    if runtime_identity(Path(binding['python']), Path(binding['root'])) != binding['runtime']:
        raise ValueError('Bound runtime changed')


def require_matching_smoke(binding):
    """Bind a full fit to the successful real smoke in this same output root."""
    output = Path(binding['output_root'])
    row = select_row(Path(binding['support_dir'])/'FIT_MATRIX_M1.csv', 'smoke', 0)
    unit = output/'units'/row['fit_id']
    paths = {name: unit/name for name in ('EXECUTION_BINDING.json', 'RESULT.json', 'WORKER_RESULT.json')}
    try:
        smoke, result, worker = (json.loads(paths[name].read_text()) for name in paths)
        e05_path = Path(result['e05_exit_receipt']).resolve(strict=True)
        if not e05_path.is_relative_to((output/'attempts').resolve()) or e05_path.name != 'EXIT.json':
            raise ValueError('Smoke E05 receipt is outside this output root')
        e05 = json.loads(e05_path.read_text())
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('A completed matching smoke is required: missing or invalid receipt') from exc
    if (result.get('schema') != 'theia.corrected_cpu.result.v1'
            or result.get('kind') != 'smoke' or result.get('status') != 'accepted'
            or result.get('fit_id') != row['fit_id'] or result.get('wrapper_exit') != 0
            or result.get('scientific_fit_count') != 0 or result.get('resource_stops')
            or result.get('technical_receipt') != str(paths['WORKER_RESULT.json'])):
        raise ValueError('A successful smoke RESULT is required')
    acceptance = worker.get('acceptance') or {}
    if (worker.get('schema') != 'theia.corrected_cpu.technical_receipt.v1'
            or worker.get('kind') != 'smoke' or worker.get('status') != 'accepted'
            or worker.get('fit_id') != row['fit_id'] or worker.get('trainer_exit') != 0
            or acceptance.get('accepted') is not True or not acceptance.get('checks')
            or not all(c.get('ok') is True for c in acceptance['checks'])
            or (worker.get('data') or {}).get('npz_rehashed') is not True):
        raise ValueError('A technically accepted smoke with rehashed real data is required')
    if (e05.get('schema') != 'theia.cpu_attempt.v1' or e05.get('task_id') != 'M03'
            or e05.get('job_key') != row['fit_id'] or e05.get('status') != 'finished'
            or e05.get('exit_code') != 0 or e05.get('signals') or e05.get('runner_error')
            or e05.get('output_errors')
            or not any(item.get('path') == str(paths['WORKER_RESULT.json'])
                       and item.get('sha256') == sha(paths['WORKER_RESULT.json'])
                       for item in e05.get('outputs', []))):
        raise ValueError('A finished zero-exit E05 smoke receipt with matching output is required')
    digest = sha(paths['EXECUTION_BINDING.json'])
    if (result.get('execution_binding_sha256') != digest
            or worker.get('execution_binding_sha256') != digest
            or worker.get('execution_binding') != str(paths['EXECUTION_BINDING.json'])
            or smoke.get('kind') != 'smoke' or smoke.get('row') != row
            or smoke.get('schema') != BINDING_SCHEMA or smoke.get('driver_version') != DRIVER_VERSION):
        raise ValueError('Smoke binding identity is stale or mismatched')
    for key in ('root', 'output_root', 'support_dir', 'source_sha256', 'runtime',
                'scientific_acceptance', 'encoder_sha256'):
        if smoke.get(key) != binding[key]:
            raise ValueError('Smoke is stale for current ' + key)
    for path in [binding[k] for k in ('manifest', 'rows', 'data_acceptance', 'splits')] + [
            str(Path(binding['support_dir'])/'FIT_MATRIX_M1.csv')]:
        if smoke.get('input_sha256', {}).get(path) != binding['input_sha256'][path]:
            raise ValueError('Smoke is stale for current input: ' + path)
    for key in ('dataset_version', 'manifest_body_sha256', 'rows_sha256', 'npz_all_sha256',
                'annotation_preprocessing_sha256', 'n_entries', 'acceptance'):
        if worker['data'].get(key) != binding['data'].get(key):
            raise ValueError('Smoke is stale for current data identity: ' + key)
    paths['E05_EXIT.json'] = e05_path
    return {str(path): sha(path) for path in paths.values()}


def prepare(args):
    if min(args.internal_reserve_bytes, args.theia_reserve_bytes, args.backing_reserve_bytes) < 0:
        raise ValueError('Resource reserves cannot be negative')
    root = args.root.resolve(strict=True)
    output = args.output_root.resolve()
    support = args.support_dir.resolve(strict=True)
    matrix = support/'FIT_MATRIX_M1.csv'
    splits = root/SPLITS
    if sha(matrix) != MATRIX_SHA or sha(splits) != SPLITS_SHA:
        raise ValueError('Frozen matrix or splits changed')
    window = load_window(args.session_window)
    row = select_row(matrix, args.kind, args.order_index)
    unit = output/'units'/row['fit_id']
    if unit.exists() or (output/'checkpoints'/row['run_id']/('fold'+row['fold'])).exists():
        raise ValueError('Unit or checkpoint path already exists; preserve it for review')
    dm = load_helper(support, 'data_manifest')
    manifest = root/PF/'source/manifest_S1_v1/CONTENT_MANIFEST.json'
    rows = manifest.parent/'rows.jsonl'
    acceptance = root/PF/'compute/M1_ACCEPTANCE_S1A_v1.json'
    acc = json.loads(acceptance.read_text())
    if acc.get('test_fixture_run') is not False or acc.get('options', {}).get('allow_test_fixtures') is not False or not acc.get('checks') or not all(c.get('status') == 'pass' for c in acc['checks']):
        raise ValueError('Real accepted source checks are required')
    data = dm.verify_manifest(manifest, rows, 'M1', root, acceptance, rehash=False)
    if dm.partition_invariance(rows, json.loads(splits.read_text())):
        raise ValueError('Frozen partition membership changed')
    for name, expected in [(acc['checker']['path'], acc['checker']['sha256']),
                           (acc['registry']['path'], acc['registry']['sha256']),
                           (acc['manifest'], acc['evidence_sha256'][acc['manifest']])]:
        if sha(root/name) != expected:
            raise ValueError('Source acceptance binding changed: ' + name)
    sources = [root/name for name in training_code(root)]
    sources += [root/'configs/default.yaml', support/'accept_fit.py', support/'data_manifest.py',
                Path(__file__).resolve(), PORTABLE/'scripts/run_cpu_task.py']
    sources += [root/name for name in ('requirements.txt', 'setup.py') if (root/name).is_file()]
    inputs = [matrix, splits, manifest, rows, acceptance, root/acc['registry']['path'], root/acc['checker']['path']]
    python = root/'.venv/bin/python'
    binding = {'schema': BINDING_SCHEMA, 'driver_version': DRIVER_VERSION, 'created_utc': now().isoformat(),
        'root': str(root), 'output_root': str(output), 'unit_dir': str(unit), 'kind': args.kind,
        'row': row, 'support_dir': str(support), 'python': str(python), 'data': data,
        'manifest': str(manifest), 'rows': str(rows), 'data_acceptance': str(acceptance), 'splits': str(splits),
        'source_sha256': {str(p): sha(p) for p in sources}, 'input_sha256': {str(p): sha(p) for p in inputs},
        'runtime': runtime_identity(python, root), 'scientific_acceptance': SCIENCE,
        'encoder_sha256': ENCODER_SHA, 'encoder_reference_provenance': 'retained accepted CPU parity reference, supplied in current execution task',
        'session_window': window, 'admission_cutoff_utc': window['admission_cutoff_utc'],
        'cooperative_stop_utc': window['cooperative_stop_utc'], 'deadline_utc': window['deadline_utc'],
        'volumes': initial_volumes(root, output, args.backing_root.resolve(strict=True)),
        'resource_policy': {'internal_reserve_bytes': args.internal_reserve_bytes,
                            'theia_reserve_bytes': args.theia_reserve_bytes,
                            'backing_reserve_bytes': args.backing_reserve_bytes,
                            'checkpoint_forecast_bytes': 1100000000, 'ac_required': True},
        'lineage': 'standalone current CPU execution; no legacy controller release or canonical adoption claimed'}
    if args.kind == 'fit':
        binding['smoke_dependency'] = require_matching_smoke(binding)
        binding['input_sha256'].update(binding['smoke_dependency'])
    binding['preflight'] = resource_check(binding, admission=True)
    binding['trainer_argv'] = build_argv(row, root, output, unit/'COOP_CONTROL.json', window['cooperative_stop_utc'])
    binding['environment'] = {**json.loads(row['env_json']), 'PYTHONDONTWRITEBYTECODE':'1',
                              'TMPDIR':str(output/'tmp')}
    binding['fold_dir'] = str(output/'checkpoints'/row['run_id']/('fold'+row['fold']))
    unit.mkdir(parents=True, exist_ok=False)
    (output/'attempts').mkdir(exist_ok=True)
    (output/'tmp').mkdir(exist_ok=True)
    save_new(unit/'SESSION.json', {'admission_cutoff_utc':window['admission_cutoff_utc'],
                                   'deadline_utc':window['deadline_utc']})
    save_new(unit/'COOP_CONTROL.json', {'deadline_utc':window['cooperative_stop_utc'], 'check_every_steps':1,
        'min_free_bytes':args.theia_reserve_bytes, 'ckpt_forecast_bytes':351227877,
        'fit_bytes_budget':1100000000, 'max_battery_pause_s':0, 'epoch_forecast_s':0})
    binding['input_sha256'][str(unit/'COOP_CONTROL.json')] = sha(unit/'COOP_CONTROL.json')
    binding['input_sha256'][str(unit/'SESSION.json')] = sha(unit/'SESSION.json')
    binding['input_sha256'][window['path']] = window['sha256']
    save_new(unit/'EXECUTION_BINDING.json', binding)
    print(json.dumps({'binding':str(unit/'EXECUTION_BINDING.json'), 'trainer_argv':binding['trainer_argv'],
                      'next_command':[str(python),str(Path(__file__).resolve()),'execute','--binding',str(unit/'EXECUTION_BINDING.json')]}))
    return 0


def expected_for(binding, data):
    dm = load_helper(binding['support_dir'], 'data_manifest')
    labels = dm.rows_labels(binding['rows'])
    splits = json.loads(Path(binding['splits']).read_text())
    mb = {fold['train_microbatches_per_epoch'] for fold in splits['folds']}
    if len(mb) != 1:
        raise ValueError('Inconsistent expected microbatch counts')
    result = {**binding['scientific_acceptance'], 'splits':splits, 'data':data,
              'rows_labels':{p:v['egfr'] for p,v in labels.items()},
              'rows_has_mask':{p:v['has_mask'] for p,v in labels.items()},
              'rows_sha256':data['rows_sha256'], 'microbatches_per_epoch':mb.pop(),
              'encoder_sha256':binding['encoder_sha256']}
    if binding['kind']=='smoke':
        result.update(max_epochs=2, allow_truncation=True)
    return result


def worker(args):
    binding=json.loads(args.binding.read_text())
    unit=Path(binding['unit_dir'])
    result={'schema':'theia.corrected_cpu.technical_receipt.v1','kind':binding['kind'],
            'fit_id':binding['row']['fit_id'],
            **{k:binding['row'][k] for k in ('run_id','cell','fold','init_seed')},
            'acceptance_helper_contract':'n08.fit_receipt.v2 technical verdict and fit identity',
            'execution_binding_sha256':sha(args.binding),
            'execution_binding':str(args.binding), 'status':'failed', 'trainer_exit':None,
            'canonical_adoption':False, 'scientific_fit_count':0}
    try:
        verify_binding(binding)
        resource_check(binding, admission=True)
        environment={**os.environ,**binding['environment']}
        # Same foreground process group as E05; no detached trainer or hidden retry.
        process=subprocess.run(binding['trainer_argv'],cwd=binding['root'],env=environment)
        result['trainer_exit']=process.returncode
        if process.returncode==0:
            verify_binding(binding)
            dm=load_helper(binding['support_dir'],'data_manifest')
            data=dm.verify_manifest(binding['manifest'],binding['rows'],'M1',binding['root'],
                                    binding['data_acceptance'],rehash=True)
            expected=expected_for(binding,data)
            accept=load_helper(binding['support_dir'],'accept_fit')
            verdict=accept.accept(binding['row'],Path(binding['fold_dir']),expected,trainer_exit=process.returncode)
            result.update(acceptance=verdict,data=data,fit_identity=accept.fit_identity(binding['row'],verdict,expected))
            result['fit_identity_sha256']=accept.canon_sha(result['fit_identity'])
            result['status']='accepted' if verdict['accepted'] else 'acceptance_failed'
            result['scientific_fit_count']=int(verdict['accepted'] and binding['kind']=='fit')
    except Exception as exc:
        result['error']=type(exc).__name__+': '+str(exc)
    result['ended_utc']=now().isoformat()
    save_new(unit/'WORKER_RESULT.json',result)
    return 0 if result['status']=='accepted' else 1


def execute(args):
    binding=json.loads(args.binding.read_text())
    if binding.get('schema')!=BINDING_SCHEMA or binding.get('driver_version')!=DRIVER_VERSION:
        raise ValueError('Only v2 execution bindings are executed by this driver')
    unit=Path(binding['unit_dir']); output=Path(binding['output_root'])
    if (unit/'RESULT.json').exists() or (unit/'WORKER_RESULT.json').exists():
        raise ValueError('Existing terminal result; no automatic retry or overwrite')
    lock=(output/'.corrected-cpu.lock').open('a+')
    with lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_binding(binding)
        save_new(unit/'LAUNCH_RESOURCE.json',resource_check(binding,admission=True))
        command=[binding['python'],str(PORTABLE/'scripts/run_cpu_task.py'),
                 '--session',str(unit/'SESSION.json'),'--attempt-root',str(output/'attempts'),
                 '--cwd',binding['root'],'--task-id','M03' if binding['kind']=='smoke' else 'M04',
                 '--job-key',binding['row']['fit_id'],'--stop-grace-seconds','60']
        for path in binding['source_sha256']:
            command += ['--source',path]
        command += ['--input',str(args.binding),'--output',str(unit/'WORKER_RESULT.json'),'--',
                    binding['python'],str(Path(__file__).resolve()),'_worker','--binding',str(args.binding)]
        stopped=[]
        attempt_pattern=('M03' if binding['kind']=='smoke' else 'M04')+'_'+binding['row']['fit_id']+'_*/EXIT.json'
        prior_attempts=set((output/'attempts').glob(attempt_pattern))
        with (unit/'WRAPPER.stdout').open('x') as stdout, (unit/'WRAPPER.stderr').open('x') as stderr:
            child=subprocess.Popen(command,cwd=binding['root'],env={**os.environ,**binding['environment']},
                                   stdout=stdout,stderr=stderr)
            save_new(unit/'OWNED_WRAPPER.json',{'pid':child.pid,'launched_utc':now().isoformat(),'command':command})
            try:
                while child.poll() is None:
                    try:
                        sample=resource_check(binding)
                        with (unit/'RESOURCE_SAMPLES.jsonl').open('a') as stream:
                            stream.write(json.dumps(sample)+'\n')
                    except Exception as exc:
                        if not stopped:
                            stopped.append(str(exc));child.send_signal(signal.SIGINT)
                    time.sleep(1 if stopped else 15)
            except KeyboardInterrupt:
                stopped.append('driver_interrupted');child.send_signal(signal.SIGINT)
            rc=child.wait()
        matches=sorted(set((output/'attempts').glob(attempt_pattern))-prior_attempts)
        if len(matches)>1:
            stopped.append('ambiguous_new_attempt_receipts')
        receipt=json.loads((unit/'WORKER_RESULT.json').read_text()) if (unit/'WORKER_RESULT.json').exists() else {}
        e05=json.loads(matches[-1].read_text()) if matches else {}
        good=rc==0 and not stopped and e05.get('status')=='finished' and receipt.get('status')=='accepted'
        result={'schema':'theia.corrected_cpu.result.v1','kind':binding['kind'],'fit_id':binding['row']['fit_id'],
                'status':'accepted' if good else 'incomplete_or_failed','wrapper_exit':rc,
                'resource_stops':stopped,'e05_exit_receipt':str(matches[-1]) if matches else None,
                'technical_receipt':str(unit/'WORKER_RESULT.json') if receipt else None,
                'execution_binding_sha256':sha(args.binding),'scientific_fit_count':int(good and binding['kind']=='fit'),
                'canonical_adoption':False,'ended_utc':now().isoformat()}
        save_new(unit/'RESULT.json',result);print(json.dumps(result))
        return 0 if good else 1


def arguments():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare',help='metadata/resource preflight only; does not load a model or launch training')
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--support-dir',type=Path,default=PORTABLE/'execution_support/n08')
    p.add_argument('--backing-root',type=Path,default=Path('/Volumes/Research'))
    p.add_argument('--kind',choices=['smoke','fit'],required=True)
    p.add_argument('--order-index',type=int,default=1)
    p.add_argument('--internal-reserve-bytes',type=int,default=1000000000)
    p.add_argument('--session-window',type=Path,required=True,
                   help='JSON with start_utc, admission_cutoff_utc, cooperative_stop_utc, deadline_utc')
    p.add_argument('--theia-reserve-bytes',type=int,default=50000000000)
    p.add_argument('--backing-reserve-bytes',type=int,default=100000000000)
    for name in ('execute','_worker'):
        p=commands.add_parser(name);p.add_argument('--binding',type=Path,required=True)
    return parser.parse_args()


if __name__=='__main__':
    try:
        args=arguments()
        raise SystemExit({'prepare':prepare,'execute':execute,'_worker':worker}[args.command](args))
    except (ValueError,OSError,KeyError,subprocess.SubprocessError) as exc:
        print('REFUSED: '+type(exc).__name__+': '+str(exc),file=sys.stderr)
        raise SystemExit(2)
