"""Small control/binding checks; no arrays, model loads, mounted drives or trainer jobs."""
import csv
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from scripts import run_corrected_cpu as driver

MATRIX=Path(__file__).resolve().parents[1]/'execution_support/n08/FIT_MATRIX_M1.csv'


class CorrectedCPU(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.row=driver.select_row(MATRIX,'fit',1)

    def test_exact_matrix_pin_and_full_fixed_roster(self):
        self.assertEqual(driver.sha(MATRIX),driver.MATRIX_SHA)
        with MATRIX.open() as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual({(r['cell'],r['init_seed'],r['fold']) for r in rows},
                         {(c,s,str(f)) for c in 'ABCDE' for s in ['42','7'] for f in range(5)})

    def test_smoke_is_exact_B_configuration_separate_and_two_epochs(self):
        row=driver.select_row(MATRIX,'smoke',1)
        self.assertEqual((row['cell'],row['region_pooling'],row['ground_target'],row['ground_weight']),
                         ('B','mean','union','0.5'))
        self.assertEqual(row['run_id'],'n08m1-smoke-cellB-init42-part1337-2ep')
        argv=json.loads(row['argv_json'])
        self.assertNotIn('--resume',argv)
        self.assertEqual(argv[-2:],['--set','train.stop_after_epochs=2'])
        self.assertIn('-smoke-',row['fit_id'])

    def test_full_fit_keeps_resume_zero_retries_and_no_truncation(self):
        argv=json.loads(self.row['argv_json'])
        self.assertIn('--resume',argv)
        self.assertEqual(argv[argv.index('--max-retries')+1],'0')
        self.assertFalse(any('stop_after_epochs' in x for x in argv))
        self.assertEqual(self.row['fit_id'],'n08m1-cellA-init42-part1337-fold0')

    def test_explicit_absolute_data_config_runtime_and_control(self):
        out=self.root/'new outputs'
        control=out/'COOP_CONTROL.json'
        argv=driver.build_argv(self.row,self.root,out,control)
        self.assertEqual(argv[0],str(self.root/'.venv/bin/python'))
        self.assertEqual(argv[argv.index('--config')+1],str(self.root/'configs/default.yaml'))
        self.assertIn('paths.processed_dir='+str(self.root/driver.PF/'source/manifest_S1_v1'),argv)
        self.assertIn('data.dataset_version=M1',argv)
        self.assertIn(str(control),argv)
        self.assertFalse(any('{' in x or '}' in x for x in argv))

    def test_mismatched_seed_or_retry_is_refused(self):
        for old,new in [('seed=42','seed=7'),('"--max-retries", "0"','"--max-retries", "1"')]:
            row=dict(self.row)
            row['argv_json']=row['argv_json'].replace(old,new)
            with self.assertRaises(ValueError):driver.validate_row(row)

    def test_missing_static_training_closure_is_refused(self):
        target=self.root/'theia/engine/train.py'
        target.parent.mkdir(parents=True)
        target.write_text('print("not imported")\n')
        with self.assertRaisesRegex(ValueError,'TRAINING_CODE'):driver.training_code(self.root)
        target.write_text('TRAINING_CODE=("theia/engine/train.py", "theia/config.py")\nraise RuntimeError("not imported")\n')
        self.assertEqual(driver.training_code(self.root),['theia/engine/train.py','theia/config.py'])

    def test_changed_bound_source_is_refused(self):
        source=self.root/'source.py';source.write_text('old')
        binding={'source_sha256':{str(source):driver.sha(source)},'input_sha256':{}}
        source.write_text('new')
        with self.assertRaisesRegex(ValueError,'Bound file changed'):driver.verify_binding(binding)

    def test_existing_artifact_is_not_overwritten(self):
        path=self.root/'binding.json';driver.save_new(path,{'first':True})
        with self.assertRaises(FileExistsError):driver.save_new(path,{'second':True})
        self.assertEqual(json.loads(path.read_text()),{'first':True})

    def smoke_fixture(self):
        """Synthetic receipt fixture only; never used as a real execution input."""
        output=self.root/'out'
        row=driver.select_row(MATRIX,'smoke',0)
        unit=output/'units'/row['fit_id'];unit.mkdir(parents=True)
        binding={'kind':'fit','root':str(self.root),'output_root':str(output),
                 'support_dir':str(MATRIX.parent),'source_sha256':{'trainer':'source-v1'},
                 'runtime':{'python':'fixture'},'scientific_acceptance':driver.SCIENCE,
                 'encoder_sha256':driver.ENCODER_SHA,'input_sha256':{},
                 'data':{'dataset_version':'M1','manifest_body_sha256':'manifest-v1',
                         'rows_sha256':'rows-v1','npz_all_sha256':'arrays-v1',
                         'annotation_preprocessing_sha256':'annotations-v1','n_entries':158,
                         'acceptance':{'decision':'accepted','sha256':'acceptance-v1'}}}
        for key in ('manifest','rows','data_acceptance','splits'):
            path=self.root/key;path.write_text('fixture')
            binding[key]=str(path);binding['input_sha256'][str(path)]=driver.sha(path)
        binding['input_sha256'][str(MATRIX)]=driver.sha(MATRIX)
        smoke=copy.deepcopy(binding);smoke.update(kind='smoke',row=row)
        bp=unit/'EXECUTION_BINDING.json';driver.save_new(bp,smoke)
        worker={'schema':'theia.corrected_cpu.technical_receipt.v1','kind':'smoke',
                'status':'accepted','fit_id':row['fit_id'],'trainer_exit':0,
                'execution_binding_sha256':driver.sha(bp),'execution_binding':str(bp),
                'acceptance':{'accepted':True,'checks':[{'ok':True}]},
                'data':{**binding['data'],'npz_rehashed':True}}
        wp=unit/'WORKER_RESULT.json';driver.save_new(wp,worker)
        e05=output/'attempts'/'fixture'/'EXIT.json';e05.parent.mkdir(parents=True)
        driver.save_new(e05,{'schema':'theia.cpu_attempt.v1','task_id':'M03',
                            'job_key':row['fit_id'],'status':'finished','exit_code':0,
                            'outputs':[{'path':str(wp),'sha256':driver.sha(wp)}]})
        result={'schema':'theia.corrected_cpu.result.v1','kind':'smoke','status':'accepted',
                'fit_id':row['fit_id'],'wrapper_exit':0,'scientific_fit_count':0,
                'technical_receipt':str(wp),'e05_exit_receipt':str(e05),
                'execution_binding_sha256':driver.sha(bp)}
        driver.save_new(unit/'RESULT.json',result)
        return binding,unit,e05

    def test_fit_requires_completed_smoke(self):
        binding,unit,_=self.smoke_fixture()
        (unit/'RESULT.json').unlink()
        with self.assertRaisesRegex(ValueError,'completed matching smoke'):
            driver.require_matching_smoke(binding)

    def test_matching_smoke_binds_all_four_receipts(self):
        binding,unit,e05=self.smoke_fixture()
        dependencies=driver.require_matching_smoke(binding)
        self.assertEqual(len(dependencies),4)
        self.assertEqual(dependencies[str(e05)],driver.sha(e05))
        self.assertEqual(dependencies[str(unit/'EXECUTION_BINDING.json')],
                         driver.sha(unit/'EXECUTION_BINDING.json'))

    def test_stale_smoke_source_runtime_manifest_or_rows_refused(self):
        binding,_,_=self.smoke_fixture()
        for key in ('source_sha256','runtime','manifest','rows'):
            changed=copy.deepcopy(binding)
            if key in ('manifest','rows'):
                changed['input_sha256'][changed[key]]='changed'
            else:
                changed[key]={'changed':True}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,'stale'):
                driver.require_matching_smoke(changed)

    def test_smoke_nonzero_e05_or_changed_worker_is_refused(self):
        binding,unit,e05=self.smoke_fixture()
        original=e05.read_text()
        value=json.loads(original);value['exit_code']=1;e05.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'zero-exit'):
            driver.require_matching_smoke(binding)
        e05.write_text(original)
        wp=unit/'WORKER_RESULT.json'
        value=json.loads(wp.read_text());value['acceptance']['accepted']=False
        wp.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'technically accepted'):
            driver.require_matching_smoke(binding)

    def resource_binding(self):
        return {'resource_policy':{'internal_reserve_bytes':100,'external_reserve_bytes':100,
                                  'checkpoint_forecast_bytes':100},
                'volumes':{name:{'path':str(self.root),'device':self.root.stat().st_dev}
                           for name in ['theia','backing','internal']}}

    def test_crossing_deadline_in_probe_refuses(self):
        current=[driver.utc('2026-10-20T12:29:59Z')]
        def power(*args,**kwargs):
            current[0]=driver.utc('2026-10-20T12:30:01Z')
            return "Now drawing from 'AC Power'"
        with mock.patch.object(driver,'now',side_effect=lambda:current[0]), \
             mock.patch.object(driver.subprocess,'check_output',side_effect=power), \
             mock.patch.object(driver.shutil,'disk_usage',return_value=types.SimpleNamespace(free=1000)):
            with self.assertRaisesRegex(ValueError,'time reached during power'):
                driver.resource_check(self.resource_binding())

    def test_changed_mount_refuses_before_power_probe(self):
        binding=self.resource_binding();binding['volumes']['theia']['device']=-1
        with mock.patch.object(driver,'now',return_value=driver.utc('2026-10-20T10:00:00Z')), \
             mock.patch.object(driver.subprocess,'check_output') as call:
            with self.assertRaisesRegex(ValueError,'mount/device changed'):driver.resource_check(binding)
            call.assert_not_called()

    def test_unknown_power_refuses(self):
        with mock.patch.object(driver,'now',return_value=driver.utc('2026-10-20T10:00:00Z')), \
             mock.patch.object(driver.subprocess,'check_output',return_value='unknown'), \
             mock.patch.object(driver.shutil,'disk_usage',return_value=types.SimpleNamespace(free=1000)):
            with self.assertRaisesRegex(ValueError,'Known AC'):driver.resource_check(self.resource_binding())


if __name__=='__main__':unittest.main()
