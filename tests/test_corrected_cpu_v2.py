"""Session-window boundary checks for the v2 corrected CPU driver; no arrays, models, mounts or jobs."""
import datetime as dt
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from scripts import run_corrected_cpu as v1
from scripts import run_corrected_cpu_v2 as driver

MATRIX=Path(__file__).resolve().parents[1]/'execution_support/n08/FIT_MATRIX_M1.csv'
WINDOW={'start_utc':'2030-01-01T00:00:00Z','admission_cutoff_utc':'2030-01-01T10:00:00Z',
        'cooperative_stop_utc':'2030-01-01T10:18:00Z','deadline_utc':'2030-01-01T10:20:00Z'}


def at(value):
    return mock.patch.object(driver,'now',return_value=driver.utc(value))


class SessionWindow(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()

    def write(self,value,name='w.json'):
        path=self.root/name;path.write_text(json.dumps(value));return path

    def test_v1_history_is_preserved_and_v2_has_no_compiled_window(self):
        self.assertEqual((v1.FIXED_CUTOFF,v1.FIXED_STOP,v1.FIXED_DEADLINE),
                         ('2026-10-20T12:10:00Z','2026-10-20T12:28:00Z','2026-10-20T12:30:00Z'))
        self.assertFalse([n for n in vars(driver) if n.startswith('FIXED_')])
        for name in ('SCIENCE','MATRIX_SHA','SPLITS_SHA','ENCODER_SHA','SPLITS','PF','BW'):
            self.assertEqual(getattr(driver,name),getattr(v1,name),name)

    def test_window_is_read_bound_and_nested_form_accepted(self):
        for value in (WINDOW,{'window':WINDOW,'other':1}):
            path=self.write(value)
            with at('2029-12-31T23:00:00Z'):
                got=driver.load_window(path)
            self.assertEqual({k:got[k] for k in driver.WINDOW_KEYS},WINDOW)
            self.assertEqual(got['sha256'],driver.sha(path))

    def test_malformed_misordered_offset_or_closed_window_is_refused(self):
        cases=[({k:v for k,v in WINDOW.items() if k!='cooperative_stop_utc'},'lacks'),
               ({**WINDOW,'cooperative_stop_utc':'2030-01-01T10:21:00Z'},'start <= cutoff'),
               ({**WINDOW,'admission_cutoff_utc':'2030-01-01T05:00:00-05:00'},'UTC')]
        for value,message in cases:
            with at('2029-12-31T23:00:00Z'),self.assertRaisesRegex(ValueError,message):
                driver.load_window(self.write(value))
        with at('2030-01-01T10:00:00Z'),self.assertRaisesRegex(ValueError,'already passed'):
            driver.load_window(self.write(WINDOW))

    def test_expired_2026_10_20_window_cannot_be_admitted(self):
        old={'start_utc':'2026-10-20T00:00:00Z','admission_cutoff_utc':v1.FIXED_CUTOFF,
             'cooperative_stop_utc':v1.FIXED_STOP,'deadline_utc':v1.FIXED_DEADLINE}
        with self.assertRaisesRegex(ValueError,'already passed'):
            driver.load_window(self.write(old))

    def test_trainer_deadline_is_the_bound_cooperative_stop(self):
        row=driver.select_row(MATRIX,'fit',1)
        argv=driver.build_argv(row,self.root,self.root/'out',self.root/'out/C.json',WINDOW['cooperative_stop_utc'])
        self.assertIn(WINDOW['cooperative_stop_utc'],' '.join(argv))
        self.assertNotIn(v1.FIXED_STOP,' '.join(argv))
        self.assertFalse(any('{' in x or '}' in x for x in argv))
        # Science-bearing argv is unchanged between versions apart from the stop time.
        old=v1.build_argv(row,self.root,self.root/'out',self.root/'out/C.json')
        self.assertEqual([x.replace(v1.FIXED_STOP,WINDOW['cooperative_stop_utc']) for x in old],argv)


class ResourceBoundary(unittest.TestCase):
    def binding(self,**window):
        dev={'theia':11,'backing':12,'internal':13}
        return {'schema':driver.BINDING_SCHEMA,'driver_version':driver.DRIVER_VERSION,
                **{k:window.get(k,WINDOW[k]) for k in ('admission_cutoff_utc','deadline_utc')},
                'volumes':{n:{'path':'/'+n,'device':d} for n,d in dev.items()},
                'resource_policy':{'internal_reserve_bytes':500,'theia_reserve_bytes':50_000,
                                   'backing_reserve_bytes':100_000,'checkpoint_forecast_bytes':1_000}}

    def run_check(self,binding,free,when,admission=True,power="Now drawing from 'AC Power'"):
        dev={'/theia':11,'/backing':12,'/internal':13}
        with at(when),mock.patch.object(driver.Path,'stat',lambda p:types.SimpleNamespace(st_dev=dev[str(p)])),\
             mock.patch.object(driver.shutil,'disk_usage',lambda p:types.SimpleNamespace(free=free[str(p)[1:]])),\
             mock.patch.object(driver.subprocess,'check_output',return_value=power):
            return driver.resource_check(binding,admission=admission)

    FREE={'theia':51_000,'backing':101_000,'internal':500}

    def test_distinct_reserves_each_enforced(self):
        snap=self.run_check(self.binding(),self.FREE,'2030-01-01T09:00:00Z')
        self.assertEqual(snap['volumes']['theia']['required_bytes'],51_000)
        self.assertEqual(snap['volumes']['backing']['required_bytes'],101_000)
        for name in ('theia','backing','internal'):
            free=dict(self.FREE);free[name]-=1
            with self.assertRaisesRegex(ValueError,name+' capacity'):
                self.run_check(self.binding(),free,'2030-01-01T09:00:00Z')

    def test_bound_cutoff_and_deadline_are_used(self):
        with self.assertRaisesRegex(ValueError,'Session time reached'):
            self.run_check(self.binding(),self.FREE,'2030-01-01T10:00:00Z',admission=True)
        self.run_check(self.binding(),self.FREE,'2030-01-01T10:10:00Z',admission=False)
        with self.assertRaisesRegex(ValueError,'Session time reached'):
            self.run_check(self.binding(),self.FREE,'2030-01-01T10:20:00Z',admission=False)

    def test_v1_binding_is_never_admitted_or_executed(self):
        old=self.binding();old['schema']='theia.corrected_cpu.execution_binding.v1'
        with self.assertRaisesRegex(ValueError,'Only v2'):
            self.run_check(old,self.FREE,'2030-01-01T09:00:00Z')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'B.json';path.write_text(json.dumps(old))
            with self.assertRaisesRegex(ValueError,'Only v2'):
                driver.execute(types.SimpleNamespace(binding=path))

    def test_battery_refused(self):
        with self.assertRaisesRegex(ValueError,'AC power'):
            self.run_check(self.binding(),self.FREE,'2030-01-01T09:00:00Z',power="Now drawing from 'Battery Power'")


class SmokeDependency(unittest.TestCase):
    """v2 fits bind only a v2 smoke; synthetic receipts only, never real execution inputs."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()

    def fixture(self,schema):
        output=self.root/'out';row=driver.select_row(MATRIX,'smoke',0)
        unit=output/'units'/row['fit_id'];unit.mkdir(parents=True)
        binding={'schema':driver.BINDING_SCHEMA,'driver_version':driver.DRIVER_VERSION,'kind':'fit',
                 'root':str(self.root),'output_root':str(output),'support_dir':str(MATRIX.parent),
                 'source_sha256':{'trainer':'s'},'runtime':{'python':'f'},'scientific_acceptance':driver.SCIENCE,
                 'encoder_sha256':driver.ENCODER_SHA,'input_sha256':{str(MATRIX):driver.sha(MATRIX)},
                 'data':{'dataset_version':'M1','manifest_body_sha256':'m','rows_sha256':'r','npz_all_sha256':'a',
                         'annotation_preprocessing_sha256':'p','n_entries':158,'acceptance':{'decision':'accepted'}}}
        for key in ('manifest','rows','data_acceptance','splits'):
            path=self.root/key;path.write_text('fixture')
            binding[key]=str(path);binding['input_sha256'][str(path)]=driver.sha(path)
        smoke=json.loads(json.dumps(binding));smoke.update(kind='smoke',row=row,schema=schema)
        bp=unit/'EXECUTION_BINDING.json';driver.save_new(bp,smoke)
        wp=unit/'WORKER_RESULT.json'
        driver.save_new(wp,{'schema':'theia.corrected_cpu.technical_receipt.v1','kind':'smoke','status':'accepted',
            'fit_id':row['fit_id'],'trainer_exit':0,'execution_binding_sha256':driver.sha(bp),'execution_binding':str(bp),
            'acceptance':{'accepted':True,'checks':[{'ok':True}]},'data':{**binding['data'],'npz_rehashed':True}})
        e05=output/'attempts'/'fixture'/'EXIT.json';e05.parent.mkdir(parents=True)
        driver.save_new(e05,{'schema':'theia.cpu_attempt.v1','task_id':'M03','job_key':row['fit_id'],'status':'finished',
                             'exit_code':0,'outputs':[{'path':str(wp),'sha256':driver.sha(wp)}]})
        driver.save_new(unit/'RESULT.json',{'schema':'theia.corrected_cpu.result.v1','kind':'smoke','status':'accepted',
            'fit_id':row['fit_id'],'wrapper_exit':0,'scientific_fit_count':0,'technical_receipt':str(wp),
            'e05_exit_receipt':str(e05),'execution_binding_sha256':driver.sha(bp)})
        return binding

    def test_v2_smoke_is_bound(self):
        self.assertEqual(len(driver.require_matching_smoke(self.fixture(driver.BINDING_SCHEMA))),4)

    def test_v1_smoke_cannot_satisfy_v2_fit(self):
        with self.assertRaisesRegex(ValueError,'stale or mismatched'):
            driver.require_matching_smoke(self.fixture('theia.corrected_cpu.execution_binding.v1'))


if __name__=='__main__':
    unittest.main()
