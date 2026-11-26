import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('steady',Path(__file__).with_name('linux-steady.py'))
steady=importlib.util.module_from_spec(spec);spec.loader.exec_module(steady)


class Steady(unittest.TestCase):
    def test_settings_are_native_uncapped_taa_and_never_relocate_saves(self):
        text=steady.settings('Setting_PC_RenderScale=3\nSetting_PC_SaveFolder=1\nSetting_PC_AntiAliasing=2\nSetting_PC_ShadowDistance=3\n')
        self.assertIn('Setting_PC_RenderScale=6',text)
        self.assertIn('Setting_PC_FrameRateLimit=2',text)
        self.assertIn('Setting_PC_AntiAliasing=0',text)
        self.assertIn('Setting_PC_ShadowDistance=3',text)
        self.assertNotIn('SaveFolder',text)
        self.assertEqual(steady.environment()['SDL_AUDIO_DRIVER'],'dummy')

    def test_clean_profile_cannot_inherit_probes_validation_or_fps_cap(self):
        inherited=dict(DS_NAN_PROBE='1',DS_PROFILE_GPU='1',DS_PB_VERIFY_CAPTURE='1',VK_INSTANCE_LAYERS='VK_LAYER_KHRONOS_validation')
        clean=steady.relay.session_environment(inherited,'steady-mprotect')
        for k in ('DS_NAN_PROBE','DS_PROFILE_GPU','DS_BATCH_STAMPS','DS_PB_VERIFY_CAPTURE','DS_RELAY_UI_PROBE'):
            self.assertEqual(clean[k],'0')
        self.assertNotIn('VK_INSTANCE_LAYERS',clean)
        self.assertEqual(clean['DS_VK_VALIDATION'],'0')
        self.assertEqual(clean['DS_LINUX_WRITE_WATCH'],'mprotect')
        diag=steady.relay.session_environment(inherited,'steady-diagnostic-uffd')
        self.assertEqual(diag['DS_PROFILE_GPU'],'1')
        self.assertEqual(diag['DS_PROFILE_GPU_STAMPS'],'seg')
        self.assertEqual(diag['DS_WB_BUCKETS_SPLIT'],'1')
        self.assertEqual(clean['DS_WB_BUCKETS'],'0')
        self.assertEqual(diag['DS_LINUX_WRITE_WATCH'],'uffd')

    def test_sampling_and_wake_candidate_are_isolated_from_clean(self):
        env={'DS_RECORD_FAST':'1','DS_LINUX_FAULT_TIMING':'1','DS_LINUX_ADDRESS_WAKE':'1'}
        clean=steady.relay.session_environment(env,'steady-mprotect')
        self.assertNotIn('DS_RECORD_FAST',clean)
        self.assertEqual(clean['DS_LINUX_FAULT_TIMING'],'0')
        self.assertEqual(clean['DS_LINUX_ADDRESS_WAKE'],'1')
        sample=steady.relay.session_environment(env,'steady-sample-mprotect')
        self.assertEqual(sample['DS_LINUX_FAULT_TIMING'],'1')
        self.assertEqual(sample['DS_PROFILE_GPU'],'0')
        self.assertEqual(sample['DS_RELAY_STEADY_DIAGNOSTIC'],'0')
        self.assertEqual(steady.relay.session_environment(env,'steady-address-wake-mprotect')['DS_LINUX_ADDRESS_WAKE'],'1')
        self.assertEqual(steady.session_plan(['mprotect'],True,True),['steady-a1-mprotect','steady-a2-mprotect','steady-sample-mprotect','steady-diagnostic-mprotect'])

    def test_whole_run_switch_and_cooled_interleaving(self):
        self.assertEqual(steady.whole_run_counts('whole-run recording: enabled true, runs 6, commands 4000'),dict(runs=6,commands=4000))
        self.assertEqual(steady.whole_run_counts(''),dict(runs=0,commands=0))
        self.assertEqual(steady.session_plan(['mprotect'],False,False,'whole-run'),
                         ['steady-a1-baseline-mprotect','steady-b1-whole-run-mprotect','steady-c1-whole-run-quick-mprotect','steady-a2-baseline-mprotect','steady-b2-whole-run-mprotect','steady-c2-whole-run-quick-mprotect','steady-validation-whole-run-quick-mprotect'])
        env={'DS_PAR_WHOLE_RUN':'1','DS_PIPELINE_CRUMBS':'1'}
        for label, expected in [('steady-a1-baseline-mprotect','0'),('steady-b1-whole-run-mprotect','1')]:
            e=steady.relay.session_environment(env,label)
            self.assertEqual(e['DS_PAR_WHOLE_RUN'],expected)
            self.assertEqual(e['DS_PIPELINE_CRUMBS'],'0')
            self.assertEqual(e['DS_LINUX_WRITE_WATCH'],'mprotect')
        invalid={'steady-a1-baseline-mprotect':dict(diagnostic=False,valid_steady_state=False,pair=1),
                 'steady-b1-whole-run-mprotect':dict(diagnostic=False,valid_steady_state=True,pair=1)}
        self.assertFalse(steady.recording_comparison(invalid)['pairs']['whole-run pair 1']['performance_valid'])

    def test_quick_wins_are_independent_and_validation_is_separate(self):
        for selected in ('tsc','static-bind','tsc,static-bind'):
            env={'DS_RELAY_QUICK_WINS':selected,'DS_RECORD_TSC':'1','DS_RECORD_STATIC_BIND':'1'}
            for label in ('steady-a1-baseline-mprotect','steady-b1-whole-run-mprotect','steady-c1-whole-run-quick-mprotect'):
                e=steady.relay.session_environment(env,label)
                for key,choice in [('DS_RECORD_TSC','tsc'),('DS_RECORD_STATIC_BIND','static-bind')]:
                    self.assertEqual(e[key],str(int('quick' in label and choice in selected.split(','))))
            e=steady.relay.session_environment(env,'steady-validation-whole-run-quick-mprotect')
            self.assertEqual(e['DS_VK_VALIDATION'],'1')
            self.assertEqual(e['DS_PROFILE_GPU'],'0')
            self.assertEqual(e['DS_RELAY_STEADY_DIAGNOSTIC'],'1')
            self.assertNotIn('VK_LOADER_LAYERS_DISABLE',e)

    def test_warm_store_requires_complete_clean_cycle_after_last_compile(self):
        calls=[]
        def run(label,attempt,name):
            calls.append(label)
            return dict(invalid_reasons=['walking compiled pipelines inside measurement'] if len(calls) in (1,3) else [])
        results,sessions=steady.warm_store(['A','B','C'],9,run,lambda *args:None)
        self.assertEqual(calls,['A','B','C','A','B','C'])
        self.assertTrue(all(r['warmup_only'] and not r['valid_steady_state'] for r in results.values()))
        self.assertEqual(len(sessions),6)

    def test_warm_store_fails_loudly_for_bad_seed_or_missing_evidence(self):
        for reason,limit in [('walking compiled pipelines inside measurement',3),('route incomplete',1)]:
            calls=[]
            def run(label,attempt,name):
                calls.append(label)
                return dict(invalid_reasons=[reason])
            with self.assertRaisesRegex(ValueError,'no performance sessions started'):
                steady.warm_store(['A','B','C'],3,run,lambda *args:None)
            self.assertEqual(len(calls),limit)

    def test_compile_invalid_summary_never_prints_fps(self):
        result=dict(diagnostic=False,valid_steady_state=False,invalid_reasons=['walking compiled pipelines inside measurement'],
            phases={'walking':dict(thermal=dict(valid=True),fully_warm=False,flips_per_second=999)},waits=[],exit_code=0)
        text=steady.summary({'candidate':result})
        self.assertIn('INVALID compile-contaminated',text)
        self.assertNotIn('999',text)

    def test_inflight_or_new_source_compile_invalidates_phase(self):
        data=dict(seconds=2,counters=dict(flips=100),intervals_us_and_flip=[],threads=[],job_workers={},upscaler='Off')
        counters=dict(source_create_calls=0,source_create_ms=0,source_create_started=0,source_create_active=0)
        self.assertTrue(steady.phase_report(data,dict(begin=counters,end=counters))['fully_warm'])
        for key in ('source_create_started','source_create_active'):
            self.assertFalse(steady.phase_report(data,dict(begin=counters,end=dict(counters,**{key:1})))['fully_warm'])
        self.assertFalse(steady.phase_report(data,dict(begin=dict(counters,source_create_active=1),end=counters))['fully_warm'])

    def test_fault_cost_uses_observed_sample_count_and_direct_faults_only(self):
        self.assertIsNone(steady.fault_cost({},10)['estimated_cpu_ms_per_flip'])
        cost=steady.fault_cost(dict(fault_samples=10,fault_sample_cpu_ns=20000,fault_sample_wall_ns=30000,direct_faults=5000,faults=7000),100)
        self.assertAlmostEqual(cost['estimated_cpu_ms_per_flip'],.1)
        self.assertAlmostEqual(cost['estimated_wall_ms_per_flip'],.15)
        self.assertIn('excludes kernel signal',cost['note'])

    def test_phase_environment_matches_monotonic_window(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'monitor.jsonl'
            rows=[dict(monotonic_s=t,event='sample',temperature_c=100+t,cpu_telemetry={'0':dict(scaling_khz=5000000,package_throttle_count=t)}) for t in range(5)]
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            r=steady.phase_cpu_environment(dict(begin_monotonic_ns=1e9,end_monotonic_ns=3e9),p)
            self.assertEqual(r['samples'],3)
            self.assertEqual(r['temperature_c']['mean'],102)
            self.assertEqual(r['cpus']['0']['package_throttle_count']['delta'],2)

    def test_thermal_measurement_restarts_from_seed_once_and_abort_does_not_retry(self):
        self.assertTrue(steady.needs_warm_retry(dict(invalid_reasons=['stationary thermal pause/abort','walking compiled pipelines inside measurement'])))
        self.assertFalse(steady.needs_warm_retry(dict(invalid_reasons=['thermal/guard abort','stationary thermal pause/abort'])))
        self.assertTrue(steady.needs_warm_retry(dict(invalid_reasons=['stationary thermal pause/abort','stationary insufficient samples','stationary no KHR completion samples'])))
        calls=[]
        def run(label,attempt,name):
            calls.append(attempt)
            return dict(valid_steady_state=False,diagnostic=False,exit_code=0,invalid_reasons=['walking thermal pause/abort'])
        results,_=steady.run_suite(['steady-mprotect'],run,lambda *args: None)
        self.assertEqual(calls,[0,1])
        self.assertTrue(results['steady-mprotect']['warmup_only'])
        self.assertNotIn('warmup_only',results['steady-warm-mprotect'])

    def test_summary_does_not_advertise_sigstop_intervals_as_performance(self):
        r=dict(diagnostic=False,valid_steady_state=False,invalid_reasons=['stationary thermal pause/abort'],
               phases={'stationary':{'thermal':{'valid':False}}},waits=[],exit_code=0)
        text=steady.summary({'paused':r})
        self.assertIn('INVALID thermal pause',text)
        self.assertNotIn('flips/s',text)

    def test_cpu_is_work_not_wall_and_real_source_keys_flag_unwarmed_phase(self):
        data=dict(seconds=2,counters=dict(flips=100,faults=200,upload_bytes=1000000),
                  intervals_us_and_flip=[[10000,1],[30000,2]],upscaler='Off',job_workers={},
                  threads=[dict(name='BPE JobWorker',cpu_s=1,runqueue_s=.1),dict(name='gpu-submit',cpu_s=2,runqueue_s=.2)])
        r=steady.phase_report(data,dict(begin=dict(source_create_calls=5,source_create_ms=100),end=dict(source_create_calls=6,source_create_ms=130)))
        self.assertEqual(r['rates_per_second']['faults'],100)
        self.assertEqual(r['per_flip']['upload_bytes'],10000)
        self.assertEqual(r['cpu_ms_per_flip_by_group'],{'job-workers':10,'renderer':20})
        self.assertFalse(r['fully_warm'])
        self.assertEqual(r['pipelines']['source_create_ms'],30)
        self.assertEqual(r['frame_times']['p99_ms'],30)
        self.assertEqual(r['frame_times']['mean_fps'],50)

    def test_latest_seed_requires_binary_evidence_not_unrelated_thermal_fps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,c in [('run-1',dict(matched_pipelines=1285,matched_hit_fraction=1,thermal_valid=False)),('run-2',dict(matched_pipelines=0,matched_hit_fraction=0))]:
                p=root/name;(p/'shaders').mkdir(parents=True)
                (p/'comparison.json').write_text(json.dumps(c))
            self.assertEqual(steady.latest_seed(root),root/'run-1')

    def test_complete_report_distinguishes_clean_diagnostic_and_missing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); user=root/'user/run';user.mkdir(parents=True)
            (user/'dsport.log').write_text('render size 3840x2160 set\nswapchain 3840x2160 B8G8\nSDL window mode 1 applied\nsteady route complete: stationary then walking\n')
            (root/'gpu-clocks.csv').write_text('elapsed_s,index,pstate,graphics_MHz,memory_MHz,gpu_util_pct,memory_util_pct,power_W,phase\n1,0,P0,2800,14000,90,2,160,steady-walking\n')
            for label,seconds in [('stationary',30),('walking',60)]:
                (user/(label+'-steady.json')).write_text(json.dumps(dict(phase=label,seconds=seconds,counters=dict(flips=100),intervals_us_and_flip=[[16667,100 if label=='stationary' else 300]],threads=[],job_workers={},upscaler='Off')))
                (user/(label+'-cpu-self.json')).write_text(json.dumps(dict(valid=True,flips=32,top10=[dict(name='packet loop',self_ms_per_flip=1,calls=200)])))
                (user/(label+'-pipeline-reuse.json')).write_text(json.dumps(dict(begin=dict(source_create_calls=0,source_create_ms=0,source_create_started=0,source_create_active=0),end=dict(source_create_calls=0,source_create_ms=0,source_create_started=0,source_create_active=0))))
                for edge in ['before','after']: (user/f'steady-{label}-{edge}.png').write_bytes(b'fixture')
            metrics=dict(exit_code=0,validation_errors=0,crash_bytes=0,thermal=dict(measurement_valid=True),effective_validation=False,
                         write_watch_requested='mprotect',write_watch='mprotect',timelines=[dict(source_file='timeline-132.json',frames=31,gpu_timestamp_spans=0),dict(source_file='timeline-332.json',frames=31,gpu_timestamp_spans=0)],presentation={p:dict(wait_complete=dict(samples=50)) for p in ('stationary','walking')})
            self.assertTrue(steady.report(root,metrics)['valid_steady_state'])
            self.assertEqual([t['phase'] for t in metrics['timelines']],['stationary','walking'])
            missing=steady.report(root,dict(metrics,timelines=[]))
            self.assertIn('walking missing adjacent CPU wait trace',missing['invalid_reasons'])
            diagnostic=steady.report(root,metrics,True)
            self.assertFalse(diagnostic['valid_steady_state'])
            self.assertIn('no diagnostic GPU timestamps',diagnostic['invalid_reasons'])
            sampling=steady.report(root,dict(metrics,profile='steady-sample-mprotect'),True)
            self.assertEqual(sampling['invalid_reasons'],[])
            self.assertFalse(sampling['valid_steady_state'])
            metrics['thermal']['measurement_valid']=False
            self.assertIn('stationary thermal pause/abort',steady.report(root,metrics)['invalid_reasons'])
            metrics['thermal']['measurement_valid']=True
            (user/'stationary-pipeline-reuse.json').write_text('{}')
            self.assertFalse(steady.report(root,metrics)['valid_steady_state'])

    def test_retry_only_for_compilation_not_thermal_or_other_failures(self):
        self.assertTrue(steady.needs_warm_retry(dict(invalid_reasons=['walking compiled pipelines inside measurement'])))
        self.assertFalse(steady.needs_warm_retry(dict(invalid_reasons=[])))
        self.assertFalse(steady.needs_warm_retry(dict(invalid_reasons=['walking compiled pipelines inside measurement','thermal pause/abort'])))

    def test_create_started_before_boundary_still_contaminates_warm_interval(self):
        data=dict(seconds=2,counters=dict(flips=100),intervals_us_and_flip=[],threads=[],job_workers={},upscaler='Off')
        r=steady.phase_report(data,dict(begin=dict(source_create_calls=5,source_create_ms=100),end=dict(source_create_calls=5,source_create_ms=130)))
        self.assertFalse(r['fully_warm'])

    def test_phase_thermal_ignores_loading_pause_but_not_overlap_or_missing_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'monitor.jsonl'
            path.write_text('\n'.join(json.dumps(dict(event=e,monotonic_s=t)) for e,t in [('pause',10),('resume',14)]))
            thermal=dict(measurement_valid=False,pauses=1,aborted=None)
            phase=dict(begin_monotonic_ns=20_000_000_000,end_monotonic_ns=30_000_000_000)
            self.assertTrue(steady.phase_thermal(phase,thermal,path)['valid'])
            phase['begin_monotonic_ns']=12_000_000_000
            r=steady.phase_thermal(phase,thermal,path)
            self.assertFalse(r['valid']);self.assertEqual(r['paused_s'],2)
            thermal['aborted']='critical heat'
            self.assertFalse(steady.phase_thermal(phase,thermal,path)['valid'])
            thermal['aborted']=None;thermal['pauses']=2
            self.assertFalse(steady.phase_thermal(phase,thermal,path)['valid'])
            self.assertFalse(steady.phase_thermal({},thermal,path)['valid'])

    def test_clean_precedes_diagnostic_and_tracking_comparison(self):
        self.assertEqual(steady.session_plan(['mprotect','uffd']),['steady-a1-mprotect','steady-b1-uffd','steady-a2-mprotect','steady-b2-uffd','steady-diagnostic-mprotect'])
        self.assertEqual(steady.session_plan(['mprotect'],False,False),['steady-a1-mprotect','steady-a2-mprotect'])

    def test_suite_attempts_every_session_after_runtime_and_analysis_failures(self):
        labels=['steady-mprotect','steady-uffd','steady-diagnostic-mprotect']
        calls=[]; snapshots=[]
        def run(label,attempt,name):
            calls.append(name)
            if label=='steady-mprotect':
                return dict(valid_steady_state=False,diagnostic=False,exit_code=13,invalid_reasons=['route incomplete'])
            if label=='steady-uffd': raise ValueError('broken metrics')
            return dict(valid_steady_state=False,diagnostic=True,exit_code=0,invalid_reasons=[])
        def persist(results,sessions): snapshots.append(json.loads(json.dumps(sessions)))
        results,sessions=steady.run_suite(labels,run,persist)
        self.assertEqual(calls,labels)
        self.assertEqual([v['state'] for v in sessions.values()],['failed','failed','completed'])
        self.assertIn('broken metrics',results['steady-uffd']['invalid_reasons'][0])
        self.assertEqual(snapshots[0]['steady-uffd']['state'],'pending')
        self.assertEqual(snapshots[-1],sessions)

    def test_summary_names_all_sessions_even_when_none_reaches_gameplay(self):
        labels=['steady-mprotect','steady-uffd','steady-diagnostic-mprotect']
        summaries=[]
        def run(label,attempt,name):
            if label=='steady-uffd': raise RuntimeError('analysis failed')
            return dict(valid_steady_state=False,diagnostic='diagnostic' in label,exit_code=13,
                        invalid_reasons=['route incomplete'],phases={},waits=[],selfplay=['FAILED: offline dialog'])
        steady.run_suite(labels,run,lambda r,s: summaries.append(steady.summary(r,s)))
        for label in labels:
            self.assertIn(label+': pending',summaries[0])
            self.assertIn(label+': failed',summaries[-1])
        self.assertIn('runtime exit: 13',summaries[-1])
        self.assertIn('analysis failed',summaries[-1])
        self.assertIn('FAILED: offline dialog',summaries[-1])

    def test_guard_failure_gets_independent_next_session_and_cooldown(self):
        calls=[]
        def run(label,attempt,name):
            calls.append(name)
            raise steady.linux_guard.GuardError('cooldown not reached')
        _,sessions=steady.run_suite(['steady-mprotect','steady-uffd'],run,lambda *args: None)
        self.assertEqual(calls,['steady-mprotect','steady-uffd'])
        self.assertTrue(all(v['state']=='failed' for v in sessions.values()))

    def test_cancellation_is_not_mistaken_for_a_retryable_failure(self):
        snapshots=[];calls=[]
        def run(label,attempt,name): calls.append(name);raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            steady.run_suite(['steady-mprotect','steady-uffd'],run,lambda r,s: snapshots.append(json.loads(json.dumps(s))))
        self.assertEqual(calls,['steady-mprotect'])
        self.assertEqual(snapshots[-1]['steady-mprotect']['state'],'cancelled')
        self.assertEqual(snapshots[-1]['steady-uffd']['state'],'not run (user cancelled)')

    def test_runtime_interrupt_code_cancels_pending_sessions(self):
        snapshots=[];calls=[]
        def run(label,attempt,name):
            calls.append(name)
            return dict(valid_steady_state=False,diagnostic=False,exit_code=130,invalid_reasons=['runtime interrupted'])
        with self.assertRaises(KeyboardInterrupt):
            steady.run_suite(['steady-mprotect','steady-uffd'],run,lambda r,s: snapshots.append(json.loads(json.dumps(s))))
        self.assertEqual(calls,['steady-mprotect'])
        self.assertEqual(snapshots[-1]['steady-mprotect']['state'],'cancelled')
        self.assertEqual(snapshots[-1]['steady-uffd']['state'],'not run (user cancelled)')

    def test_warm_retry_is_bounded_and_then_runs_remaining_sessions(self):
        calls=[]
        def run(label,attempt,name):
            calls.append((label,attempt,name))
            return dict(valid_steady_state=False,diagnostic=False,exit_code=0,invalid_reasons=['walking compiled pipelines inside measurement'])
        results,_=steady.run_suite(['steady-mprotect','steady-uffd'],run,lambda *args: None)
        self.assertEqual([a for _,a,_ in calls],[0,1,0,1])
        self.assertTrue(results['steady-mprotect']['warmup_only'])
        self.assertNotIn('warmup_only',results['steady-warm-mprotect'])

    def test_interleaved_pairs_keep_temperatures_and_never_substitute_other_pair(self):
        p=dict(flips_per_second=40,rates_per_second={},cpu_ms_per_flip_by_group={},frame_times={},cpu_environment=dict(temperature_c=dict(mean=99)))
        a=dict(diagnostic=False,valid_steady_state=True,actual_tracking='mprotect',pair=1,phases=dict(stationary=p,walking=p))
        b=dict(a,actual_tracking='uffd',phases=dict(stationary=dict(p,flips_per_second=42),walking=p))
        results=dict(a1=a,b1=b,a2=dict(a,pair=2,valid_steady_state=False),b2=dict(b,pair=2))
        c=steady.tracking_comparison(results)
        self.assertAlmostEqual(c['pairs']['1']['phases']['stationary']['fps_change_pct'],5)
        self.assertEqual(c['pairs']['1']['phases']['stationary']['temperature_before']['mean'],99)
        self.assertFalse(c['pairs']['2']['performance_valid'])
        labels=steady.session_plan(['mprotect','uffd'],False)
        calls=[]
        def run(label,attempt,name):
            calls.append(name)
            return dict(diagnostic=False,valid_steady_state=False,exit_code=0,invalid_reasons=['walking compiled pipelines inside measurement'])
        steady.run_suite(labels,run,lambda *a:None)
        self.assertEqual(calls,labels) # no A,A warm retry inserted into A,B,A,B

    def test_failed_pair_two_is_not_omitted_from_comparison(self):
        labels=steady.session_plan(['mprotect','uffd'],False)
        def run(*args): raise ValueError('cooldown timed out')
        results,sessions=steady.run_suite(labels,run,lambda *a:None)
        pairs=steady.tracking_comparison(results)['pairs']
        self.assertEqual(set(pairs),{'1','2'})
        self.assertFalse(pairs['1']['performance_valid'])
        self.assertFalse(pairs['2']['performance_valid'])
        self.assertTrue(all(s['state']=='failed' for s in sessions.values()))

    def test_cooldown_requires_evidence_at_or_below_seventy(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cooldown.jsonl'
            self.assertFalse(steady.cooldown_report(p)['valid'])
            p.write_text(json.dumps(dict(event='cooldown',temperature_c=70,target_c=70,monotonic_s=5))+'\n'+json.dumps(dict(event='sample',temperature_c=103,monotonic_s=9)))
            self.assertTrue(steady.cooldown_report(p)['valid'])
            p.write_text(json.dumps(dict(event='cooldown',temperature_c=71,target_c=70,monotonic_s=5)))
            self.assertFalse(steady.cooldown_report(p)['valid'])

    def test_tracking_comparison_requires_valid_warm_pairs_and_not_visual_acceptance(self):
        p=dict(flips_per_second=40,rates_per_second=dict(faults=58000),cpu_ms_per_flip_by_group={'renderer':42},frame_times={})
        a=dict(diagnostic=False,valid_steady_state=True,actual_tracking='mprotect',phases=dict(stationary=p,walking=p))
        q=dict(p,flips_per_second=44,rates_per_second=dict(faults=0))
        b=dict(a,actual_tracking='uffd',phases=dict(stationary=q,walking=q))
        r=steady.tracking_comparison(dict(a=a,b=b))
        self.assertTrue(r['performance_valid']);self.assertFalse(r['correctness_verified'])
        self.assertAlmostEqual(r['phases']['walking']['fps_change_pct'],10)
        b['valid_steady_state']=False
        self.assertFalse(steady.tracking_comparison(dict(a=a,b=b))['performance_valid'])
