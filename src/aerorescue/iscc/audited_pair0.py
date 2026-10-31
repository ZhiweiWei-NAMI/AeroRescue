"""Leak-isolated heterogeneous RGB-D ISCC training and measured evaluation.

All online work is in distributed.py, whose endpoints never accept evaluator
records. This module alone consumes branch labels and geometric truth for scoring.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .rgbd import bilinear,project
from . import rgbd,wire_protocol as W
from .distributed import OURS,METHODS,cache_model_input,new_state,run_wire_step,task_embedding,shrink_task_packet
from .pair0_qwen_decoder import Pair0QwenVLMRuntime,fit_independent_reference_mapping
from .gating_and_baselines import compute_fisher_and_vertical_by,evaluate_safety_and_regret_metrics
from .geometry import se3_exp,transport_pose_covariance


def depth_correspondence_anchors(tc,T,sigma):
    """Measured joint-depth geometry diagnostic; never used by online selection."""
    K=tc['K'];ug=tc['ug'];rr,cc=np.mgrid[4:384:8,4:640:8]
    uv=np.column_stack([cc.ravel()+.5,rr.ravel()+.5]);depth=ug['depth'][rr,cc].ravel()
    points=(np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T)*depth[:,None]
    normals=ug['normal_cam'][rr,cc].reshape(-1,3);pred=points@T[:3,:3].T+T[:3,3]
    _,ids=tc['tree'].query(pred);observed=tc['cloud'][ids]
    from .geometry import j_se3_left
    J=np.array([j_se3_left(p) for p in pred]);tolerance=.15+.03*np.maximum(pred[:,2],0)
    covariance=np.einsum('nij,jk,nlk->nil',J,sigma,J)+tolerance[:,None,None]**2*np.eye(3)
    residual=pred-observed;mahal=np.einsum('ni,ni->n',residual,np.linalg.solve(covariance,residual[...,None])[...,0])
    normal_ok=np.sum((normals@T[:3,:3].T)*tc['cloud_normals'][ids],axis=1)>=.7
    valid=(depth>.5)&(pred[:,2]>.5)&normal_ok&(mahal<11.345)
    token=(rr.ravel()//32)*20+cc.ravel()//32;chosen={}
    for i in np.flatnonzero(valid):
        key=int(token[i])
        if key not in chosen or mahal[i]<mahal[chosen[key]]:chosen[key]=int(i)
    return [dict(token_idx=key,p_i=points[i],source_uv=uv[i]) for key,i in chosen.items()]


def q_values(record):
    labels=record['supervision_and_eval']['branch_evaluations']
    return np.array([labels[c['candidate_id']]['q_value_120s'] for c in record['model_input']['public_candidate_menu']])


def fit_train_head(records,cache,rt,ref):
    """Train-only visual bootstrap for the common four-dimensional readout space."""
    train=[r for r in records if r['metadata']['split']=='train']
    ego=[];target=[];features=[]
    for record in train:
        tc=cache[record['metadata']['record_id']];K=len(record['model_input']['public_candidate_menu'])
        ego.append(rt.score_residual_calibrated_decoder(tc['receiver']['tokens'],tc['prompt'],K)['z_ego'])
        target.append(q_values(record))
        features.append((ref['U_rec'].T@tc['teacher']['tokens'][:,tc['slot_ids']]).T)
    E=np.asarray(ego);Q=np.asarray(target);Y=np.asarray(features)
    Ec=E-E.mean(0);Qc=Q-Q.mean(0)
    Wego=np.linalg.solve(Ec.T@Ec+np.eye(E.shape[1]),Ec.T@Qc);bias=Q.mean(0)-E.mean(0)@Wego
    residual=Q-(E@Wego+bias);N,K,p=Y.shape
    flat=Y.reshape(N,K*p)
    # Dual ridge avoids a Kp by Kp inverse when the train set is small.
    W=(flat.T@np.linalg.solve(flat@flat.T+10*np.eye(N),residual)).T.reshape(K,K,p)
    _,_,vt=np.linalg.svd(W.reshape(K*K,p),full_matrices=False);basis=vt[:4].T
    latent=np.einsum('njp,pl->njl',Y,basis).reshape(N,K*4)
    weights=(latent.T@np.linalg.solve(latent@latent.T+np.eye(N),residual)).T.reshape(K,K,4)
    scale=float(np.std(Q)/max(np.std(E),1e-6))
    rt.configure_decision_head(ref['U_rec']@basis,weights,Wego,bias,scale)
    return dict(train_record_ids=[r['metadata']['record_id'] for r in train],communication_dropout_probability=.5,
        training_form='train-only ridge bootstrap followed by actual Decoder adapter optimization',decoder_residual_scale=scale)


def _bootstrap_query(cache,r):
    return W.decode_query(W.encode_query(tag=0,seq=0,T_e_i=cache['T_prior'],covariance=cache['covariance'],
        K_e=cache['receiver']['K'],anchors_e=cache['anchors_e'],r=r))


def _teacher_slots(cache,teacher_tokens,query):
    slots=rgbd.sender_slots(cache['source'],None,query,feature_tokens=teacher_tokens)
    output=np.zeros((teacher_tokens.shape[0],cache['K']))
    mask=np.zeros(cache['K'],bool)
    for slot in slots:output[:,slot['slot_id']]=slot['x_k'];mask[slot['slot_id']]=True
    return output,mask


def _independent_reference(records,caches,teachers):
    X=[];Y=[]
    for record in records:
        if record['metadata']['split']!='train':continue
        rid=record['metadata']['record_id'];cache=caches[rid];query=_bootstrap_query(cache,32)
        slots=rgbd.sender_slots(cache['source'],None,query)
        for slot in slots:
            uv=slot['roi_uv'];uv=uv[np.linspace(0,len(uv)-1,min(8,len(uv))).astype(int)]
            X.append(bilinear(cache['source']['tokens'],uv));Y.append(bilinear(teachers[rid],uv))
    X=np.hstack(X);Y=np.hstack(Y)
    ref=fit_independent_reference_mapping(X,Y,32,32,2.)
    xs=ref['U_src'].T@X;ys=ref['U_rec'].T@Y
    residual=ys-ref['A_ref_sub']@xs
    ref['sigma_obs']=float(max(np.sqrt(np.mean(residual**2)),.1))
    ref['prior_information']=(xs@xs.T+2*np.eye(32))/ref['sigma_obs']**2
    ref['basis_tag']=W.basis_tag(ref['U_src'],ref['U_rec'])
    return ref


def _prepare_reader(records,caches,teachers,rt,ref):
    bootstrap={}
    for record in records:
        if record['metadata']['split']!='train':continue
        rid=record['metadata']['record_id'];cache=caches[rid]
        target,_=_teacher_slots(cache,teachers[rid],_bootstrap_query(cache,32))
        bootstrap[rid]=dict(receiver={'tokens':cache['receiver']['tokens']},teacher={'tokens':target},
                            prompt=cache['prompt'],slot_ids=np.arange(cache['K']))
    info=fit_train_head(records,bootstrap,rt,ref)
    ref['U_dec']=rt.U_dec.copy();ref['static_source_task']=ref['A_ref_sub'].T@(ref['U_rec'].T@rt.U_dec)
    rows=[];source_x=[];teacher_y=[];height=[]
    train=[r for r in records if r['metadata']['split']=='train']
    # Actual decoded payloads cover every adapted strategy. Readout supervision is
    # shared and equal; no baseline is forced through a reader trained only on Ours.
    strategies=[OURS,'Where2comm-Adapted','Which2comm-Adapted','RDcomm-Adapted','BOLT-Adapted','KVComm-Adapted']
    rng=np.random.default_rng(42)
    for index,record in enumerate(train):
        rid=record['metadata']['record_id'];cache=caches[rid]
        for budget in (512,2048):
            for method in strategies:
                augmented=cache
                if budget==512:
                    delta=np.r_[rng.normal(0,np.deg2rad(.3),3),rng.normal(0,.15,3)]
                    augmented=dict(cache,T_prior=se3_exp(delta)@cache['T_prior'],
                                   covariance=cache['covariance']+np.diag([np.deg2rad(.3)**2]*3+[.15**2]*3))
                packet=run_wire_step(augmented,ref,rt.U_dec,np.ones(4),budget,new_state(ref),method)
                row=dict(record=record,cache=cache,packet=packet,method=method,budget=budget)
                rows.append(row)
                if method==OURS:
                    source_x.append(packet['X']);height.append(packet['aux'][:,0]);
                    query=dict(packet['query_decoded'])
                    reply=packet['reply_decoded']
                    if reply['T'] is not None:query.update(T=reply['T'],covariance=reply['covariance'])
                    teacher,_=_teacher_slots(cache,teachers[rid],query);teacher_y.append(teacher)
        print('  train wire examples',index+1,'/',len(train),flush=True)
    ours=[row for row in rows if row['method']==OURS]
    K=caches[train[0]['metadata']['record_id']]['K'];p=ref['U_rec'].shape[1]
    masks=np.array([row['packet']['mask'] for row in ours])
    mapped=np.array([ref['U_rec'].T@row['packet']['Y'] for row in ours])
    denominators=np.maximum(masks.sum(0),1)
    ref['wire_mean']=np.sum(mapped*masks[:,None,:],axis=0)/denominators[None]
    ref['height_mean']=np.sum(np.array(height)*masks,axis=0)/denominators
    actual_teacher=np.array([ref['U_rec'].T@y for y in teacher_y])
    teacher_mean=np.sum(actual_teacher*masks[:,None,:],axis=0)/denominators[None]
    errors=(mapped-ref['wire_mean'][None]-(actual_teacher-teacher_mean[None]))
    samples=np.concatenate([errors[i][:,mask] for i,mask in enumerate(masks)],axis=1)
    ref['model_covariance']=samples@samples.T/max(samples.shape[1],1)
    ref['geometry_axis']=rt.U_dec[:,0].copy()
    if not samples.shape[1]:raise RuntimeError('training received no actual blind-slot observations')
    visual=[];relief=[]
    for row in ours:
        packet=row['packet'];mask=packet['mask']
        visual.extend((ref['geometry_axis']@ref['U_rec']@(ref['U_rec'].T@packet['Y']-ref['wire_mean']))[mask].tolist())
        relief.extend((packet['aux'][:,0]-ref['height_mean'])[mask].tolist())
    correlation=float(np.dot(visual,relief))
    ref['geometry_scale']=float(np.sign(correlation or 1)*np.std(visual)/max(np.std(relief),.1))
    ref['geometry_scale']=float(np.clip(ref['geometry_scale'],-10,10))
    features=[];targets=[];raw_ego=[];examples=[]
    for row in rows:
        record=row['record'];cache=row['cache'];packet=row['packet']
        values=task_embedding(packet,ref)
        features.append((rt.U_dec.T@values).T.reshape(-1));targets.append(q_values(record))
        ego=rt.score_residual_calibrated_decoder(cache['receiver']['tokens'],cache['prompt'],K)['z_ego']
        raw_ego.append(ego)
        examples.append(dict(v=values,ground=cache['receiver']['tokens'],prompt=cache['prompt'],positions=packet['positions'],q=q_values(record)))
    F=np.array(features);R=np.array(targets)-np.array(raw_ego)
    weights=np.linalg.solve(F.T@F+.1*np.eye(F.shape[1]),F.T@R).T.reshape(K,K,4)
    rt.configure_decision_head(rt.U_dec,weights,rt.W_ego,rt.ego_bias,rt.decoder_scale)
    info['wire_training_strategies']=strategies;info['feature_centering']='train payload means, missing slots remain exactly zero'
    info['adapter_training']=rt.train_soft_reader(examples,epochs=2,learning_rate=.003,dropout_probability=.5)
    return info


def _probe(rt,cache):
    out=rt.score_residual_calibrated_decoder(cache['receiver']['tokens'],cache['prompt'],cache['K'],
        np.zeros((rt.d_e,cache['K']),np.float32),cache['anchors_e'].T.astype(np.float32),compute_shadow_jacobian=True)
    fisher=compute_fisher_and_vertical_by(out['probs'],out['J_Y_slots'],4)
    if fisher['discarded_fisher_energy_fraction']>1e-8:raise AssertionError('deployed head left its common rank-4 space')
    return out,fisher


def _score(rt,cache,values,positions,diagnostics=False):
    return rt.score_residual_calibrated_decoder(cache['receiver']['tokens'],cache['prompt'],cache['K'],
        values.astype(np.float32),positions.T.astype(np.float32),collect_layer_diagnostics=diagnostics)


def _episode_key(record):
    meta=record['metadata']
    return (meta['scene_id'],meta['seed'],meta['sample_kind'],meta['record_id'] if meta['sample_kind']=='challenge_matched' else None)


def _packets(records,caches,probes,ref,budget,methods):
    output={method:[] for method in methods}
    for method in methods:
        previous=None;state=None
        for record in records:
            key=_episode_key(record)
            if key!=previous:state=new_state(ref,method=='Ours-NoWoodbury');previous=key
            rid=record['metadata']['record_id'];cache=caches[rid];_,fisher=probes[rid]
            packet=run_wire_step(cache,ref,fisher['U_act'],fisher['lambda_act'],budget,state,method)
            output[method].append(packet)
    return output


def _calibrate_gate(records,caches,probes,ref,rt):
    val=[r for r in records if r['metadata']['split']=='val']
    stored={b:_packets(val,caches,probes,ref,b,[OURS])[OURS] for b in (512,1024,2048,4096)}
    grid=[]
    for kappa in (.125,.25,.5,1.,2.,4.,8.):
        regrets=[];negative=[];active=[]
        for budget,packets in stored.items():
            for record,packet in zip(val,packets):
                rid=record['metadata']['record_id'];cache=caches[rid];own,fisher=probes[rid]
                gate=shrink_task_packet(packet,ref,fisher['U_act'],kappa)
                prediction=own['z_ego'] if gate['all_off'] else _score(rt,cache,gate['Y_gated'],packet['positions'])['z_dec']
                q=q_values(record);chosen=int(np.argmax(prediction));ego=int(np.argmax(own['z_ego']))
                regrets.append(float(q.max()-q[chosen]));negative.append(bool(q[chosen]<q[ego]-1e-4));active.append(not gate['all_off'])
        grid.append(dict(kappa=kappa,mean_regret=float(np.mean(regrets)),ntr=float(np.mean(negative)),coverage=float(np.mean(active))))
    safe=[row for row in grid if row['ntr']<=.05]
    chosen=min(safe or grid,key=lambda row:(row['mean_regret'],row['ntr'],-row['coverage'],row['kappa']))
    return chosen['kappa'],grid


def _true_metrics(record,cache,packet):
    truth=np.asarray(record['supervision_and_eval']['true_pose_e_from_i']['T_e_from_i'])
    # An evaluator-only comparison at the same actual transmitted source points.
    points=[]
    if packet.get('raw_messages'):
        points=[s['p_i'] for s in W.decode_up(packet['raw_messages'][3])['slots']]
    if not points:return dict(pose_error_prior_m=None,pose_error_post_m=None)
    p=np.array(points);star=p@truth[:3,:3].T+truth[:3,3]
    before=p@cache['T_prior'][:3,:3].T+cache['T_prior'][:3,3]
    after=p@packet['Tpost'][:3,:3].T+packet['Tpost'][:3,3]
    return dict(pose_error_prior_m=float(np.linalg.norm(before-star,axis=1).mean()),pose_error_post_m=float(np.linalg.norm(after-star,axis=1).mean()))


def sequential_trajectory(sequence,caches,ref,rt,run_dir,kappa):
    state=new_state(ref);rows=[];previous=None;correction=np.eye(4);cov=None;normal=[];threshold=float('inf');was_high=False
    for record in sequence:
        rid=record['metadata']['record_id'];cache=dict(caches[rid]);prior=cache['T_prior'].copy()
        if cov is None:cov=cache['covariance'].copy()
        if previous is not None:
            delta=prior@np.linalg.inv(previous);turn=float(np.linalg.norm(delta[:3,:3]-np.eye(3)))
            cov=transport_pose_covariance(cov,delta,np.diag([1e-5+.001*turn]*3+[.001+.02*turn]*3))
        cache['T_prior']=correction@prior;cache['covariance']=cov
        own,fisher=_probe(rt,cache)
        packet=run_wire_step(cache,ref,fisher['U_act'],fisher['lambda_act'],2048,state,OURS)
        gate=shrink_task_packet(packet,ref,fisher['U_act'],kappa)
        cov=packet['Spost'];previous=prior
        if packet['locked']:correction=packet['Tpost']@np.linalg.inv(prior)
        uncertainty=float(np.mean(np.where(np.isfinite(gate['modelled_variance']),gate['modelled_variance'],0))) if packet['locked'] else float('inf')
        step=record['trajectory']['step']
        if 4<=step<=8 and np.isfinite(uncertainty):normal.append(uncertainty)
        if step==8 and normal:threshold=float(np.median(normal)+5*max(np.std(normal),.05*np.median(normal)))
        high=step>=9 and uncertainty>threshold;trigger=packet['zero_overlap'] or (high and not was_high);was_high=high
        ledger=packet['ledger'];total=max(1,sum(ledger[k] for k in ('b_g_bytes','b_a_bytes','b_s_bytes')))
        rows.append(dict(step=step,source_record_id=rid,sim_time_s=record['metadata']['sim_time_s'],provenance='continuous_native_simulator',
            bg_pct=100*ledger['b_g_bytes']/total,ba_pct=100*ledger['b_a_bytes']/total,bs_pct=100*ledger['b_s_bytes']/total,
            zero_overlap=packet['zero_overlap'],trigger_active_view=bool(trigger),all_off=gate['all_off'],
            rep_uncertainty_trace=uncertainty if np.isfinite(uncertainty) else None,active_view_threshold=threshold if np.isfinite(threshold) else None,
            pose_covariance_trace=float(np.trace(cov)),byte_ledger=ledger,**_true_metrics(record,cache,packet)))
        print('  sequential actual wire',step,'locked',packet['locked'],'all off',gate['all_off'],flush=True)
    (run_dir/'sequential_filter_trace.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows


def run_audited_pair0(records,run_dir,plotter,runtime=None,pair_description=None):
    run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    rt=runtime or Pair0QwenVLMRuntime('cuda:0')
    caches={};teachers={}
    data_dir=Path(records[0]['model_input']['aerial_ground_views']['uav_view']['depth_path']).parent.parent
    sequence=[json.loads(line) for line in (data_dir/'sequential_diagnostic_records.jsonl').read_text().splitlines() if line.strip()]
    for i,record in enumerate(records+sequence):
        rid=record['metadata']['record_id'];cache=cache_model_input(record['model_input'],rt);caches[rid]=cache
        if record in records:teachers[rid]=rt.extract_post_merger_tokens(Image.fromarray(cache['source']['rgb']),'ground')['tokens']
        print('  real sensor/model cache',i+1,'/',len(records)+len(sequence),rid,flush=True)
    ref=_independent_reference(records,caches,teachers)
    training=_prepare_reader(records,caches,teachers,rt,ref)
    probes={}
    for record in records:
        rid=record['metadata']['record_id'];probes[rid]=_probe(rt,caches[rid])
        print('  actual deployed Jacobian',rid,probes[rid][1]['discarded_fisher_energy_fraction'],flush=True)
    kappa,calibration=_calibrate_gate(records,caches,probes,ref,rt)
    test=[r for r in records if r['metadata']['split']=='test'];results={};detailed=[];spectra=[];points=[];layer_raw=None;layer_gate=None
    raw_sv=np.linalg.svd(ref['A_ref_unconstrained'],compute_uv=False)
    for record in test:
        BY=probes[record['metadata']['record_id']][1]['B_Y'];sv=np.linalg.svd(BY@ref['A_ref_unconstrained'],compute_uv=False)
        spectra.append(dict(srank_raw=float(np.sum(raw_sv**2)/raw_sv[0]**2),srank_pullback=float(np.sum(sv**2)/max(sv[0]**2,1e-30)),raw_singular_values=raw_sv.tolist(),pullback_singular_values=sv.tolist()))
    for budget in (512,1024,2048,4096):
        packets=_packets(test,caches,probes,ref,budget,METHODS);summary={};allq=np.array([q_values(r) for r in test]);ego=np.array([int(np.argmax(probes[r['metadata']['record_id']][0]['z_ego'])) for r in test])
        for method in METHODS:
            selected=[];active=[];used=[];nmse=[];rows=[]
            for record,packet in zip(test,packets[method]):
                rid=record['metadata']['record_id'];cache=caches[rid];own,fisher=probes[rid]
                gated=method in (OURS,'Ours-NoWoodbury')
                gate=shrink_task_packet(packet,ref,fisher['U_act'],kappa) if gated else None
                values=gate['Y_gated'] if gated else task_embedding(packet,ref)
                if method=='Ego-Only':values[:]=0
                on=bool(np.any(values));diagnostic=budget==2048 and method=='Ours-NoGate' and layer_raw is None and record['metadata']['sample_kind']=='challenge_matched'
                output=_score(rt,cache,values,packet['positions'],diagnostic) if on or diagnostic else dict(z_dec=own['z_ego'])
                choice=int(np.argmax(output['z_dec']));selected.append(choice);active.append(on);used.append(packet['ledger']['total_used_bytes'])
                query=packet.get('query_decoded',_bootstrap_query(cache,32));reply=packet.get('reply_decoded')
                if reply and reply['T'] is not None:query=dict(query,T=reply['T'],covariance=reply['covariance'])
                target,valid=_teacher_slots(cache,teachers[rid],query)
                error=float(np.sum((fisher['U_act'].T@(packet['Y']-target))**2)/max(np.sum((fisher['U_act'].T@target)**2),1e-9))
                nmse.append(error)
                row=dict(record_id=rid,budget_bytes=budget,method=method,selected_idx=choice,logits=np.asarray(output['z_dec']).tolist(),
                    regret=float(q_values(record).max()-q_values(record)[choice]),active=on,nmse_sub=error,
                    byte_ledger=packet['ledger'],zero_overlap=packet['zero_overlap'],pose_covariance_trace=float(np.trace(packet['Spost'])),
                    discarded_fisher_energy_fraction=fisher['discarded_fisher_energy_fraction'],**_true_metrics(record,cache,packet))
                rows.append(row);detailed.append(row)
                if diagnostic:
                    layer_raw=output['layer_diagnostics'];good=shrink_task_packet(packet,ref,fisher['U_act'],kappa)
                    layer_gate=_score(rt,cache,good['Y_gated'],packet['positions'],True)['layer_diagnostics']
                if budget==2048 and method==OURS:
                    for anchor in packet['source_trace']:
                        if 'grad_norm' in anchor:points.append(dict(grad_norm=anchor['grad_norm'],subspace_energy=float(np.linalg.norm(anchor['x_k'])**2),edge_energy=anchor['edge_energy']))
            selected=np.array(selected);metrics=evaluate_safety_and_regret_metrics(allq,ego,selected,np.array(active),np.array(used))
            metrics['mean_subspace_nmse']=float(np.mean(nmse));metrics['by_sample_kind']={}
            for kind in ('natural','challenge_matched'):
                ids=np.array([i for i,r in enumerate(test) if r['metadata']['sample_kind']==kind],int)
                if len(ids):metrics['by_sample_kind'][kind]=evaluate_safety_and_regret_metrics(allq[ids],ego[ids],selected[ids],np.array(active)[ids],np.array(used)[ids])
            summary[method]=metrics
        results[str(budget)]=dict(methods=summary,test_samples=len(test))
        print('  actual test budget',budget,{m:round(v['mean_regret'],6) for m,v in summary.items()},flush=True)
    trace=sequential_trajectory(sequence,caches,ref,rt,run_dir,kappa)
    measurements=dict(srank_records=spectra,phase_plane_points=points,layer_drift_no_gate=layer_raw,layer_drift_gated=layer_gate,results_by_budget=results,seq_phase_trace=trace)
    plotter(run_dir/'diagnostics',**measurements)
    payload={key:value for key,value in ref.items() if isinstance(value,np.ndarray)}
    np.savez_compressed(run_dir/'train_only_parameters.npz',**payload,W_dec=rt.W_dec,W_ego=rt.W_ego,ego_bias=rt.ego_bias,reader_mix=rt.reader_mix.detach().cpu().numpy())
    accepted=all(results[str(b)]['methods'][OURS]['mean_regret']<results[str(b)]['methods']['Ego-Only']['mean_regret'] and results[str(b)]['methods'][OURS]['active_coverage_pct']>0 for b in (1024,2048,4096))
    zero_safe=all(row['all_off'] for row in trace if row['step'] in (9,10))
    attention_measured=bool(layer_raw and any(row.get('full_attention_ratio') is not None for row in layer_raw))
    adapter_trained=training['adapter_training']['adapter_gradient_max']>0
    accepted=accepted and zero_safe and attention_measured and adapter_trained
    output=dict(schema='aerorescue.phase1.local_eval/v4',audit_status='ACCEPTED' if accepted else 'NOT_ACCEPTED',
        pair=pair_description or 'Qwen3.5-0.8B-Base -> Qwen3.5-2B-Base',evaluation_split='test',training=training,
        gate_calibration=dict(split='val',kappa=kappa,grid=calibration),results_by_budget_bytes=results,
        independent_A_ref_train_nmse=ref['train_fit_nmse'],protocol='actual binary query/calibration/ACK/blind packets',
        sensor_boundary='own RGB-D only; depth-derived normals; evaluator truth never passed to endpoints',
        basis_tag=ref['basis_tag'],zero_overlap_safe=zero_safe,
        acceptance_checks=dict(positive_communication_gain=accepted,zero_overlap_safe=zero_safe,actual_attention_measured=attention_measured,actual_decoder_adapter_gradient=adapter_trained),
        adaptation_note='Mechanism-faithful budget-matched adaptations, not claims of full original paper reproductions',
        untruncated_srank_summary=dict(mean_srank_raw_A=float(np.mean([r['srank_raw'] for r in spectra])),mean_srank_pullback_BY_A=float(np.mean([r['srank_pullback'] for r in spectra]))))
    for filename,value in [('evaluation_results.json',output),('per_record_metrics.json',detailed),('diagnostic_measurements.json',measurements)]:
        (run_dir/filename).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    return output
