"""Distributed RGB-D ISCC execution. Endpoints consume only their local observations.

The evaluator may inspect returned traces, but no evaluator truth is accepted by
query, source selection, registration, adaptation or reconstruction functions.
"""
from __future__ import annotations
from copy import deepcopy
import json
import numpy as np
from . import wire_protocol as W
from . import rgbd
from .eiv_filter import FixedSubspaceEIVFilter,woodbury_correlated_precision
from .geometry import compute_pose_ray_jacobian,j_se3_left
from .budget_solver import solve_ternary_budget_algorithm1
from .gating_and_baselines import apply_positive_part_shrinkage_gate

OURS='Ours (Agentic-Native ISCC)'
METHODS=['Ego-Only','Where2comm-Adapted','Which2comm-Adapted','RDcomm-Adapted',
         'BOLT-Adapted','KVComm-Adapted','Ours-NoWoodbury','Ours-NoGate',OURS]


def compact_prompt(model_input):
    edge=model_input['edge_telemetry_and_belief']
    observed={k:v for k,v in edge['believed_entities'].items() if v['source']!='prior'}
    context=model_input['center_delivered_context']
    reports=[{k:r[k] for k in ('entity_id','kind','source','sim_time_s')} for r in context['delivered_reports'][-3:]]
    candidates=[{k:c[k] for k in ('candidate_id','option_id','task_type','target_anchor')}|{'anchor_enu':c['params']['anchor_enu']}
                for c in model_input['public_candidate_menu']]
    return model_input['mission_brief']+'\n'+json.dumps(dict(
        edge={k:edge[k] for k in ('kind','mobility','pose_enu','battery_Wh','nominal_capabilities')},
        observed_belief=dict(list(observed.items())[:8]),delivered_reports=reports,
        public_objectives=context['public_objectives'],candidates=candidates),ensure_ascii=False,sort_keys=True)


def cache_model_input(model_input,runtime):
    """Strict online signature; this function has no record/supervision argument."""
    from PIL import Image
    views=model_input['aerial_ground_views']
    source=rgbd.load_sensor(views['uav_view']);receiver=rgbd.load_sensor(views['ground_view'])
    source=rgbd.attach_features(source,runtime.extract_post_merger_tokens(Image.fromarray(source['rgb']),'uav'))
    receiver=rgbd.receiver_cloud(rgbd.attach_features(receiver,runtime.extract_post_merger_tokens(Image.fromarray(receiver['rgb']),'ground')))
    T=np.linalg.inv(receiver['T_w_c'])@source['T_w_c']
    anchors=np.array([c['params']['anchor_enu'] for c in model_input['public_candidate_menu']])
    inverse=np.linalg.inv(receiver['T_w_c']);anchors_e=anchors@inverse[:3,:3].T+inverse[:3,3]
    covariance=np.asarray(views['uav_view']['prior_covariance_xi'],float)
    return dict(source=source,receiver=receiver,T_prior=T,covariance=covariance,
                anchors_e=anchors_e,prompt=compact_prompt(model_input),K=len(anchors))


def new_state(ref,diagonal=False):
    flt=FixedSubspaceEIVFilter(ref['U_src'],ref['U_rec'],sigma_obs=ref['sigma_obs'])
    flt.reset(ref['A_ref_sub'],ref['prior_information'])
    return dict(filter=flt,P_pred=np.linalg.inv(ref['prior_information']),steps=0,locked=False,
                diagonal=diagonal,kv_gain=np.ones(flt.p),kv_bias=np.zeros(flt.p))


def _query(cache,ref,Uact,lambdas,state,budget,method):
    adaptive=method in (OURS,'Ours-NoGate','Ours-NoWoodbury')
    mode=adaptive and state['steps']>0 and state['locked'] and budget>512
    V=state['filter'].A_hat.T@(ref['U_rec'].T@Uact) if mode else None
    return W.encode_query(tag=ref['basis_tag'],seq=state['steps'],T_e_i=cache['T_prior'],covariance=cache['covariance'],
        K_e=cache['receiver']['K'],anchors_e=cache['anchors_e'],r=ref['U_src'].shape[1],
        V=V,lambdas=lambdas if mode else None)


def _all_source_blocks(source,U_src):
    X=U_src.T@source['tokens'];uv=source['uv'];points=source['cloud']
    ids=(uv[:,1].astype(int)//32)*20+uv[:,0].astype(int)//32
    center=source['tokens'].mean(1,keepdims=True)
    blocks=[]
    for token in np.unique(ids):
        indices=np.flatnonzero(ids==token);pixel=np.array([(token%20+.5)*32,(token//20+.5)*32])
        index=indices[np.argmin(np.sum((uv[indices]-pixel)**2,axis=1))]
        height=0.;sigma=3.
        if source['plane'] is not None:
            n,b,var=source['plane'];height=float(points[index]@n+b);sigma=float(np.sqrt(var+.01**2))
        blocks.append(dict(slot_id=255,token_idx=int(token),p_i=points[index],x_k=rgbd.bilinear(X,uv[index][None])[:,0],
            sensor_aux=np.array([height,sigma]),saliency=float(np.linalg.norm(source['tokens'][:,token:token+1]-center))))
    return blocks


def _fit_payload(source,slots,query_bytes,calibration_bytes,ack_bytes,budget,r):
    remaining=budget-len(query_bytes)-len(calibration_bytes)-len(ack_bytes)-W.UP_HEADER.size
    feasible=[bits for bits in W.BITS if len(slots)*W.SLOT_RECORD.size+len(slots)*((r*bits+7)//8)<=remaining]
    if slots and not feasible:raise ValueError('source selection exceeded actual payload budget')
    bits=max(feasible) if slots else 0
    return W.encode_up(r=r,slots=slots,bits=bits or 2)


def _simple_plan(source,query,ref,budget,method,query_bytes):
    r=ref['U_src'].shape[1];slots=rgbd.sender_slots(source,ref['U_src'],query)
    geom,rep=rgbd.sender_candidates(source,ref['U_src'],query)
    if method=='Where2comm-Adapted':
        blocks=sorted(_all_source_blocks(source,ref['U_src']),key=lambda item:item['saliency'],reverse=True)
        available=budget-len(query_bytes)-W.ACK_HEADER.size-W.UP_HEADER.size
        count=max(0,available//(W.SLOT_RECORD.size+r))
        return [],[],blocks[:count],8
    if method=='Which2comm-Adapted':
        slots=[s for s in slots if s['sensor_aux'][0]>.25 and s['sensor_aux'][1]<s['sensor_aux'][0]]
    if method=='RDcomm-Adapted':
        V=ref['static_source_task']
        slots.sort(key=lambda s:float(np.linalg.norm(V.T@s['x_k'])**2),reverse=True)
    if method in ('Which2comm-Adapted','RDcomm-Adapted'):
        remaining=budget-len(query_bytes)-W.ACK_HEADER.size-W.UP_HEADER.size
        while slots and len(slots)*(W.SLOT_RECORD.size+(r*2+7)//8)>remaining:slots.pop()
        return [],[],slots,None
    # These operators adapt on high-response/full-pool observations, without
    # the geometry-vs-low-gradient representation split used by Algorithm 1.
    if method=='BOLT-Adapted':rep.sort(key=lambda a:float(np.linalg.norm(a['x_k'])**2),reverse=True)
    else:rep.sort(key=lambda a:float(a['reliability_weight'])/(1+a['grad_norm']),reverse=True)
    minimum=len(slots)*(W.SLOT_RECORD.size+(r*2+7)//8)
    available=budget-len(query_bytes)-W.ACK_HEADER.size-2*W.UP_HEADER.size-minimum
    selected=rep[:min(32,max(0,available//(W.REP_RECORD.size+r)))]
    return [],selected,slots,None


def _source_initial(source,query,ref,state,Uact,lambdas,budget,method,query_bytes):
    if method not in (OURS,'Ours-NoGate','Ours-NoWoodbury'):
        g,a,slots,bits=_simple_plan(source,query,ref,budget,method,query_bytes)
        return g,a,slots,dict(zero_overlap=not(g or a),selected_rep_indices=[],M_proxy_selected=None)
    geometry,representation=rgbd.sender_candidates(source,ref['U_src'],query)
    slots=rgbd.sender_slots(source,ref['U_src'],query)
    r=ref['U_src'].shape[1];K=query['K'];XT=np.zeros((r,K));GT=np.zeros((K,2,6))
    for slot in slots:
        j=slot['slot_id'];XT[:,j]=slot['x_k']
        p=query['T'][:3,:3]@slot['p_i']+query['T'][:3,3]
        if p[2]>.5:GT[j]=compute_pose_ray_jacobian(p,query['K_e'])[1]
    sol=solve_ternary_budget_algorithm1(budget,query['covariance'],state['P_pred'],XT,GT,
        geometry,representation,mode=query['mode'],V_query=query['V'],lambda_act=query['lambdas'],
        geom_counts_grid=(0,2,3,6,12,24,32),max_rep_anchors=32,sigma_obs=ref['sigma_obs'],
        wire_costs=W.cost_model(query_bytes,r),codec_bits=W.BITS)
    return [geometry[i] for i in sol['selected_geom_indices']],[representation[i] for i in sol['selected_rep_indices']],slots,sol


def _receiver_calibrate(receiver,decoded,state,query,ref,method):
    flt=state['filter'];T=query['T'];S=query['covariance'];ok_g=[]
    if decoded['geometry']:
        T,S,ok_g=rgbd.receiver_registration(receiver,decoded['geometry'],T,S)
    X=[];Y=[];D=[];G=[];accepted=[];matches=[]
    for index,anchor in enumerate(decoded['representation']):
        match=rgbd.receiver_match(receiver,anchor,T,S)
        if match is None:continue
        pixel=match['pixel_e'][None]
        X.append(anchor['x_k']);Y.append(ref['U_rec'].T@rgbd.bilinear(receiver['tokens'],pixel)[:,0])
        D.append(np.stack([ref['U_rec'].T@rgbd.bilinear(receiver['spatial_grad'][:,:,axis],pixel)[:,0] for axis in range(2)],axis=-1))
        p=T[:3,:3]@anchor['p_i']+T[:3,3];G.append(compute_pose_ray_jacobian(p,receiver['K'])[1])
        accepted.append(index);matches.append(match)
    diagnostics=None
    if X:
        xa=np.array(X).T;ya=np.array(Y).T;de=np.stack(D,axis=1);gf=np.array(G)
        if method=='KVComm-Adapted' and xa.shape[1]>=2:
            predicted=ref['A_ref_sub']@xa
            denominator=np.sqrt(np.var(predicted,axis=1)+ref['sigma_obs']**2/xa.shape[1])
            state['kv_gain']=np.std(ya,axis=1)/np.maximum(denominator,1e-8)
            state['kv_bias']=np.mean(ya,axis=1)-state['kv_gain']*np.mean(predicted,axis=1)
        else:
            diagnostics=flt.step_b_woodbury_rls_update(xa,ya,de,gf,S,
                diagonal_pose_noise=method in ('Ours-NoWoodbury','BOLT-Adapted'))
    else:flt.step_count+=1
    return T,S,ok_g,accepted,dict(X=X,Y=Y,D=D,G=G,matches=matches,diagnostics=diagnostics)


def _ack_sender_update(state,selected_rep,decoded_calibration,ack,query,ref):
    """Only source-local derivatives, actual decoded feature codes and ACK ids enter."""
    kept=ack['representation_ok']
    if not kept:return
    r=ref['U_src'].shape[1]
    X=np.column_stack([decoded_calibration['representation'][i]['x_k'] for i in kept])
    V=query['V']
    if V is None:
        _,_,vt=np.linalg.svd(X.T,full_matrices=False);V=vt[:min(4,vt.shape[0])].T
    prior=np.linalg.inv(state['P_pred']);post=[]
    for direction in V.T:
        M=np.array([direction@selected_rep[i]['D_i_k']@selected_rep[i]['H_f_pinv']@selected_rep[i]['G_f'] for i in kept])
        precision=woodbury_correlated_precision(ref['sigma_obs']**2,M,ack['covariance'] if ack['covariance'] is not None else query['covariance'])
        post.append(np.linalg.inv(prior+X@precision@X.T))
    state['P_pred']=np.mean(post,axis=0)


def run_wire_step(cache,ref,Uact,lambdas,budget,state,method=OURS):
    """One real query/calibration/ACK/task exchange; no supervision is accepted."""
    K=cache['K'];r=ref['U_src'].shape[1];flt=state['filter']
    if method=='Ego-Only':
        return dict(Y=np.zeros((flt.d_e,K)),X=np.zeros((r,K)),mask=np.zeros(K,bool),aux=np.zeros((K,2)),positions=cache['anchors_e'],
                    var=np.full((4,K),np.inf),sigma_q=np.zeros(K),Tpost=cache['T_prior'],Spost=cache['covariance'],locked=False,
                    ledger=dict(total_budget_bytes=budget,total_used_bytes=0,b_g_bytes=0,b_a_bytes=0,b_s_bytes=0,downlink_bytes=0),
                    source_trace=[],receiver_trace=None,zero_overlap=False)
    query_bytes=_query(cache,ref,Uact,lambdas,state,budget,method)
    query=W.decode_query(query_bytes,ref['basis_tag'])
    g,a,initial_slots,sol=_source_initial(cache['source'],query,ref,state,Uact,lambdas,budget,method,query_bytes)
    calibration=W.encode_up(r=r,geometry=g,representation=a) if g or a or method in (OURS,'Ours-NoGate','Ours-NoWoodbury','BOLT-Adapted','KVComm-Adapted') else b''
    decoded=W.decode_up(calibration) if calibration else dict(r=r,geometry=[],representation=[])
    T,S,ok_g,ok_a,receiver_trace=_receiver_calibrate(cache['receiver'],decoded,state,query,ref,method)
    adaptive=method in (OURS,'Ours-NoGate','Ours-NoWoodbury')
    ack_bytes=W.encode_ack(ok_g,ok_a,T,S) if adaptive else W.encode_ack(ok_g,ok_a,None,None)
    ack=W.decode_ack(ack_bytes)
    _ack_sender_update(state,a,decoded,ack,query,ref)
    # The second source decision sees only the encoded reply, not receiver arrays.
    updated=dict(query)
    if ack['T'] is not None:updated.update(T=ack['T'],covariance=ack['covariance'])
    if adaptive:slots=rgbd.sender_slots(cache['source'],ref['U_src'],updated)
    else:slots=initial_slots
    blind_bytes=_fit_payload(cache['source'],slots,query_bytes,calibration,ack_bytes,budget,r)
    payload=W.decode_up(blind_bytes)
    ledger=W.byte_ledger(query_bytes,calibration,ack_bytes,blind_bytes,budget)
    X=np.zeros((r,K));mask=np.zeros(K,bool);aux=np.zeros((K,2));positions=cache['anchors_e'].copy();sigma_q=np.zeros(K)
    candidate_assignments=[]
    for item in payload['slots']:
        point=T[:3,:3]@item['p_i']+T[:3,3]
        if item['slot_id']==255:
            distances=np.linalg.norm(cache['anchors_e']-point,axis=1)
            j=int(np.argmin(distances))
            if distances[j]>4.5:continue
        else:j=item['slot_id']
        distance=float(np.linalg.norm(cache['anchors_e'][j]-point))
        if mask[j] and distance>=candidate_assignments[j]:continue
        if not candidate_assignments:candidate_assignments=[float('inf')]*K
        candidate_assignments[j]=distance
        X[:,j]=item['x_k'];mask[j]=True;aux[j]=item['sensor_aux'];positions[j]=point;sigma_q[j]=item['quantization_variance']
    mapped=flt.A_hat@X
    if method=='KVComm-Adapted':mapped=state['kv_gain'][:,None]*mapped+state['kv_bias'][:,None]
    Y=ref['U_rec']@mapped;Y[:,~mask]=0
    _,variance=flt.query_direction_uncertainty(Uact,X)
    state['steps']+=1
    current_lock=bool(ok_g or ok_a)
    state['locked']=current_lock
    return dict(Y=Y,X=X,mask=mask,aux=aux,positions=positions,var=variance,sigma_q=sigma_q,
                Tpost=T,Spost=S,locked=current_lock,ledger=ledger,source_trace=g+a,
                receiver_trace=receiver_trace,zero_overlap=not current_lock,
                query_decoded=query,reply_decoded=ack,raw_messages=(query_bytes,calibration,ack_bytes,blind_bytes))


def task_embedding(packet,ref,geometry=True,visual=True):
    """Train-centred real visual evidence plus separately typed RGB-D relief evidence."""
    mask=packet['mask'];K=len(mask);basis=ref['U_rec']
    projected=basis.T@packet['Y']
    centered=(projected-ref['wire_mean'])*mask[None]
    result=basis@centered if visual else np.zeros((basis.shape[0],K))
    if geometry:
        height=(packet['aux'][:,0]-ref['height_mean'])*mask
        result+=ref['geometry_axis'][:,None]*height[None]*ref['geometry_scale']
    result[:,~mask]=0
    return result


def shrink_task_packet(packet,ref,Uact,kappa,geometry=True,visual=True):
    slots=task_embedding(packet,ref,geometry,visual)
    direction=ref['U_rec'].T@Uact
    model_var=np.einsum('pl,pq,ql->l',direction,ref['model_covariance'],direction)[:,None]
    variance=packet['var']+model_var
    axis=Uact.T@ref['geometry_axis']
    if geometry:
        geometry_noise=np.zeros(len(packet['mask']))
        for j,point in enumerate(packet['positions']):
            J=j_se3_left(point)
            association=float(np.trace(J@packet['Spost']@J.T))/(4.5**2)
            geometry_noise[j]=ref['geometry_scale']**2*(packet['aux'][j,1]**2+association)
        variance+=axis[:,None]**2*geometry_noise[None]
    if not packet['locked']:variance[:]=np.inf
    variance[:,~packet['mask']]=np.inf
    mu=Uact.T@slots
    qnorm=np.sum((ref['A_ref_sub'].T@direction)**2,axis=0)
    result=apply_positive_part_shrinkage_gate(slots,Uact,mu,variance,qnorm,packet['sigma_q'],kappa_gate=kappa)
    result['modelled_variance']=variance
    return result
