"""Main RGB-D input and actual-message accounting regressions."""
import struct
import numpy as np
import pytest
from PIL import Image
from aerorescue.iscc import solve_ternary_budget_algorithm1,se3_exp
from aerorescue.iscc import wire_protocol as W
from aerorescue.iscc.rgbd import load_sensor,normals_from_depth
from rescue_scenes.assets import normalization_parameters


def query(rank=4):
    return W.encode_query(tag=1234,seq=7,T_e_i=se3_exp(np.array([.02,-.01,.03,1.,2.,30.])),
        covariance=np.diag([.002]*3+[.6]*3),K_e=np.array([[450.,0,320.],[0,450.,192.],[0,0,1.]]),
        anchors_e=np.array([[0.,0.,20.],[4.,0.,20.]]),r=32,
        V=np.linspace(-1.,1.,32*rank).reshape(32,rank) if rank else None,
        lambdas=np.arange(1,rank+1) if rank else None)


def anchor(index=0):
    return dict(token_idx=index,p_i=np.array([1.,2.,30.]),normal=np.array([0.,0.,-1.]),
                descriptor=np.arange(16,dtype=np.uint8),x_k=np.linspace(-2,3,32))


def test_query_and_ack_round_trip_and_basis_identity():
    q=query();d=W.decode_query(q,1234)
    assert d['r']==32 and d['K']==2 and d['V'].shape==(32,4)
    assert np.linalg.det(d['T'][:3,:3])==pytest.approx(1.,abs=1e-10)
    assert np.linalg.eigvalsh(d['covariance']).min()>0
    with pytest.raises(ValueError,match='version'):W.decode_query(q,42)
    ack=W.encode_ack([0,2],[1],d['T'],d['covariance'])
    restored=W.decode_ack(ack)
    assert restored['geometry_ok']==[0,2] and restored['representation_ok']==[1]
    assert len(ack)==78


def test_every_actual_uplink_field_is_quantized_decoded_and_accounted():
    a=anchor();slot=dict(a,slot_id=0,sensor_aux=np.array([1.2,.02]))
    for bits in W.BITS:
        up=W.encode_up(r=32,geometry=[a],representation=[a],slots=[slot],bits=bits)
        decoded=W.decode_up(up)
        expected=W.UP_HEADER.size+W.GEOM_RECORD.size+W.REP_RECORD.size+32+W.SLOT_RECORD.size+32*bits//8
        assert len(up)==expected
        assert np.allclose(decoded['geometry'][0]['p_i'],a['p_i'])
        assert decoded['representation'][0]['x_k'].shape==(32,)
        assert decoded['slots'][0]['slot_id']==0
        assert np.isfinite(decoded['slots'][0]['x_k']).all()
        assert decoded['slots'][0]['sensor_aux'][0]==pytest.approx(1.2,abs=.001)
    with pytest.raises(ValueError,match='length'):W.decode_up(up+b'!')


def test_message_does_not_serialize_extra_oracle_keys_and_budget_is_actual_length():
    a=anchor();b=dict(a,oracle=999,instance_id=7,covis_mask=True,q_value=12.)
    first=W.encode_up(r=32,representation=[a]);second=W.encode_up(r=32,representation=[b])
    assert first==second
    d=W.decode_query(query());ack=W.encode_ack([0],[0],d['T'],d['covariance'])
    blind=W.encode_up(r=32,slots=[dict(a,slot_id=0)],bits=4)
    total=sum(map(len,[query(),first,ack,blind]))
    ledger=W.byte_ledger(query(),first,ack,blind,total)
    assert ledger['total_used_bytes']==total
    assert ledger['downlink_bytes']+ledger['uplink_header_bytes']+ledger['b_g_bytes']+ledger['b_a_bytes']+ledger['b_s_bytes']==total
    with pytest.raises(ValueError,match='over budget'):W.byte_ledger(query(),first,ack,blind,total-1)


def test_algorithm1_categorical_allocation_uses_real_packet_costs():
    q=query(rank=0);cost=W.cost_model(q,32)
    solution=solve_ternary_budget_algorithm1(512,np.eye(6),np.eye(32),np.ones((32,2)),np.zeros((2,2,6)),
        [{'G_f':np.zeros((2,6)),'Gamma_k':np.eye(2),'p_i':np.array([0.,0.,20.])}],[],
        mode='bootstrap',geom_counts_grid=(0,),wire_costs=cost,codec_bits=W.BITS)
    bits=int(solution['byte_ledger']['b_s_bits_per_component'])
    assert bits in W.BITS
    assert solution['byte_ledger']['b_s_bytes']>=128
    slots=[dict(anchor(i),slot_id=i) for i in range(2)]
    calibration=W.encode_up(r=32);blind=W.encode_up(r=32,slots=slots,bits=bits)
    d=W.decode_query(q);ack=W.encode_ack([],[],d['T'],d['covariance'])
    ledger=W.byte_ledger(q,calibration,ack,blind,512)
    assert ledger['total_used_bytes']==solution['byte_ledger']['total_used_bytes']


def test_sensor_depth_rejects_truth_container_and_derives_normal_from_own_depth(tmp_path):
    image=tmp_path/'rgb.png';Image.fromarray(np.full((384,640,3),120,np.uint8)).save(image)
    depth=tmp_path/'depth.npy';np.save(depth,np.full((384,640),10,np.float32))
    view=dict(rgb_path=str(image),depth_path=str(depth),intrinsics=dict(fx=450,fy=450,cx=320,cy=192),extrinsics_w_from_c=np.eye(4))
    sensor=load_sensor(view)
    assert np.allclose(sensor['normal'][100,100],[0,0,-1],atol=1e-6)
    private=tmp_path/'truth.npz';np.savez(private,depth=np.ones((384,640)),instance_id=np.ones((384,640)),covis_mask=np.ones((384,640)))
    with pytest.raises(ValueError,match='single'):load_sensor(dict(view,depth_path=str(private)))


def test_engine_origin_is_translated_before_heading_including_shared_reference():
    row=dict(up='+y',forward='+z',boundsM=dict(min=[10.,1.,20.],max=[14.,5.,24.]),sizeM=[4,4,4])
    reference=dict(row,origin_m=[12.,1.,22.])
    translation,turn,scale=normalization_parameters(row,reference)
    assert translation==[-12.,-1.,-22.]
    R=np.array([[np.cos(turn),0,np.sin(turn)],[0,1,0],[-np.sin(turn),0,np.cos(turn)]])
    # The authored common origin must land at the world placement for every heading.
    assert np.allclose(R@(np.array(reference['origin_m'])+translation),0)
