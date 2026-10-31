"""Actual binary RGB-D ISCC messages. Sizes come from serialized buffers.

All points/normals are in their observing sensor's optical frame. No instance,
world-pose truth, co-visibility mask or task label has a field in this protocol.
"""
from __future__ import annotations
import struct
import zlib
import numpy as np

QUERY_HEADER=struct.Struct('<4sIHHHHBB')
UP_HEADER=struct.Struct('<4sHHHHBB')
GEOM_RECORD=struct.Struct('<H3e3bB16B')
REP_RECORD=struct.Struct('<H3e3bBe16B')
SLOT_RECORD=struct.Struct('<BH3e2f2e')
ACK_HEADER=struct.Struct('<4sII')
BITS=(2,4,8,16)


def basis_tag(U_src,U_rec):
    return zlib.crc32(np.asarray(U_src,np.float32).tobytes()+np.asarray(U_rec,np.float32).tobytes())


def _pose_bytes(T):
    return np.asarray(T[:3,:],dtype='<f2').tobytes()


def _pose_read(buffer,offset):
    T=np.eye(4)
    T[:3]=np.frombuffer(buffer,dtype='<f2',count=12,offset=offset).reshape(3,4)
    u,_,vt=np.linalg.svd(T[:3,:3])
    sign=np.eye(3);sign[-1,-1]=np.linalg.det(u@vt)
    T[:3,:3]=u@sign@vt
    return T,offset+24


def _cov_bytes(cov):
    return np.asarray(np.asarray(cov)[np.triu_indices(6)],dtype='<f2').tobytes()


def _cov_read(buffer,offset):
    upper=np.frombuffer(buffer,dtype='<f2',count=21,offset=offset).astype(float)
    covariance=np.zeros((6,6));indices=np.triu_indices(6)
    covariance[indices]=upper;covariance[(indices[1],indices[0])]=upper
    values,vectors=np.linalg.eigh(covariance)
    # Encoding a correlated PSD matrix in FP16 can leave a tiny negative tail.
    covariance=(vectors*np.maximum(values,1e-10))@vectors.T
    return covariance,offset+42


def _int8_encode(matrix):
    matrix=np.asarray(matrix,float)
    scales=np.maximum(np.max(np.abs(matrix),axis=0)/127,np.finfo(np.float16).tiny).astype('<f2')
    if not np.isfinite(scales).all():raise ValueError('unencodable feature scale')
    codes=np.clip(np.rint(matrix/scales[None]),-127,127).astype(np.int8)
    return scales,codes


def encode_query(*,tag,seq,T_e_i,covariance,K_e,anchors_e,r,V=None,lambdas=None):
    anchors=np.asarray(anchors_e,float)
    L=0 if V is None else np.asarray(V).shape[1]
    payload=bytearray(QUERY_HEADER.pack(b'ISQ4',int(tag),int(seq),int(r),len(anchors),L,int(L>0),0))
    payload.extend(_pose_bytes(T_e_i));payload.extend(_cov_bytes(covariance))
    payload.extend(np.asarray([K_e[0,0],K_e[1,1],K_e[0,2],K_e[1,2]],'<f2').tobytes())
    payload.extend(np.asarray(anchors,'<f2').tobytes())
    if L:
        scales,codes=_int8_encode(V)
        payload.extend(scales.tobytes());payload.extend(codes.tobytes())
        payload.extend(np.asarray(lambdas,dtype='<f2').tobytes())
    return bytes(payload)


def decode_query(payload,expected_tag=None):
    magic,tag,seq,r,K,L,mode,flags=QUERY_HEADER.unpack_from(payload)
    if magic!=b'ISQ4' or expected_tag is not None and tag!=expected_tag:raise ValueError('query/basis version mismatch')
    offset=QUERY_HEADER.size
    T,offset=_pose_read(payload,offset);covariance,offset=_cov_read(payload,offset)
    fx,fy,cx,cy=np.frombuffer(payload,dtype='<f2',count=4,offset=offset).astype(float);offset+=8
    intrinsics=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
    anchors=np.frombuffer(payload,dtype='<f2',count=3*K,offset=offset).astype(float).reshape(K,3);offset+=6*K
    V=lambdas=None
    if L:
        scales=np.frombuffer(payload,dtype='<f2',count=L,offset=offset).astype(float);offset+=2*L
        codes=np.frombuffer(payload,dtype=np.int8,count=r*L,offset=offset).reshape(r,L);offset+=r*L
        V=codes*scales[None]
        lambdas=np.frombuffer(payload,dtype='<f2',count=L,offset=offset).astype(float);offset+=2*L
    if offset!=len(payload):raise ValueError('query packet length mismatch')
    # Pose quantization has a real finite uncertainty, including translation ULPs.
    pose_floor=np.r_[np.full(3,1e-7),np.spacing(T[:3,3].astype(np.float16)).astype(float)**2/12]
    covariance+=np.diag(pose_floor)
    return dict(tag=tag,seq=seq,r=r,K=K,mode='query' if mode else 'bootstrap',T=T,covariance=covariance,K_e=intrinsics,anchors_e=anchors,V=V,lambdas=lambdas)


def _normal_codes(normal):
    n=np.asarray(normal,float);n=n/max(np.linalg.norm(n),1e-12)
    return np.clip(np.rint(n*127),-127,127).astype(np.int8).tolist()


def _normal_decoded(values):
    n=np.asarray(values,float)/127
    return n/max(np.linalg.norm(n),1e-12)


def _packed_codes(values,bits):
    values=np.asarray(values,np.uint16)
    return np.packbits(((values[:,None]>>np.arange(bits))&1).astype(np.uint8).ravel(),bitorder='little').tobytes()


def _unpacked_codes(payload,r,bits):
    data=np.frombuffer(payload,np.uint8)
    binary=np.unpackbits(data,bitorder='little')[:r*bits].reshape(r,bits)
    return binary@(1<<np.arange(bits))


def encode_up(*,r,geometry=(),representation=(),slots=(),bits=8):
    if slots and bits not in BITS:raise ValueError('unsupported blind codec precision')
    output=bytearray(UP_HEADER.pack(b'ISU4',len(geometry),len(representation),len(slots),r,bits if slots else 0,0))
    for anchor in geometry:
        output.extend(GEOM_RECORD.pack(int(anchor['token_idx']),*anchor['p_i'],*_normal_codes(anchor['normal']),0,*np.asarray(anchor['descriptor'],np.uint8)))
    for anchor in representation:
        scale,codes=_int8_encode(np.asarray(anchor['x_k'])[:,None])
        output.extend(REP_RECORD.pack(int(anchor['token_idx']),*anchor['p_i'],*_normal_codes(anchor['normal']),0,float(scale[0]),*np.asarray(anchor['descriptor'],np.uint8)))
        output.extend(codes[:,0].tobytes())
    for slot in slots:
        x=np.asarray(slot['x_k'],float)
        low=float(np.float32(x.min()));high=float(np.float32(x.max()))
        aux=np.asarray(slot.get('sensor_aux',[0.,0.]),float)
        output.extend(SLOT_RECORD.pack(int(slot['slot_id']),int(slot['token_idx']),*slot['p_i'],low,high,*aux))
        step=max(high-low,1e-6)/((1<<bits)-1)
        codes=np.clip(np.rint((x-low)/step),0,(1<<bits)-1).astype(np.uint16)
        output.extend(_packed_codes(codes,bits))
    return bytes(output)


def decode_up(payload):
    magic,ng,na,ns,r,bits,flags=UP_HEADER.unpack_from(payload)
    if magic!=b'ISU4' or ns and bits not in BITS:raise ValueError('invalid uplink header')
    offset=UP_HEADER.size;geometry=[];representation=[];slots=[]
    for _ in range(ng):
        row=GEOM_RECORD.unpack_from(payload,offset);offset+=GEOM_RECORD.size
        geometry.append(dict(token_idx=row[0],p_i=np.array(row[1:4]),normal=_normal_decoded(row[4:7]),descriptor=np.array(row[8:],np.uint8)))
    for _ in range(na):
        row=REP_RECORD.unpack_from(payload,offset);offset+=REP_RECORD.size
        codes=np.frombuffer(payload,np.int8,count=r,offset=offset).astype(float);offset+=r
        representation.append(dict(token_idx=row[0],p_i=np.array(row[1:4]),normal=_normal_decoded(row[4:7]),descriptor=np.array(row[9:],np.uint8),x_k=codes*row[8]))
    size=(r*bits+7)//8 if ns else 0
    for _ in range(ns):
        row=SLOT_RECORD.unpack_from(payload,offset);offset+=SLOT_RECORD.size
        codes=_unpacked_codes(payload[offset:offset+size],r,bits);offset+=size
        step=max(float(row[6]-row[5]),1e-6)/((1<<bits)-1)
        slots.append(dict(slot_id=row[0],token_idx=row[1],p_i=np.array(row[2:5]),x_k=row[5]+codes*step,
                          sensor_aux=np.array(row[7:9]),quantization_variance=step**2/12))
    if offset!=len(payload):raise ValueError('uplink packet length mismatch')
    return dict(r=r,bits=bits,geometry=geometry,representation=representation,slots=slots)


def encode_ack(geometry_ok,representation_ok,T_e_i,covariance):
    g=sum(1<<int(i) for i in geometry_ok);a=sum(1<<int(i) for i in representation_ok)
    header=ACK_HEADER.pack(b'ISA4',g,a)
    return header if T_e_i is None else header+_pose_bytes(T_e_i)+_cov_bytes(covariance)


def decode_ack(payload):
    magic,g,a=ACK_HEADER.unpack_from(payload)
    if magic!=b'ISA4':raise ValueError('invalid acknowledgement')
    if len(payload)==ACK_HEADER.size:
        return dict(geometry_ok=[i for i in range(32) if g>>i&1],representation_ok=[i for i in range(32) if a>>i&1],T=None,covariance=None)
    T,offset=_pose_read(payload,ACK_HEADER.size);covariance,offset=_cov_read(payload,offset)
    if offset!=len(payload):raise ValueError('ACK packet length mismatch')
    return dict(geometry_ok=[i for i in range(32) if g>>i&1],representation_ok=[i for i in range(32) if a>>i&1],T=T,covariance=covariance)


def cost_model(query,r):
    return dict(fixed=len(query)+ACK_HEADER.size+24+42+2*UP_HEADER.size,
                query=len(query),ack=ACK_HEADER.size+24+42,up_headers=2*UP_HEADER.size,
                geometry=GEOM_RECORD.size,representation=REP_RECORD.size+r,slot_header=SLOT_RECORD.size)


def byte_ledger(query,calibration,ack,blind,budget):
    task=decode_up(blind)
    up=decode_up(calibration) if calibration else dict(r=task['r'],geometry=[],representation=[])
    geom=len(up['geometry'])*GEOM_RECORD.size
    rep=len(up['representation'])*(REP_RECORD.size+up['r'])
    blind_body=len(blind)-UP_HEADER.size
    total=sum(map(len,(query,calibration,ack,blind)))
    if total>budget:raise ValueError(f'actual protocol over budget: {total} > {budget}')
    return dict(total_budget_bytes=int(budget),downlink_bytes=len(query)+len(ack),
                uplink_header_bytes=(int(bool(calibration))+int(bool(blind)))*UP_HEADER.size,query_bytes=len(query),ack_bytes=len(ack),
                calibration_uplink_bytes=len(calibration),blind_uplink_bytes=len(blind),
                b_g_bytes=geom,b_a_bytes=rep,b_s_bytes=blind_body,b_s_reserved_bytes=blind_body,
                b_s_bits_per_component=task['bits'],b_s_emitted_bits_per_component=task['bits'],
                total_used_bytes=total,unused_bytes=int(budget)-total)
