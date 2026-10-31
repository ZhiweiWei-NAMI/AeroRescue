"""Sensor-local RGB-D geometry and spatial feature sampling (no evaluator inputs)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
from .geometry import compute_homography_jacobian,compute_pose_ray_jacobian,j_se3_left


def bilinear(tokens,uv):
    xy=np.asarray(uv)/32.-.5
    x=np.clip(xy[:,0],0,19);y=np.clip(xy[:,1],0,11)
    x0=np.floor(x).astype(int);y0=np.floor(y).astype(int)
    x1=np.minimum(x0+1,19);y1=np.minimum(y0+1,11);dx=x-x0;dy=y-y0
    return (tokens[:,y0*20+x0]*((1-dx)*(1-dy))[None]+tokens[:,y0*20+x1]*(dx*(1-dy))[None]
            +tokens[:,y1*20+x0]*((1-dx)*dy)[None]+tokens[:,y1*20+x1]*(dx*dy)[None])


def project(points,K):
    p=np.asarray(points)
    return p[:,:2]/np.maximum(p[:,2:3],1e-6)*np.array([K[0,0],K[1,1]])+K[:2,2]


def optical_points(depth,K):
    h,w=depth.shape;row,col=np.mgrid[:h,:w]
    return np.stack([(col+.5-K[0,2])/K[0,0]*depth,(row+.5-K[1,2])/K[1,1]*depth,depth],axis=-1)


def normals_from_depth(depth,K):
    points=optical_points(depth,K)
    dx=points[1:-1,2:]-points[1:-1,:-2]
    dy=points[2:,1:-1]-points[:-2,1:-1]
    normal=np.cross(dx,dy);length=np.linalg.norm(normal,axis=-1)
    center=depth[1:-1,1:-1];limit=.15+.03*center
    valid=(center>0)&(length>1e-10)
    for neighbor in (depth[:-2,1:-1],depth[2:,1:-1],depth[1:-1,:-2],depth[1:-1,2:]):
        valid&=(neighbor>0)&(np.abs(neighbor-center)<=limit)
    normal/=np.maximum(length[...,None],1e-12)
    normal[np.sum(normal*points[1:-1,1:-1],axis=-1)>0]*=-1
    result=np.zeros_like(points,np.float32);result[1:-1,1:-1]=np.where(valid[...,None],normal,0)
    return result


def load_sensor(view):
    """Only the view allowlist is consumed; depth containers with extra arrays fail."""
    rgb=np.asarray(Image.open(view['rgb_path']).convert('RGB'))
    path=Path(view['depth_path'])
    if path.suffix!='.npy':raise ValueError('sensor depth must be a single .npy array')
    depth=np.load(path,allow_pickle=False)
    if depth.dtype!=np.float32 or depth.shape!=(384,640) or not np.isfinite(depth).all() or np.any(depth<0):
        raise ValueError('invalid optical-Z sensor depth')
    if rgb.shape!=(384,640,3):raise ValueError('invalid RGB sensor dimensions')
    intr=view['intrinsics'];K=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]])
    normal=normals_from_depth(depth,K);points=optical_points(depth,K)
    rr,cc=np.mgrid[2:384:4,2:640:4];valid=depth[rr,cc]>.5
    gray=rgb.astype(float).mean(2)/255;gy,gx=np.gradient(gray)
    pose_key='prior_extrinsics_w_from_c' if 'prior_extrinsics_w_from_c' in view else 'extrinsics_w_from_c'
    return dict(rgb=rgb,depth=depth,K=K,T_w_c=np.asarray(view[pose_key],float),normal=normal,
        cloud=points[rr,cc][valid],normals=normal[rr,cc][valid],uv=np.column_stack([cc[valid]+.5,rr[valid]+.5]),
        gray=gray,gray_gradient=np.stack([gx,gy],axis=-1)*32)


def descriptor(sensor,uv):
    u,v=np.rint(np.asarray(uv)-.5).astype(int)
    u=int(np.clip(u,0,639));v=int(np.clip(v,0,383))
    crop=Image.fromarray(sensor['rgb']).convert('L').crop((u-16,v-16,u+16,v+16)).resize((4,4),Image.Resampling.BILINEAR)
    return np.asarray(crop,np.uint8).reshape(16)


def ground_plane(sensor):
    """Fit a dominant upward surface from this sensor's own measured cloud."""
    points=sensor['cloud'];normals=sensor['normals'];up=sensor['T_w_c'][:3,:3].T@np.array([0.,0.,1.])
    mask=np.abs(normals@up)>.85
    pool=points[mask]
    if len(pool)<12:return None
    pool=pool[np.linspace(0,len(pool)-1,min(len(pool),2048)).astype(int)]
    rng=np.random.default_rng(0);best=None
    for _ in range(64):
        a,b,c=pool[rng.choice(len(pool),3,replace=False)]
        n=np.cross(b-a,c-a);size=np.linalg.norm(n)
        if size<1e-8:continue
        n/=size
        if abs(n@up)<.85:continue
        if n@up<0:n=-n
        offset=-n@a;error=np.abs(pool@n+offset);inliers=error<.15
        score=int(inliers.sum())
        if best is None or score>best[0]:best=(score,n,offset,inliers)
    if best is None or best[0]<12:return None
    selected=pool[best[3]];center=selected.mean(0)
    _,_,vt=np.linalg.svd(selected-center,full_matrices=False);n=vt[-1]
    if n@up<0:n=-n
    offset=-n@center;variance=float(np.mean((selected@n+offset)**2))
    return n,offset,variance


def attach_features(sensor,features):
    sensor=dict(sensor,tokens=np.asarray(features['tokens'],float),spatial_grad=np.asarray(features['spatial_grad'],float))
    sensor['plane']=ground_plane(sensor)
    return sensor


def projected_source(source,U_src):
    return U_src.T@source['tokens'],np.einsum('dr,dka->rka',U_src,source['spatial_grad'])


def sender_candidates(source,U_src,query):
    """Source-only frustum/normal proxy; never reads a receiver image or cloud."""
    T=query['T'];cov=query['covariance'];K_e=query['K_e']
    points=source['cloud'];normal=source['normals'];uv=source['uv']
    target=points@T[:3,:3].T+T[:3,3];projected=project(target,K_e)
    target_norm=normal@T[:3,:3].T
    margin=3*K_e[0,0]*(np.sqrt(np.trace(cov[:3,:3]))+np.sqrt(np.trace(cov[3:,3:]))/np.maximum(target[:,2],1.))
    facing=np.sum(target_norm*target,axis=1)<3*np.sqrt(np.trace(cov[3:,3:])+np.sum(target**2,axis=1)*np.trace(cov[:3,:3]))
    valid=(target[:,2]>.5)&(projected[:,0]>=-margin)&(projected[:,0]<640+margin)&(projected[:,1]>=-margin)&(projected[:,1]<384+margin)&facing&(np.linalg.norm(normal,axis=1)>.9)
    X,Di=projected_source(source,U_src)
    token=(uv[:,1].astype(int)//32)*20+uv[:,0].astype(int)//32
    gradient=source['gray_gradient'][np.clip(uv[:,1].astype(int),0,383),np.clip(uv[:,0].astype(int),0,639)]
    strength=np.sum(gradient**2,axis=1)
    chosen={}
    for index in np.flatnonzero(valid):
        key=int(token[index])
        if key not in chosen or strength[index]>strength[chosen[key]]:chosen[key]=int(index)
    geometry=[];representation=[]
    for key,index in chosen.items():
        point=points[index];pixel=uv[index];n=normal[index]
        hom=compute_homography_jacobian(point,n,T,source['K'],K_e)
        _,G=compute_pose_ray_jacobian(hom['p_e'],K_e)
        x=bilinear(X,pixel[None])[:,0]
        D=np.stack([bilinear(Di[:,:,axis],pixel[None])[:,0] for axis in range(2)],axis=-1)
        gx=gradient[index]
        Gamma=np.outer(gx,gx)/.025**2+np.eye(2)*.01
        base=dict(token_idx=key,p_i=point,normal=n,descriptor=descriptor(source,pixel),G_f=G,Gamma_k=Gamma,edge_energy=float(np.trace(Gamma)),x_k=x)
        geometry.append(base)
        representation.append(dict(base,D_i_k=D,H_f_pinv=hom['H_f_pinv'],reliability_weight=hom['reliability_weight'],grad_norm=float(np.linalg.norm(D@hom['H_f_pinv']@G))))
    # Pools contain all spatial cells; the optimizer chooses actual transmitted ids.
    geometry.sort(key=lambda c:c['edge_energy'],reverse=True)
    representation.sort(key=lambda c:c['grad_norm'])
    return geometry[:64],representation[:64]


def sender_slots(source,U_src,query,feature_tokens=None):
    """Observe each public corridor ROI locally, including an honest missing-data case."""
    T=query['T'];anchors=(query['anchors_e']-T[:3,3])@T[:3,:3]
    points=source['cloud'];uv=source['uv'];normal=source['normals'];plane=source['plane']
    tokens=source['tokens'] if feature_tokens is None else np.asarray(feature_tokens)
    projected=U_src.T@tokens if U_src is not None else tokens
    slots=[]
    for slot_id,anchor in enumerate(anchors):
        distances=np.linalg.norm(points-anchor,axis=1)
        region=np.flatnonzero(distances<4.5)
        if len(region)<6:continue
        if plane is not None:
            n,offset,var=plane;heights=points[region]@n+offset
            height=float(np.percentile(heights,95))
            weights=np.exp(-.5*(distances[region]/2.5)**2)*(1+2*np.clip(heights,0,3))
            representative=region[int(np.argmin(np.abs(heights-height)))]
            aux=np.array([height,np.sqrt(var+.01**2)])
        else:
            weights=np.exp(-.5*(distances[region]/2.5)**2)
            representative=region[np.argmin(distances[region])]
            aux=np.array([0.,3.])
        weights/=weights.sum()
        feature=bilinear(projected,uv[region])@weights
        pixel=uv[representative];token_idx=int(pixel[1]//32)*20+int(pixel[0]//32)
        slots.append(dict(slot_id=slot_id,token_idx=token_idx,p_i=points[representative],normal=normal[representative],
            x_k=feature,sensor_aux=aux,roi_uv=uv[region],roi_weights=weights,
            anchor_distance_m=float(distances[region].min())))
    return slots


def receiver_cloud(receiver):
    from scipy.spatial import cKDTree
    result=dict(receiver)
    valid=np.linalg.norm(result['normals'],axis=1)>.9
    result['cloud']=result['cloud'][valid];result['normals']=result['normals'][valid];result['uv']=result['uv'][valid]
    result['tree']=cKDTree(result['cloud'])
    return result


def receiver_match(receiver,anchor,T,covariance):
    point=T[:3,:3]@anchor['p_i']+T[:3,3]
    if point[2]<=.5:return None
    observed_pixel=project(point[None],receiver['K'])[0]
    if np.any(observed_pixel<[-80,-80]) or np.any(observed_pixel>=[720,464]):return None
    normal=T[:3,:3]@anchor['normal']
    count=min(12,len(receiver['cloud']))
    _,ids=receiver['tree'].query(point,k=count);ids=np.atleast_1d(ids)
    J=j_se3_left(point)
    tolerance=.15+.03*point[2]
    covariance_point=J@covariance@J.T+tolerance**2*np.eye(3)
    residual=receiver['cloud'][ids]-point
    mahal=np.einsum('ni,ij,nj->n',residual,np.linalg.inv(covariance_point),residual)
    compatible=(receiver['normals'][ids]@normal>.7)&(mahal<11.345)
    target_desc=np.asarray(anchor['descriptor'],float);target_desc-=target_desc.mean()
    best=None
    for order in np.flatnonzero(compatible):
        index=int(ids[order]);pixel=receiver['uv'][index]
        desc=descriptor(receiver,pixel).astype(float);desc-=desc.mean()
        norm=np.linalg.norm(desc)*np.linalg.norm(target_desc)
        correlation=float(desc@target_desc/norm) if norm>1e-6 else 0.
        # A flat patch supplies point-plane geometry, not invented lateral precision.
        if norm>1 and correlation<.1:continue
        score=float(mahal[order])+2*(1-correlation)
        if best is None or score<best[0]:best=(score,index,correlation,norm)
    if best is None:return None
    _,index,correlation,norm=best;pixel=receiver['uv'][index]
    return dict(point_e=receiver['cloud'][index],normal_e=receiver['normals'][index],pixel_e=pixel,
                photometric=norm>1,correlation=correlation,mahalanobis=float(best[0]))


def receiver_registration(receiver,anchors,T_prior,covariance,iterations=3):
    """MAP point-plane refinement; weak/null directions keep the finite prior."""
    from .geometry import se3_exp,se3_log
    prior=np.linalg.inv(covariance);T=np.array(T_prior,copy=True);accepted=[];information=prior.copy()
    for iteration in range(iterations):
        information=prior.copy();rhs=-prior@se3_log(T@np.linalg.inv(T_prior));accepted=[]
        for index,anchor in enumerate(anchors):
            match=receiver_match(receiver,anchor,T,covariance)
            if match is None:continue
            p=T[:3,:3]@anchor['p_i']+T[:3,3];n=match['normal_e'];J=j_se3_left(p)
            row=n@J;res=float(n@(match['point_e']-p))
            variance=(.03+.005*np.linalg.norm(p))**2
            information+=np.outer(row,row)/variance;rhs+=row*res/variance
            if match['photometric']:
                _,G=compute_pose_ray_jacobian(p,receiver['K'])
                u,v=match['pixel_e'].astype(int)
                grad=receiver['gray_gradient'][v,u]
                Gamma=np.outer(grad,grad)/.025**2
                delta=(match['pixel_e']-project(p[None],receiver['K'])[0])/32
                information+=G.T@Gamma@G;rhs+=G.T@Gamma@delta
            accepted.append(index)
        posterior=np.linalg.inv(information)
        step=posterior@rhs
        T=se3_exp(step)@T
        if np.linalg.norm(step)<1e-5:break
    return T,.5*(posterior+posterior.T),accepted
