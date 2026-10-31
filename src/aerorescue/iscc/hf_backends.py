"""Real HuggingFace vision backends for the five heterogeneous Colab pairs.

Native spatial features are deterministically sampled onto the 12x20 logical
sensor atlas. Native grid and processed image dimensions remain reported.
"""
from __future__ import annotations
import inspect
import math
import numpy as np
from PIL import Image
from .pair0_qwen_decoder import Pair0QwenVLMRuntime,compute_spatial_feature_gradients


def source_family(name):
    value=name.lower()
    for marker,family in [('qwen2.5-vl','qwen'),('smolvlm','smol'),('internvl','internvl'),('dinov2','dino'),('siglip2','siglip')]:
        if marker in value:return family
    raise ValueError('unsupported heterogeneous vision family: '+name)


def _features(output):
    if hasattr(output,'pooler_output') and output.pooler_output is not None:
        value=output.pooler_output
    elif hasattr(output,'last_hidden_state'):value=output.last_hidden_state
    elif isinstance(output,(list,tuple)):value=output[0]
    else:value=output
    while isinstance(value,(list,tuple)):value=value[0]
    return value


class HFVisionEncoder:
    def __init__(self,name,device,dtype,model=None,processor=None,revision=None):
        import torch
        from transformers import AutoModel,AutoModelForImageTextToText,AutoProcessor,AutoImageProcessor
        self.torch=torch;self.name=name;self.family=source_family(name);self.device=device;self.dtype=dtype
        common=dict(revision=revision,trust_remote_code=self.family=='internvl')
        if model is None:
            loader=AutoModelForImageTextToText if self.family in ('qwen','smol') else AutoModel
            model=loader.from_pretrained(name,torch_dtype=dtype,**common).to(device)
        self.model=model.eval()
        for parameter in self.model.parameters():parameter.requires_grad_(False)
        if processor is None and self.family!='internvl':
            factory=AutoProcessor if self.family in ('qwen','smol') else AutoImageProcessor
            processor=factory.from_pretrained(name,**common)
        self.processor=processor
        config=model.config
        self.dimension=(getattr(getattr(config,'text_config',None),'hidden_size',None)
            or getattr(getattr(config,'llm_config',None),'hidden_size',None)
            or getattr(getattr(config,'vision_config',None),'hidden_size',None)
            or getattr(config,'hidden_size',None))
        self.commit=getattr(config,'_commit_hash',None)

    def extract(self,image):
        torch=self.torch;family=self.family;native={};extent=None
        with torch.no_grad():
            if family=='qwen':
                batch=self.processor(text=['<|vision_start|><|image_pad|><|vision_end|>'],images=[image],return_tensors='pt')
                grid=batch['image_grid_thw'].to(self.device)
                result=self.model.model.get_image_features(batch['pixel_values'].to(self.device,self.dtype),grid)
                value=_features(result)
                _,h,w=[int(v) for v in grid[0]]
                merge=int(getattr(self.model.config.vision_config,'spatial_merge_size',2));h//=merge;w//=merge
                native['image_grid_thw']=grid[0].tolist();native['merge_size']=merge
            elif family=='internvl':
                config=self.model.config
                size=int(getattr(config,'force_image_size',None) or config.vision_config.image_size)
                array=np.asarray(image.resize((size,size),Image.Resampling.BICUBIC)).astype(np.float32)/255
                pixel=torch.as_tensor(array,device=self.device).permute(2,0,1)[None]
                mean=torch.tensor([.485,.456,.406],device=self.device)[None,:,None,None]
                std=torch.tensor([.229,.224,.225],device=self.device)[None,:,None,None]
                value=self.model.extract_feature(((pixel-mean)/std).to(self.dtype))
                ratio=float(config.downsample_ratio);patch=int(config.vision_config.patch_size)
                h=w=int(size/patch*ratio);native['processed_hw']=[size,size]
            elif family=='smol':
                batch=self.processor(images=[image],text=['<image>'],do_image_splitting=False,return_tensors='pt')
                pixel=batch['pixel_values'];pixel=pixel.reshape(-1,*pixel.shape[-3:]).to(self.device,self.dtype)
                if pixel.shape[0]!=1:raise ValueError('SmolVLM must use one unsplit sensor image')
                base=self.model.model;vision=base.vision_model
                mask=batch.get('pixel_attention_mask')
                patch=int(base.config.vision_config.patch_size)
                kwargs={'pixel_values':pixel}
                if mask is not None:
                    mask=mask.reshape(-1,*mask.shape[-2:]).to(self.device)
                    patch_mask=mask.unfold(1,patch,patch).unfold(2,patch,patch).sum((-1,-2))>0
                    kwargs['patch_attention_mask']=patch_mask
                    pos=torch.nonzero(mask[0],as_tuple=False)
                    extent=(float(pos[:,1].min()),float(pos[:,0].min()),float(pos[:,1].max()+1),float(pos[:,0].max()+1),pixel.shape[-1],pixel.shape[-2])
                value=base.connector(vision(**kwargs).last_hidden_state)
                factor=int(getattr(base.config,'scale_factor',getattr(base.connector,'scale_factor',2)))
                h=pixel.shape[-2]//patch//factor;w=pixel.shape[-1]//patch//factor
                native['processed_hw']=list(pixel.shape[-2:]);native['connector_factor']=factor
            else:
                options={'do_center_crop':False} if family=='dino' else {}
                batch=self.processor(images=image,return_tensors='pt',**options)
                vision=self.model.vision_model if family=='siglip' else self.model
                accepted=inspect.signature(vision.forward).parameters
                kwargs={key:value.to(self.device,self.dtype if value.is_floating_point() else value.dtype)
                        for key,value in batch.items() if key in accepted}
                value=vision(**kwargs).last_hidden_state
                config=getattr(self.model.config,'vision_config',self.model.config)
                patch=int(config.patch_size)
                if 'spatial_shapes' in batch:
                    h,w=[int(x) for x in batch['spatial_shapes'][0]]
                    value=value[:,:h*w];native['spatial_shapes']=[h,w]
                else:
                    height,width=batch['pixel_values'].shape[-2:];h=height//patch;w=width//patch
                    extent=(0.,0.,float(width),float(height),w*patch,h*patch)
                    native['processed_hw']=[height,width]
                if value.shape[-2]==h*w+1:value=value[:,1:]
            value=value.reshape(-1,value.shape[-1])
            if value.shape[0]!=h*w:raise ValueError(f'{self.name}: spatial feature count {value.shape[0]} != {h}x{w}')
            self.dimension=int(value.shape[-1]);grid=value.T.reshape(1,self.dimension,h,w).float()
            yy,xx=torch.meshgrid(torch.arange(12,device=self.device),torch.arange(20,device=self.device),indexing='ij')
            x=(xx+.5)/20;y=(yy+.5)/12
            if extent is not None:
                x0,y0,x1,y1,covered_w,covered_h=extent
                x=(x0+x*(x1-x0))/covered_w;y=(y0+y*(y1-y0))/covered_h
            coordinates=torch.stack([2*x-1,2*y-1],dim=-1)[None]
            logical=torch.nn.functional.grid_sample(grid,coordinates,mode='bilinear',padding_mode='border',align_corners=False)
            array=logical[0].cpu().numpy();tokens=array.reshape(self.dimension,240)
        return dict(tokens=tokens,grid=array,spatial_grad=compute_spatial_feature_gradients(array),
                    image_grid_thw=[1,24,40],native_grid_hw=[h,w],native_processor=native,model=self.name)


class HeterogeneousQwenRuntime(Pair0QwenVLMRuntime):
    def __init__(self,sender,receiver,device='cuda:0',revision=None):
        import torch
        from transformers import AutoModelForImageTextToText,AutoProcessor
        self.torch=torch;self.device=device
        self.dtype=(torch.bfloat16 if torch.cuda.get_device_capability(device)[0]>=8 else torch.float16) if str(device).startswith('cuda') else torch.float32
        if source_family(receiver)!='qwen':raise ValueError('the shared receiver backend is Qwen2.5-VL')
        self.proc_2b=AutoProcessor.from_pretrained(receiver,revision=revision)
        self.model_2b=AutoModelForImageTextToText.from_pretrained(receiver,revision=revision,torch_dtype=self.dtype,attn_implementation='eager').to(device).eval()
        for parameter in self.model_2b.parameters():parameter.requires_grad_(False)
        self.source_encoder=HFVisionEncoder(sender,device,self.dtype,revision=revision)
        self.receiver_encoder=HFVisionEncoder(receiver,device,self.dtype,model=self.model_2b,processor=self.proc_2b)
        self.model_08b=self.source_encoder.model
        config=getattr(self.model_2b.config,'text_config',self.model_2b.config)
        self._initialize_reader(self.source_encoder.dimension,int(config.hidden_size))
        self.pair_description=sender+' -> '+receiver
        self.model_provenance=dict(sender=sender,receiver=receiver,source_commit=self.source_encoder.commit,
                                   receiver_commit=getattr(self.model_2b.config,'_commit_hash',None),native_atlas='12x20 sensor-coordinate resampling')

    def extract_post_merger_tokens(self,pil_image,role='uav'):
        encoder=self.source_encoder if role=='uav' else self.receiver_encoder
        output=encoder.extract(pil_image)
        if role=='uav':self.d_i=encoder.dimension
        elif encoder.dimension!=self.d_e:raise ValueError('receiver feature space differs from its language embedding')
        return output
