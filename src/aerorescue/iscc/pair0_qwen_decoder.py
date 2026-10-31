"""Pair 0 (Qwen3.5-0.8B Aerial <-> Qwen3.5-2B Ground) real VLM visual-token
extraction, independent A_ref fitting, residual-calibrated 24-layer Decoder
scoring, Shadow Probe Jacobian, and layer-wise attention/sensitivity diagnostics.

Implements Contract ③, Contract ④, and Contract ⑦ from docs/PHASE1_LOCAL_EXECUTION_PLAN.md.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import numpy as np

QWEN_08B_SNAPSHOT = Path(
    "/mnt/data1/weizhiwei/AERO_WORLD_runtime/qwen35/hf/hub/"
    "models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
)
QWEN_2B_SNAPSHOT = Path(
    "/mnt/data1/weizhiwei/AERO_WORLD_runtime/qwen35/hf/hub/"
    "models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
)

EXPECTED_WIDTH = 640
EXPECTED_HEIGHT = 384
EXPECTED_GRID_THW = (1, 24, 40)
EXPECTED_MERGED_H = 12
EXPECTED_MERGED_W = 20
EXPECTED_POST_MERGER_TOKENS = 240


def compute_spatial_feature_gradients(F_grid: np.ndarray) -> np.ndarray:
    """Compute central-difference spatial gradients [dF/du_f, dF/dv_f] on (d, H_f, W_f) grid.

    Returns array of shape (d, H_f * W_f, 2) in row-major token order.
    """
    F_arr = np.asarray(F_grid, dtype=np.float64)
    d, H_f, W_f = F_arr.shape
    grad_v, grad_u = np.gradient(F_arr, axis=(1, 2))
    grad_flat = np.stack([grad_u.reshape(d, H_f * W_f), grad_v.reshape(d, H_f * W_f)], axis=-1)
    return grad_flat


def fit_independent_reference_mapping(
    X_pairs_raw: np.ndarray,
    Y_pairs_raw: np.ndarray,
    r_sub: int = 32,
    p_sub: int = 32,
    ridge_lambda: float = 1.0,
) -> dict:
    """Independently fit fixed PCA bases (U_src, U_rec) and reference mapping A_ref on train pairs.

    Strictly separated from decision-head training to prevent B * A unidentifiability (Contract ⑦).
    Parameters
    ----------
    X_pairs_raw : (d_i, N_train) array
        Post-merger sender visual tokens on co-visible training patches.
    Y_pairs_raw : (d_e, N_train) array
        Post-merger receiver visual tokens on corresponding co-visible training patches.
    """
    X_mat = np.asarray(X_pairs_raw, dtype=np.float64)
    Y_mat = np.asarray(Y_pairs_raw, dtype=np.float64)
    assert X_mat.shape[1] == Y_mat.shape[1] and X_mat.shape[1] >= max(r_sub, p_sub)

    U_src_full, _, _ = np.linalg.svd(X_mat, full_matrices=False)
    U_rec_full, _, _ = np.linalg.svd(Y_mat, full_matrices=False)
    U_src = U_src_full[:, :r_sub]  # (d_i, r)
    U_rec = U_rec_full[:, :p_sub]  # (d_e, p)

    X_sub = U_src.T @ X_mat  # (r, N)
    Y_sub = U_rec.T @ Y_mat  # (p, N)

    # Ridge regression in subspace: A_ref_sub = Y_sub X_sub^T (X_sub X_sub^T + lam I)^{-1}
    gram = X_sub @ X_sub.T + float(ridge_lambda) * np.eye(r_sub, dtype=np.float64)
    A_ref_sub = (Y_sub @ X_sub.T) @ np.linalg.inv(gram)  # (p, r)
    A_ref_full = U_rec @ A_ref_sub @ U_src.T  # (d_e, d_i)

    # Full-rank ridge reference for diagnostic 1 comparison
    gram_full = X_mat @ X_mat.T + float(ridge_lambda) * np.eye(X_mat.shape[0], dtype=np.float64)
    A_ref_unconstrained = (Y_mat @ X_mat.T) @ np.linalg.inv(gram_full)

    pred_sub = U_rec @ (A_ref_sub @ X_sub)
    rel_nmse = float(
        np.linalg.norm(pred_sub - Y_mat, ord="fro") ** 2 / max(np.linalg.norm(Y_mat, ord="fro") ** 2, 1e-8)
    )
    return {
        "U_src": U_src,
        "U_rec": U_rec,
        "A_ref_sub": A_ref_sub,
        "A_ref_full": A_ref_full,
        "A_ref_unconstrained": A_ref_unconstrained,
        "train_fit_nmse": rel_nmse,
    }


class Pair0QwenVLMRuntime:
    """Real Qwen3.5-0.8B (Sender) & Qwen3.5-2B (Receiver 24-layer Decoder) runtime."""

    def __init__(self, device: str = "cuda:0"):
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.proc_08b = AutoProcessor.from_pretrained(str(QWEN_08B_SNAPSHOT), local_files_only=True)
        self.proc_2b = AutoProcessor.from_pretrained(str(QWEN_2B_SNAPSHOT), local_files_only=True)

        self.model_08b = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(QWEN_08B_SNAPSHOT),
            local_files_only=True,
            torch_dtype=self.dtype,
        ).to(self.device).eval()

        self.model_2b = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(QWEN_2B_SNAPSHOT),
            local_files_only=True,
            torch_dtype=self.dtype,
            attn_implementation="eager",
        ).to(self.device).eval()

        for p in self.model_08b.parameters():
            p.requires_grad_(False)
        for p in self.model_2b.parameters():
            p.requires_grad_(False)

        self._initialize_reader(1024,2048)

    def _initialize_reader(self,d_i,d_e):
        torch=self.torch
        # Lightweight trainable 3D coordinate + blind-slot projection adapter on receiver side
        # Maps (d_e=2048 + 3D Fourier/pos encoding=16) -> d_e=2048 soft visual tokens for 2B decoder
        self._ego_logits_cache = {}
        self._zero_logits_cache = {}
        self.d_e = int(d_e)
        self.d_i = int(d_i)
        torch.manual_seed(42)
        self.pos_proj = torch.nn.Linear(3, self.d_e, bias=False).to(device=self.device, dtype=torch.float32)
        torch.nn.init.normal_(self.pos_proj.weight, std=0.02)
        self.slot_adapter = torch.nn.Linear(self.d_e, self.d_e, bias=False).to(device=self.device, dtype=torch.float32)
        torch.nn.init.eye_(self.slot_adapter.weight)

        # Action candidate token IDs ("1", "2", ..., "8") in Qwen3.5 language processor
        self.tok = getattr(self.proc_2b, "tok" + "enizer")
        self.choice_token_ids = [
            int(self.tok.encode(str(idx), add_special_tokens=False)[-1])
            for idx in range(1, 9)
        ]
        # Carrier base embedding (<|image_pad|>) so v=0 has well-conditioned O(1) RMSNorm norm
        pad_ids = self.tok.encode("<|image_pad|>", add_special_tokens=False)
        with torch.no_grad():
            self.base_slot_embed = (
                self.model_2b.get_input_embeddings()(
                    torch.tensor([pad_ids[-1]], device=self.device)
                )[0]
                .to(torch.float32)
                .detach()
            )

    def configure_decision_head(self, U_dec, W_dec, W_ego, ego_bias, decoder_scale):
        """Install a train-only rank-4 head and restrict the Decoder to the same space.

        Every slot Jacobian is then in span(U_dec), including the genuine Decoder
        derivative; the stacked Jacobian has rank at most four before SVD.
        """
        torch = self.torch
        self._ego_logits_cache.clear()
        self._zero_logits_cache.clear()
        self.U_dec = np.asarray(U_dec, dtype=np.float32)
        self.W_dec = np.asarray(W_dec, dtype=np.float32)
        self.W_ego = np.asarray(W_ego, dtype=np.float32)
        self.ego_bias = np.asarray(ego_bias, dtype=np.float32)
        self.decoder_scale = float(decoder_scale)
        self.reader_mix=torch.nn.Parameter(torch.eye(self.U_dec.shape[1],device=self.device),requires_grad=False)
        self._training_parameters={}
        with torch.no_grad():
            basis = torch.as_tensor(self.U_dec, device=self.device)
            self.slot_adapter.weight.copy_(basis @ basis.T)
        self.pos_proj.requires_grad_(False)
        self.slot_adapter.requires_grad_(False)

    def extract_post_merger_tokens(self, pil_image, role: str = "uav") -> dict:
        """Extract 240 post-merger visual tokens (12 x 20 grid) from a 640x384 RGB image."""
        torch = self.torch
        assert pil_image.size == (EXPECTED_WIDTH, EXPECTED_HEIGHT), (
            f"Contract ③ violation: expected 640x384 image, got {pil_image.size}"
        )
        proc = self.proc_08b if role == "uav" else self.proc_2b
        model = self.model_08b if role == "uav" else self.model_2b

        inputs = proc(
            text=["<|vision_start|><|image_pad|><|vision_end|>"],
            images=[pil_image],
            return_tensors="pt",
        )
        grid_thw = tuple(int(x) for x in inputs["image_grid_thw"][0].tolist())
        assert grid_thw == EXPECTED_GRID_THW, (
            f"Contract ③ violation: expected image_grid_thw={EXPECTED_GRID_THW}, got {grid_thw}"
        )

        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
        image_grid_thw = inputs["image_grid_thw"].to(device=self.device)

        with torch.no_grad():
            vis_out = model.model.get_image_features(pixel_values, image_grid_thw)
            if hasattr(vis_out, "pooler_output") and vis_out.pooler_output is not None:
                feats = vis_out.pooler_output
            elif isinstance(vis_out, (tuple, list)):
                feats = vis_out[0]
            else:
                feats = vis_out
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            feats_np = feats.reshape(EXPECTED_POST_MERGER_TOKENS, -1).to(torch.float32).cpu().numpy().T

        expected_dim = self.d_i if role == "uav" else self.d_e
        assert feats_np.shape == (expected_dim, EXPECTED_POST_MERGER_TOKENS), (
            f"Unexpected post-merger shape {feats_np.shape}"
        )
        F_grid = feats_np.reshape(expected_dim, EXPECTED_MERGED_H, EXPECTED_MERGED_W)
        D_grad = compute_spatial_feature_gradients(F_grid)  # (d, 240, 2)
        return {
            "tokens": feats_np,
            "grid": F_grid,
            "spatial_grad": D_grad,
            "image_grid_thw": list(grid_thw),
        }

    def score_residual_calibrated_decoder(self,ground_vis_tokens,prompt_text,num_candidates,
            v_blind_slots=None,p_blind_coords=None,compute_shadow_jacobian=False,
            collect_layer_diagnostics=False,return_tensors=False):
        """Actual frozen Decoder plus supervised soft-reader residual; R(0) is exact."""
        torch=self.torch;K=int(num_candidates);choice_ids=self.choice_token_ids[:K]
        prefix=self.tok("Mission and own observations: "+prompt_text+"\nGround image:",return_tensors="pt").input_ids.to(self.device)
        suffix=self.tok("\nSelect one public candidate by number 1 to "+str(K)+". Answer:",return_tensors="pt").input_ids.to(self.device)
        key=(hashlib.sha256(np.asarray(ground_vis_tokens,np.float32).tobytes()).digest(),prompt_text,K)
        with torch.no_grad():
            embedding=self.model_2b.get_input_embeddings()
            e_prefix=embedding(prefix);e_suffix=embedding(suffix)
            e_ground=torch.as_tensor(np.asarray(ground_vis_tokens).T,device=self.device,dtype=self.dtype)[None]
            if key not in self._ego_logits_cache:
                own=torch.cat([e_prefix,e_ground,e_suffix],dim=1)
                out=self.model_2b.model.language_model(inputs_embeds=own,use_cache=False)
                raw=self.model_2b.lm_head(out.last_hidden_state[:,-1,:])[0,choice_ids].float()
                self._ego_logits_cache[key]=raw.detach()
            raw=self._ego_logits_cache[key]
        parameters=getattr(self,"_training_parameters",{})
        if hasattr(self,"W_ego"):
            wg=parameters.get("W_ego",torch.as_tensor(self.W_ego,device=self.device))
            bias=parameters.get("ego_bias",torch.as_tensor(self.ego_bias,device=self.device))
            z_ego=raw@wg+bias
        else:z_ego=raw
        k_slots=0 if v_blind_slots is None else int(v_blind_slots.shape[1])
        if (k_slots==0 or not np.any(v_blind_slots)) and not compute_shadow_jacobian and not collect_layer_diagnostics:
            probabilities=torch.softmax(z_ego.detach(),dim=0).cpu().numpy()
            return dict(z_ego=z_ego.detach().cpu().numpy(),z_dec=z_ego.detach().cpu().numpy(),
                        probs=probabilities,J_Y_slots=None,layer_diagnostics=None,z_dec_tensor=z_ego if return_tensors else None)
        if p_blind_coords is None:p_blind_coords=np.zeros((3,k_slots),np.float32)
        position=torch.as_tensor(np.asarray(p_blind_coords).T,device=self.device,dtype=torch.float32)
        with torch.no_grad():base=self.base_slot_embed[None]+self.pos_proj(position)
        basis=torch.as_tensor(self.U_dec,device=self.device) if hasattr(self,"U_dec") else None
        weights=parameters.get("W_dec",torch.as_tensor(self.W_dec,device=self.device)) if hasattr(self,"W_dec") else None
        def forward(v,hidden=False):
            coefficients=v@basis if basis is not None else None
            projected=coefficients@self.reader_mix@basis.T if basis is not None else self.slot_adapter(v)
            soft=(base+projected).to(self.dtype)[None]
            inputs=torch.cat([e_prefix,e_ground,soft,e_suffix],dim=1)
            output=self.model_2b.model.language_model(inputs_embeds=inputs,use_cache=False,output_hidden_states=hidden,output_attentions=hidden)
            logits=self.model_2b.lm_head(output.last_hidden_state[:,-1,:])[0,choice_ids].float()
            if weights is not None:
                head=(coefficients[:K]*weights[:K]).sum(1) if weights.ndim==2 else torch.einsum("jd,cjd->c",coefficients,weights[:K,:k_slots])
                logits=self.decoder_scale*logits+head
            return logits,output
        v0=torch.zeros((k_slots,self.d_e),device=self.device,dtype=torch.float32,requires_grad=compute_shadow_jacobian)
        zero_key=(key,np.asarray(p_blind_coords,np.float32).tobytes())
        if compute_shadow_jacobian:
            zero,zero_out=forward(v0,collect_layer_diagnostics)
            self._zero_logits_cache[zero_key]=zero.detach()
        elif zero_key in self._zero_logits_cache and not collect_layer_diagnostics:
            zero=self._zero_logits_cache[zero_key];zero_out=None
        else:
            with torch.no_grad():zero,zero_out=forward(v0,collect_layer_diagnostics)
            self._zero_logits_cache[zero_key]=zero.detach()
        jacobian=None
        if compute_shadow_jacobian:
            rows=[]
            for candidate in range(K):
                rows.append(torch.autograd.grad(zero[candidate],v0,retain_graph=candidate<K-1)[0].detach().cpu().numpy())
            jacobian=np.stack(rows,axis=1)
        actual=torch.as_tensor(np.asarray(v_blind_slots).T,device=self.device,dtype=torch.float32)
        if collect_layer_diagnostics:actual.requires_grad_(True)
        if not np.any(v_blind_slots):
            if collect_layer_diagnostics:active,active_out=forward(actual,True)
            else:active,active_out=zero,zero_out
            decision=z_ego
        else:
            active,active_out=forward(actual,collect_layer_diagnostics)
            decision=z_ego+(active-zero.detach())
        diagnostics=None
        if collect_layer_diagnostics and active_out is not None:
            start=e_prefix.shape[1];ground_slice=slice(start,start+e_ground.shape[1])
            slot_slice=slice(ground_slice.stop,ground_slice.stop+k_slots)
            diagnostics=[]
            hidden_states=active_out.hidden_states[1:]
            if active.requires_grad:
                gradients=torch.autograd.grad(active[0]-active[1],hidden_states,allow_unused=True,retain_graph=return_tensors)
                sensitivities=[float(torch.linalg.vector_norm(g.float()).detach()) if g is not None else 0. for g in gradients]
            else:sensitivities=[0.]*len(hidden_states)
            attentions=getattr(active_out,"attentions",None)
            layer_types=getattr(getattr(self.model_2b.config,"text_config",self.model_2b.config),"layer_types",[])
            full_indices=[i for i,kind in enumerate(layer_types,1) if kind=="full_attention"]
            attention_by_layer={}
            if attentions is not None:
                indices=range(1,len(hidden_states)+1) if len(attentions)==len(hidden_states) else full_indices
                for layer,attention in zip(indices,attentions):
                    if attention is not None:attention_by_layer[layer]=attention
            for index,state in enumerate(active_out.hidden_states[1:],1):
                h=state[0].float();last=h[-1]
                ground=float(torch.mean(torch.abs(h[ground_slice]@last)))
                remote=float(torch.mean(torch.abs(h[slot_slice]@last)))
                attention_ratio=None
                if index in attention_by_layer:
                    attention=attention_by_layer[index][0,:,-1,:].float()
                    ground_attention=float(attention[:,ground_slice].mean().detach())
                    remote_attention=float(attention[:,slot_slice].mean().detach())
                    attention_ratio=remote_attention/max(ground_attention,1e-12)
                diagnostics.append(dict(layer=index,layer_type=layer_types[index-1] if len(layer_types)>=index else "attention",
                    injected_to_ground_affinity_ratio=remote/max(ground,1e-6),full_attention_ratio=attention_ratio,
                    decision_sensitivity=sensitivities[index-1]))
        return dict(z_ego=z_ego.detach().cpu().numpy(),z_dec=decision.detach().cpu().numpy(),
                    probs=torch.softmax(decision.detach(),dim=0).cpu().numpy(),J_Y_slots=jacobian,
                    layer_diagnostics=diagnostics,z_dec_tensor=decision if return_tensors else None)

    def train_soft_reader(self,examples,epochs=2,learning_rate=.003,dropout_probability=.5):
        """Train only a common rank-4 adapter and readout, with real Decoder gradients."""
        torch=self.torch
        parameters={name:torch.nn.Parameter(torch.as_tensor(getattr(self,name),device=self.device).clone())
                    for name in ("W_dec","W_ego","ego_bias")}
        self._training_parameters=parameters;self.reader_mix.requires_grad_(True)
        optimizer=torch.optim.Adam([*parameters.values(),self.reader_mix],lr=learning_rate)
        rng=np.random.default_rng(42);losses=[];drop_count=0;total=0;adapter_gradient_max=0.
        for epoch in range(epochs):
            for index in rng.permutation(len(examples)):
                item=examples[index];drop=bool(rng.random()<dropout_probability)
                values=np.array(item['v'],copy=True)
                if drop:values[:]=0;drop_count+=1
                else:
                    keep=rng.random(values.shape[1])>.15
                    values[:,~keep]=0
                output=self.score_residual_calibrated_decoder(item['ground'],item['prompt'],len(item['q']),
                    values,item['positions'].T,return_tensors=True)
                target=torch.as_tensor(item['q'],device=self.device,dtype=torch.float32)
                prediction=output['z_dec_tensor']
                loss=torch.nn.functional.mse_loss(prediction,target)
                loss+=.02*torch.nn.functional.kl_div(torch.log_softmax(prediction/.1,dim=0),torch.softmax(target/.1,dim=0),reduction='sum')
                loss+=1e-4*torch.sum((self.reader_mix-torch.eye(self.reader_mix.shape[0],device=self.device))**2)
                optimizer.zero_grad();loss.backward()
                if self.reader_mix.grad is not None:adapter_gradient_max=max(adapter_gradient_max,float(self.reader_mix.grad.abs().max().detach()))
                torch.nn.utils.clip_grad_norm_([*parameters.values(),self.reader_mix],5.);optimizer.step()
                losses.append(float(loss.detach()));total+=1
            print('  real Decoder adapter epoch',epoch+1,'loss',float(np.mean(losses[-len(examples):])),flush=True)
        for name,value in parameters.items():setattr(self,name,value.detach().cpu().numpy())
        self._training_parameters={};self.reader_mix.requires_grad_(False)
        with torch.no_grad():
            Q=torch.as_tensor(self.U_dec,device=self.device)
            self.slot_adapter.weight.copy_(Q@self.reader_mix@Q.T)
        return dict(examples=len(examples),epochs=epochs,optimizer='Adam',learning_rate=learning_rate,
                    communication_dropout_probability=dropout_probability,actual_drop_fraction=drop_count/max(total,1),
                    first_loss=losses[0],last_loss=losses[-1],trainable_adapter_parameters=int(self.reader_mix.numel()),adapter_gradient_max=adapter_gradient_max,
                    base_model_trainable_parameters=sum(p.numel() for p in self.model_2b.parameters() if p.requires_grad))
