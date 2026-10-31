"""Regression checks for result-changing findings from the Phase-1 sol audit."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

from aerorescue.iscc import apply_positive_part_shrinkage_gate, compute_fisher_and_vertical_by, se3_exp, solve_ternary_budget_algorithm1
from aerorescue.iscc.budget_solver import quantize_and_dequantize_blind_features
from aerorescue.iscc.audited_pair0 import bilinear, fit_train_head
from aerorescue.phase1.branch_collector import se3_log


def test_gate_uses_frozen_positive_part_formula_without_signal_noise_denominator():
    result = apply_positive_part_shrinkage_gate(np.array([[2.]]), np.eye(1), np.array([[2.]]),
        np.array([[1.]]), np.array([0.]), np.array([0.]), kappa_gate=2.)
    assert result["weights"][0, 0] == pytest.approx(.5)
    assert result["Y_gated"][0, 0] == pytest.approx(1.)
    off = apply_positive_part_shrinkage_gate(np.ones((1,1)), np.eye(1), np.ones((1,1)),
        np.ones((1,1)), np.zeros(1), np.zeros(1), kappa_gate=2.)
    assert off["all_off"] and np.array_equal(off["Y_gated"], np.zeros((1,1)))


def test_stacked_multislot_jacobian_only_has_zero_loss_when_common_space_is_constrained():
    rng=np.random.default_rng(4)
    basis=np.linalg.qr(rng.normal(size=(24,4)))[0]
    jac=rng.normal(size=(5,5,4))@basis.T
    fi=compute_fisher_and_vertical_by(np.ones(5)/5,jac,4)
    assert fi["discarded_fisher_energy_fraction"]<1e-25
    assert np.linalg.norm(fi["B_Y"]@(np.eye(24)-fi["U_act"]@fi["U_act"].T))<1e-12
    arbitrary=compute_fisher_and_vertical_by(np.ones(5)/5,rng.normal(size=(5,5,24)),4)
    assert arbitrary["discarded_fisher_energy_fraction"]>.1


def test_optical_token_bilinear_correspondence_uses_projection_not_same_index():
    values=np.arange(240,dtype=float)[None,:]
    uv=np.array([[16.,16.],[48.,16.],[32.,16.],[624.,368.]])
    assert np.allclose(bilinear(values,uv),[[0.,1.,.5,239.]])


def test_se3_log_roundtrip_for_nonzero_rotation_and_translation():
    xi=np.array([.24,-.15,.06,2.,-.4,1.])
    assert np.allclose(se3_log(se3_exp(xi)),xi,atol=1e-12)


def test_quantizer_never_exceeds_bit_budget_and_improves_precision():
    x=np.linspace(-2,3,160).reshape(32,5)
    errors=[]
    for bits in (1.,2.,4.,8.,16.):
        recon,variance,used=quantize_and_dequantize_blind_features(x,bits)
        assert used==5*14+int(np.ceil(x.size*int(bits)/8))
        assert np.all(variance>=0)
        errors.append(np.sum((recon-x)**2))
    assert np.all(np.diff(errors)<0)
    recon,_,used=quantize_and_dequantize_blind_features(x,.8)
    assert used==0 and not np.any(recon)


def test_solver_rejects_control_header_overflow_instead_of_reporting_overbudget():
    with pytest.raises(ValueError,match="budget"):
        solve_ternary_budget_algorithm1(5,np.eye(6),np.eye(2),np.ones((2,1)),np.zeros((1,2,6)),[],[])


def test_head_parameters_do_not_change_when_validation_and_test_rewards_are_poisoned():
    class Runtime:
        def score_residual_calibrated_decoder(self,tokens,prompt,k):
            return {"z_ego":np.asarray(tokens[:k,0])}
        def configure_decision_head(self,*params):
            self.params=params
    rng=np.random.default_rng(5); records=[]; cache={}
    for i in range(12):
        split="train" if i<10 else ("val" if i==10 else "test")
        records.append({"metadata":{"record_id":str(i),"split":split},"model_input":{"public_candidate_menu":[{"candidate_id":str(c)} for c in range(5)]},
            "supervision_and_eval":{"branch_evaluations":{str(c):{"q_value_120s":float(rng.normal())} for c in range(5)}}})
        cache[str(i)]={"receiver":{"tokens":rng.normal(size=(8,8))},"teacher":{"tokens":rng.normal(size=(8,8))},"prompt":"public", "slot_ids":np.arange(5)}
    ref={"U_src":np.eye(8),"U_rec":np.eye(8)}
    a=Runtime(); b=Runtime()
    fit_train_head(records,cache,a,ref)
    poisoned=deepcopy(records)
    for r in poisoned[10:]:
        for v in r["supervision_and_eval"]["branch_evaluations"].values():
            v["q_value_120s"]=1e20
    fit_train_head(poisoned,cache,b,ref)
    for x,y in zip(a.params,b.params):
        assert np.array_equal(x,y)


def test_native_target_crn_does_not_shift_when_an_unrelated_target_is_sampled():
    from random import Random
    from aerorescue.phase1.branch_collector import StableExogenousRNG
    from rescue_scenes.core.sim.actions import _exogenous_rng
    sim=SimpleNamespace(exogenous_crn=StableExogenousRNG("scene",3),t=10.)
    agent=SimpleNamespace(id="edge")
    first=_exogenous_rng(sim,agent,Random(1),"sensing","field_dropout","person-A").random()
    _exogenous_rng(sim,agent,Random(123),"sensing","field_dropout","person-B").random()
    after=_exogenous_rng(sim,agent,Random(999),"sensing","field_dropout","person-A").random()
    assert first==after


def test_emitted_blind_packet_length_matches_ledger_and_decodes_exactly():
    from aerorescue.iscc.budget_solver import encode_blind_packet
    x=np.linspace(-3.,2.,160).reshape(32,5)
    coords=np.array([[1.,2.,3.,4.,5.],[0.,1.,2.,3.,4.],[10.,11.,12.,13.,14.]])
    for bits in (1.,2.,4.,8.,16.):
        packet,decoded,variance=encode_blind_packet(x,coords,bits)
        reference,expected_var,bytes_used=quantize_and_dequantize_blind_features(x,bits)
        assert len(packet)==bytes_used
        assert np.array_equal(decoded,reference)
        assert np.array_equal(variance,expected_var)


def test_grazing_homography_keeps_exact_forward_differential():
    from aerorescue.iscc.geometry import finite_difference_check_jacobians
    K=np.array([[450.,0,320.],[0,450.,192.],[0,0,1.]])
    result=finite_difference_check_jacobians(np.array([.02,0.,1.]),np.array([1.,0.,0.]),
        se3_exp(np.array([0.,.15,0.,.1,.2,10.])),K,K)
    assert result["H_max_abs_err"]<1e-4


def test_registration_rejects_out_of_frame_and_opposing_normals():
    from aerorescue.iscc.audited_pair0 import depth_correspondence_anchors
    class Tree:
        def query(self,points):
            return np.zeros(len(points)),np.zeros(len(points),dtype=int)
    K=np.array([[100.,0,320.],[0,100.,192.],[0,0,1.]])
    ug={"depth":np.ones((384,640)),"normal_cam":np.tile([0.,0.,-1.],(384,640,1))}
    base={"K":K,"tree":Tree(),"ug":ug,"cloud":np.array([[0.,0.,1.]]),"cloud_uv":np.array([[320.,192.]])}
    far=np.eye(4);far[0,3]=100.
    for transform,normal in ((far,[0.,0.,-1.]),(np.eye(4),[0.,0.,1.])):
        tc=dict(base,cloud_normals=np.array([normal]))
        assert depth_correspondence_anchors(tc,transform,np.eye(6)*1e-8)==[]


def test_budget_quantization_risk_respects_emitted_codec_precision_ceiling():
    solution=solve_ternary_budget_algorithm1(1024,np.eye(6),np.eye(1),np.ones((1,1)),
        np.zeros((1,2,6)),[{"G_f":np.zeros((2,6)),"Gamma_k":np.eye(2),"p_i":np.array([0.,0.,1.])}],[],
        mode="bootstrap",geom_counts_grid=(0,))
    quantization_floor=solution["proxy_objective"]-solution["rep_uncertainty_trace"]
    assert quantization_floor==pytest.approx(2.**-32,rel=1e-7,abs=0.)
