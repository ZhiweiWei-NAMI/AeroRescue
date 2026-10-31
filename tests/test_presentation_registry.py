"""Compile rescue types through AAS and retain explicit overlay descriptors."""
from copy import deepcopy

from aeroagentsim.scenario.loader import load_registry
from rescue_scenes.engine import command_descriptors
from rescue_scenes.presentation import AGENT, BASE, BELIEF, FIELDS, registry_snapshot, scenario_registry


def test_compiled_registry_retains_types_fields_messages_and_local_inheritance(tmp_path):
    snapshot=registry_snapshot()
    snapshot['registry']['types'].append({'id':'rescue:Upstream_test','parents':[AGENT],'abstract':False})
    snapshot['registry']['messages'].extend(command_descriptors())
    original=deepcopy(snapshot)
    spec=scenario_registry(tmp_path/'aerograph',snapshot)
    compiled,registry=load_registry(spec,base=tmp_path)
    assert snapshot==original
    assert compiled.registry.revision=='aerograph-compiler/1'
    assert {r.id for r in registry.types}=={r['id'] for r in snapshot['registry']['types']}
    assert {r.id for r in registry.fields}==set(FIELDS)
    assert {r.id for r in registry.messages}=={r['id'] for r in snapshot['registry']['messages']}
    assert registry.is_a('rescue:Upstream_test',AGENT)
    assert {f.id for f in registry.fields if registry.is_a('rescue:Upstream_test',f.declaring_type)}==set(FIELDS)
    assert not registry.is_a(BASE,registry.field(BELIEF).declaring_type)
    provenance=compiled.details['provenance']['descriptors']['types']
    assert BASE in provenance and AGENT in provenance
    assert all(row['file'] and row['pointer'].startswith('/') for row in provenance.values())
