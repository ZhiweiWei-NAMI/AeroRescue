"""Camera envelopes must not enter the legacy data_json reducer protocol."""
import json
from types import SimpleNamespace
from pathlib import Path

from aeroagentsim.packs.lowalt.metrics import RescueReducer, RescueState


def message(schema, payload):
    return SimpleNamespace(source_kind='partition', schema_id=schema, payload=payload, at_ns=1_000_000_000)


def test_camera_payload_does_not_enter_legacy_rescue_json_protocol():
    reducer=RescueReducer()
    state=RescueState()
    recorded=json.loads((Path(__file__).parent/'fixtures/rescue_camera_request.json').read_text())
    payload=recorded['payload']
    for schema in (recorded['schema_id'], 'rescue.camera.result'):
        assert schema not in reducer.input_schemas
        reducer.consume(state,message(schema,payload))
    assert state.rescue_seen is False
    assert state.rescue_values['rescue_bus_messages']==0
    assert state.rescue_values['rescue_bus_bytes']==0
    assert 'data_json' not in payload


def test_existing_rescue_event_still_counts_legacy_protocol_and_outcome():
    reducer=RescueReducer()
    state=RescueState()
    data={'kind':'outcome','event':'person-1','success':True,
          'time_to_confirmation_s':2.0,'time_to_delivery_s':5.0}
    raw=json.dumps(data)
    reducer.consume(state,message('rescue.event',{'data_json':raw}))
    assert state.rescue_values['rescue_bus_messages']==1
    assert state.rescue_values['rescue_bus_bytes']==len(raw.encode('utf-8'))
    assert reducer.snapshot(state)['rescue_success']==1
    assert reducer.snapshot(state)['rescue_time_to_delivery_s']==5.0
