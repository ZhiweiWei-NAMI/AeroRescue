"""Lifecycle regression for the bounded preemption return (audit F-A finding 1).

Reproduces the reachable sequence on the real ``AgentRuntime``:

    A053 (safe-boundary preemption request) -> A056 suspends the executing
    original but leaves ``current`` naming it -> the replacement's second A053
    check fails (or the replacement is canceled in that window).

Before the fix the terminal transition checked ``current is None``, saw the
suspended original's stale pointer, and suppressed the resume; ``_advance_task``
then cleared the pointer, nothing drained ``_resume_stack``, and the original
stayed suspended forever.

The WP1 guarantee is kept: a replacement refused while the preempting task
*really* executes must still not resume the suspended original.
"""
from copy import deepcopy

import pytest

from rescue_scenes.core.agents.runtime import AgentRuntime, may_resume_preempted

AGENT = 'uav-1'


REPLACEMENT = {'id': 'T027@uav-1#4', 'type': 'T027', 'assigned_to': AGENT}
STACK = ['T002@coordinator-1#1']


def test_pointer_to_a_suspended_original_is_not_ownership():
    # A056 leaves current naming the suspended original; that must not
    # suppress the bounded return path.
    tasks = {'T002@coordinator-1#1': {'id': 'T002@coordinator-1#1', 'type': 'T002',
                                      'assigned_to': AGENT, 'status': 'suspended'}}
    assert may_resume_preempted(REPLACEMENT, AGENT, 'T002@coordinator-1#1', STACK, tasks)


def test_a_really_executing_owned_task_blocks_resume():
    tasks = {'T002@coordinator-1#1': {'id': 'T002@coordinator-1#1', 'type': 'T002',
                                      'assigned_to': AGENT, 'status': 'suspended'},
             'T027@uav-1#2': {'id': 'T027@uav-1#2', 'type': 'T027',
                              'assigned_to': AGENT, 'status': 'executing'}}
    assert not may_resume_preempted(REPLACEMENT, AGENT, 'T027@uav-1#2', STACK, tasks)


def test_an_accepted_replacement_also_blocks_resume():
    tasks = {'T002@coordinator-1#1': {'id': 'T002@coordinator-1#1', 'type': 'T002',
                                      'assigned_to': AGENT, 'status': 'suspended'},
             'T027@uav-1#5': {'id': 'T027@uav-1#5', 'type': 'T027',
                              'assigned_to': AGENT, 'status': 'accepted'}}
    assert not may_resume_preempted(REPLACEMENT, AGENT, None, STACK, tasks)


def test_another_agents_activity_never_blocks_this_agents_resume():
    tasks = {'T002@coordinator-1#1': {'id': 'T002@coordinator-1#1', 'type': 'T002',
                                      'assigned_to': AGENT, 'status': 'suspended'},
             'T027@uav-2#7': {'id': 'T027@uav-2#7', 'type': 'T027',
                              'assigned_to': 'uav-2', 'status': 'executing'}}
    assert may_resume_preempted(REPLACEMENT, AGENT, None, STACK, tasks)


def test_meta_task_never_releases_a_preemption():
    meta = {'id': 'T047@uav-1#1', 'type': 'T047', 'assigned_to': AGENT}
    assert not may_resume_preempted(meta, AGENT, None, STACK, {})


def test_meta_work_does_not_block_the_return():
    # Control/meta protocols run in their own lane: accepted meta work neither
    # releases a physical preemption nor keeps a suspended original parked.
    tasks = {'T002@coordinator-1#1': {'id': 'T002@coordinator-1#1', 'type': 'T002',
                                      'assigned_to': AGENT, 'status': 'suspended'},
             'T047@uav-1#1': {'id': 'T047@uav-1#1', 'type': 'T047',
                              'assigned_to': AGENT, 'status': 'accepted'}}
    assert may_resume_preempted(REPLACEMENT, AGENT, None, STACK, tasks)


def test_another_holder_cannot_release_this_agents_stack():
    foreign = {'id': 'T027@coordinator-1#9', 'type': 'T027', 'assigned_to': 'coordinator-1'}
    assert not may_resume_preempted(foreign, AGENT, None, STACK, {})


class StubSim:
    """Only the surface ``_status``/``_emit`` touch."""

    def __init__(self):
        self.t = 0.0
        self.events = []

    def emit(self, **kw):
        self.events.append(kw)


def task(tid, typ='T002', status='received', priority=70, target='b-1', **extra):
    row = {
        'id': tid, 'type': typ, 'status': status, 'rev': 1, 'priority': priority,
        'target': target, 'assigned_to': AGENT, 'accepted_by': None,
        'history': [], 'progress': {},
        'method': {'seq_id': 'Q010', 'resolved_steps': ['A001'], 'step_index': 0, 'details': []},
        'interruptibility': 'interruptible', 'deadline_s': 999.0, 'parameters': {},
        'dependency_ids': [], 'commitment_window': None, 'failure_reason': None,
        'assignment': {'delivery_state': 'delivered', 'decision': 'unknown', 'retry_count': 0},
    }
    row.update(extra)
    return row


def runtime():
    rt = AgentRuntime({'kind': 'uav', 'coordinator_id': AGENT}, AGENT, [0.0, 0.0], None)
    # Environment-dependent acceptance is the subject of other suites; these
    # tests drive the task-lifecycle state machine itself.
    rt._refresh_telemetry = lambda sim: None
    rt._select_method = lambda sim, task: None
    return rt


def install(rt, *rows):
    for row in rows:
        rt._store_task(deepcopy(row))


def resumes(rt):
    return [c for c in rt._controls if c[0] == 'A057']


def next_control(rt, sim):
    """Run the next queued control the way the kernel's control slot does."""
    action, params, _tid, _purpose, _context = rt._controls.popleft()
    return rt.apply_task_action(sim, action, params)


def test_refused_replacement_resumes_the_suspended_original():
    rt, sim = runtime(), StubSim()
    original = task('T002@coordinator-1#1', typ='T002', status='executing')
    replacement = task('T027@uav-1#4', typ='T027', target='rescue-camera/uav-1/periodic/1/person/0')
    install(rt, original, replacement)
    rt.current = original['id']

    # 1. A053 at the safe action boundary queues A056 for the original and a
    #    second A053 for the replacement (the WP1 order is preserved).
    rt._acceptance_check = lambda candidate, resuming=False: (True, 'none')
    outcome = rt.apply_task_action(sim, 'A053', {'task_id': replacement['id'], 'rev': 1})
    assert outcome.status == 'partial'
    assert rt._resume_stack == [original['id']]
    assert [c[0] for c in rt._controls] == ['A056', 'A053']

    # 2. A056 suspends the original, but the pointer still names it.
    outcome = next_control(rt, sim)  # A056 for the original
    assert outcome.status == 'succeeded'
    assert rt.tasks[original['id']]['status'] == 'suspended'
    assert rt.current == original['id']

    # 3. The replacement's second acceptance check fails. The stale pointer
    #    must not suppress the bounded return path.
    rt._acceptance_check = lambda candidate, resuming=False: (False, 'busy')
    outcome = next_control(rt, sim)  # the re-queued A053 for the replacement
    assert outcome.status == 'succeeded'
    assert rt.tasks[replacement['id']]['status'] == 'failed'
    assert rt._resume_stack == []
    assert [c[1] for c in resumes(rt)] == [{'task_id': original['id']}]

    # 4. The existing A057 path really resumes it, once.
    rt._acceptance_check = lambda candidate, resuming=False: (True, 'none')
    outcome = next_control(rt, sim)  # A057 for the original
    assert outcome.status == 'succeeded'
    assert rt.tasks[original['id']]['status'] == 'executing'
    assert rt.current == original['id']
    assert rt._resume_stack == []


def test_replacement_canceled_after_the_suspend_resumes_the_original():
    rt, sim = runtime(), StubSim()
    original = task('T002@coordinator-1#1', typ='T002', status='executing')
    replacement = task('T027@uav-1#4', typ='T027', target='candidate-1')
    install(rt, original, replacement)
    rt.current = original['id']
    rt._acceptance_check = lambda candidate, resuming=False: (True, 'none')

    rt.apply_task_action(sim, 'A053', {'task_id': replacement['id'], 'rev': 1})
    next_control(rt, sim)  # A056 suspends the original
    assert rt.tasks[original['id']]['status'] == 'suspended'
    assert rt.current == original['id']

    outcome = rt.apply_task_action(sim, 'A055',
                                   {'task_id': replacement['id'], 'reason': 'reassigned'})
    assert outcome.status == 'succeeded'
    assert rt.tasks[replacement['id']]['status'] == 'canceled'
    assert [c[1] for c in resumes(rt)] == [{'task_id': original['id']}]
    assert rt._resume_stack == []


def test_cancel_before_the_queued_suspend_is_drained_by_execute():
    """The replacement is canceled while A056 is still queued. The queued A056
    then suspends the original with no further terminal transition to drain the
    stack, so the periodic ``_execute`` safety net must return it."""
    rt, sim = runtime(), StubSim()
    original = task('T002@coordinator-1#1', typ='T002', status='executing')
    replacement = task('T027@uav-1#4', typ='T027', target='candidate-1')
    install(rt, original, replacement)
    rt.current = original['id']
    rt._acceptance_check = lambda candidate, resuming=False: (True, 'none')

    rt.apply_task_action(sim, 'A053', {'task_id': replacement['id'], 'rev': 1})
    rt.apply_task_action(sim, 'A055', {'task_id': replacement['id'], 'reason': 'reassigned'})
    assert rt.tasks[replacement['id']]['status'] == 'canceled'
    # The preempting task is still executing, so nothing resumes yet.
    assert rt._resume_stack == [original['id']]
    assert resumes(rt) == []

    next_control(rt, sim)  # the already queued A056 still suspends it
    assert rt.tasks[original['id']]['status'] == 'suspended'
    assert rt.current == original['id']

    rt._execute(sim)
    assert [c[1] for c in resumes(rt)] == [{'task_id': original['id']}]
    assert rt._resume_stack == []


def test_no_resume_while_the_preempting_task_really_executes():
    """GT04 regression (WP1): a refused replacement must not resume the
    suspended original while its preemptor is still executing."""
    rt, sim = runtime(), StubSim()
    original = task('T002@coordinator-1#1', typ='T002', status='suspended')
    preemptor = task('T027@uav-1#2', typ='T027', status='executing', target='candidate-2')
    replacement = task('T027@uav-1#4', typ='T027', status='received', target='candidate-4')
    install(rt, original, preemptor, replacement)
    rt.current = preemptor['id']
    rt._resume_stack = [original['id']]

    rt._status(sim, rt.tasks[replacement['id']], 'failed', 'replacement refused')

    assert rt.tasks[replacement['id']]['status'] == 'failed'
    assert rt.tasks[preemptor['id']]['status'] == 'executing'
    assert rt._resume_stack == [original['id']]
    assert resumes(rt) == []


def test_suspended_pointer_does_not_hide_an_executing_preemptor():
    """Ownership is the real status of this agent's tasks, not the pointer: a
    pointer naming the suspended original must not authorize a resume while the
    preemptor still executes."""
    rt, sim = runtime(), StubSim()
    original = task('T002@coordinator-1#1', typ='T002', status='suspended')
    preemptor = task('T027@uav-1#2', typ='T027', status='executing', target='candidate-2')
    replacement = task('T027@uav-1#4', typ='T027', status='received', target='candidate-4')
    install(rt, original, preemptor, replacement)
    rt.current = original['id']  # the A056 stale pointer
    rt._resume_stack = [original['id']]

    rt._status(sim, rt.tasks[replacement['id']], 'failed', 'replacement refused')

    assert rt._resume_stack == [original['id']]
    assert resumes(rt) == []


def test_return_path_is_bounded_and_never_retries():
    """A stack entry that is no longer suspended is consumed, not retried."""
    rt, sim = runtime(), StubSim()
    done = task('T002@coordinator-1#1', typ='T002', status='completed')
    install(rt, done)
    rt._resume_stack = [done['id']]

    assert rt._resume_preempted(sim) is False
    assert rt._resume_stack == []
    assert resumes(rt) == []
    # A second call with an empty stack is a no-op.
    assert rt._resume_preempted(sim) is False


def test_meta_work_does_not_block_the_resume_path():
    rt, sim = runtime(), StubSim()
    original = task('T002@coordinator-1#1', typ='T002', status='suspended')
    meta = task('T047@uav-1#1', typ='T047', status='accepted', target=AGENT)
    install(rt, original, meta)
    rt._resume_stack = [original['id']]

    assert rt._resume_preempted(sim) is True
    assert [c[1] for c in resumes(rt)] == [{'task_id': original['id']}]
    assert rt._resume_stack == []


@pytest.mark.parametrize('status', ['accepted', 'executing'])
def test_any_owned_live_task_blocks_the_return(status):
    rt, sim = runtime(), StubSim()
    original = task('T002@coordinator-1#1', typ='T002', status='suspended')
    live = task('T027@uav-1#5', typ='T027', status=status, target='candidate-5')
    install(rt, original, live)
    rt._resume_stack = [original['id']]

    assert rt._resume_preempted(sim) is False
    assert rt._resume_stack == [original['id']]
    assert resumes(rt) == []
