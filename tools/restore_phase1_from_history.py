"""Recover explicit Phase-1 text writes/edits from the authorized DSH history.

Restoration is staged first. Historical shell commands are never executed: only
selected pathlib text-mutation AST segments run with restricted builtins/paths.
"""
from pathlib import Path
import ast
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[1]
RECOVERY=ROOT/'runs/main_recovery'
STAGE=RECOVERY/'restored'
TARGETS={
    'src/aerorescue/phase1/schema_v2.py','src/aerorescue/phase1/branch_collector.py',
    'src/aerorescue/phase1/scene_renderer.py','src/aerorescue/phase1/visual_qa.py',
    'src/aerorescue/phase1/collector.py','src/aerorescue/phase1/schema.py','src/aerorescue/phase1/iscc_core.py',
    'tools/run_phase1_full_local.py','tools/collect_phase1_pilot.py',
    'tools/colab/run_colab_5pairs.py','tools/colab/AeroRescue_Phase1_5Pairs_Colab.ipynb',
    'tools/audit_phase1_artifacts.py','tools/verify_phase1_bundle.py',
    'src/aerorescue/iscc/geometry.py','src/aerorescue/iscc/eiv_filter.py',
    'src/aerorescue/iscc/budget_solver.py','src/aerorescue/iscc/gating_and_baselines.py',
    'src/aerorescue/iscc/pair0_qwen_decoder.py','src/aerorescue/iscc/audited_pair0.py',
    'tests/test_iscc_core_math.py','tests/test_phase1_audit_regressions.py','tests/test_phase1_collector.py',
}

def relpath(value):
    p=Path(value)
    if p.is_absolute():
        try:p=p.relative_to(ROOT)
        except ValueError:return None
    return str(p) if str(p) in TARGETS else None


def events(filename):
    return [json.loads(line) for line in (RECOVERY/filename).read_text().splitlines() if line.strip()]


def args(call):
    a=call['data']['arguments']
    return json.loads(a) if isinstance(a,str) else a


def text_of(event):
    return '\n'.join(p.get('text','') for p in event['data']['message'].get('content',[]) if p.get('type')=='text')


def stage_path(name):
    assert name in TARGETS,name
    result=(STAGE/name).resolve()
    assert STAGE.resolve() in result.parents
    result.parent.mkdir(parents=True,exist_ok=True)
    return result


def main():
    history=events('11c30ac7-40d6-4d7b-b6ab-c42521afdb31.jsonl')
    earlier=events('7ae11296-5c23-4457-903c-7f2ac61d70e3.jsonl')
    # Earliest complete reads are the exact pre-audit baseline.
    lines={}; incomplete={}; original_lengths={}
    for event in history:
        if event.get('type')!='tool/result':continue
        text=text_of(event)
        path=re.search(r'<path>(.*?)</path>',text)
        if not path:continue
        name=relpath(path.group(1))
        if not name:continue
        count=re.search(r'\(End of file - total (\d+) lines\)|\(Showing lines \d+-\d+ of (\d+)',text)
        if count and name not in original_lengths:
            original_lengths[name]=int(count.group(1) or count.group(2))
        fragment=re.search(r'<content>\n(.*?)(?:\n\(End of file|\n\(Showing lines|\n\(Output capped)',text,re.S)
        if not fragment:continue
        target=lines.setdefault(name,{})
        for line in fragment.group(1).splitlines():
            m=re.match(r'^(\d+): ?(.*)$',line)
            if m and '(line truncated' not in m.group(2):target.setdefault(int(m.group(1)),m.group(2))
    # Full original writes fill any lines that the read-output cap omitted.
    for event in earlier:
        if event.get('type')=='tool/call' and event['data']['name'].split('.')[-1]=='write':
            a=args(event);name=relpath(a.get('file_path',''))
            if name:
                target=lines.setdefault(name,{})
                for n,line in enumerate(a['content'].splitlines(),1):target.setdefault(n,line)
    for name,numbered in lines.items():
        if name in original_lengths:
            numbered={n:line for n,line in numbered.items() if n<=original_lengths[name]}
        missing=sorted(set(range(1,max(numbered)+1))-set(numbered))
        if missing:incomplete[name]=missing[:12];continue
        stage_path(name).write_text('\n'.join(numbered[i] for i in range(1,max(numbered)+1))+'\n')
    branch=stage_path('src/aerorescue/phase1/branch_collector.py')
    if branch.is_file() and branch.read_text().rstrip().endswith('def main(argv=None):'):
        for event in earlier:
            if event.get('type')!='tool/call' or event['data']['name']!='bash':continue
            command=args(event).get('command','')
            if 'new3=' not in command or 'def main(argv=None)' not in command:continue
            body=re.search(r"python - <<'PY'\n(.*?)\nPY(?:\n|$)",command,re.S).group(1)
            assignment=next(n for n in ast.parse(body).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='new3' for t in n.targets))
            cli=ast.literal_eval(assignment.value)
            branch.write_text(branch.read_text().rsplit('def main(argv=None):',1)[0]+cli+'\n')
            break
    operations=[]
    safe_builtins={key:getattr(__import__('builtins'),key) for key in ('str','int','float','bool','len','list','dict','set','tuple','range','enumerate','zip','min','max','abs','isinstance','print')}
    def mapped_path(name):
        relative=relpath(str(name))
        if relative is None:raise ValueError('non-restoration path: '+str(name))
        return stage_path(relative)
    for event in history:
        if event.get('type')!='tool/call':continue
        name=event['data']['name'].split('.')[-1];a=args(event)
        if name=='write':
            path=relpath(a.get('file_path',''))
            if path:stage_path(path).write_text(a['content']);operations.append(['write',path])
        elif name=='edit':
            path=relpath(a.get('file_path',''))
            if path:
                p=stage_path(path);s=p.read_text();old=a['old_string']
                assert old in s,(event['seq'],path,'edit target missing')
                p.write_text(s.replace(old,a['new_string'],-1 if a.get('replace_all') else 1));operations.append(['edit',path])
        elif name=='bash':
            command=a.get('command','')
            m=re.search(r"(?:python|python3) - <<'PY'\n(.*?)\nPY(?:\n|$)",command,re.S)
            if not m:continue
            block=m.group(1)
            # Each path assignment starts an independent original text edit.
            chunks=re.split(r"(?m)^(?=p\s*=\s*Path\()",block)
            for chunk in chunks:
                pathmatch=re.match(r"p\s*=\s*Path\(['\"]([^'\"]+)['\"]\)",chunk)
                if not pathmatch or relpath(pathmatch.group(1)) is None:continue
                if 'p.write_text' not in chunk:continue
                tree=ast.parse(chunk)
                if any(isinstance(n,ast.Attribute) and n.attr.startswith('__') for n in ast.walk(tree)):
                    raise ValueError('unexpected dunder in restoration mutation')
                # Imports in mutation tails only carry over the next independent section.
                tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
                namespace={'__builtins__':safe_builtins,'Path':mapped_path}
                exec(compile(tree,'<authorized-text-restoration>','exec'),namespace)
                operations.append(['text_mutation',pathmatch.group(1),event['seq']])
    comparisons={}
    for name in sorted(TARGETS):
        staged=STAGE/name;current=ROOT/name
        if staged.is_file():
            raw=staged.read_bytes()
            syntax_error=None
            if name.endswith('.py'):
                try:compile(raw,name,'exec')
                except SyntaxError as exc:syntax_error=str(exc)
            comparisons[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                'syntax_error':syntax_error,
                'matches_current':current.is_file() and current.read_bytes()==raw}
    result={'stage':str(STAGE),'files':comparisons,'incomplete_initial_reads':incomplete,'operations':operations}
    (RECOVERY/'restoration_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'files':comparisons,'incomplete_initial_reads':incomplete,'operations':len(operations)},indent=2))


if __name__=='__main__':main()
