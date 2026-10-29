"""Loopback-only HTTP explorer over the public Environment interface."""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sys
import subprocess
import os
import logging
import math
import mimetypes
from pathlib import Path
import re
import threading
import time
from urllib.parse import parse_qs, unquote, urlsplit
import uuid

from . import catalog, contracts
from .agents.embodiment import EMBODIMENTS
from .api import DEMO_CONFIGS, Environment
from .world.profiles import OVERLAYS, PROFILES
from .provenance import source_revision
from .world.demo import BOUNDS as DEMO_BOUNDS, DEMO_NOTE_ZH, DEMO_SOURCE_ID, DEMO_TITLE_ZH


WEB_ROOT = Path(__file__).resolve().parent.parent / "web"
DISRUPTION_KINDS = (
    "aftershock_reblock", "link_outage", "low_battery", "service_failure",
    "target_moves", "equipment_fault", "wrong_evidence", "fire_spread",
    "water_rise", "cancel_task",
)
LAYERS = {
    "world", "knowledge", "task", "allocation", "action", "comms",
    "compute", "evaluation", "upstream",
}
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:#@-]{0,199}\Z")
SESSION_ID = re.compile(r"[0-9a-f]{32}\Z")
MAX_BODY_BYTES = 2 * 1024 * 1024


class APIError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def require_id(value, label="id"):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise APIError(400, "invalid_id", f"无效标识 / Invalid {label}")
    return value


def number(value, label, minimum=0):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or (isinstance(value, float) and not math.isfinite(value)) or value < minimum):
        raise APIError(400, "invalid_parameter", f"无效数值 / Invalid {label}")
    return value


def integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise APIError(400, "invalid_parameter", f"无效整数 / Invalid {label}")
    return value


def reject_constant(value):
    raise ValueError(f"非有限 JSON 数值 / Non-finite JSON number: {value}")


def finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        reject_constant(value)
    return result


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("重复 JSON 键 / Duplicate JSON key")
        result[key] = value
    return result


@dataclass
class Session:
    env: Environment
    lock: threading.RLock = field(default_factory=threading.RLock)


class Application:
    """Own the bounded session registry and startup-only coverage cache."""

    def __init__(self):
        self.catalog = catalog.load()
        self.sessions = OrderedDict()
        self.sessions_lock = threading.Lock()
        self.coverage_lock = threading.Lock()
        self.coverage_thread = None
        self.demos = {}
        demo_meta = []
        display_keys = {"id", "title_zh", "name", "name_zh"}
        for item in DEMO_CONFIGS:
            demo_id = require_id(item["id"], "demo id")
            cfg = item.get("config")
            if cfg is None:
                cfg = {key: value for key, value in item.items() if key not in display_keys}
            self.demos[demo_id] = deepcopy(cfg)
            title = item.get("title_zh") or item.get("name_zh") or item.get("name") or demo_id
            demo_meta.append({"id": demo_id, "title_zh": title})
        scenarios = []
        for sid, profile in PROFILES.items():
            scenarios.append({
                "id": sid, "name_zh": profile["name_zh"],
                "family": profile["family"],
                "task_count": len(profile["catalog_tasks"]),
                "hazards": profile["hazards"], "agent_mix": profile["agent_mix"],
            })
        actions = []
        for aid, row in self.catalog.actions.items():
            actions.append({
                "id": aid, "name": row["name"],
                "name_zh": row.get("name_zh", row["name"]),
                "layer": row["layer"], "capabilities": row["capabilities"],
            })
        # Base-map semantic sources: the labelled authored fallback plus every converted upstream
        # sidecar in ./upstream (read-only intake by rescue_env.world.upstream_gt). Clients may only
        # pick one of these ids; arbitrary file paths are never accepted from the browser.
        self.semantic_sources = {"authored": None}
        sources_meta = [{"id": "authored", "title_zh": "作者标注底图（未接入上游）", "kind": "authored"}]
        upstream_dir = Path(__file__).resolve().parents[1] / "upstream"
        for path in sorted(upstream_dir.glob("*.semantics.json")):
            try:
                doc = json.loads(path.read_text())
                sid = path.name[: -len(".semantics.json")]
                self.semantic_sources[sid] = str(path)
                sources_meta.append({"id": sid, "kind": "upstream", "bounds_m": doc["frame"]["bounds_m"],
                                     "title_zh": f"{sid}（上游只读二维语义布局，未用高程/网格）", "producer": doc["source"]["producer"],
                                     "note_zh": doc["source"]["note"],
                                     "entity_counts_by_class": dict(sorted(Counter(e["class"] for e in doc["entities"]).items())),
                                     # Of those, authored here (buildings on candidate plots, trees from optional anchors, stations).
                                     "authored_counts_by_class": dict(sorted(Counter(e["class"] for e in doc["entities"]
                                                                                     if e.get("label_source") == "authored").items()))})
            except (OSError, ValueError, KeyError):
                logging.exception("Skipping unreadable semantic source %s", path.name)
        # Compact authored rescue demo: a separate, clearly labelled world preset (not a background
        # import, not a random scenario); hidden occupants must still be discovered by agents.
        self.semantic_sources[DEMO_SOURCE_ID] = "demo"
        sources_meta.append({"id": DEMO_SOURCE_ID, "kind": "demo", "bounds_m": list(DEMO_BOUNDS[:2]) + list(DEMO_BOUNDS[2:]),
                             "title_zh": DEMO_TITLE_ZH, "note_zh": DEMO_NOTE_ZH,
                             "recommended": {"scenario_id": "SC01", "seed": 1, "until_s": 2400}})
        self.meta = deepcopy({
            "semantic_sources": sources_meta,
            "catalog": self.catalog.verify(), "scenarios": scenarios,
            "overlays": list(OVERLAYS.values()),
            "embodiments": list(EMBODIMENTS.values()),
            "disruption_kinds": list(DISRUPTION_KINDS),
            "demo_configs": demo_meta, "state_machines": contracts.STATE_MACHINES,
            "actions": actions,
            "capabilities": list(self.catalog.capabilities.values()),
        })

    COVERAGE_FILE = Path(__file__).resolve().parents[1] / "evidence" / "coverage" / "coverage.json"

    def start_coverage(self):
        """Coverage is measured out of process (rescue_env.coverage_job) and read from its file.

        The job writes partial results after every demo run, so the explorer stays usable while
        coverage is still being measured; interactive sessions never share the sweep's CPU/GIL.
        """
        with self.coverage_lock:
            if self.coverage_thread is not None:
                return
            self.coverage_thread = "file"
        # Reuse a sweep only if it measured THIS source revision; a partial file of this revision
        # that was updated recently belongs to a job that is still running (no duplicate job).
        done = False
        try:
            measured = json.loads(self.COVERAGE_FILE.read_text()).get("measured", {})
            same = measured.get("source_revision") == source_revision() and "dirty" not in measured.get("source_revision", "dirty")
            running = measured.get("partial", True) and time.time() - self.COVERAGE_FILE.stat().st_mtime < 900
            done = same and (not measured.get("partial", True) or running)
        except (OSError, ValueError):
            done = False
        if not done:
            self.COVERAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
            log = open(self.COVERAGE_FILE.parent / "job.log", "ab")
            subprocess.Popen([sys.executable, "-m", "rescue_env.coverage_job"], cwd=str(self.COVERAGE_FILE.parents[2]),
                             stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True,
                             preexec_fn=lambda: os.nice(10))

    def read_coverage(self):
        try:
            data = json.loads(self.COVERAGE_FILE.read_text())
        except (OSError, ValueError):
            return 200, {"pending": True}
        return 200, data


    def create_session(self, body):
        if "demo" in body:
            if set(body) != {"demo"}:
                raise APIError(400, "invalid_config", "演示配置不可混用 / Demo request must contain only demo")
            demo_id = require_id(body["demo"], "demo id")
            if demo_id not in self.demos:
                raise APIError(404, "unknown_demo", "演示不存在 / Unknown demo")
            cfg = deepcopy(self.demos[demo_id])
        else:
            cfg = deepcopy(body)
        # Semantic input: only listed source ids ({"semantics": {"source_id": ...}}) are accepted.
        # Server-side demo presets carry their own (trusted, path-free) semantics; translate to an id.
        if "demo" in body and isinstance(cfg.get("semantics"), dict) and cfg["semantics"].get("source") == "demo":
            cfg["semantics"] = {"source_id": DEMO_SOURCE_ID}
        sem = cfg.pop("semantics", None)
        if sem is not None:
            if not isinstance(sem, dict) or set(sem) - {"source_id"} or sem.get("source_id") not in self.semantic_sources:
                raise APIError(400, "invalid_semantics", "底图语义来源无效 / Unknown semantic source")
            path = self.semantic_sources[sem["source_id"]]
            cfg["semantics"] = ({"source": "authored", "path": None} if path is None else
                                {"source": "demo", "path": None} if path == "demo" else {"source": "upstream", "path": path})
        # Reject unaddressable client IDs before allocating simulation state.
        agents = cfg.get("agents", "profile")
        if isinstance(agents, list):
            for agent in agents:
                if isinstance(agent, dict):
                    require_id(agent.get("id"), "agent id")
                    require_id(agent.get("kind"), "embodiment kind")
        disruptions = cfg.get("disruptions", [])
        if isinstance(disruptions, list):
            for disruption in disruptions:
                if isinstance(disruption, dict):
                    self.validate_disruption(disruption, scheduled=True)
        # Configuration defaults, capabilities and semantic shapes belong to Environment.
        try:
            env = Environment.from_config(cfg)
        except OSError as exc:
            raise APIError(400, "invalid_semantics", "语义文件无法读取 / Cannot read semantic input") from exc
        session = Session(env)
        with session.lock:
            response = deepcopy({"config": env.config, "snapshot": env.snapshot(view="truth"), "meta": self.meta})
        sid = uuid.uuid4().hex
        with self.sessions_lock:
            self.sessions[sid] = session
            while len(self.sessions) > 8:
                self.sessions.popitem(last=False)
        response["session_id"] = sid
        return response

    def session(self, sid):
        if not SESSION_ID.fullmatch(sid):
            raise APIError(400, "invalid_session_id", "会话标识无效 / Invalid session id")
        with self.sessions_lock:
            if sid not in self.sessions:
                raise APIError(404, "unknown_session", "会话不存在或已淘汰 / Session missing or evicted")
            self.sessions.move_to_end(sid)
            return self.sessions[sid]

    @staticmethod
    def validate_disruption(body, scheduled=False):
        kind = body.get("kind")
        allowed = {"kind", "target", "params"}
        if scheduled:
            allowed.add("t")
        elif kind == "cancel_task":
            allowed.update({"agent", "task"})
        if (set(body) - allowed or not isinstance(kind, str)
                or kind not in DISRUPTION_KINDS):
            raise APIError(400, "invalid_disruption", "扰动无效 / Invalid disruption")
        params = body.get("params", {})
        if not isinstance(params, dict):
            raise APIError(400, "invalid_disruption", "params 需要对象 / params must be an object")
        if scheduled:
            number(body.get("t", 0), "disruption.t")
        if body.get("target") is not None:
            require_id(body["target"], "disruption target")
        for key in ("agent", "task", "entity"):
            if key in params:
                require_id(params[key], key)
        if kind == "cancel_task":
            for key in ("agent", "task"):
                if key in body:
                    require_id(body[key], key)
                    if key in params and body[key] != params[key]:
                        raise APIError(400, "invalid_disruption", "取消参数冲突 / Conflicting cancellation ids")
        return body

    def catalog_record(self, kind, record_id):
        kinds = {
            "scenarios", "overlays", "tasks", "sequences", "steps", "actions",
            "capabilities", "fields", "sources", "quotes", "rules",
        }
        if kind not in kinds:
            raise APIError(404, "unknown_catalog_kind", "目录类别不存在 / Unknown catalog kind")
        require_id(record_id, "catalog id")
        if kind == "steps" and ":" in record_id:
            seq_id, step_id = record_id.rsplit(":", 1)
            if not re.fullmatch(r"Q[0-9]{3}", seq_id) or not re.fullmatch(r"[1-9][0-9]*", step_id):
                raise APIError(400, "invalid_id", "步骤标识无效 / Invalid sequence step id")
            # Use only Catalog's public normalized steps interface; actor/rule/
            # terminal retain the catalog text, without reaching into loader internals.
            record = next((dict(row, sequence=seq_id)
                           for row in self.catalog.steps.get(seq_id, [])
                           if str(row["step"]) == step_id), None)
            if record is None:
                raise APIError(404, "unknown_catalog_id", "目录条目不存在 / Unknown catalog id")
            return record
        table = getattr(self.catalog, kind)
        record = table.get(record_id) if isinstance(table, dict) else next(
            (row for row in table if row["id"] == record_id), None,
        )
        if record is None:
            raise APIError(404, "unknown_catalog_id", "目录条目不存在 / Unknown catalog id")
        return record


class Handler(BaseHTTPRequestHandler):
    server_version = "RescueEnvironment/0.5"

    @property
    def app(self):
        return self.server.app

    def send_json(self, data, status=200):
        payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def send_error(self, code, message=None, explain=None):
        # BaseHTTPRequestHandler also uses this for malformed HTTP / unknown verbs.
        try:
            phrase = HTTPStatus(code).phrase
        except ValueError:
            phrase = "Request failed"
        self.send_json({"error": {"code": "http_error", "message": message or phrase}}, code)

    def read_json(self):
        if self.headers.get("Transfer-Encoding"):
            raise APIError(400, "invalid_body", "不支持分块请求 / Transfer encoding unsupported")
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not lengths[0].isdigit():
            raise APIError(400, "invalid_body", "需要有效长度 / Valid Content-Length required")
        length = int(lengths[0])
        if length > MAX_BODY_BYTES:
            raise APIError(413, "body_too_large", "请求过大 / Request body too large")
        content_type = self.headers.get("Content-Type", "application/json").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise APIError(415, "invalid_content_type", "需要 JSON / Expected application/json")
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Incomplete JSON body")
            body = json.loads(raw.decode("utf-8"), parse_constant=reject_constant,
                              parse_float=finite_float, object_pairs_hook=unique_object)
        except (UnicodeError, ValueError, RecursionError) as exc:
            raise APIError(400, "invalid_json", "JSON 无效 / Invalid JSON") from exc
        if not isinstance(body, dict):
            raise APIError(400, "invalid_body", "需要 JSON 对象 / Expected JSON object")
        return body

    def do_GET(self):
        self.handle_request()

    def do_HEAD(self):
        self.handle_request()

    def do_POST(self):
        self.handle_request()

    def handle_request(self):
        try:
            parsed = urlsplit(self.path)
            path = unquote(parsed.path, encoding="utf-8", errors="strict")
            if "\x00" in path or "\\" in path:
                raise APIError(400, "invalid_path", "路径无效 / Invalid path")
            query = parse_qs(parsed.query, keep_blank_values=True, max_num_fields=20)
            if path == "/api" or path.startswith("/api/"):
                status, data = self.api_request(path, query)
                self.send_json(data, status)
            elif self.command in {"GET", "HEAD"}:
                self.serve_static(path)
            else:
                raise APIError(405, "method_not_allowed", "请求方法不支持 / Method not allowed")
        except APIError as exc:
            self.close_connection = True
            self.send_json({"error": {"code": exc.code, "message": exc.message}}, exc.status)
        except (ValueError, TypeError, OverflowError) as exc:
            self.close_connection = True
            self.send_json({"error": {"code": "invalid_request", "message": str(exc)}}, 400)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            logging.exception("HTTP request failed")
            self.close_connection = True
            self.send_json({"error": {
                "code": "internal_error", "message": "请求处理失败 / Internal server error",
            }}, 500)

    def api_request(self, path, query):
        method = "GET" if self.command == "HEAD" else self.command
        if path == "/api/meta":
            self.require_method(method, "GET")
            return 200, self.app.meta
        if path == "/api/coverage":
            self.require_method(method, "GET")
            return self.app.read_coverage()
        if path == "/api/sessions":
            self.require_method(method, "POST")
            return 201, self.app.create_session(self.read_json())
        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[:2] == ["api", "catalog"]:
            self.require_method(method, "GET")
            return 200, self.app.catalog_record(parts[2], parts[3])
        if len(parts) not in {4, 5} or parts[:2] != ["api", "sessions"]:
            raise APIError(404, "unknown_endpoint", "接口不存在 / Unknown endpoint")
        session = self.app.session(parts[2])
        with session.lock:
            # Freeze public outputs while locked, including lists that may be live views.
            result = self.session_request(session.env, parts[3:], method, query)
            return 200, deepcopy(result)

    @staticmethod
    def require_method(actual, expected):
        if actual != expected:
            raise APIError(405, "method_not_allowed", "请求方法不支持 / Method not allowed")

    @staticmethod
    def query_value(query, name, default):
        values = query.get(name, [default])
        if len(values) != 1:
            raise APIError(400, "invalid_query", "查询参数重复 / Duplicate query parameter")
        return values[0]

    def session_request(self, env, route, method, query):
        endpoint = route[0]
        if len(route) == 2:
            if endpoint != "task":
                raise APIError(404, "unknown_endpoint", "接口不存在 / Unknown endpoint")
            self.require_method(method, "GET")
            task_id = require_id(route[1], "task id")
            records = [row for row in env.tasks() if row.get("id") == task_id]
            if not records:
                raise APIError(404, "unknown_task", "任务不存在 / Unknown task")
            all_events = env.events()
            direct = [event for event in all_events
                      if event.get("task") == task_id or event.get("subject") == task_id]
            instances = {event["action_instance"] for event in direct if event.get("action_instance")}
            events = [event for event in all_events
                      if event.get("task") == task_id or event.get("subject") == task_id
                      or event.get("action_instance") in instances
                      or event.get("subject") in instances]
            return {"task_id": task_id, "records": records, "events": events}
        if endpoint == "step":
            self.require_method(method, "POST")
            body = self.read_json()
            if set(body) - {"n", "until_s"} or ("n" in body and "until_s" in body):
                raise APIError(400, "invalid_step", "指定 n 或 until_s / Specify n or until_s")
            if "until_s" in body:
                until_s = number(body["until_s"], "until_s", env.snapshot(view="truth")["t"])
                events = env.run(until_s)
            else:
                events = env.step(n=integer(body.get("n", 1), "n", 0))
            snapshot = env.snapshot(view="truth")
            return {"t": snapshot["t"], "events": events, "tasks": env.tasks(),
                    "allocation": env.allocation(), "objectives": env.objectives(),
                    "agents": snapshot["agents"]}
        if endpoint == "inject":
            self.require_method(method, "POST")
            body = self.app.validate_disruption(self.read_json())
            # The public API already returns exactly the newly emitted events.
            # Omitted cancel ids intentionally select current work, as in the demos.
            return env.inject(body)
        self.require_method(method, "GET")
        if endpoint == "snapshot":
            view = self.query_value(query, "view", "truth")
            if view not in {"truth", "central"}:
                if not view.startswith("agent:"):
                    raise APIError(400, "invalid_view", "视角无效 / Invalid view")
                agent_id = require_id(view[6:], "agent id")
                if agent_id not in {agent["id"] for agent in env.snapshot(view="truth")["agents"]}:
                    raise APIError(404, "unknown_agent", "智能体不存在 / Unknown agent")
            return env.snapshot(view=view)
        if endpoint == "events":
            raw_since = self.query_value(query, "since", "0")
            if not re.fullmatch(r"[0-9]+", raw_since):
                raise APIError(400, "invalid_since", "事件序号无效 / Invalid event sequence")
            since = int(raw_since)
            layer = self.query_value(query, "layer", None) or None
            if layer is not None and layer not in LAYERS:
                raise APIError(400, "invalid_layer", "事件层无效 / Invalid event layer")
            return env.events(since_seq=since, layer=layer)
        if endpoint == "tasks":
            return env.tasks()
        raise APIError(404, "unknown_endpoint", "接口不存在 / Unknown endpoint")

    def serve_static(self, path):
        parts = path.split("/")
        if ".." in parts:
            raise APIError(403, "unsafe_path", "禁止目录穿越 / Path traversal forbidden")
        root = WEB_ROOT.resolve()
        target = (root / (path.lstrip("/") or "index.html")).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise APIError(403, "unsafe_path", "路径越界 / Path outside web root") from exc
        if not target.is_file():
            raise APIError(404, "not_found", "文件不存在 / File not found")
        payload = target.read_bytes()
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if content_type.startswith("text/") or target.suffix in {".js", ".json", ".svg"}:
            content_type += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, app):
        self.app = app
        super().__init__(address, Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rescue environment explorer (loopback only)")
    parser.add_argument("--port", type=int, default=8851)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    app = Application()
    with Server(("127.0.0.1", args.port), app) as server:
        app.start_coverage()
        logging.info("Rescue explorer: http://127.0.0.1:%s", args.port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
