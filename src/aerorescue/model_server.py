"""Serve local Qwen3.5 base weights as an OpenAI-compatible endpoint.

Run it with the AERO_WORLD Qwen interpreter, which owns the pinned torch and
transformers build::

    /mnt/data1/weizhiwei/AERO_WORLD_runtime/qwen35/python/bin/python \
        -m aerorescue.model_server --model Qwen3.5-2B-Base --port 8788

The weights live in the runtime's Hugging Face cache
(``/mnt/data1/weizhiwei/AERO_WORLD_runtime/qwen35/hf``); ``--weights-root``
overrides the root and ``--model`` also accepts an explicit snapshot directory.
Only ``/v1/chat/completions``, ``/v1/models`` and ``/healthz`` are served, with
one generation at a time. This is a development endpoint, not a hardened server.
"""
import argparse
import base64
import binascii
from io import BytesIO
import json
import os
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_ROOT = os.environ.get('AERO_QWEN_ROOT', '/mnt/data1/weizhiwei/AERO_WORLD_runtime/qwen35')
MODELS = {'Qwen3.5-2B-Base': 'Qwen/Qwen3.5-2B-Base',
          'Qwen3.5-0.8B-Base': 'Qwen/Qwen3.5-0.8B-Base'}
MAX_BODY_BYTES = 32 * 1024 * 1024


def resolve_model(name, root=DEFAULT_ROOT):
    """Resolve a model name to its local snapshot directory (never downloads)."""
    direct = Path(name).expanduser()
    if direct.is_dir():
        return direct
    repo = MODELS.get(name, name)
    if repo.count('/') != 1:
        raise SystemExit(f'unknown model {name!r}; choose one of {sorted(MODELS)} or a snapshot path')
    cache = Path(root).expanduser() / 'hf' / 'hub'
    folder = 'models--' + repo.replace('/', '--')
    snapshots = sorted((cache / folder / 'snapshots').glob('*/config.json'))
    if not snapshots:
        raise SystemExit(f'model snapshot not found under {cache / folder}; '
                         f'fetch it with HF_HOME={Path(root) / "hf"} huggingface-cli download {repo}')
    return snapshots[-1].parent


def _content_parts(content):
    """Split OpenAI message content into (text, image data URLs)."""
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        raise ValueError('message content must be a string or a list of parts')
    texts, images = [], []
    for part in content:
        if not isinstance(part, dict):
            raise ValueError('content parts must be objects')
        if part.get('type') == 'text' and isinstance(part.get('text'), str):
            texts.append(part['text'])
        elif part.get('type') == 'image_url':
            url = (part.get('image_url') or {}).get('url')
            if not isinstance(url, str):
                raise ValueError('image_url part requires a url string')
            images.append(url)
        else:
            raise ValueError(f"unsupported content part type: {part.get('type')!r}")
    return '\n'.join(texts), images


def _load_image(url):
    from PIL import Image

    if url.startswith('data:'):
        header, _, payload = url.partition(',')
        if ';base64' not in header:
            raise ValueError('data URLs must be base64 encoded')
        try:
            raw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError('invalid base64 image payload') from exc
        return Image.open(BytesIO(raw)).convert('RGB')
    if url.startswith(('http://', 'https://')):
        raise ValueError('remote image URLs are not fetched; send a data URL')
    return Image.open(Path(url).expanduser()).convert('RGB')


class LocalQwen:
    """One loaded model behind a single generation lock."""

    def __init__(self, path, device='auto', dtype='bfloat16'):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.torch = torch
        self.lock = threading.Lock()
        self.processor = AutoProcessor.from_pretrained(str(path))
        # Qwen3.5's processor ships without its own template; the tokenizer
        # carries the exact chat template used by the pinned weights.
        if getattr(self.processor, 'chat_template', None) is None:
            self.processor.chat_template = self.processor.tokenizer.chat_template
        kwargs = {'dtype': getattr(torch, dtype), 'device_map': device}
        self.model = AutoModelForImageTextToText.from_pretrained(str(path), **kwargs)
        self.model.eval()
        self.name = Path(path).name

    def _prompt(self, messages):
        converted, images = [], []
        for message in messages:
            role = message.get('role')
            if role not in {'system', 'user', 'assistant'}:
                raise ValueError(f'unsupported role: {role!r}')
            text, urls = _content_parts(message.get('content', ''))
            parts = []
            for url in urls:
                images.append(_load_image(url))
                parts.append({'type': 'image'})
            if text:
                parts.append({'type': 'text', 'text': text})
            converted.append({'role': role, 'content': parts or text})
        prompt = self.processor.apply_chat_template(converted, add_generation_prompt=True,
                                                    tokenize=False)
        return prompt, images

    def complete(self, messages, max_tokens, temperature):
        prompt, images = self._prompt(messages)
        with self.lock:
            inputs = self.processor(text=[prompt], images=images or None, return_tensors='pt')
            inputs = inputs.to(self.model.device)
            with self.torch.inference_mode():
                options = {'max_new_tokens': max_tokens, 'do_sample': temperature > 0}
                if temperature > 0:
                    options['temperature'] = temperature
                generated = self.model.generate(**inputs, **options)
            new_tokens = generated[:, inputs['input_ids'].shape[1]:]
            text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, (list, tuple)) else [eos]
        stopped = bool(new_tokens.shape[1] and int(new_tokens[0, -1]) in eos)
        return {'text': text.strip(), 'prompt_tokens': int(inputs['input_ids'].shape[1]),
                'completion_tokens': int(new_tokens.shape[1]),
                'finish_reason': 'length' if new_tokens.shape[1] >= max_tokens and not stopped else 'stop'}


def _completion(model, text, prompt_tokens, completion_tokens, finish_reason='stop'):
    return {'id': f'chatcmpl-{int(time.time() * 1000)}', 'object': 'chat.completion',
            'created': int(time.time()), 'model': model,
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text},
                         'finish_reason': finish_reason}],
            'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens,
                      'total_tokens': prompt_tokens + completion_tokens}}


def make_handler(engine, default_max_tokens=512):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, fmt, *args):
            if os.environ.get('AERO_QWEN_VERBOSE'):
                super().log_message(fmt, *args)

        def _send(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status, message, code='BAD_REQUEST'):
            self._send(status, {'error': {'message': message, 'type': code}})

        def do_GET(self):
            if self.path == '/healthz':
                self._send(200, {'status': 'ok', 'model': engine.name})
            elif self.path == '/v1/models':
                self._send(200, {'object': 'list', 'data': [
                    {'id': engine.name, 'object': 'model', 'owned_by': 'local'}]})
            else:
                self._error(404, 'not found', 'NOT_FOUND')

        def do_POST(self):
            if self.path != '/v1/chat/completions':
                self._error(404, 'not found', 'NOT_FOUND')
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_BODY_BYTES:
                    raise ValueError('request body length is outside the accepted range')
                request = json.loads(self.rfile.read(length))
                messages = request['messages']
                if not isinstance(messages, list) or not messages:
                    raise ValueError('messages must be a nonempty list')
                max_tokens = int(request.get('max_tokens', default_max_tokens))
                if not 0 < max_tokens <= 8192:
                    raise ValueError('max_tokens must be in [1, 8192]')
                temperature = float(request.get('temperature', 0.0))
                if not 0.0 <= temperature <= 2.0:
                    raise ValueError('temperature must be in [0, 2]')
            except (ValueError, KeyError, TypeError) as exc:
                self._error(400, str(exc))
                return
            try:
                result = engine.complete(messages, max_tokens, temperature)
            except (ValueError, OSError) as exc:
                self._error(400, str(exc))
                return
            except Exception as exc:  # noqa: BLE001 -- report the typed fault, keep serving
                self._error(500, f'{type(exc).__name__}: {exc}', 'INFERENCE_ERROR')
                return
            self._send(200, _completion(engine.name, result['text'], result['prompt_tokens'],
                                        result['completion_tokens'], result.get('finish_reason', 'stop')))

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description='Local Qwen3.5 OpenAI-compatible chat server')
    parser.add_argument('--model', default='Qwen3.5-2B-Base',
                        help='model name from %s or an explicit snapshot directory' % sorted(MODELS))
    parser.add_argument('--weights-root', default=DEFAULT_ROOT)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8788)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--max-tokens', type=int, default=512, help='default completion cap')
    parser.add_argument('--check', action='store_true', help='resolve the weights and exit')
    args = parser.parse_args(argv)
    path = resolve_model(args.model, args.weights_root)
    if args.check:
        print(json.dumps({'model': args.model, 'path': str(path)}))
        return 0
    engine = LocalQwen(path, device=args.device, dtype=args.dtype)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine, args.max_tokens))
    print(json.dumps({'status': 'serving', 'model': args.model, 'path': str(path),
                      'url': f'http://{args.host}:{args.port}/v1'}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
