"""SAM jobs use the native /prompt validation and queue submission path."""

from __future__ import annotations

import asyncio
import copy
import json
import threading
import uuid

from aiohttp import web

_JOBS = {}
_LOCK = threading.RLock()
MAX_JOBS = 32


class _PromptRequest:
    def __init__(self, request, payload):
        self._request = request
        self._payload = payload

    async def json(self):
        return self._payload

    def __getattr__(self, name):
        return getattr(self._request, name)


async def _submit_prompt(server, request, payload):
    # Calling the registered handler preserves version-specific validation,
    # replacement hooks, sensitive fields and queue tuple layout.
    handler = next((
        route.handler for route in server.routes
        if route.method == "POST" and getattr(route, "path", None) == "/prompt"
    ), None)
    if handler is None:
        raise RuntimeError("Native /prompt handler is unavailable.")
    response = await handler(_PromptRequest(request, payload))
    if not isinstance(response, web.Response):
        raise RuntimeError("Unsupported native /prompt response.")
    data = json.loads(response.body)
    if response.status >= 400 or not data.get("prompt_id"):
        raise ValueError(json.dumps(data.get("error", data), ensure_ascii=False))
    return data


async def sam31_capabilities(request):
    from .sam31 import capabilities
    return web.json_response(await asyncio.to_thread(capabilities))


async def sam31_submit(request):
    job_id = None
    try:
        from server import PromptServer
        from .sam31 import capabilities, prepare_request
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("SAM request must be a JSON object.")
        status = await asyncio.to_thread(capabilities)
        if not status["available"]:
            raise ValueError(status["reason"])
        spec = await asyncio.to_thread(prepare_request, body)
        if not spec["enabled"]:
            raise ValueError("Enable SAM3.1 before running analysis.")
        server = PromptServer.instance
        if server is None:
            raise RuntimeError("ComfyUI PromptServer is unavailable.")
        job_id = str(uuid.uuid4())
        with _LOCK:
            if len(_JOBS) >= MAX_JOBS:
                finished = next((key for key, job in _JOBS.items()
                                 if job["status"] in {"complete", "error"}), None)
                if finished is None:
                    raise ValueError("Too many pending SAM jobs.")
                _JOBS.pop(finished)
            _JOBS[job_id] = dict(status="submitting", spec=spec, prompt_id=job_id)
        prompt = {"sam31": {
            "class_type": "DirectorSAM31Job", "inputs": {"job_id": job_id},
        }}
        submitted = await _submit_prompt(server, request, {
            "prompt": prompt, "prompt_id": job_id, "extra_data": {},
        })
        with _LOCK:
            job = _JOBS[job_id]
            job["prompt_id"] = submitted["prompt_id"]
            if job["status"] == "submitting":
                job["status"] = "queued"
            result = dict(job_id=job_id, prompt_id=job["prompt_id"], status=job["status"])
        return web.json_response(result)
    except Exception as exc:
        with _LOCK:
            job = _JOBS.get(job_id)
            if job is not None and job["status"] == "submitting":
                job.update(status="error", error=str(exc))
                job.pop("spec", None)
        return web.json_response({"error": str(exc)}, status=400)


def execute_job(job_id):
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None or job["status"] not in {"submitting", "queued"}:
            raise ValueError("Unknown or already executed SAM job.")
        job["status"] = "running"
        spec = copy.deepcopy(job["spec"])
    try:
        from .sam31 import run_request
        result = run_request(spec)
        with _LOCK:
            job.update(status="complete", result=result)
            job.pop("spec", None)
        return result
    except Exception as exc:
        with _LOCK:
            job.update(status="error", error=f"{type(exc).__name__}: {exc}")
            job.pop("spec", None)
        raise


async def sam31_job_status(request):
    from server import PromptServer
    job_id = str(request.query.get("job_id") or "")
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return web.json_response({"error": "SAM job is unknown or expired."}, status=404)
        observed_status, prompt_id = job["status"], job["prompt_id"]
    if observed_status in {"queued", "running"}:
        server = PromptServer.instance
        if server is None:
            return web.json_response({"error": "PromptServer unavailable."}, status=503)
        running, queued = await asyncio.to_thread(server.prompt_queue.get_current_queue)
        present = any(item[1] == prompt_id for item in running + queued)
        with _LOCK:
            # Execution may complete or start between snapshot and this lock.
            # Never overwrite a terminal result, or act on a changed prompt ID.
            if (not present and _JOBS.get(job_id) is job
                    and job["status"] == observed_status
                    and job["prompt_id"] == prompt_id):
                job.update(status="error", error="SAM job was removed or failed before internal execution.")
                job.pop("spec", None)
    with _LOCK:
        result = {key: copy.deepcopy(value) for key, value in job.items() if key != "spec"}
    return web.json_response(result)


async def sam31_preview(request):
    try:
        from .sam31 import preview_request
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("SAM preview request must be a JSON object.")
        return web.json_response(await asyncio.to_thread(preview_request, body))
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)
