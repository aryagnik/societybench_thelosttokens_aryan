# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import ssl
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError

TERMINAL_RUN_STATUSES = {
    "SUCCEEDED",
    "FAILED",
    "ABORTED",
    "TIMED-OUT",
}


class ApifyAPIError(RuntimeError):
    def __init__(
        self,
        *,
        status_code: int,
        message: str,
        error_type: Optional[str] = None,
        payload: Optional[Any] = None,
    ) -> None:
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.payload = payload

        parts = [f"status={status_code}"]
        if error_type:
            parts.append(f"type={error_type}")
        parts.append(f"message={message}")
        super().__init__(", ".join(parts))


class ApifyClient:
    def __init__(self, token: str, base_url: str = "https://api.apify.com", timeout: int = 60) -> None:
        if not token:
            raise ValueError("APIFY token is required")
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.ssl_context = self._build_ssl_context()

    def whoami(self) -> Dict[str, Any]:
        payload = self._request("GET", "/v2/users/me")
        return self._unwrap_data(payload)

    def run_actor(
        self,
        actor_id: str,
        *,
        input_data: Optional[Dict[str, Any]] = None,
        memory_mbytes: Optional[int] = None,
        timeout_secs: Optional[int] = None,
    ) -> Dict[str, Any]:
        query: Dict[str, Any] = {}
        if memory_mbytes is not None:
            query["memory"] = memory_mbytes
        if timeout_secs is not None:
            query["timeout"] = timeout_secs

        actor_key = self._normalize_actor_id(actor_id)
        payload = self._request(
            "POST",
            f"/v2/acts/{actor_key}/runs",
            query=query,
            json_body=input_data,
        )
        return self._unwrap_data(payload)

    def get_run(self, run_id: str, wait_for_finish: int = 0) -> Dict[str, Any]:
        payload = self._request(
            "GET",
            f"/v2/actor-runs/{urllib.parse.quote(run_id, safe='')}",
            query={"waitForFinish": max(0, min(wait_for_finish, 60))},
        )
        return self._unwrap_data(payload)

    def wait_for_run_finished(
        self,
        run_id: str,
        *,
        timeout_secs: int = 600,
        poll_interval_secs: int = 3,
    ) -> Dict[str, Any]:
        deadline = time.time() + timeout_secs
        run = self.get_run(run_id)

        while run.get("status") not in TERMINAL_RUN_STATUSES:
            remaining = int(deadline - time.time())
            if remaining <= 0:
                raise TimeoutError(f"Run {run_id} did not finish within {timeout_secs}s")

            wait_window = max(1, min(60, remaining, poll_interval_secs))
            run = self.get_run(run_id, wait_for_finish=wait_window)
            if run.get("status") in TERMINAL_RUN_STATUSES:
                break

            time.sleep(max(0, min(poll_interval_secs, int(deadline - time.time()))))

        return run

    def get_dataset_items(
        self,
        dataset_id: str,
        *,
        limit: int = 20,
        offset: int = 0,
        clean: bool = True,
        desc: bool = False,
    ) -> Any:
        query = {
            "limit": max(1, limit),
            "offset": max(0, offset),
            "clean": int(bool(clean)),
            "desc": int(bool(desc)),
            "format": "json",
        }
        return self._request(
            "GET",
            f"/v2/datasets/{urllib.parse.quote(dataset_id, safe='')}/items",
            query=query,
        )

    def run_actor_sync_get_dataset_items(
        self,
        actor_id: str,
        *,
        input_data: Optional[Dict[str, Any]] = None,
        timeout_secs: int = 300,
        limit: int = 20,
        clean: bool = True,
    ) -> Any:
        actor_key = self._normalize_actor_id(actor_id)
        query = {
            "timeout": max(1, timeout_secs),
            "limit": max(1, limit),
            "clean": int(bool(clean)),
            "format": "json",
        }
        return self._request(
            "POST",
            f"/v2/acts/{actor_key}/run-sync-get-dataset-items",
            query=query,
            json_body=input_data,
        )

    def run_actor_and_fetch_items(
        self,
        actor_id: str,
        *,
        input_data: Optional[Dict[str, Any]] = None,
        item_limit: int = 20,
        wait_timeout_secs: int = 600,
    ) -> Dict[str, Any]:
        run = self.run_actor(actor_id, input_data=input_data)
        run_id = run.get("id")
        if not run_id:
            raise RuntimeError(f"Missing run id in response: {run}")

        finished_run = self.wait_for_run_finished(run_id, timeout_secs=wait_timeout_secs)
        status = finished_run.get("status")
        if status != "SUCCEEDED":
            raise RuntimeError(f"Run finished with status={status}, runId={run_id}")

        dataset_id = finished_run.get("defaultDatasetId")
        if not dataset_id:
            return {"run": finished_run, "items": []}

        items = self.get_dataset_items(dataset_id, limit=item_limit)
        return {"run": finished_run, "items": items}

    # HTTP status codes that are safe to retry (transient server/rate-limit errors)
    _RETRYABLE_STATUS_CODES = {429, 502, 503, 504}

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
    ) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query, doseq=True)}"

        headers = {
            "Authorization": f"Bearer {self.token}",
        }
        body_bytes = None
        if json_body is not None:
            body_bytes = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout, context=self.ssl_context) as resp:
                    raw = resp.read().decode("utf-8")
                    return self._decode_json(raw)
            except HTTPError as e:
                if e.code in self._RETRYABLE_STATUS_CODES and attempt < max_retries:
                    e.read()  # drain response body
                    retry_after = 0
                    try:
                        retry_after = int(e.headers.get("Retry-After", "0") or "0")
                    except (ValueError, TypeError):
                        pass
                    delay = max(retry_after, 2 ** attempt)  # 1s, 2s, 4s
                    time.sleep(delay)
                    last_exc = e
                    continue
                raw = e.read().decode("utf-8", errors="replace")
                payload = self._decode_json(raw)
                error_obj = payload.get("error") if isinstance(payload, dict) else None
                message = (
                    error_obj.get("message")
                    if isinstance(error_obj, dict)
                    else f"HTTP {e.code} from Apify API"
                )
                raise ApifyAPIError(
                    status_code=e.code,
                    message=message,
                    error_type=error_obj.get("type") if isinstance(error_obj, dict) else None,
                    payload=payload,
                ) from e
            except URLError as e:
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    last_exc = e
                    continue
                raise RuntimeError(f"Network error while calling Apify API: {e}") from e

        raise RuntimeError(f"All {max_retries + 1} attempts failed") from last_exc

    @staticmethod
    def _decode_json(raw: str) -> Any:
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except ValueError:
            return {"raw": raw}

    @staticmethod
    def _unwrap_data(payload: Any) -> Any:
        if isinstance(payload, dict) and "data" in payload:
            return payload["data"]
        return payload

    @staticmethod
    def _normalize_actor_id(actor_id: str) -> str:
        actor_id = actor_id.strip()
        if not actor_id:
            raise ValueError("actor_id is required")
        if "/" in actor_id and "~" not in actor_id:
            actor_id = actor_id.replace("/", "~", 1)
        return urllib.parse.quote(actor_id, safe="~")

    @staticmethod
    def _build_ssl_context() -> ssl.SSLContext:
        """
        SSL behavior:
        - default: verify certificates
        - APIFY_SSL_NO_VERIFY=1: disable verification (debug only)
        """
        if os.getenv("APIFY_SSL_NO_VERIFY", "").strip() in {"1", "true", "TRUE"}:
            return ssl._create_unverified_context()
        return ssl.create_default_context()
