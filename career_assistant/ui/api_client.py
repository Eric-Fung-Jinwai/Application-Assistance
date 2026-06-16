"""HTTP client the Streamlit UI uses to talk to the FastAPI backend (Phase 13).

The UI never imports the pipeline directly — it goes through the API, so the browser flow
exercises the same contract as any other client. The client is constructor-injectable: production
points it at a base URL; tests pass an ``httpx.Client`` wired to the ASGI app in-process (no
servers, no network), so the whole upload→…→export path is testable deterministically.
"""

from __future__ import annotations

import os

import httpx

DEFAULT_BASE_URL = os.environ.get("CAREER_ASSISTANT_API_URL", "http://localhost:8000")
DEFAULT_TIMEOUT = 180.0  # parsing + embedding + LLM calls can be slow


class ApiError(Exception):
    """A non-2xx response from the API, carrying the status code and server detail."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API error {status_code}: {detail}")


class ApiClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        client: httpx.Client | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    # --- pipeline endpoints ------------------------------------------------------

    def parse_resume(self, *, filename: str, content: bytes) -> dict:
        resp = self._client.post(
            "/parse_resume",
            files={"file": (filename, content, "application/octet-stream")},
        )
        return self._json(resp)

    def analyze_jd(self, text: str) -> dict:
        return self._json(self._client.post("/analyze_jd", json={"text": text}))

    def fit_score(self, resume_version_id: str, jd_id: str) -> dict:
        return self._json(
            self._client.post("/fit_score", json=self._pair(resume_version_id, jd_id))
        )

    def recommendation(self, resume_version_id: str, jd_id: str) -> dict:
        return self._json(
            self._client.post("/recommendation", json=self._pair(resume_version_id, jd_id))
        )

    def tailor(self, resume_version_id: str, jd_id: str) -> list[dict]:
        body = self._json(self._client.post("/tailor", json=self._pair(resume_version_id, jd_id)))
        return body["suggestions"]

    def review(
        self,
        *,
        suggestion_id: str,
        action: str,
        base_version_id: str | None = None,
        edit: str | None = None,
        instruction: str | None = None,
    ) -> dict:
        payload = {"suggestion_id": suggestion_id, "action": action}
        if base_version_id is not None:
            payload["base_version_id"] = base_version_id
        if edit is not None:
            payload["edit"] = edit
        if instruction is not None:
            payload["instruction"] = instruction
        return self._json(self._client.post("/review", json=payload))

    def budget(self, *, tailored: int = 8, jds: int = 100) -> dict:
        return self._json(self._client.get("/budget", params={"tailored": tailored, "jds": jds}))

    def accept_all(self, suggestion_ids: list[str], base_version_id: str | None = None) -> dict:
        payload: dict = {"suggestion_ids": suggestion_ids}
        if base_version_id is not None:
            payload["base_version_id"] = base_version_id
        return self._json(self._client.post("/accept_all", json=payload))

    def generate_html(self, resume_version_id: str, template: str, name: str | None = None) -> str:
        resp = self._client.post(
            "/generate_html",
            json={"resume_version_id": resume_version_id, "template": template, "name": name},
        )
        self._raise_for_status(resp)
        return resp.text

    def generate_pdf(self, resume_version_id: str, template: str, name: str | None = None) -> bytes:
        resp = self._client.post(
            "/generate_pdf",
            json={"resume_version_id": resume_version_id, "template": template, "name": name},
        )
        self._raise_for_status(resp)
        return resp.content

    # --- helpers -----------------------------------------------------------------

    @staticmethod
    def _pair(resume_version_id: str, jd_id: str) -> dict:
        return {"resume_version_id": resume_version_id, "jd_id": jd_id}

    def _json(self, resp: httpx.Response) -> dict:
        self._raise_for_status(resp)
        return resp.json()

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise ApiError(resp.status_code, str(detail))
