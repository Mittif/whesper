from __future__ import annotations

import json
from dataclasses import asdict

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from whesper.api.application import (
    AgentApplication,
    SessionAlreadyExists,
    SessionBusy,
    SessionNotFound,
)
from whesper.api.events import (
    EVENT_DONE,
    EVENT_ERROR,
    chat_message_to_dict,
    decision_to_dict,
)
from whesper.chat import GenerationInterrupted
from whesper.client import ProviderError
from whesper.config import ConfigError
from whesper.server.auth import get_application, require_auth
from whesper.server.schemas import (
    CapabilitiesResponse,
    ChatMessageModel,
    ErrorBody,
    HealthResponse,
    MessageListResponse,
    OkResponse,
    SessionCreateRequest,
    SessionListResponse,
    SessionSummaryResponse,
    TurnRequest,
    TurnResponse,
)
from whesper.tools import ToolExecutionError


def create_app(application: AgentApplication) -> FastAPI:
    """Build a FastAPI app bound to the given AgentApplication.

    The application instance owns long-lived state (config + stores + chat
    service); FastAPI only dispatches requests to it.
    """

    app = FastAPI(
        title="Whesper Agent API",
        version=application.version,
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url=None,
    )
    app.state.application = application

    _configure_cors(app, application)

    @app.get(
        "/v1/health",
        response_model=HealthResponse,
        tags=["meta"],
        summary="Liveness probe; no auth required.",
    )
    def health() -> HealthResponse:
        return HealthResponse(ok=True, version=application.version)

    @app.get(
        "/v1/capabilities",
        response_model=CapabilitiesResponse,
        tags=["meta"],
        summary="Describe what this backend supports.",
        dependencies=[Depends(require_auth)],
    )
    def capabilities(
        app_ref: AgentApplication = Depends(get_application),
    ) -> CapabilitiesResponse:
        payload = app_ref.capabilities()
        return CapabilitiesResponse(**payload)  # type: ignore[arg-type]

    @app.get(
        "/v1/sessions",
        response_model=SessionListResponse,
        tags=["sessions"],
        dependencies=[Depends(require_auth)],
    )
    def list_sessions(
        app_ref: AgentApplication = Depends(get_application),
    ) -> SessionListResponse:
        items = [
            SessionSummaryResponse(**asdict(summary))
            for summary in app_ref.list_sessions()
        ]
        return SessionListResponse(items=items, next_before=None)

    @app.post(
        "/v1/sessions",
        response_model=SessionSummaryResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["sessions"],
        dependencies=[Depends(require_auth)],
        responses={
            409: {"model": ErrorBody, "description": "Session already exists"},
            400: {"model": ErrorBody, "description": "Invalid session_id"},
        },
    )
    def create_session(
        body: SessionCreateRequest,
        app_ref: AgentApplication = Depends(get_application),
    ) -> SessionSummaryResponse:
        try:
            session = app_ref.create_session(body.session_id)
        except SessionAlreadyExists as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "CONFLICT",
                    "message": f"Session already exists: {exc}",
                },
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "VALIDATION_ERROR", "message": str(exc)},
            )
        return SessionSummaryResponse(**asdict(app_ref.summarize(session)))

    @app.get(
        "/v1/sessions/{session_id}",
        response_model=SessionSummaryResponse,
        tags=["sessions"],
        dependencies=[Depends(require_auth)],
        responses={404: {"model": ErrorBody, "description": "Session not found"}},
    )
    def get_session(
        session_id: str,
        app_ref: AgentApplication = Depends(get_application),
    ) -> SessionSummaryResponse:
        try:
            session = app_ref.get_session(session_id)
        except SessionNotFound:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "NOT_FOUND",
                    "message": f"Session not found: {session_id}",
                },
            )
        return SessionSummaryResponse(**asdict(app_ref.summarize(session)))

    @app.delete(
        "/v1/sessions/{session_id}",
        response_model=OkResponse,
        tags=["sessions"],
        dependencies=[Depends(require_auth)],
        responses={404: {"model": ErrorBody, "description": "Session not found"}},
    )
    def delete_session(
        session_id: str,
        app_ref: AgentApplication = Depends(get_application),
    ) -> OkResponse:
        deleted = app_ref.delete_session(session_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "NOT_FOUND",
                    "message": f"Session not found: {session_id}",
                },
            )
        return OkResponse(ok=True)

    @app.get(
        "/v1/sessions/{session_id}/messages",
        response_model=MessageListResponse,
        tags=["sessions"],
        dependencies=[Depends(require_auth)],
        responses={
            404: {"model": ErrorBody, "description": "Session not found"},
            400: {"model": ErrorBody, "description": "Invalid pagination params"},
        },
    )
    def get_messages(
        session_id: str,
        limit: int = 50,
        before: str | None = None,
        app_ref: AgentApplication = Depends(get_application),
    ) -> MessageListResponse:
        return _list_messages(app_ref, session_id, limit, before, transcript=False)

    @app.get(
        "/v1/sessions/{session_id}/transcript",
        response_model=MessageListResponse,
        tags=["sessions"],
        dependencies=[Depends(require_auth)],
        responses={
            404: {"model": ErrorBody, "description": "Session not found"},
            400: {"model": ErrorBody, "description": "Invalid pagination params"},
        },
    )
    def get_transcript(
        session_id: str,
        limit: int = 50,
        before: str | None = None,
        app_ref: AgentApplication = Depends(get_application),
    ) -> MessageListResponse:
        return _list_messages(app_ref, session_id, limit, before, transcript=True)

    @app.post(
        "/v1/sessions/{session_id}/turn",
        response_model=TurnResponse,
        tags=["turn"],
        dependencies=[Depends(require_auth)],
        responses={
            404: {"model": ErrorBody, "description": "Session not found"},
            409: {"model": ErrorBody, "description": "Session busy"},
            502: {"model": ErrorBody, "description": "Provider failure"},
            500: {
                "model": ErrorBody,
                "description": "Tool execution failure or server configuration error",
            },
        },
    )
    async def send_turn(
        session_id: str,
        body: TurnRequest,
        app_ref: AgentApplication = Depends(get_application),
    ) -> TurnResponse:
        try:
            result = await app_ref.send_turn(
                session_id, text=body.text, mode=body.mode,
            )
        except SessionNotFound:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "NOT_FOUND",
                    "message": f"Session not found: {session_id}",
                },
            )
        except SessionBusy:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "CONFLICT",
                    "message": f"Session is busy: {session_id}",
                },
            )
        except GenerationInterrupted as exc:
            raise HTTPException(
                status_code=499,
                detail={
                    "code": "GENERATION_INTERRUPTED",
                    "message": str(exc) or "Generation interrupted.",
                },
            )
        except ProviderError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={"code": "PROVIDER_ERROR", "message": str(exc)},
            )
        except ToolExecutionError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"code": "TOOL_EXECUTION_ERROR", "message": str(exc)},
            )
        except ConfigError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"code": "CONFIG_ERROR", "message": str(exc)},
            )
        payload = {
            "decision": decision_to_dict(result.decision),
            "assistant_message": chat_message_to_dict(result.assistant_message),
            "ask_user": (
                None
                if result.ask_user is None
                else asdict(result.ask_user)
            ),
        }
        return TurnResponse(**payload)  # type: ignore[arg-type]

    @app.post(
        "/v1/sessions/{session_id}/turn:stream",
        tags=["turn"],
        dependencies=[Depends(require_auth)],
        responses={
            200: {
                "description": "SSE stream of route/step/chunk/ask_user/final/error/done events",
                "content": {"text/event-stream": {}},
            },
            404: {"model": ErrorBody, "description": "Session not found"},
            409: {"model": ErrorBody, "description": "Session busy"},
        },
    )
    async def send_turn_stream(
        session_id: str,
        body: TurnRequest,
        request: Request,
        app_ref: AgentApplication = Depends(get_application),
    ) -> StreamingResponse:
        # Raise synchronously (before starting the stream) for preconditions
        # so clients see a proper HTTP error code, not a half-baked stream.
        try:
            app_ref.get_session(session_id)
        except SessionNotFound:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "NOT_FOUND",
                    "message": f"Session not found: {session_id}",
                },
            )
        if app_ref.is_busy(session_id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "CONFLICT",
                    "message": f"Session is busy: {session_id}",
                },
            )

        async def event_stream():
            try:
                async for event_name, data in app_ref.send_turn_events(
                    session_id, text=body.text, mode=body.mode,
                ):
                    if await request.is_disconnected():
                        break
                    yield _format_sse(event_name, data)
            except SessionBusy:
                yield _format_sse(
                    EVENT_ERROR,
                    {"code": "CONFLICT", "message": f"Session is busy: {session_id}"},
                )
                yield _format_sse(EVENT_DONE, {})
            except BaseException as exc:  # noqa: BLE001
                yield _format_sse(
                    EVENT_ERROR,
                    {"code": "INTERNAL_ERROR", "message": str(exc) or exc.__class__.__name__},
                )
                yield _format_sse(EVENT_DONE, {})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail:
            body = detail
        else:
            body = {"code": "ERROR", "message": str(detail)}
        return JSONResponse(status_code=exc.status_code, content=body)

    return app


def _list_messages(
    app_ref: AgentApplication,
    session_id: str,
    limit: int,
    before: str | None,
    *,
    transcript: bool,
) -> MessageListResponse:
    try:
        items, next_before = app_ref.list_messages(
            session_id, limit=limit, before=before, transcript=transcript,
        )
    except SessionNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "NOT_FOUND",
                "message": f"Session not found: {session_id}",
            },
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "VALIDATION_ERROR", "message": str(exc)},
        )
    return MessageListResponse(
        items=[ChatMessageModel(**asdict(m)) for m in items],
        next_before=next_before,
    )


def _format_sse(event_name: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event_name}\ndata: {payload}\n\n"


def _configure_cors(app: FastAPI, application: AgentApplication) -> None:
    origins = list(application.config.server.cors_origins)
    if not origins:
        return
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
