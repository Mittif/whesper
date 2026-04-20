from __future__ import annotations

from fastapi import Depends, Header, HTTPException, Request, status

from whesper.api.application import AgentApplication


def get_application(request: Request) -> AgentApplication:
    application: AgentApplication | None = getattr(
        request.app.state, "application", None
    )
    if application is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "CONFIG_ERROR",
                "message": "AgentApplication not attached to app.state.",
            },
        )
    return application


def require_auth(
    request: Request,
    authorization: str | None = Header(default=None),
    application: AgentApplication = Depends(get_application),
) -> None:
    server = application.config.server
    if not server.require_auth:
        return

    expected = server.resolved_api_token()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "CONFIG_ERROR",
                "message": (
                    "Server requires auth but no api_token is configured. "
                    "Set [server].api_token or [server].api_token_env."
                ),
            },
        )

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "UNAUTHORIZED",
                "message": "Missing or malformed Authorization header.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization[len("Bearer ") :].strip()
    if token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "UNAUTHORIZED",
                "message": "Invalid bearer token.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )
