from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import subprocess
import time

from whesper.config import ShellSandboxSettings


DEFAULT_ALLOWED_COMMAND_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("pwd",),
    ("ls",),
    ("rg",),
    ("cat",),
    ("sed", "-n"),
    ("head",),
    ("tail",),
    ("wc",),
    ("git", "status"),
    ("git", "diff", "--stat"),
    ("git", "diff"),
)

_DISALLOWED_RAW_MARKERS = (
    "\n",
    "\r",
    "\x00",
    "&&",
    "||",
    ";",
    "|",
    ">",
    "<",
    "`",
    "$(",
)


class ShellSandboxError(RuntimeError):
    pass


@dataclass(slots=True)
class ShellCommandResult:
    command: str
    argv: tuple[str, ...]
    cwd: str
    exit_code: int | None
    stdout: str
    stderr: str
    ok: bool
    truncated: bool
    duration_ms: int

    def as_payload(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "command": self.command,
            "argv": list(self.argv),
            "cwd": self.cwd,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "truncated": self.truncated,
            "duration_ms": self.duration_ms,
        }


class ShellSandboxExecutor:
    def __init__(self, settings: ShellSandboxSettings, *, workspace_root: Path) -> None:
        self.settings = settings
        self.workspace_root = workspace_root.resolve()
        configured_roots = settings.allowed_roots or (".",)
        self.allowed_roots = tuple(
            self._resolve_root(root)
            for root in configured_roots
        )
        self.allowed_prefixes = tuple(
            sorted(
                settings.allowed_command_prefixes or DEFAULT_ALLOWED_COMMAND_PREFIXES,
                key=len,
                reverse=True,
            )
        )

    def describe_supported_commands(self) -> str:
        return ", ".join(" ".join(prefix) for prefix in self.allowed_prefixes)

    def execute(self, command: str, *, cwd: str | None = None) -> ShellCommandResult:
        cleaned_command = command.strip()
        if not cleaned_command:
            raise ShellSandboxError("Shell command cannot be empty.")
        for marker in _DISALLOWED_RAW_MARKERS:
            if marker in cleaned_command:
                raise ShellSandboxError(
                    "Shell sandbox does not allow pipes, redirection, command chaining, "
                    "or other shell operators."
                )

        try:
            argv = tuple(shlex.split(cleaned_command, posix=True))
        except ValueError as exc:
            raise ShellSandboxError(f"Shell command could not be parsed: {exc}") from exc

        if not argv:
            raise ShellSandboxError("Shell command cannot be empty.")

        prefix = self._match_prefix(argv)
        command_cwd = self._resolve_cwd(cwd)
        self._validate_arguments(prefix, argv, command_cwd)

        started_at = time.monotonic()
        try:
            completed = subprocess.run(
                argv,
                cwd=command_cwd,
                env=self._execution_env(command_cwd),
                capture_output=True,
                text=True,
                timeout=self.settings.timeout_seconds,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration_ms = int((time.monotonic() - started_at) * 1000)
            stdout, stdout_truncated = self._truncate_output(exc.stdout or "")
            stderr, stderr_truncated = self._truncate_output(exc.stderr or "")
            return ShellCommandResult(
                command=cleaned_command,
                argv=argv,
                cwd=str(command_cwd),
                exit_code=None,
                stdout=stdout,
                stderr=stderr or "Command timed out.",
                ok=False,
                truncated=stdout_truncated or stderr_truncated,
                duration_ms=duration_ms,
            )

        duration_ms = int((time.monotonic() - started_at) * 1000)
        stdout, stdout_truncated = self._truncate_output(completed.stdout)
        stderr, stderr_truncated = self._truncate_output(completed.stderr)
        return ShellCommandResult(
            command=cleaned_command,
            argv=argv,
            cwd=str(command_cwd),
            exit_code=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            ok=completed.returncode == 0,
            truncated=stdout_truncated or stderr_truncated,
            duration_ms=duration_ms,
        )

    def _match_prefix(self, argv: tuple[str, ...]) -> tuple[str, ...]:
        for prefix in self.allowed_prefixes:
            if len(argv) >= len(prefix) and argv[: len(prefix)] == prefix:
                return prefix
        supported = self.describe_supported_commands()
        raise ShellSandboxError(
            "Command prefix is not allowed by the sandbox. "
            f"Supported prefixes: {supported}."
        )

    def _resolve_root(self, root: str) -> Path:
        candidate = Path(root).expanduser()
        if not candidate.is_absolute():
            candidate = (self.workspace_root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if not self._is_within(candidate, self.workspace_root):
            raise ShellSandboxError(
                "shell_sandbox.allowed_roots must stay within the workspace root."
            )
        return candidate

    def _resolve_cwd(self, cwd: str | None) -> Path:
        if cwd is None or not cwd.strip():
            candidate = self.workspace_root
        else:
            raw = Path(cwd.strip()).expanduser()
            candidate = raw.resolve() if raw.is_absolute() else (self.workspace_root / raw).resolve()
        if not candidate.exists() or not candidate.is_dir():
            raise ShellSandboxError("Shell sandbox cwd must point to an existing directory.")
        if not any(self._is_within(candidate, root) for root in self.allowed_roots):
            raise ShellSandboxError("Shell sandbox cwd is outside the allowed roots.")
        return candidate

    def _execution_env(self, cwd: Path) -> dict[str, str]:
        return {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(self.workspace_root),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "PWD": str(cwd),
            "PAGER": "cat",
            "GIT_PAGER": "cat",
            "GIT_CONFIG_NOSYSTEM": "1",
            "NO_COLOR": "1",
            "CLICOLOR": "0",
        }

    def _validate_arguments(
        self,
        prefix: tuple[str, ...],
        argv: tuple[str, ...],
        cwd: Path,
    ) -> None:
        remainder = argv[len(prefix) :]
        if prefix == ("pwd",):
            if remainder:
                raise ShellSandboxError("pwd does not accept additional arguments.")
            return
        if prefix == ("ls",):
            self._validate_path_args(remainder, cwd)
            return
        if prefix == ("cat",):
            self._validate_required_paths(remainder, cwd, command_name="cat")
            return
        if prefix == ("sed", "-n"):
            if not remainder:
                raise ShellSandboxError("sed -n requires a script and at least one file.")
            script, *paths = remainder
            if not script.strip():
                raise ShellSandboxError("sed -n requires a non-empty script.")
            self._validate_required_paths(paths, cwd, command_name="sed -n")
            return
        if prefix in {("head",), ("tail",), ("wc",)}:
            self._validate_paths_with_simple_options(remainder, cwd, command_name=" ".join(prefix))
            return
        if prefix == ("rg",):
            self._validate_rg_args(remainder, cwd)
            return
        if prefix == ("git", "status"):
            if remainder:
                raise ShellSandboxError("git status is limited to the base command in the sandbox.")
            return
        if prefix in {("git", "diff"), ("git", "diff", "--stat")}:
            self._validate_git_diff_args(remainder, cwd)
            return
        if remainder:
            raise ShellSandboxError(
                "Custom sandbox command prefixes currently allow only the exact command without extra arguments."
            )

    def _validate_path_args(self, args: list[str] | tuple[str, ...], cwd: Path) -> None:
        for token in args:
            if token.startswith("-"):
                raise ShellSandboxError("This sandbox command does not allow additional flags.")
            self._ensure_allowed_path(token, cwd)

    def _validate_required_paths(
        self,
        args: list[str] | tuple[str, ...],
        cwd: Path,
        *,
        command_name: str,
    ) -> None:
        if not args:
            raise ShellSandboxError(f"{command_name} requires at least one file path.")
        self._validate_path_args(args, cwd)

    def _validate_paths_with_simple_options(
        self,
        args: list[str] | tuple[str, ...],
        cwd: Path,
        *,
        command_name: str,
    ) -> None:
        paths: list[str] = []
        index = 0
        while index < len(args):
            token = args[index]
            if token in {"-n", "-c"}:
                if index + 1 >= len(args):
                    raise ShellSandboxError(f"{command_name} requires a value after {token}.")
                index += 2
                continue
            if token.startswith("-") and token[1:].isdigit():
                index += 1
                continue
            if token.startswith("-"):
                raise ShellSandboxError(f"{command_name} only allows simple count flags in the sandbox.")
            paths.append(token)
            index += 1
        self._validate_required_paths(paths, cwd, command_name=command_name)

    def _validate_rg_args(self, args: list[str] | tuple[str, ...], cwd: Path) -> None:
        if not args:
            raise ShellSandboxError("rg requires a search pattern.")
        pattern, *paths = args
        if not pattern.strip() or pattern.startswith("-"):
            raise ShellSandboxError(
                "rg in the sandbox expects the first argument to be a plain search pattern."
            )
        self._validate_path_args(paths, cwd)

    def _validate_git_diff_args(
        self, args: list[str] | tuple[str, ...], cwd: Path
    ) -> None:
        if not args:
            return
        if args[0] == "--":
            self._validate_required_paths(args[1:], cwd, command_name="git diff")
            return
        for token in args:
            if token.startswith("-"):
                raise ShellSandboxError(
                    "git diff in the sandbox only allows optional path arguments."
                )
        self._validate_path_args(args, cwd)

    def _ensure_allowed_path(self, token: str, cwd: Path) -> None:
        candidate = Path(token).expanduser()
        resolved = candidate.resolve() if candidate.is_absolute() else (cwd / candidate).resolve()
        if not any(self._is_within(resolved, root) for root in self.allowed_roots):
            raise ShellSandboxError(
                f"Path '{token}' is outside the allowed shell sandbox roots."
            )

    def _truncate_output(self, text: str) -> tuple[str, bool]:
        limit = max(self.settings.max_output_chars, 0)
        if len(text) <= limit:
            return text, False
        if limit == 0:
            return "", True
        notice = "\n...[truncated]"
        trimmed = text[: max(limit - len(notice), 0)] + notice
        return trimmed, True

    @staticmethod
    def _is_within(candidate: Path, root: Path) -> bool:
        return candidate == root or root in candidate.parents
