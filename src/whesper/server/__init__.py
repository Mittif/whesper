"""HTTP/SSE transport for the Whesper Agent.

All code in this package depends on FastAPI, which is an *optional* install
extra (`pip install -e ".[server]"`). Keep heavy imports confined to this
package so plain CLI users are not forced to install FastAPI.
"""

from whesper.server.http import create_app

__all__ = ["create_app"]
