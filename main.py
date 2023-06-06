# Copyright (c) 2023 California Institute of Technology ("Caltech"). U.S.
# Government sponsorship acknowledged.
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Caltech nor its operating division, the Jet Propulsion
#   Laboratory, nor the names of its contributors may be used to endorse or
#   promote products derived from this software without specific prior written
#   permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from lwll_api.blueprints.authentication import authentication
from lwll_api.blueprints.base import base, authenticated_base
import traceback
from lwll_api.utils.logger import get_module_logger
from lwll_api.utils.config import config
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from fastapi_cache import FastAPICache
from fastapi_cache.decorator import cache
import json

import sentry_sdk


sentry_sdk.init(
    dsn=config.sentry_uri,
    traces_sample_rate=0.5,
    environment=config.lwll_mode
)


log = get_module_logger(__name__)


@cache()
async def get_cache() -> int:
    return 1


def make_app() -> FastAPI:
    app = FastAPI()

    app.include_router(authentication)
    app.include_router(base)
    app.include_router(authenticated_base)

    origins = ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def index() -> str:
        return "Index Hit"

    @app.on_event("startup")
    async def startup() -> None:
        FastAPICache.init(backend="InMemoryBackend", prefix="fastapi-cache")

    return app


app = make_app()


@app.exception_handler(StarletteHTTPException)
async def common_error_exceptions(request: Request, exception: StarletteHTTPException) -> JSONResponse:
    # Only trace back if it's a non-critical exception
    if exception.status_code not in [400, 401, 404, 300, 301]:
        track = traceback.format_exc()
        log.info(str(track))
        log.error(json.dumps({
            "detail": str(exception.detail),
            "path": request['path'],
            "status_code": exception.status_code,
            "headers": str(request.headers)
        }))
        return JSONResponse(
            status_code=500,
            content={'Status': 'Error', 'trace': str(track)},
        )
    elif exception.status_code == 404:
        return JSONResponse(
            status_code=404,
            content={'Status': 'Error', 'message': 'Route not found'},
        )
    if isinstance(exception.detail, str):
        return JSONResponse({"Error": str(exception.detail)}, status_code=exception.status_code)
    else:
        return JSONResponse({*exception.detail}, status_code=exception.status_code)
