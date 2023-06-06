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

from fastapi import HTTPException, Request
from lwll_api.utils.config import config
from typing import Any
from lwll_api.classes.authentication import authenticate_secret
from lwll_api.utils.logger import get_module_logger
from lwll_api.utils.database import auth
from sentry_sdk import set_user

log = get_module_logger(__name__)


def valid_govteam_secret(secret: str) -> bool:
    return config.govteam_secret == secret and config.govteam_secret != ''


def verify_user_secret(request: Request) -> Any:
    log.debug('Authenticating User Secret')
    user_secret = request.headers.get('user_secret', "")
    valid, _ = authenticate_secret(user_secret)
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid User Secret")
        return {"Error": "Invalid User Secret"}

    team = auth.get_user(user_secret)
    set_user({"team": team.display_name})

    log.debug('Authenticated User Secret')
    return user_secret


def verify_govteam_secret(request: Request) -> Any:
    govteam_secret = request.headers.get('govteam_secret', "")
    if config.lwll_mode == 'EVAL':
        if not valid_govteam_secret(govteam_secret):
            raise HTTPException(
                status_code=401, detail="Running in EVAL model without providing the valid `govteam_secret`")
            return {"Error": "Running in EVAL model without providing the valid `govteam_secret`"}
        log.debug('Authenticated Govteam Secret')
    return govteam_secret
