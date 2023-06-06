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

from typing import Any, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from lwll_api.classes.authentication import authenticate_secret
from lwll_api.classes.models import Session
from lwll_api.utils.config import config
from lwll_api.utils.decorators import verify_user_secret
from lwll_api.utils.logger import get_module_logger

authentication = APIRouter(
    prefix="/auth",
    tags=["authentication"],
)

log = get_module_logger(__name__)


@authentication.get("/")
def index() -> Any:
    return {"Status": "Success"}


class Auth(BaseModel):
    task_id: str
    data_type: Optional[str] = "full"
    session_name: Optional[str] = None
    zsl: Optional[bool] = False


@authentication.post("/create_session", dependencies=[Depends(verify_user_secret)])
def auth_create_session(data: Auth, request: Request) -> Any:
    """Unique session token ID for given task. data_type should either be `full` or `sample`.

        .. :quickref: Authentication; Get session token
        :reqheader user_secret: secret

        .. :quickref: Base; POST for session token

        PARAMS
        ------
        task_id: str
            The task ID

        data_type: Optional[str]
            either 'full' or 'sample', defaults to 'full'

        session_name: Optional[str]
            customize name

        **Example response**:

        .. sourcecode:: http

          HTTP/1.1 200 OK
          Vary: Accept
          Content-Type: application/json

          {
            'session_token': 'RSsFBzwLFiC0QjpAQqWW'
          }

        """

    task_id: str = data.task_id
    data_type: Optional[str] = data.data_type
    session_name: Optional[str] = data.session_name
    zsl: Optional[bool] = data.zsl

    log.info(f"Creating session token for\nProblem: {task_id}")
    if data_type not in ['full', 'sample']:
        raise Exception(
            f"Invalid `data_type` {data_type}: "
            f"argument needs to be one of `['full', 'sample']`"
        )
    if data_type == 'sample' and config.lwll_mode == 'EVAL':
        raise Exception(
            "`sample` data types are only valid in SANDBOX mode. "
            "Samples are not available in EVAL mode."
        )
    log.info(f"Using `{data_type}` dataset version")

    # Validate user_secret
    # We are already doing this via the decorator for consistency, but we will end up using the
    # memoized result such that we have access to the team name for `Session` data creation
    user_secret = request.headers.get('user_secret', '')
    authenticated, team = authenticate_secret(user_secret)
    if not authenticated:
        raise Exception("Invalid User Secret")

    # create session in database
    new_session = {
        'task_id': task_id,
        'user_name': team,
        'using_sample_datasets': data_type == 'sample',
        'ZSL': zsl
    }

    if session_name:
        new_session['session_name'] = session_name

    session: Session = Session.create(new_session)

    return {"session_token": session.uid}
