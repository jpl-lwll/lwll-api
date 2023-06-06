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

import firebase_admin
from firebase_admin import credentials, firestore
import os
from lwll_api.utils.logger import get_module_logger
import sys


class FB_Auth(object):
    def __init__(self) -> None:
        # Use a service account
        self.log = get_module_logger(__name__)
        if "FB_CREDS" in os.environ:
            creds_path = os.environ.get("FB_CREDS")
            self.log.info(f"loading creds from env var: {creds_path}")
        else:
            creds_path = f'{os.path.dirname(__file__)}/service_accounts/firebase_creds.json'  #
            self.log.info(f"env var not found, loading from path {creds_path}")
        try:
            cred = credentials.Certificate(creds_path)
            firebase_admin.initialize_app(cred)
        except Exception as e:
            self.log.error(f"Error loading credentials file: {e}")
            sys.exit()

        self.db_firestore = firestore.client()
        self.db_realtime = ""
        pass


firebase = None
fb_store = None
fb_realtime = None
if not os.environ.get("LWLL_LOCAL_DB_FOLDER", False):
    firebase = FB_Auth()
    fb_store = firebase.db_firestore
    fb_realtime = firebase.db_realtime
