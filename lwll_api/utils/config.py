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

import os
import logging


class Config:

    def __init__(self) -> None:
        self.lwll_mode = self._get_mode()
        self.log_level = self._get_log_level(mode=self.lwll_mode)
        self.db_prefix = self._get_db_prefix(mode=self.lwll_mode)
        self.auth_cache_timeout = 120
        self.govteam_secret = self._get_govteam_secret()
        self.sentry_uri = self._get_sentry_uri()

    def _get_mode(self) -> str:
        logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                            datefmt="%m-%d:%H:%M:%S")
        logger = logging.getLogger("config_logger")
        logger.setLevel(logging.INFO)
        mode = os.environ.get('LWLL_SESSION_MODE')
        if mode is None:
            mode = 'DEV'
        logger.info(f"LWLL_SESSION_MODE set to {mode}")
        if mode not in ['DEV', 'SANDBOX', 'EVAL']:
            raise Exception('Invalid LwLL Session Mode Identified')
        return mode

    def _get_log_level(self, mode: str) -> str:
        if mode == 'DEV':
            return 'DEBUG'
        else:
            return 'INFO'

    def _get_db_prefix(self, mode: str) -> str:
        if mode == 'EVAL':
            return 'prod_'
        elif mode == 'SANDBOX':
            return 'staging_'
        else:
            return 'dev_'

    def _get_govteam_secret(self) -> str:
        logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                            datefmt="%m-%d:%H:%M:%S")
        logger = logging.getLogger("config_logger")
        logger.setLevel(logging.INFO)
        logger.info("SERVERTYPE set to %s" % os.environ.get('SERVERTYPE'))
        logger.info("GOVTEAM_SECRET set to %s" % os.environ.get('GOVTEAM_SECRET'))
        logger.info("AWS_REGION_NAME set to %s" % os.environ.get('AWS_REGION_NAME'))
        return os.environ.get('GOVTEAM_SECRET', '')

    def _get_sentry_uri(self) -> str:
        return os.environ.get('SENTRY_URI', '')


config = Config()
