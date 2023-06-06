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

import time
from typing import TypeVar, Any
from lwll_api.utils.logger import get_module_logger
from lwll_api.utils.config import config
from lwll_api.utils.database import database

C = TypeVar("C")


class BaseFBObject(object):
    def __init__(self, uid: str, base_ref: str, new: bool = False, load: bool = True, data: dict = {}, overridden_create: bool = False) -> None:
        self.uid = uid
        if base_ref != 'DatasetMetadata':
            base_ref = config.db_prefix + base_ref
        # Ignoring type below line for now until we get the Firebase account provisioned
        self.base_ref = database.collection(base_ref)
        self.data = data
        self.retry_number = 0
        self.log = get_module_logger(__name__)
        if new:
            self.ref = self.base_ref.document()
            if not overridden_create:
                self.uid = self.ref.id
            self.data["date_created"] = int(time.time()) * 1000
        else:
            if load:
                self.ref = self.base_ref.document(self.uid)
                self.load_from_db()

        if not isinstance(self.data, dict):
            raise Exception(f"BaseFBObject (uid: {uid}) data is not a dict")
        self.date_created = self.data.get("date_created")

    def load_from_db(self) -> None:
        try:
            self.data = self.ref.get().to_dict()
        except Exception as e:
            self.log.exception(e)
            time.sleep(1)
            self.retry_number += 1
            if self.retry_number < 8:
                self.load_from_db()
        return

    @classmethod
    def create(cls, data: dict = {}, overridden_create: bool = False) -> C:
        # Meant to call subclass constructor that then in turn calls this base class constructor
        u = cls("", new=True, data=data, overridden_create=overridden_create)  # type: ignore
        u._set_ref_to_db(obj=u)
        return u  # type: ignore

    def save(self) -> None:
        try:
            # Subclass will need to implement a `to_dict()` method
            data = self.to_dict()  # type: ignore
            if 'date_last_interacted' in data.keys():
                data['date_last_interacted'] = int(time.time() * 1000)
            self.ref.update(data)
        except Exception as e:
            self.log.exception(e)
            time.sleep(1)
            self.retry_number += 1
            if self.retry_number < 8:
                self.save()
        return

    def _set_ref_to_db(self, obj: C) -> None:
        obj.ref.set(obj.to_dict())  # type: ignore
        return

    def delete(self) -> None:
        self.base_ref.document(self.uid).delete()
        return

    def update(self, data: dict) -> None:
        for key, val in data.items():
            if key != "date_created":
                setattr(self, key, val)
        self.save()

    def unwrap_query_data(self, data: dict) -> dict:
        data_dict = {}
        for d in data:
            data_dict[d.id] = d.to_dict()
        return data_dict

    def clean_empty(self, d: Any) -> Any:
        if not isinstance(d, (dict, list)):
            return d
        if isinstance(d, list):
            return [v for v in (self.clean_empty(v) for v in d) if (v or v == "")]
        return {k: v for k, v in ((k, self.clean_empty(v)) for k, v in d.items()) if ((v or v == "") or v is False)}
