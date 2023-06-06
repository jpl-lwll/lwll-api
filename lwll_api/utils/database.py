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

from lwll_api.classes.fb_auth import fb_store
from lwll_api.utils.config import config
import os
import json
import uuid
from typing import Any, List
from firebase_admin import auth as firebase_admin_auth

LOCAL_JSON_FOLDER = os.environ.get("LWLL_LOCAL_DB_FOLDER", None)


class Team():
    display_name: str

    def __init__(self, team_id: str):
        self.team_id = team_id
        with open(f"{LOCAL_JSON_FOLDER}/users.json", "r") as user_file:
            users = json.load(user_file)
            team_json = users.get(team_id, {})
            self.display_name = team_json.get("display_name", "")


class ObjectAbstract():
    def __init__(self, path: str, id: str = None):
        self.path = path
        self.id = id

    def to_dict(self) -> Any:
        with open(self.path, "r") as file:
            docs = json.load(file)
            doc = docs.get(self.id, None)
            return doc


class LocalDocument():
    def __init__(self, collection_name: str, id: str = None):
        self.collection_name = collection_name
        self.id = id
        self.json_folder = LOCAL_JSON_FOLDER
        self.path = f"{self.json_folder}/{collection_name}.json"
        if not os.path.exists(self.path):
            with open(self.path, "w") as file:
                json.dump({}, file)

        if id is None:
            self.id = str(uuid.uuid4())

    def get(self) -> ObjectAbstract:
        return ObjectAbstract(self.path, self.id)

    def update(self, data: dict) -> None:
        with open(self.path, "r") as file:
            docs = json.load(file)
            with open(self.path, "w") as file:
                docs[self.id] = data
                json.dump(docs, file)

    def set(self, data: dict) -> None:
        self.update(data)

    def delete(self) -> None:
        with open(self.path, "r") as file:
            docs = json.load(file)
            with open(self.path, "w") as file:
                del docs[self.id]
                json.dump(docs, file)


class LocalCollection():
    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    def document(self, doc_id: str = None) -> LocalDocument:
        return LocalDocument(self.collection_name, doc_id)


class Database():
    def __init__(self, json_folder: str = None):
        self.local = json_folder is not None
        self.json_folder = json_folder

    def get_tasks(self) -> List[str]:
        task_ids = []
        if self.local:
            path = f"{self.json_folder}/{config.db_prefix}Task.json"
            if not os.path.exists(path):
                with open(path, "w") as file:
                    json.dump({}, file)
            with open(path, "r") as file:
                tasks = json.load(file)
                task_ids = [id for id, task in tasks.items()]
        elif fb_store:
            tasks_gen = fb_store.collection(f"{config.db_prefix}Task").stream()
            task_ids = [t.id for t in tasks_gen]

        return task_ids

    def get_active_sessions(self, team: Team) -> List[str]:
        docs = []
        if self.local:
            path = f"{self.json_folder}/{config.db_prefix}Session.json"
            if not os.path.exists(path):
                with open(path, "w") as file:
                    json.dump({}, file)
            with open(path, "r") as file:
                sessions = json.load(file)
                docs = [session["id"] for session in sessions if session["user_name"] == team.display_name and session["active"] == "In Progress"]
        elif fb_store:
            session_ref = fb_store.collection(f"{config.db_prefix}Session")
            doc_stream = session_ref.where(u'user_name', u'==', team.display_name).where(
                u'active', u'==', u'In Progress').stream()
            docs = [doc.id for doc in doc_stream]

        return docs

    def collection(self, base_ref: str) -> Any:
        if self.local:
            return LocalCollection(base_ref)
        elif fb_store:
            return fb_store.collection(base_ref)


class LocalAuth():
    def __init__(self, json_folder: str = None):
        self.json_folder = json_folder
        self.UserNotFoundError = firebase_admin_auth.UserNotFoundError

    def get_user(self, user_secret: str) -> Team:
        team = Team(user_secret)
        return team


database = Database(LOCAL_JSON_FOLDER)
auth = firebase_admin_auth if LOCAL_JSON_FOLDER is None else LocalAuth(LOCAL_JSON_FOLDER)
