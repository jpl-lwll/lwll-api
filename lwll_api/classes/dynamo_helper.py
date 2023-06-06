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

# from lwll_api.utils.logger import log
from typing import List, Any
import asyncio
import aiobotocore
from itertools import chain
import os


async def fetch_from_dynamo(table: str, keys: list, dynamodb: Any) -> Any:
    # log.info(f"Keys:: {keys}")
    response = await dynamodb.batch_get_item(RequestItems={
        table: {'Keys': keys}
    })
    return response


async def launch_query(session: Any, table: str, query: list) -> Any:
    async with session.create_client('dynamodb', verify=False) as dynamodb:
        MAX_DYNAMO_BATCH_GET = 100
        tasks = []
        for i in range(int(len(query) / MAX_DYNAMO_BATCH_GET) + 1):
            keys = query[MAX_DYNAMO_BATCH_GET * i:MAX_DYNAMO_BATCH_GET * (i + 1)]
            if len(keys) > 0:
                task = await fetch_from_dynamo(table, keys, dynamodb)
                tasks.append(task)
        responses = [_res['Responses'][table] for _res in tasks]
        return responses


class DynamoHelper:

    def __init__(self) -> None:
        self.table = 'machine_translation_target_lookup'
        if os.environ.get('SERVERTYPE') == 'REMOTE':
            self.session = aiobotocore.get_session()
        else:
            self.session = aiobotocore.AioSession(profile='saml-pub')

    def query_ids(self, dataset_id: str, dataset_type: str, id_list: List[str]) -> List[dict]:
        query = [{'datasetid_sentenceid': {"S": f"{dataset_id}_{dataset_type}_{_id}"}} for _id in id_list]
        # log.info(f"Query: {query}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        future = asyncio.ensure_future(launch_query(self.session, self.table, query))
        items = loop.run_until_complete(future)
        loop.close()
        # log.debug(f"After Loop: {items}")
        flat_items = list(chain.from_iterable(items))

        # log.info(f"Query Response: {flat_items}")
        items_tfm: List[dict] = [{'id': _d['datasetid_sentenceid']['S'].split(
            '_')[-1], 'text': _d['target']['S'], 'size': _d['size']['N']} for _d in flat_items]
        return items_tfm


dynamo_helper = DynamoHelper()
