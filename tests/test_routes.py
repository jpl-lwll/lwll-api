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

from fastapi.testclient import TestClient
import json
import os
# from conftest import client
import pandas as pd
import pytest
from pathlib import Path
from lwll_api.classes.models import DatasetMetadata
from typing import Any
from lwll_api.classes.models import Session
# from test_session import pseudo_mnist_load

def psuedo_mnist_load(*args: Any, **kwargs: Any) -> DatasetMetadata:
    data = {
        'name': 'mnist',
        'dataset_type': 'image_classification',
        'sample_number_of_samples_train': 5000,
        'sample_number_of_samples_test': 1000,
        'sample_number_of_classes': 10,
        'sample_data_url': '/datasets/lwll_datasets/mnist/mnist_sample/train',
        'full_number_of_samples_train': 60000,
        'full_number_of_samples_test': 10000,
        'full_number_of_classes': 10,
        'full_data_url': '/datasets/lwll_datasets/mnist/mnist_full/train',
        'date_created': 2342342,
        'uid': 'mnist',
        'classes': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
    }
    dataset = DatasetMetadata("", data=data, load=False)
    return dataset


@pytest.fixture()
def mnist_sess_token(client: TestClient) -> Any:
    """
    Creates a session that gets stored in firebase
    :return:
    """
    headers = {'user_secret': os.environ.get('TESTS_SECRET'), 'govteam_secret': os.environ.get('GOVTEAM_SECRET')}
    data = {
        'task_id': 'problem_test_image_classification',
        'user_name': 'JPL',
        'data_type': 'full',
        'session_name': 'pytest'
    }
    # create the session
    response = client.post('/auth/create_session', json=data, headers=headers)
    assert response.status_code == 200
    data = response.json()
    session_token = data['session_token']
    return session_token


@pytest.fixture()
def prob_preds_missing_classes() -> pd.DataFrame:
    with open(Path.cwd() / 'tests' / 'test_data' / 'test_ids_ic_full.json', 'r') as f:
        ids = json.load(f)['id']
    dataset_metadata = psuedo_mnist_load()
    current_dataset_classes = dataset_metadata.classes
    if current_dataset_classes:
        columns = current_dataset_classes[:-2]
    predictions = pd.DataFrame({'id': ids, columns[0]: [1.0] * len(ids)})
    for col in columns[1:]:
        predictions[col] = [0.0] * len(ids)
    return predictions


@pytest.fixture()
def prob_preds_bad_probs() -> pd.DataFrame:
    with open(Path.cwd() / 'tests' / 'test_data' / 'test_ids_ic_full.json', 'r') as f:
        ids = json.load(f)['id']
    dataset_metadata = psuedo_mnist_load()
    classes = dataset_metadata.classes
    if classes:
        prob = 1.0 / len(classes)
        preds = {classes[i]: [prob] * len(ids) for i in range(len(classes))}
        preds['id'] = ids
        predictions = pd.DataFrame(preds)
        # make an example not sum to one
        predictions.loc[0, classes[0]] += 0.1
    return predictions

def noop(*args: Any, **kwargs: Any) -> None:
    return


Session.save = noop  # type: ignore
Session._set_ref_to_db = noop  # type: ignore
Session._get_current_dataset = psuedo_mnist_load  # type: ignore

@pytest.fixture()
def sess() -> Session:

    data = {
        'task_id': 'problem_test_image_classification',
        'user_name': 'JPL'
    }
    s: Session = Session.create(data)
    return s

class TestRoutes:

    def test_prob_preds_missing_classes(self, sess: Session,
                                        prob_preds_missing_classes: pd.DataFrame) -> None:
        """
        :param sess:
        :param prob_preds_missing_classes:
        :return:
        """
        _, r = sess._valid_probabilistic(prob_preds_missing_classes)
        assert r.startswith("Probabilistic predictions are missing classes")

    def test_preds_bad_probs(self, sess: Session, prob_preds_bad_probs: pd.DataFrame) -> None:
        """
        :param sess:
        :param prob_preds_bad_prob:
        :return:
        """
        _, r = sess._valid_probabilistic(prob_preds_bad_probs)
        assert r.startswith("Probabilistic predictions do not sum to 1.0")

    def test_skip_checkpoints(self, client: TestClient, mnist_sess_token: str) -> None:
        """
        :param client:
        :param mnist_sess_token:
        :return:
        """
        assert isinstance(mnist_sess_token, str) and len(mnist_sess_token) > 0
        secret = os.environ.get('TESTS_SECRET')
        headers = {'user_secret': secret, 'session_token': mnist_sess_token,
                   'govteam_secret': os.environ.get('GOVTEAM_SECRET')}

        response = client.get('/skip_checkpoint', headers=headers)

        # Test Client doesn't send outbound call, so we can't test external creation locally,
        # but allow 200 in case of remote API
        assert response.status_code in [200, 400]
