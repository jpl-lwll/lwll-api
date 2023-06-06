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

import pytest
from lwll_api.classes.models import Session, DatasetMetadata
from typing import Any
import pandas as pd
import json
import random
from pathlib import Path

def noop(*args: Any, **kwargs: Any) -> None:
    return

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

@pytest.fixture()
def predictions() -> pd.DataFrame:
    with open(Path.cwd() / 'tests' / 'test_data' / 'test_ids_ic_full.json', 'r') as f:
        ids = json.load(f)['id']
    dataset_metadata = psuedo_mnist_load()
    current_dataset_classes = dataset_metadata.classes
    rand_lbls = ['0'] * len(ids)
    if current_dataset_classes:
        rand_lbls = [str(random.choice(current_dataset_classes)) for _ in range(len(ids))]
    predictions = pd.DataFrame({'id': ids, "class": rand_lbls})
    return predictions

class TestSession:
    # Base and adaptation budgets are the same for the test problem
    base_budget_stages = [10, 20, 40, 80, 419, 2191, 11465, 60000]
    adapt_budget_stages = [10, 20, 40, 80, 419, 2191, 11465, 60000]

    def test_initial_budget(self, sess: Session) -> None:
        assert sess.get_available_budget() == 10

    def test_initial_stage(self, sess: Session) -> None:
        assert sess.pair_stage == 'base'

    def test_initial_metadata(self, sess: Session) -> None:
        d = sess.session_status_dict()
        assert d['current_label_budget_stages'] == TestSession.base_budget_stages
        # test that we only get the keys that we expect
        expected_keys = set({'budget_left_until_checkpoint', 'current_dataset', 'current_label_budget_stages',
                             'task_id', 'using_sample_datasets', 'pair_stage', 'budget_used', 'domain_adaptation_submitted',
                             'active', 'date_created', 'date_last_interacted', 'uid', 'session_name', 'user_name', 'ZSL'})
        difference = expected_keys.symmetric_difference(set(d.keys()))
        assert len(difference) == 0, f"session keys are {d.keys()}"

    def test_step_through(self, sess: Session) -> None:
        # Step through all of the stages of base
        for i in range(len(TestSession.base_budget_stages)):
            # self.one_step(i, Session, 'base')
            if i == 0:
                assert sess.get_available_budget() == TestSession.base_budget_stages[i]
            else:
                assert sess.get_available_budget() == TestSession.base_budget_stages[i] - TestSession.base_budget_stages[i - 1], \
                    f"i: {i}, available budget is {sess.get_available_budget()}"
            assert sess.current_stage == i
            assert sess.pair_stage == 'base'
            if i == 0:
                assert sess.budget_used == 0, f"budget used is {sess.budget_used}"
            else:
                assert sess.budget_used == TestSession.base_budget_stages[i - 1], f"step: {i}, budget_used: {sess.budget_used}"
            sess._advance_stage()

        # Step through all the stages of adaptation
        for i in range(len(TestSession.adapt_budget_stages)):
            if i == 0:
                assert sess.get_available_budget() == TestSession.adapt_budget_stages[i]
            else:
                assert sess.get_available_budget() == TestSession.adapt_budget_stages[i] - \
                       TestSession.adapt_budget_stages[i - 1], \
                    f"i: {i}, available budget is {sess.get_available_budget()}"  # noqa E126
            assert sess.current_stage == len(TestSession.adapt_budget_stages) + i
            assert sess.pair_stage == 'adaptation'
            if i == 0:
                assert sess.budget_used == 0, f"budget used is {sess.budget_used}"
            else:
                assert sess.budget_used == TestSession.adapt_budget_stages[
                    i - 1], f"step: {i}, budget_used: {sess.budget_used}"
            sess._advance_stage()
        assert sess.active == 'Complete'

    # TODO: Test getting seed labels
    def test_seed_labels(self, sess: Session) -> None:
        for i in range(4):
            seeds = pd.DataFrame(sess.get_current_seed_labels())
            if i == 0:
                assert seeds['id'].nunique() == TestSession.base_budget_stages[i]
            else:
                assert seeds['id'].nunique() == TestSession.base_budget_stages[i] - TestSession.base_budget_stages[i - 1], \
                    f"i: {i}, got  {seeds['id'].nunique()} unique seeds"

            sess._advance_stage()

        # Test that we get an error after the first 4 checkpoints
        with pytest.raises(Exception):
            sess.get_current_seed_labels()

        # Advance to adaptation
        for i in range(4):
            sess._advance_stage()
        assert sess.pair_stage == 'adaptation'
        assert sess.budget_used == 0

        for i in range(4):
            seeds = pd.DataFrame(sess.get_current_seed_labels())
            if i == 0:
                assert seeds['id'].nunique() == TestSession.base_budget_stages[i]
            else:
                assert seeds['id'].nunique() == TestSession.base_budget_stages[i] - TestSession.base_budget_stages[
                    i - 1], \
                    f"i: {i}, got  {seeds['id'].nunique()} unique seeds"
            sess._advance_stage()

        # Test that we get an error after the first 4 checkpoints
        with pytest.raises(Exception):
            sess.get_current_seed_labels()

    '''
    def test_domain_adaptation(self, sess: Session, predictions: pd.DataFrame) -> None:
        assert not sess.domain_adaptation_submitted, f"domain_adaptation_submitted is {sess.domain_adaptation_submitted}"
        sess.submit_checkpoint(predictions, UDA=True)
        # Test successful submission
        assert sess.domain_adaptation_submitted
        assert sess.domain_adaptation_score'''

    # TODO zero shot tests in API:
    # calling zero shot twice
    # calling zero shot after submission
    # after querying by id

    def test_submission_validation(self, sess: Session) -> None:
        dataset = psuedo_mnist_load()
        mnist_submission_validation_df = pd.DataFrame({'id': ['img1.png', 'img2.png', 'img3.png'], 'class': ['3', '2', '4']})

        # Test Validation goes through where our val df has correct labels
        sess._validate_class_labels(mnist_submission_validation_df, dataset, False)

    def test_submission_validation_fail(self, sess: Session) -> None:
        dataset = psuedo_mnist_load()
        mnist_submission_validation_df = pd.DataFrame({'id': ['img1.png', 'img2.png', 'img3.png'], 'class': ['13', '2', '4']})

        # Test Validation raises exception when we supply an incorrect label
        with pytest.raises(Exception):
            sess._validate_class_labels(mnist_submission_validation_df, dataset, False)
