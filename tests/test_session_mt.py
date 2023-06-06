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


def noop(*args: Any, **kwargs: Any) -> None:
    return

def pseudo_wikimatrix_load(*args: Any, **kwargs: Any) -> DatasetMetadata:
    data = {
        'name': 'wikimatrix-fas',
        'dataset_type': 'machine_translation',
        'sample_data_url': '/datasets/lwll_datasets/wikimatrix-fas/wikimatrix-fassample/train',
        'full_data_url': '/datasets/lwll_datasets/wikimatrix-fas/wikimatrix-fas_full/train',
        'uid': 'wikimatrix-fas',
        'date_created': 2342342,  # Note: This value is not correct. Could not find the right value
        'sample_number_of_samples_train': 28000,
        'sample_number_of_samples_test': 5000,
        'full_number_of_samples_train': 80000,
        'full_number_of_samples_test': 5000,
    }
    dataset = DatasetMetadata("", data=data, load=False)
    return dataset


Session.save = noop  # type: ignore
Session._set_ref_to_db = noop  # type: ignore
Session._get_current_dataset = pseudo_wikimatrix_load  # type: ignore

@pytest.fixture()
def sess() -> Session:

    data = {
        'task_id': 'problem_test_machine_translation',
        'user_name': 'JPL'
    }
    s: Session = Session.create(data)
    return s


'''''@pytest.fixture()
def predictions() -> pd.DataFrame:
    with open(Path.cwd() / 'tests' / 'test_data' / 'test_ids_mt_full.json', 'r') as f:
        testids = json.load(f)['id']
    dataset_metadata = pseudo_wikimatrix_load()

    pred = 'The quick brown fox jumps over the lazy dog'
    pred_list = [pred for _ in range(len(test_df))]
    df = pd.DataFrame({'id': testids, 'text': pred_list})
    return df'''

class TestSession:
    # Base and adaptation budgets are the same for the test problem
    base_budget_stages = [5000, 14693, 43177, 126882, 372858, 1095690, 3219816, 9461816]
    adapt_budget_stages = [5000, 14693, 43177, 126882, 372858, 1095690, 3219816, 9461816]

    def test_initial_budget(self, sess: Session) -> None:
        assert sess.get_available_budget() == 5000

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
                print('Stage', i, sess.get_available_budget(),
                      TestSession.base_budget_stages[i] - TestSession.base_budget_stages[i - 1])
                assert sess.get_available_budget() == TestSession.base_budget_stages[i] - \
                    TestSession.base_budget_stages[i - 1], f"i: {i}, " \
                    f"available budget is {sess.get_available_budget()}"
            assert sess.current_stage == i
            assert sess.pair_stage == 'base'
            if i == 0:
                assert sess.budget_used == 0, f"budget used is {sess.budget_used}"
            else:
                assert sess.budget_used == TestSession.base_budget_stages[i - 1], f"step: {i}, " \
                                                                                  f"budget_used: {sess.budget_used}"
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
