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
from typing import Any, List, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd

from lwll_api.classes import metrics
from lwll_api.classes.base import BaseFBObject
from lwll_api.classes.dynamo_helper import dynamo_helper
from lwll_api.classes.s3_cls import s3_operator
from lwll_api.utils.config import config
from lwll_api.utils.logger import get_module_logger

C = TypeVar("C")


class DatasetMetadata(BaseFBObject):
    def __init__(self, uid: str, new: bool = False, data: dict = {}, **kwargs: Any) -> None:
        super().__init__(uid, "DatasetMetadata", new=new, data=data, **kwargs)
        self.name: str = self.data["name"]
        self.dataset_type: str = self.data["dataset_type"]
        self.sample_number_of_samples_train: int = self.data[
            "sample_number_of_samples_train"
        ]
        self.sample_number_of_samples_test: int = self.data[
            "sample_number_of_samples_test"
        ]
        self.sample_number_of_classes: Optional[int] = self.data.get(
            "sample_number_of_classes"
        )
        self.full_number_of_samples_train: int = self.data[
            "full_number_of_samples_train"
        ]
        self.full_number_of_samples_test: int = self.data["full_number_of_samples_test"]
        self.full_number_of_classes: Optional[int] = self.data.get(
            "full_number_of_classes"
        )
        self.language_from: Optional[str] = self.data.get("language_from")
        self.language_to: Optional[str] = self.data.get("language_to")
        self.sample_total_codecs: Optional[int] = self.data.get("sample_total_codecs")
        self.full_total_codecs: Optional[int] = self.data.get("full_total_codecs")
        self.number_of_channels: Optional[int] = self.data.get("number_of_channels")
        self.classes: List[str] = self.data.get("classes", [])
        self.license_link: str = self.data.get("license_link", "")
        self.license_requirements: str = self.data.get("license_requirements", "")
        self.license_citation: str = self.data.get("license_citation", "")
        self.zsl_description: dict = self.data.get("zsl_description", None)
        self.log = get_module_logger(__name__)

        if "seen_classes" in self.data and "unseen_classes" in self.data:
            self.seen_classes = self.data["seen_classes"]
            self.unseen_classes = self.data["unseen_classes"]

    # Overwriting the create from the superclass so that we can set the appropriate ref since we know the uid on create only for the Dataset
    @classmethod
    def create(cls, data: dict = {}) -> C:  # type: ignore
        u = cls(data["uid"], new=True, data=data, overridden_create=True)
        u.ref = u.base_ref.document(u.uid)
        u.ref.set(u.to_dict())
        return u  # type: ignore

    def to_dict(self) -> dict:
        data = {
            'name': self.name,
            'dataset_type': self.dataset_type,
            'sample_number_of_samples_train': self.sample_number_of_samples_train,
            'sample_number_of_samples_test': self.sample_number_of_samples_test,
            'sample_number_of_classes': self.sample_number_of_classes,
            'full_number_of_samples_train': self.full_number_of_samples_train,
            'full_number_of_samples_test': self.full_number_of_samples_test,
            'full_number_of_classes': self.full_number_of_classes,
            'language_from': self.language_from,
            'language_to': self.language_to,
            'sample_total_codecs': self.sample_total_codecs,
            'full_total_codecs': self.full_total_codecs,
            'number_of_channels': self.number_of_channels,
            'classes': self.classes,
            'license_link': self.license_link,
            'license_requirements': self.license_requirements,
            'license_citation': self.license_citation,
            'uid': self.uid,
            'zsl_description': self.zsl_description
        }
        if hasattr(self, "seen_classes"):
            data["seen_classes"] = self.seen_classes
        if hasattr(self, "unseen_classes"):
            data["unseen_classes"] = self.unseen_classes
        data = self.clean_empty(data)
        return data


class Task(BaseFBObject):
    def __init__(self, uid: str, new: bool = False, data: dict = {}, **kwargs: Any) -> None:
        super().__init__(uid, "Task", new=new, data=data)

        self.task_id: str = self.data['task_id']
        self.problem_type: str = self.data['problem_type']
        self.base_dataset: str = self.data['base_dataset']
        self.whitelist: List[str] = self.data['whitelist']

        # Add a conditional block for ZSL since ZSL doesn't have budgets
        self.optional_sub_tasks: list = self.data.get("optional_sub_tasks", [])
        self.data_fields: list = list(self.data.keys()) + ["optional_sub_tasks"]
        self.__dict__.update(self.data)
        if "zsl" in self.optional_sub_tasks:
            self.base_label_budget_full: List[int] = []
            self.base_label_budget_sample: List[int] = []
            self.base_seed_labels: dict = dict()
            self.adaptation_evaluation_metrics: List[str] = []
            self.adaptation_label_budget_full: List[int] = []
            self.adaptation_label_budget_sample: List[int] = []
            self.adaptation_seed_labels: dict = {}
        else:
            if self.problem_type != "machine_translation":
                self.base_seed_labels = self.data["base_seed_labels"]
                self.adaptation_seed_labels = self.data["adaptation_seed_labels"]
                self.uda_base_to_adapt_overlap_ratio = self.data[
                    "uda_base_to_adapt_overlap_ratio"
                ]
                self.uda_adapt_to_base_overlap_ratio = self.data[
                    "uda_adapt_to_base_overlap_ratio"
                ]
            self.base_label_budget_full = self.data["base_label_budget_full"]
            self.base_label_budget_sample = self.data["base_label_budget_sample"]
            self.adaptation_label_budget_full = self.data[
                "adaptation_label_budget_full"
            ]
            self.adaptation_label_budget_sample = self.data[
                "adaptation_label_budget_sample"
            ]
            self.adaptation_evaluation_metrics = self.data[
                "adaptation_evaluation_metrics"
            ]
            self.base_evaluation_metrics: List[str] = self.data[
                "base_evaluation_metrics"
            ]
            self.adaptation_dataset: str = self.data["adaptation_dataset"]

    def to_dict(self) -> dict:
        data = self.clean_empty({key: self.__dict__[key] for key in self.data_fields})
        if "zsl" in self.optional_sub_tasks:
            data["base_label_budget_full"] = []
            data["base_label_budget_sample"] = []
            data["base_seed_labels"] = dict()
            data["adaptation_evaluation_metrics"] = []
            data["adaptation_label_budget_full"] = []
            data["adaptation_label_budget_sample"] = []
            data["adaptation_seed_labels"] = {}
        return dict(data)

    def to_public_dict(self) -> dict:
        """
        Removes seed labels from the Session to create a performer safe view.
        :return:
        """
        # exclude any keys that contain "seed_label" in the key name
        data = {k: v for k, v in self.to_dict().items() if "seed_label" not in k}
        return data


class Session(BaseFBObject):

    # Dictionary of scoring metrics
    score_metrics = {
        "accuracy": metrics.accuracy,
        "top_5_accuracy": metrics.top_5_accuracy,
        "roc_auc": metrics.roc_auc,
        "mAP": metrics.mAP,
        "cross_entropy_logloss": metrics.cross_entropy_logloss,
        "brier_score": metrics.brier_score,
        "bleu": metrics.bleu_sacre,
        "weighted_accuracy": metrics.weighted_accuracy,
        "precision_at_50": metrics.precision_at_k,
        "average_precision_at_50": metrics.average_precision_at_50
    }

    def __init__(self, uid: str, new: bool = False, data: dict = {}, **kwargs: Any) -> None:
        super().__init__(uid, "Session", new=new, data=data)
        self.task_id: str = self.data['task_id']
        self.user_name: str = self.data['user_name']
        self.using_sample_datasets: bool = self.data.get('using_sample_datasets', False)
        self.date_last_interacted: int = self.data.get('date_last_interacted', int(time.time()) * 1000)
        self.session_name: str = self.data.get('session_name', None)

        # Our additional keys that will get filled in with defaults upon creation
        self.pair_stage: str = self.data.get('pair_stage', 'base')  # Valid values here will be 'base' and 'adaptation'
        self.current_stage: int = self.data.get('current_stage', 0)
        self.budget_used: int = self.data.get('budget_used', 0)
        self.checkpoint_scores: List[dict] = self.data.get('checkpoint_scores', [])
        self.active: str = self.data.get("active", "In Progress")
        self.domain_adaptation_score: List[dict] = self.data.get(
            "domain_adaptation_score", []
        )
        self.domain_adaptation_submitted: bool = self.data.get(
            "domain_adaptation_submitted", False
        )

        # Lookups
        self._current_task = Task(self.task_id)
        self.zsl = "zsl" in self._current_task.optional_sub_tasks
        self._base_budget: List[int] = self._current_task.base_label_budget_full
        self._adaptation_budget: List[int] = self._current_task.adaptation_label_budget_full
        self._seed_labels: dict = {}
        if self._current_task.problem_type != 'machine_translation':
            if self.pair_stage == "base":
                self.log.info("getting base seed labels")
                self._seed_labels = self._current_task.base_seed_labels
            elif self.pair_stage == "adaptation":
                self.log.info("getting adaptation labels")
                self._seed_labels = self._current_task.adaptation_seed_labels

        if self.using_sample_datasets:
            self._base_budget = self._current_task.base_label_budget_sample
            self._adaptation_budget = self._current_task.adaptation_label_budget_sample
        # Stages are the checkpoints at which we care about submissions, indexed by checkpoint index
        self._num_stages: int = len(self._base_budget) + len(self._adaptation_budget)
        self._base_s3_path, self._adaptation_s3_path = self._get_dataset_s3_paths(
            include_adaptation=(not self.zsl)
        )

        if self.zsl:
            self.zsl_std_scores: dict = self.data.get("standard_zsl_scores", {})

        # Track if labels has been requested
        self._requested_labels: bool = self.data.get("requested_labels", False)

    def to_dict(self) -> dict:
        data = {
            "task_id": self.task_id,
            "user_name": self.user_name,
            "using_sample_datasets": self.using_sample_datasets,
            "pair_stage": self.pair_stage,
            "current_stage": self.current_stage,
            "budget_used": self.budget_used,
            "checkpoint_scores": self.checkpoint_scores,
            "active": self.active,
            "date_created": self.date_created,
            "date_last_interacted": self.date_last_interacted,
            "uid": self.uid,
            "session_name": self.session_name,
            "domain_adaptation_score": self.domain_adaptation_score,
            "domain_adaptation_submitted": self.domain_adaptation_submitted,
            "ZSL": self.zsl,
            "requested_labels": self._requested_labels,
        }
        if self.zsl:
            data["standard_zsl_scores"] = self.zsl_std_scores
        return data

    def public_to_dict(self, data: dict = {}) -> dict:
        black_list = {'current_stage', 'checkpoint_scores', 'base_seed_labels', 'adaptation_seed_labels',
                      'domain_adaptation_score', 'requested_labels'}
        if self.active in ['Complete', 'Deactivated']:
            black_list.discard('checkpoint_scores')
            black_list.discard('domain_adaptation_score')
        elif self.current_stage == len(self._base_budget):
            black_list.discard('checkpoint_scores')
        public_data = {key: val for key, val in data.items() if key not in black_list}
        return public_data

    def deactivate_session(self) -> None:
        if self.active == 'In Progress':
            self.active = 'Deactivated'
            self.save()
        else:
            raise Exception("Attempting to deactivate a session that is already Deactivated or Complete")
        return

    def skip_checkpoint(self) -> None:
        """
        Skips the current checkpoint and report None as the score. Final checkpoints cannot be skipped.
        """
        if self.current_stage == (len(self._base_budget) - 1) or self.current_stage == (self._num_stages - 1):
            raise Exception("The last checkpoint cannot be skipped")

        metric_dict: dict = {}
        for idx, metric in enumerate(getattr(self._current_task, f'{self.pair_stage}_evaluation_metrics')):
            metric_dict[metric] = None

        self.checkpoint_scores.append(metric_dict)
        # Advance the stage
        self._advance_stage()
        return

    def session_status_dict(self) -> dict:
        data = self.to_dict()
        data['budget_left_until_checkpoint'] = self.get_available_budget()
        dataset: DatasetMetadata = self._get_current_dataset()
        data['current_dataset'] = dataset.to_dict()
        data['current_dataset'] = self._subset_dataset_metadata(data=data['current_dataset'])
        data['current_label_budget_stages'] = getattr(self, f"_{self.pair_stage}_budget")
        public_data = self.public_to_dict(data)
        return public_data

    def get_available_budget(self) -> int:
        """
        Method to do appropriate lookup into our tiered budgeting and get the max amount left
        before having to submit a model
        """
        self.log.info(f"current stage is: {self.current_stage}")
        if self.current_stage > len(self._base_budget) + len(self._adaptation_budget) - 1:
            return 0
        if self.current_stage < len(self._base_budget):
            self.log.info(f"calculating budget left: {self._base_budget[self.current_stage]} - {self.budget_used}")
            return self._base_budget[self.current_stage] - self.budget_used
        else:
            return self._adaptation_budget[self.current_stage - len(self._base_budget)] - self.budget_used

    def get_current_seed_labels(self) -> Any:
        """
        Returns the seed labels for the current dataset of the session.
        """
        # Check that the current task is image_classification or object_detection since there is no concept of seed labels
        # for machine translation.
        if self._current_task.problem_type == 'machine_translation':
            warning = f"You have requested seed labels for {self._current_task.problem_type}, which are not available for this task."
            # warnings.warn(warning)
            # return None
            raise Exception(warning)
        elif self.zsl:
            raise Exception(
                "You cannot request seed labels for a ZSL session. Please use the get_seen_labels endpoint instead."
            )
        # Check if the budget left matches the budget for the current checkpoint. If not, seed labels cannot be returned
        budget_left = self.get_available_budget()
        stage_budget = self._base_budget if self.pair_stage == 'base' else self._adaptation_budget
        curr_stage = self.current_stage if self.pair_stage == 'base' else self.current_stage - len(self._base_budget)
        curr_stage_budget = stage_budget[curr_stage]
        prev_stage_budget = 0 if curr_stage == 0 else stage_budget[curr_stage - 1]
        # The number of additional labels that will be given at this checkpoint
        budget_delta = curr_stage_budget - prev_stage_budget

        if budget_left == 0:
            raise Exception("Budget left is 0. You must submit predictions before calling seed_labels again.")
        elif budget_left != budget_delta:
            error_msg = f"Budget left is {budget_left} but you must have {budget_delta} available at this " + \
                        "checkpoint to get seed labels."
            if curr_stage < 4:
                error_msg = error_msg + "You may query for labels until you reach the label budget and you may " + \
                            "request seed labels again after submitting."  # noqa: E126
            raise Exception(error_msg)
        _seeds = self._seed_labels[str(curr_stage)]

        if not _seeds:
            raise Exception(f"Task seed labels are not valid for this stage (stage {curr_stage}).")

        # increment the budget used
        seed_ids = set([item['id'] for item in _seeds])
        self.budget_used = self.budget_used + len(seed_ids)
        self.save()
        self.log.info(f'budget_used is {self.budget_used}')
        return _seeds

    def request_labels(self, example_ids: List[str]) -> List[dict]:
        """
        Performs the operation of single label lookups and subtracts the appropriate amount from the budget
        """
        current_dataset = self.session_status_dict()['current_dataset']
        dataset_type = current_dataset['dataset_type']
        dataset_id = current_dataset['uid']
        budget_left = self.get_available_budget()

        if dataset_type != 'machine_translation':
            # Get our labels and available budget
            lbls = self._get_train_labels()

            # If we are requesting more than we have left in our budget, we truncate to first N until
            # we hit the budget we have left
            example_ids = list(set(example_ids))
            if len(example_ids) > budget_left:
                example_ids = example_ids[:budget_left]

            # Do the actual lookup in our data
            df_labels = lbls.loc[lbls['id'].isin(example_ids)]
            df_list: List[dict] = df_labels.to_dict(orient='records') if not df_labels.empty else None

            # subtract from our budget
            self.budget_used += len(example_ids)

        else:
            sample_full = 'sample' if self.using_sample_datasets else 'full'
            self.log.info(f"Querying ids: {example_ids}")
            lbls = dynamo_helper.query_ids(dataset_id, sample_full, example_ids)
            returned_tot = 0
            df_list = []
            for _lbl in lbls:
                if budget_left > returned_tot:
                    df_list.append(_lbl)
                    returned_tot += int(_lbl['size'])
                else:
                    break

            # We might be missing some spill over, but we make the assumption that small
            # spillover is fine.
            self.budget_used += min(returned_tot, budget_left)

        self._requested_labels = True
        self.save()
        return df_list

    def get_seen_labels(self) -> List[dict]:
        """
        Gets the labels for all seen classes for a ZSL (zero shot learning) session. This is done by
        retrieving the label file, and subtracting any ids from the metafile containing unseen ids.
        Returns:
            A list of labels. Example:
            [
                {'id': 'a.png', 'class': '0'},
                {'id': 'b.png', 'class': '0'},
                {'id': 'c.png', 'class': '1'}, ...
            ]
        """
        # Get the label files and the metafiles
        label_df = self._get_train_labels()
        id_label_dict = label_df.set_index("id").to_dict()["class"]
        all_ids, unseen_ids = set(id_label_dict.keys()), set(self._get_unseen_ids())
        seen_ids = all_ids - unseen_ids
        seen_lbls_list = [
            {"id": seen_id, "class": id_label_dict[seen_id]} for seen_id in seen_ids
        ]
        return seen_lbls_list

    def _advance_stage(self) -> None:
        """
        Advances to the next stage of performance

        If wanting to advance before the end of a stage, then we advance the counter up to the point where it
        should be in order to avoid 'saving' labels for adaptation run.
        """
        available_budget = self.get_available_budget()
        self.budget_used += available_budget
        self.current_stage += 1
        # switch from base to adaptation and reset budget and seed label cache
        if self.current_stage == len(self._base_budget) and self.pair_stage == 'base':
            self.pair_stage = 'adaptation'
            self.budget_used = 0
            # Reset requested labels flag
            self._requested_labels = False
        if self.current_stage == self._num_stages:
            self.active = 'Complete'
        self.save()
        return

    def check_accuracy(self, df: pd.DataFrame, probabilistic: bool = False) -> float:
        """
        Takes the submission file and checks the accuracy before submission.
        Args:
            df: pd.DataFrame, a dataframe of predictions
            probabilistic: bool, indicates whether the scores are in probabilistic format
        Returns accuracy
        """
        accuracy = self._get_submission_accuracy(df, probabilistic)

        return accuracy

    def submit_checkpoint(
        self,
        df: pd.DataFrame,
        UDA: bool = False,
        probabilistic: bool = False,
        standard_zsl: bool = False,
    ) -> dict:

        """
        Takes the submission file and scores the output against our label file
        Args:
            df: pd.DataFrame, a dataframe of predictions
            UDA: bool, indicates whether to score the submission as unsupervised domain adaptation
            probabilistic: bool, indicates whether the scores are in probabilistic format
            standard_zsl: bool, indicates whether to score the submission as standard zero-shot learning, meaning
                that the model only outputs unseen classes, so we should only evaluate it on the unseen classes.
        Returns data regarding performance and stage
        """

        if UDA and self._requested_labels:
            raise Exception("Attempting to submit UDA but labels have been requested already.")

        # Current dataset before advancing the stage
        dataset: DatasetMetadata = self._get_current_dataset()
        if dataset.dataset_type != 'machine_translation' and not probabilistic:
            # probabilistic predictions already validated and doesn't make sense for MT
            self._validate_class_labels(df, dataset, probabilistic)

        # Score submission
        scores: dict = self._score_submission(df, probabilistic)
        self.log.info(str(scores))
        if UDA:
            self.domain_adaptation_score.append(scores)
            self.domain_adaptation_submitted = True
            self.save()
            # Save raw prediction file out for archive purposes
            # Example format: 'DEV/<task_id>/<session_id>/base/\
            # <timestamp>_<ckpt-no>.feather'
            s3_operator.save_prediction_df(df, f"{config.lwll_mode}/\
{self.task_id}/{self.uid}/{self.pair_stage}/{int(time.time())}_UDA.feather")
        elif self.zsl:
            s3_save_path = (
                f"{config.lwll_mode}/{self.task_id}/{self.uid}/{self.pair_stage}/"
                f"{int(time.time())}"
                "{zsl}.feather"
            )
            if not standard_zsl:
                self.checkpoint_scores.append(scores)
                self.save()
                self.active = "Complete"
                s3_operator.save_prediction_df(df, s3_save_path.format(zsl="_zsl"))
            else:
                unseen_scores = {  # rename unseen scores to unseen_std
                    k.replace("_unseen", "_unseen_std"): v
                    for k, v in scores.items()
                    if k.endswith("_unseen")
                }
                self.zsl_std_scores = unseen_scores
                self.save()
                s3_operator.save_prediction_df(df, s3_save_path.format(zsl="_zsl_std"))
        else:
            self.checkpoint_scores.append(scores)

            # Save raw prediction file before advancing stage
            # Example format: 'DEV/<task_id>/<session_id>/base/\
            # <timestamp>_<ckpt-no>.feather'
            s3_operator.save_prediction_df(df, f"{config.lwll_mode}/\
{self.task_id}/{self.uid}/{self.pair_stage}/{int(time.time())}_ckpt\
{self.current_stage+1}.feather")

            # Advance the stage
            self._advance_stage()

        # Getting the dataset for this next stage, if moving to the 'adaptation' pair_stage from 'base', this will
        # have new metadata to be returned.
        new_dataset: DatasetMetadata = self._get_current_dataset()
        return {'scores': scores, 'current_dataset_metadata': new_dataset.to_dict(), 'session_status': self.session_status_dict()}

    def _validate_class_labels(self, df: pd.DataFrame, dataset: DatasetMetadata, probabilistic: bool) -> None:
        """
        We do validation on the user submitted checkpoint and throw a hard 500 error if there exists a label that is not within the
        valid dataset class labels
        Args:
            df: pd.DataFrame, performer submitted predictions
            dataset: DatasetMetadata, metadata for the ground truth data
            probabilistic: bool, Whether we are validating a probabilistic predictions dataframe
        """
        classes = dataset.classes
        if classes is None or len(classes) == 0:
            raise Exception("Attempting to validate an empty class list")
        if probabilistic:
            non_valid_classes = set(df.columns).difference(set(classes))
        else:
            if "class" in df.columns:
                non_valid_classes = set(df['class']).difference(set(classes))
            else:
                raise Exception("Attempting to validate a non-probabilistic submission without a 'class' column")
        if len(non_valid_classes) > 0:
            raise Exception(f"Presence of invalid class(es): {non_valid_classes}")
        return

    def _valid_probabilistic(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Checks that probabilistic predictions sum to 1.0 within a tolerance and that all class labels (and no extras) are
        present
        Args:
            df: pd.DataFrame, A dataframe of performer predictions
        :return:
         (valid, warning): (bool, str)
            valid: Indicates whether the probabilistic predictions meet both requirements: all class labels present
        and probabilities sum to 1.
            warning: str, A message to raise to the user
        """
        if "video_id" in df.columns:
            view = df.drop(columns=["id", "video_id", "end_frame", "start_frame"])
        else:
            view = df.drop(columns="id")

        # Check that there are no extra classes. This will raise an error and won't continue if there are classes that
        # don't exist in the ground truth
        dataset: DatasetMetadata = self._get_current_dataset()
        self._validate_class_labels(view, dataset, True)

        # Check that all classes exist
        if dataset.classes is not None and len(dataset.classes) > 0:
            missing_classes = set(dataset.classes).difference(view.columns)
        all_classes = True if len(missing_classes) == 0 else False
        warning = ""
        if not all_classes:
            # TODO: check that warning is raised but program continues
            warning = f"Probabilistic predictions are missing classes {missing_classes}. \
            Falling back to hard assignment for majority class."
            self.log.info(warning)
        # Check that predictions all sum to one
        view['sum'] = view.sum(axis=1)
        # print(f"view looks like: {view.head()}")
        allclose = np.allclose(view['sum'].values, [1.0] * view.shape[0])
        if not allclose:
            warning = "Probabilistic predictions do not sum to 1.0. Falling back to hard assignment for majority class."
            self.log.info(warning)
        # TODO: test for this
        valid = True if allclose and all_classes else False
        return valid, warning

    def _get_submission_accuracy(self, df: pd.DataFrame, probabilistic: bool) -> float:
        df['id'] = df['id'].apply(lambda x: str(x))
        dataset: DatasetMetadata = self._get_current_dataset()
        accuracy = 0.0
        if dataset.dataset_type != 'machine_translation':
            actuals: pd.DataFrame = self._get_actual_labels()
            if dataset.dataset_type == 'video_classification':
                actuals['id'] = actuals['id'].apply(lambda x: str(x))
                actuals = actuals.drop(columns=["video_id", "end_frame", "start_frame"])

            # transform dataframes to one-hot encoded numpy arrays
            if dataset.dataset_type != "object_detection":
                metrics._validate_input_ids_one_to_one(df, actuals)
                # Get one hot encoded vectors
                if probabilistic:
                    df = metrics._probabilistic_to_hard_assignment(df)
                accuracy = metrics.accuracy(df, actuals)
                if accuracy == 0.0:
                    accuracy = 0.01
        return accuracy

    @staticmethod
    def _reformat_submission(preds: pd.DataFrame) -> pd.DataFrame:
        """
        Reformats an incorrectly formatted probabilistic prediction submission to hard assignment
        Returns:
            reformat_df: {pd.DataFrame} The reformatted dataframe
        """
        class_cols = preds.drop(columns='id').columns
        if "video_id" in preds.columns:
            class_cols = preds.drop(columns=["id", "video_id", "end_frame", "start_frame"]).columns
        class_col = pd.Series(preds.loc[:, class_cols].idxmax(axis=1), name="class")
        reformat_df = pd.merge(class_col, preds['id'], left_index=True, right_index=True)
        return reformat_df

    def _score_submission(self, df: pd.DataFrame, probabilistic: bool) -> dict:
        """
        Scores the performer submissions for all of the metric types listed in a task. If the submission is not
        probabilistic, top-5 accuracy is skipped
        Args:
            df: pd.DataFrame, The performer's predictions
            probabilistic: bool, True if predictions are probabilistic, False if hard assignment
        :return:
        """
        df['id'] = df['id'].apply(lambda x: str(x))
        dataset: DatasetMetadata = self._get_current_dataset()
        scores: List[dict] = []
        metric_dict = {}

        if dataset.dataset_type != 'machine_translation':
            actuals: pd.DataFrame = self._get_actual_labels()
            if dataset.dataset_type == 'video_classification':
                actuals['id'] = actuals['id'].apply(lambda x: str(x))
                actuals = actuals.drop(columns=["video_id", "end_frame", "start_frame"])

            # transform dataframes to one-hot encoded numpy arrays
            if dataset.dataset_type != "object_detection":
                metrics._validate_input_ids_one_to_one(df, actuals)
                classes = dataset.classes

                metric_dict = self.compute_metrics(df, actuals, classes, probabilistic)

                if self.zsl:  # For ZSL, we also include metrics for (un)seen classes
                    metric_dict = {
                        f"{k}_all_classes": v for k, v in metric_dict.items()
                    }

                    if hasattr(dataset, "seen_classes") and hasattr(
                        dataset, "unseen_classes"
                    ):
                        seen_classes = sorted(dataset.seen_classes)
                        unseen_classes = sorted(dataset.unseen_classes)
                    else:
                        unseen_classes = sorted(dataset.zsl_description.keys())
                        seen_classes = sorted(set(classes) - set(unseen_classes))

                    # Compute metrics only for test images that belong to seen classes
                    metric_dict_seen = self.compute_metrics(
                        df, actuals, seen_classes, probabilistic, suffix="_seen"
                    )

                    # Compute metrics only for test images that belong to unseen classes
                    metric_dict_unseen = self.compute_metrics(
                        df, actuals, unseen_classes, probabilistic, suffix="_unseen"
                    )
                    metric_dict.update(metric_dict_seen)
                    metric_dict.update(metric_dict_unseen)

            else:
                # Object detection has mAP and mAP at different levels as metrics
                score = metrics.mAP(df, actuals)
                if score == 0.0:
                    score = 0.01
                metric_dict["mAP"] = score

                precisions = metrics.mAP_range(df, actuals)
                for precision_level, value in precisions.items():
                    metric_dict[f"mAP_level_{precision_level}"] = float(value)

        else:
            # For 'machine_tranlsation', we assume only metric is going to be bleu
            # for more complicated metrics they actually return multiple scores like BLEU so we have to accomodate
            # slightly differently
            actuals_mt: pd.DataFrame = pd.DataFrame(self._get_mt_actual_labels())
            for idx, metric in enumerate(getattr(self._current_task, f'{self.pair_stage}_evaluation_metrics')):
                if metric == 'bleu':
                    bleu_sacre_score, bleu_sacre_str = self._score_mt_with_metric(df, actuals_mt, metric)[0]
                    metric_dict['bleu'] = bleu_sacre_score if bleu_sacre_score != 0.0 else 0.001
                    metric_dict['bleu_str'] = bleu_sacre_str

        scores.append(metric_dict)

        flattened_scores: dict = {k: v for d in scores for k, v in d.items()}
        # print(f" flattened scores: {flattened_scores}")
        return flattened_scores

    def compute_metrics(
        self,
        df: pd.DataFrame,
        actuals: pd.DataFrame,
        classes: List[str],
        probabilistic: bool,
        suffix: str = "",
    ) -> dict:

        original_df = df.copy()

        metrics_dict = {}
        actuals = actuals[actuals["class"].isin(classes)].copy()  # filter classes

        if probabilistic:
            df = metrics._probabilistic_to_hard_assignment(df)

        df = df[df["id"].isin(actuals["id"])].copy()
        # get y_true and y_pred by joining the actuals and df on id
        merged_on_ids = pd.merge(actuals, df, on="id", how="left")
        y_true, y_pred = merged_on_ids.drop(columns="id").values.T

        # throw an exception if there are any missing ids
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Length of y_true and y_pred do not match. y_true: {len(y_true)}, y_pred: {len(y_pred)}"
            )

        y_true_oh, y_pred_oh = metrics._preprocess_df_to_array(original_df, actuals, classes)

        metrics_ = getattr(self._current_task, f"{self.pair_stage}_evaluation_metrics")
        for idx, metric in enumerate(metrics_):  # TODO: move this to another line
            # print(f"checking on metric {metric}")
            if metric == "accuracy":
                score = Session.score_metrics[metric](df, actuals)  # type: ignore
                if score == 0.0:
                    score = 0.01
            elif metric == "average_per_class_recall":
                score = metrics.recall(y_true, y_pred, labels=classes, average="macro")
            elif metric == "weighted_accuracy":
                try:
                    # y_true, y_pred = metrics._align_ids(df, actuals)
                    score = Session.score_metrics[metric](y_true, y_pred.flatten())  # type: ignore
                except Exception:
                    self.log.info(
                        "failed to compute {metric} calculating accuracy instead"
                    )
                    score = None
                if score == 0.0:
                    score = 0.01
            elif metric == "top_5_accuracy" or metric == 'roc_auc' or metric == 'precision_at_50' or metric == 'average_precision_at_50':
                if probabilistic:
                    # TOP k doesn't work with one-hot encoding
                    try:
                        score = Session.score_metrics[metric](y_true, original_df)  # type: ignore
                    except Exception:
                        self.log.info(
                            f"failed to compute {metric} calculating accuracy instead"
                        )
                        score = None
                    if score == 0.0:
                        score = 0.01
                else:
                    self.log.info(f"{metric} can only be computed with probabilities.")
                    score = None
            elif metric == "mean_average_precision_50":
                # Skip it here, it is computed after all of this.
                continue
            elif metric in Session.score_metrics.keys():
                # print(f"calculating metric {metric}")
                try:
                    score = Session.score_metrics[metric](y_true_oh, y_pred_oh)  # type: ignore
                except Exception:
                    self.log.info(f"skipping {metric} failed to compute")
                    score = None
                    # When storing more advanced nested dicts within lists in Firestore, it evaluates all "falsey" values to null,
                    # thus removing the key val pair from the data structure. To get around this so that we never lose a metric we do a
                    # check of the only "falsey" numeric value that would do this and set to 0.01 instead. This case should never happen
                    # other than running test scripts for proof of concept.
                if score == 0.0:
                    score = 0.01
                    # print(f"score is {score}")
            else:
                raise Exception("Invalid metric")
            metrics_dict[f"{metric}{suffix}"] = score
        return metrics_dict

    def _score_mt_with_metric(self, df: pd.DataFrame, actuals: pd.DataFrame, metric: str) -> List[Any]:
        """
        Calculates the appropriate score on the DataFarme for MT
        """
        if metric == 'bleu':
            return [Session.score_metrics["bleu"](df, actuals)]  # type: ignore
        else:
            raise Exception('Invalid metric')

    def _get_actual_labels(self) -> pd.DataFrame:
        """
        Get actual labels from S3 datastore
        """
        _path = self._base_s3_path if self.pair_stage == 'base' else self._adaptation_s3_path
        actuals = s3_operator.read_path(f"{_path}/labels_test.feather")
        return actuals

    def _get_mt_actual_labels(self) -> pd.DataFrame:
        """
        Get actual labels from S3 datastore
        """
        _path = self._base_s3_path if self.pair_stage == 'base' else self._adaptation_s3_path
        test_ids = s3_operator.read_path(f"{_path}/test_label_ids.feather")['id'].tolist()
        sample_full = 'sample' if self.using_sample_datasets else 'full'
        current_dataset = self.session_status_dict()['current_dataset']
        dataset_id = current_dataset['uid']

        actuals = dynamo_helper.query_ids(dataset_id, sample_full, test_ids)
        return actuals

    def _get_train_labels(self) -> pd.DataFrame:
        _path = self._base_s3_path if self.pair_stage == 'base' else self._adaptation_s3_path
        lbls = s3_operator.read_path(f"{_path}/labels_train.feather")
        # Check if any ids are missing labels
        if lbls['class'].dtype == object:
            lbls.loc[lbls['class'] == '', 'class'] = None
            lbls.loc[lbls['class'].str.isspace(), 'class'] = None
        lbls.loc[lbls['class'].isna(), 'class'] = None
        return lbls

    def _get_unseen_ids(self, train: bool = True) -> List:
        """
        For ZSL, gets the metafile of unseen ids from S3 and returns the list of unseen ids
        Returns: unseen_ids {List}
        """
        _path = self._base_s3_path if self.pair_stage == 'base' else self._adaptation_s3_path
        # Todo: check for train or test
        filename = "meta_zsl_train.feather"
        if not train:
            filename = "meta_zsl_test.feather"
        meta_data = s3_operator.read_path(f"{_path}/{filename}")
        unseen_ids: List = meta_data["unseen_ids"].tolist()
        return unseen_ids

    def _get_current_dataset(self) -> DatasetMetadata:
        dataset_name = getattr(self._current_task, f'{self.pair_stage}_dataset')
        self.log.info(f"Looking for dataset: {dataset_name}")
        dataset: DatasetMetadata = DatasetMetadata(dataset_name)
        return dataset

    def _get_num_classes(self) -> Optional[int]:
        dataset = self._get_current_dataset()
        try:
            num_classes = dataset.full_number_of_classes
            if self.using_sample_datasets:
                num_classes = dataset.sample_number_of_classes
            if num_classes:
                return int(num_classes)
            else:
                return None
        except KeyError as e:
            self.log.exception(e)
            return None

    def _get_dataset_s3_paths(
        self, include_adaptation: bool = True
    ) -> Tuple[str, Optional[str]]:
        # TODO: This function will need to be updated for MT once it has its new home.
        prefix = "live"
        suffix = 'sample' if self.using_sample_datasets else 'full'
        base_dataset_name = self._current_task.base_dataset
        base_path = f"{prefix}/labels/{base_dataset_name}/{suffix}"
        if include_adaptation:
            adaptation_dataset_name = self._current_task.adaptation_dataset
            adaptation_path: Optional[
                str
            ] = f"{prefix}/labels/{adaptation_dataset_name}/{suffix}"
        else:
            adaptation_path = None
        return base_path, adaptation_path

    def _subset_dataset_metadata(self, data: dict) -> dict:
        _data = data.copy()
        for key, val in data.items():
            if not self.using_sample_datasets:
                # Use full dataset keys and adjust names
                if key[:6] == 'sample':
                    del _data[key]
                if key[:4] == 'full':
                    _data[key[5:]] = _data.pop(key)
            else:
                # Use sample dataset keys and adjust names
                if key[:4] == 'full':
                    del _data[key]
                if key[:6] == 'sample':
                    _data[key[7:]] = _data.pop(key)
        return _data
