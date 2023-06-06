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

from typing import Any, List

import pandas as pd
from lwll_api.utils.database import database, auth
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from lwll_api.classes.models import DatasetMetadata, Session, Task
from lwll_api.utils.decorators import verify_govteam_secret, verify_user_secret
from lwll_api.utils.logger import get_module_logger

base = APIRouter(
    tags=["base"],
)

authenticated_base = APIRouter(
    tags=["authenticated base"],
    dependencies=[
        Depends(verify_user_secret),
        Depends(verify_govteam_secret)
    ]
)

log = get_module_logger(__name__)


@base.get("/")
def index() -> Any:
    return {'Status': 'Success'}


@base.get("/health")
def health() -> Any:
    return 'ok'


@authenticated_base.get("/list_tasks")
def list_tasks() -> Any:
    """List all tasks

        .. :quickref: Base; List all tasks

        :reqheader user_secret: secret

        **Example request**:

        .. sourcecode:: http

          GET /list_tasks HTTP/1.1
          Accept: application/json

        **Example response**:

        .. sourcecode:: http

          HTTP/1.1 200 OK
          Vary: Accept
          Content-Type: application/json

          {
            "tasks": [
                "problem_test"
            ]
          }
        """
    task_ids = database.get_tasks()
    return {'tasks': task_ids}


@authenticated_base.get("/list_active_sessions")
def list_active_sessions(request: Request) -> Any:
    """List all active sessions

        .. :quickref: Base; List all active sessions

        :reqheader user_secret: secret

        **Example request**:

        .. sourcecode:: http

          GET /list_active_sessions HTTP/1.1
          Accept: application/json

        **Example response**:

        .. sourcecode:: http

          HTTP/1.1 200 OK
          Vary: Accept
          Content-Type: application/json

          {
            "active_sessions": [
                '04N8Oqa7OOfVOpxEoM9W',
                '05hftG8z5QZ9r59prtUF'
            ]
          }
        """
    user_secret = request.headers.get('user_secret', "")
    team = auth.get_user(user_secret)

    active_session_ids = database.get_active_sessions(team)

    return {"active_sessions": active_session_ids}


@authenticated_base.get("/task_metadata/{task_id}")
def get_task_metadata(task_id: str) -> Any:
    """Get the metadata of a particular task. Includes the datasets and budgets.

        .. :quickref: Base; Get task metadata

        :reqheader user_secret: secret

        **Example request**:

        .. sourcecode:: http

          GET /task_metadata/problem_test HTTP/1.1
          Accept: application/json

        **Example response**:

        .. sourcecode:: http

          HTTP/1.1 200 OK
          Vary: Accept
          Content-Type: application/json

          {
            "task_metadata": {
                "adaptation_can_use_pretrained_model": false,
                "adaptation_dataset": "mnist",
                "adaptation_evaluation_metrics": [
                    "accuracy"
                ],
                "adaptation_label_budget": [
                    5,
                    2000,
                    3000
                ],
                "base_can_use_pretrained_model": true,
                "base_dataset": "mnist",
                "base_evaluation_metrics": [
                    "accuracy"
                ],
                "base_label_budget": [
                    3000,
                    6000,
                    8000
                ],
                "problem_id": "problem_test"
            }
          }
        """
    try:
        _task = Task(task_id)
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Error getting task '{task_id}' ({err})")

    return {'task_metadata': _task.to_public_dict()}


@authenticated_base.get("/dataset_metadata/{dataset_id}")
def get_dataset_metadata(dataset_id: str) -> Any:
    """Get the metadata of a particular dataset. Used in particular for external datasets associated
    with tasks.

        .. :quickref: Base; Get dataset metadata

        :reqheader user_secret: secret

        **Example request**:

        .. sourcecode:: http

          GET /dataset_metadata/domain_net-painting HTTP/1.1
          Accept: application/json

        **Example response**:

        .. sourcecode:: http

            HTTP/1.1 200 OK
            Vary: Accept
            Content-Type: application/json

            {
                "dataset_metadata": {
                    "classes": [
                    "lollipop",
                    "apple",
                    "diamond",
                    ...
                    ],
                    "dataset_type": "image_classification",
                    "full_number_of_classes": 345,
                    "full_number_of_samples_test": 22892,
                    "full_number_of_samples_train": 52867,
                    "license_citation": "{{@article{peng2018moment,title={Moment Matching for Multi-Source
                                            Domain Adaptation},author={Peng, Xingchao and Bai, Qinxun and Xia, Xide and Huang,
                                            Zijun and Saenko, Kate and Wang, Bo},journal={arXiv preprint arXiv:1812.01754},year={2018}}}}",
                    "license_link": "http://ai.bu.edu/M3SDA/",
                    "license_requirements": "None",
                    "name": "domain_net-painting",
                    "number_of_channels": 3,
                    "sample_number_of_classes": 345,
                    "sample_number_of_samples_test": 2289,
                    "sample_number_of_samples_train": 5286,
                    "uid": "domain_net-painting"
                }
                }

    """
    try:
        _dataset = DatasetMetadata(dataset_id)
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Error getting dataset '{dataset_id}' ({err})")
    return {'dataset_metadata': _dataset.to_dict()}


@authenticated_base.get("/session_status")
def get_session_status(request: Request) -> Any:
    """Get the metadata of the current session

        .. :quickref: Base; Get session status

        :reqheader user_secret: secret
        :reqheader session_token: session_token

        **Example request**:

        .. sourcecode:: http

          GET /session_status HTTP/1.1
          Accept: application/json

        **Example response**:

        .. sourcecode:: http

          HTTP/1.1 200 OK
          Vary: Accept
          Content-Type: application/json

          {
            Session_Status': {
                'active': 'In Progress',
                'budget_left_until_checkpoint': 3000,
                'current_dataset': {
                    '<See `/dataset_metadata` route for schema>'
                },
                'current_label_budget_stages': [3000, 6000, 8000],
                'date_created': 1573865321000,
                'date_last_interacted': 1573865321000,
                'pair_stage': 'base',
                'task_id': 'problem_test',
                'uid': 'RSsFBzwLFiC0QjpAQqWW',
                'user_name': 'DEMO_TEAM',
                'using_sample_datasets': True
            }
          }
        """
    session_token = request.headers.get('session_token', "")
    if not session_token:
        return {"Error": "No 'session_token' header provided"}

    try:
        sess = Session(session_token)
    except Exception as err:
        log.error(f"Error getting session: {err}")
        raise HTTPException(status_code=400, detail=str(err))
        return {'Error': str(err)}
    else:
        return {"Session_Status": sess.session_status_dict()}


@authenticated_base.get("/seed_labels")
def seed_labels(request: Request) -> Any:
    """Get the initial seed labels for the current dataset you are on.

        .. :quickref: Base; Get seed labels

        :reqheader user_secret: secret
        :reqheader session_token: session_token

        **Example request**:

        .. sourcecode:: http

          GET /seed_labels HTTP/1.1
          Accept: application/json

        **Example response**:

        .. sourcecode:: http

          HTTP/1.1 200 OK
          Vary: Accept
          Content-Type: application/json

          {
            'Labels': [
                {
                    'id': '56847.png',
                    'label': '2'
                },
                {
                    'id': '45781.png',
                    'label': '3'
                },
                {
                    'id': '40214.png',
                    'label': '7'
                },
                {
                    'id': '49851.png',
                    'label': '8'
                },
                {
                    'id': '46024.png',
                    'label': '6'
                },
                {
                    'id': '13748.png',
                    'label': '1'
                },
                {
                    'id': '13247.png',
                    'label': '9'
                },
                {
                    'id': '39791.png',
                    'label': '4'
                },
                {
                    'id': '37059.png',
                    'label': '0'
                },
                {
                    'id': '46244.png',
                    'label': '5'
                }
            ]
          }
    """
    session_token = request.headers.get('session_token', "")
    try:
        sess = Session(session_token)
    except Exception as err:
        log.error(f"Error getting session: {err}")
        raise HTTPException(status_code=400, detail=str(err))
        return {'Error': str(err)}
    log.debug(f"Current Session Metadata on seed labels: {sess.to_dict()}")
    log.debug(f"Current Session `current_stage` on seed labels: {sess.current_stage}")
    if sess.current_stage not in list(range(4)) + list(range(8, 8 + 4)):
        return {'Error': 'You are attempting to request seed labels outside of the first four checkpoints'}
    try:
        labels = sess.get_current_seed_labels()
    except Exception as err:
        log.error(f"Error getting seed labels: {err}")
        return {'Error': str(err)}
    else:
        return {"Labels": labels}


class Labels(BaseModel):
    example_ids: List[str]


@authenticated_base.post("/query_labels")
def get_label(data: Labels, request: Request) -> Any:
    """Get labels for array of image examples

    .. :quickref: Base; Get labels for examples

    PARAMS
    ------
    example_ids: List[str]
        The ids you are requesting labels for

    **Example response**:

        .. sourcecode:: http

          HTTP/1.1 200 OK
          Vary: Accept
          Content-Type: application/json

          {
            'Labels': [
                {
                    'id': '56847.png',
                    'label': '2'
                },
                {
                    'id': '45781.png',
                    'label': '3'
                }
            ],
            'Session_Status': {
                '<See `/session_status` route for schema>'
            }
          }

          HTTP/1.1 400 Error
          Vary: Accept
          Content-Type: application/json

          {
              'Labels': [],
              'Session_stats': {
                  '<See `/session_stats` route for schema>'
              },
              'Error': 'Machine Translation has a cap of requesting 25k translation records at one time.'
          }
    """
    session_token = request.headers.get('session_token', "")
    sess = Session(session_token)
    example_ids: List[str] = data.example_ids

    # Validation for large MT requests
    current_dataset = sess.session_status_dict()['current_dataset']
    dataset_type = current_dataset['dataset_type']
    if dataset_type == 'machine_translation':
        MAX_MT_LOOKUP = 25000
        if len(example_ids) > MAX_MT_LOOKUP:
            raise HTTPException(status_code=400, detail={'Labels': [], 'Session_Status': sess.session_status_dict(
            ), 'Error': 'Machine Translation has a cap of requesting 25k translation records at one time.'})
            return {'Labels': [], 'Session_Status': sess.session_status_dict(),
                    'Error': 'Machine Translation has a cap of requesting 25k translation records at one time.'}
    try:
        df_list = sess.request_labels(example_ids=example_ids)
    except Exception as err:
        log.error(f"Error querying labels: {err}")
        return {'Labels': [], 'Error': str(err)}

    return {"Labels": df_list, "Session_Status": sess.session_status_dict()}


class Predictions(BaseModel):
    predictions: dict


def verify_zsl(sess: Session) -> Any:
    if not sess.zsl:
        raise HTTPException(
            status_code=400,
            detail={
                "Labels": [],
                "Session_Status": sess.session_status_dict(),
                "Error": (
                    "All labels can only be requested for a ZSL session. "
                    "If you want to do ZSL, please pass `ZSL:True` "
                    "in the header when creating a session."
                ),
            },
        )

    if "zsl" not in sess._current_task.optional_sub_tasks:
        raise HTTPException(
            status_code=400,
            detail={
                "Labels": [],
                "Session_Status": sess.session_status_dict(),
                "Error": (
                    "A ZSL session was initiated (session.zsl = True), "
                    "but the session's current task does not support ZSL."
                ),
            },
        )

    # Check that the specific task has ZSL supported
    log.debug(f"pair stage is {sess.pair_stage}")
    log.debug(f"optional subtasks: {sess._current_task.optional_sub_tasks}")
    if sess.pair_stage == "base" and "zsl" not in sess._current_task.optional_sub_tasks:
        raise HTTPException(
            status_code=400,
            detail={
                "Labels": [],
                "Session_Status": sess.session_status_dict(),
                "Error": "ZSL is not supported for the base dataset of this task.",
            },
        )
    elif (
        sess.pair_stage == "adaptation" and "adaptation_zsl" not in sess._current_task.optional_sub_tasks
    ):
        raise HTTPException(
            status_code=400,
            detail={
                "Labels": [],
                "Session_Status": sess.session_status_dict(),
                "Error": "ZSL is not supported for the adaptation dataset of this task.",
            },
        )
    return {}, 200


@authenticated_base.get("/get_seen_labels")
def get_seen_labels(request: Request) -> Any:
    """Get labels for array of image examples

    .. :quickref: Base; Get labels for examples

    PARAMS
    ------
    example_ids: List[str]
        The ids you are requesting labels for

    **Example response**:

        .. sourcecode:: http

          HTTP/1.1 200 OK
          Vary: Accept
          Content-Type: application/json

          {
            'Labels': [
                {
                    'id': '56847.png',
                    'label': '2'
                },
                {
                    'id': '45781.png',
                    'label': '3'
                }
            ],
            'Session_Status': {
                '<See `/session_status` route for schema>'
            }
          }

          HTTP/1.1 400 Error
          Vary: Accept
          Content-Type: application/json

          {
              'Labels': [],
              'Session_stats': {
                  '<See `/session_stats` route for schema>'
              },
              'Error': 'Machine Translation has a cap of requesting 25k translation records at one time.'
          }
    """
    session_token = request.headers.get("session_token", "")
    try:
        sess = Session(session_token)
    except Exception as err:
        log.error(f"Error getting session: {err}")
        raise HTTPException(status_code=400, detail=str(err))

    if sess.active != "In Progress":
        raise HTTPException(
            status_code=400, detail={"Error": "This session is no longer active."}
        )
    res_dict, res_code = verify_zsl(sess)
    if res_code == 400:
        return res_dict, res_code
    df_list = sess.get_seen_labels()

    return {"Labels": df_list, "Session_Status": sess.session_status_dict()}


@authenticated_base.get("/get_unseen_ids")
def get_unseen_ids(request: Request) -> Any:
    """For ZSL, return a list of filenames (ids) that belong to the unseen classes."""
    session_token = request.headers.get("session_token", "")
    try:
        sess = Session(session_token)
    except Exception as err:
        log.error(f"Error getting session: {err}")
        raise HTTPException(status_code=400, detail=str(err))

    if sess.active != "In Progress":
        raise HTTPException(
            status_code=400, detail={"Error": "This session is no longer active."}
        )
    res_dict, res_code = verify_zsl(sess)
    if res_code == 400:
        return res_dict, res_code
    file_list = sess._get_unseen_ids()

    return {"ids": file_list, "Session_Status": sess.session_status_dict()}


@authenticated_base.post("/submit_predictions")
def submit_predictions(data: Predictions, request: Request) -> Any:
    """Submit predictions file for scoring and advancing to the next stage. Predictions for image classification can be
    submitted as hard assignment or probabilistic predictions. Probabilistic predictions must sum to 1 for each observation.

    .. :quickref: Base; Submit prediction labels.

    :reqheader user_secret: secret
    :reqheader session_token: session_token
    :<json List[dict] predictions: submission predictions list

    **Example Submission format for Image Classification Hard Assignment**

    ========  =====
    id        class
    ========  =====
    6831.png  '3'
    1186.png  '9'
    8149.png  '6'
    4773.png  '3'
    3752.png  '10'
    ========  =====

    **Example Submission format for Image Classification Probabilistic Predictions**
    The following example is based on MNIST. The columns are the string names of the classes.

    ========  ====  ====  ====  ====  ====  ====  ====  ====  ====
    id        '1'   '2'   '3'   '4'   '5'   '6'   '7'   '8'   '9'
    ========  ====  ====  ====  ====  ====  ====  ====  ====  ====
    6831.png  0.01  0.09  0.0   0.25  0.65  0.0   0.0   0.0   0.0
    1186.png  0.15  0.0   0.20  0.25  0.05  0.35  0.0   0.0   0.0
    8149.png  0.80  0.10  0.0   0.05  0.0   0.05  0.0   0.0   0.0
    4773.png  0.0   0.7   0.0   0.15  0.15  0.0   0.0   0.0   0.0
    3752.png  0.0   0.10  0.0   0.0   0.0   0.9   0.0   0.0   0.0
    ========  ====  ====  ====  ====  ====  ====  ====  ====  ====

    **Example Submission format for Object Detection**

    =========  ====================================  ======  ==========
    id         bbox                                  class   confidence
    =========  ====================================  ======  ==========
    img_1.png  '<x_min>, <y_min>, <x_max>, <y_max>'  'face'  0.83
    img_1.png  '<x_min>, <y_min>, <x_max>, <y_max>'  'face'  0.83
    img_2.png  '<x_min>, <y_min>, <x_max>, <y_max>'  'face'  0.83
    =========  ====================================  ======  ==========

    **Example Submission format for Video Classification**

    ============  =======
    segment_id     class
    ============  =======
        2678        '0'
        1995        '4'

    PARAMS
    ------
    predictions: List[dict]
        The predictions you are submitting. All ids must be accounted for.
        This should be sent up simply as a `df.to_dict()` output with the original Dataframe
        having the columns
    """
    session_token = request.headers.get('session_token', "")

    try:
        sess = Session(session_token)
    except Exception as err:
        log.error(f"Error getting session: {err}")
        raise HTTPException(status_code=400, detail=str(err))
        return {'Error': str(err)}

    if sess.active != 'In Progress':
        raise HTTPException(status_code=400, detail='This session is no longer active.')
        return {'Error': 'This session is no longer active.'}

    try:
        preds = pd.DataFrame(data.predictions)
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))
        return {'Error': err}

    is_probabilistic = False
    warning = ""
    # Check if the predictions are in hard assignment or probabilistic format
    if sess._current_task.problem_type == "image_classification" or sess._current_task.problem_type == "video_classification":
        if "class" not in data.predictions.keys():
            # Validate that all classes are present and that predictions sum to 1 (or very close)
            is_valid, warning = sess._valid_probabilistic(df=preds)
            if not is_valid:
                # Reformat as hard assignment
                class_cols = preds.drop(columns='id').columns
                if "video_id" in preds.columns:
                    class_cols = preds.drop(columns=["id", "video_id", "end_frame", "start_frame"]).columns
                class_col = pd.Series(preds.loc[:, class_cols].idxmax(axis=1), name="class")
                reformat_df = pd.merge(class_col, preds['id'], left_index=True, right_index=True)
                preds = reformat_df
            else:
                is_probabilistic = True
    try:
        sess.submit_checkpoint(df=preds, probabilistic=is_probabilistic)
    except Exception as err:
        log.error(f"Error submitting checkpoint: {err}")
        raise HTTPException(status_code=400, detail=str(err))
        return {'Error': str(err)}
    else:
        if warning:
            return {"Warning": warning, "Session_Status": sess.session_status_dict()}
        return {"Session_Status": sess.session_status_dict()}


@authenticated_base.post("/submit_standard_zsl_predictions")
def submit_standard_zsl_predictions(data: Predictions, request: Request) -> Any:
    """Submit predictions from a standard ZSL model (i.e., a model that only predicts
    classes in the set of unseen classes) on the full dataset. We will only calculate
    metrics on the images in the test set that belong to the unseen classes.

    Args:
        data (Predictions): The predictions you are submitting. All ids must be accounted for.
            This is the same as the `submit_predictions` endpoint, except that the
            `class` column should contain only unseen classes predicted by the standard ZSL model.
        request (Request): The request object from FastAPI. This is used to get the session token.

    Returns:
        response (dict): A dictionary containing the session status.
    """
    session_token = request.headers.get("session_token", "")

    try:
        sess = Session(session_token)
    except Exception as err:
        log.error(f"Error getting session: {err}")
        raise HTTPException(status_code=400, detail=str(err))
        return {'Error': str(err)}  # Need to figure out if we can remove these returns

    if sess.active != "In Progress":
        raise HTTPException(status_code=400, detail="This session is no longer active.")
        return {'Error': 'This session is no longer active.'}
    try:
        preds = pd.DataFrame(data.predictions)
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))
        return {'Error': err}

    # raise exception if we're not in a ZSL session
    if "zsl" not in sess._current_task.optional_sub_tasks:
        raise HTTPException(
            status_code=400, detail="This session is not a ZSL session."
        )

    if "class" not in data.predictions.keys():
        # Validate that all classes are present and that predictions sum to 1 (or very close)
        is_valid, warning = sess._valid_probabilistic(df=preds)
        if not is_valid:
            # Reformat as hard assignment
            class_cols = preds.drop(columns="id").columns
            if "video_id" in preds.columns:
                class_cols = preds.drop(
                    columns=["id", "video_id", "end_frame", "start_frame"]
                ).columns
            class_col = pd.Series(preds.loc[:, class_cols].idxmax(axis=1), name="class")
            reformat_df = pd.merge(
                class_col, preds["id"], left_index=True, right_index=True
            )
            preds = reformat_df
    else:
        warning = ""

    try:
        sess.submit_checkpoint(df=preds, probabilistic=False, standard_zsl=True)
    except Exception as err:
        log.error(f"Error submitting checkpoint: {err}")
        raise HTTPException(status_code=400, detail=str(err))
        return {'Error': str(err)}
    else:
        if warning:
            return {"Warning": warning, "Session_Status": sess.session_status_dict()}
        return {"Session_Status": sess.session_status_dict()}


@authenticated_base.post("/submit_UDA_predictions")
def submit_UDA_predictions(data: Predictions, request: Request) -> Any:
    """Submit predictions file before first checkpoint in adaptation stage
    for scoring unsupervised domain adaptation

    .. :quickref: Base; Submit prediction labels.

    :reqheader user_secret: secret
    :reqheader session_token: session_token
    :<json List[dict] predictions: submission predictions list

    **Example Submission format for Image Classification**

    ========  =====
    id        class
    ========  =====
    6831.png  '3'
    1186.png  '9'
    8149.png  '6'
    4773.png  '3'
    3752.png  '10'
    ========  =====

    **Example Submission format for Object Detection**

    =========  ====================================  ======  ==========
    id         bbox                                  class   confidence
    =========  ====================================  ======  ==========
    img_1.png  '<x_min>, <y_min>, <x_max>, <y_max>'  'face'  0.83
    img_1.png  '<x_min>, <y_min>, <x_max>, <y_max>'  'face'  0.83
    img_2.png  '<x_min>, <y_min>, <x_max>, <y_max>'  'face'  0.83
    =========  ====================================  ======  ==========

    **Example Submission format for Video Classification**

    ============  =======
    segment_id     class
    ============  =======
        2678        '0'
        1995        '4'

    PARAMS
    ------
    predictions: List[dict]
        The predictions you are submitting. All ids must be accounted for.
        This should be sent up simply as a `df.to_dict()` output with the original Dataframe
        having the columns
    """
    session_token = request.headers.get('session_token', "")

    try:
        sess = Session(session_token)
    except Exception as err:
        log.error(f"Error getting session: {err}")
        raise HTTPException(status_code=400, detail=str(err))

    if sess.active != 'In Progress':
        raise HTTPException(status_code=400, detail='This session is no longer active.')

    # if sess.pair_stage != "adaptation" or sess.budget_used != 0:
    #     raise HTTPException(
    #         status_code=400,
    #         detail='This endpoint can only be used to submit unsupervised domain adaptation predictions when transitioning from base to adaptation '
    #         'before requesting any adaptation labels')

    # if sess.domain_adaptation_submitted:
    #     return {'Warning': 'Predictions for unsupervised domain adaptation have already been scored.'
    #             'This submission will not be scored.'}

    is_probabilistic = False
    preds = pd.DataFrame(data.predictions)

    warning = ""
    # Check if the predictions are in hard assignment or probabilistic format
    if sess._current_task.problem_type == "image_classification" or sess._current_task.problem_type == "video_classification":
        if "class" not in data.predictions.keys():
            # Validate that all classes are present and that predictions sum to 1 (or very close)
            is_valid, warning = sess._valid_probabilistic(df=preds)
            if not is_valid:
                # Reformat as hard assignment
                class_cols = preds.drop(columns='id').columns
                if "video_id" in preds.columns:
                    class_cols = preds.drop(columns=["id", "video_id", "end_frame", "start_frame"]).columns
                class_col = pd.Series(preds.loc[:, class_cols].idxmax(axis=1), name="class")
                reformat_df = pd.merge(class_col, preds['id'], left_index=True, right_index=True)
                preds = reformat_df
            else:
                is_probabilistic = True
    try:
        sess.submit_checkpoint(df=preds, UDA=True, probabilistic=is_probabilistic)
    except Exception as err:
        log.error(f"Error submitting UDA checkpoint: {err}")
        raise HTTPException(status_code=400, detail=str(err))
    else:
        if warning:
            return {"Warning": warning, "Session_Status": sess.session_status_dict()}
        return {"Session_Status": sess.session_status_dict()}


@authenticated_base.get("/skip_checkpoint")
def skip_checkpoint(request: Request) -> Any:
    """Skip up to 2 checkpoints

    .. :quickref: Base; Skip Checkpoint.

    :reqheader user_secret: secret
    :reqheader session_token: session_token

    """
    session_token = request.headers.get('session_token', "")
    try:
        sess = Session(session_token)
    except Exception as err:
        log.error(f"Error getting session: {err}")
        raise HTTPException(status_code=400, detail=str(err))
        return {'Error': str(err)}

    if sess.active != 'In Progress':
        raise HTTPException(status_code=400, detail='This session is no longer active.')
        return {'Error': 'This session is no longer active.'}

    sess.skip_checkpoint()

    return {"Session_Status": sess.session_status_dict()}


class DeactivateToken(BaseModel):
    session_token: str


@authenticated_base.post("/deactivate_session")
def deactivate_session(data: DeactivateToken) -> Any:
    """Deactivate session
        .. :quickref: Base; Deactivate Session

    PARAMS
    ------
    session_token: str
        The session token
    """
    try:
        sess = Session(data.session_token)
    except Exception as err:
        log.error(f"Error getting session: {err}")
        raise HTTPException(status_code=400, detail=str(err))
        return {'Error': str(err)}
    sess.deactivate_session()

    return {'Status': 'Deactivated'}
