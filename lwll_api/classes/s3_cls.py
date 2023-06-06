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

import abc
import pandas as pd
from lwll_api.utils.logger import get_module_logger
import boto3
from botocore.exceptions import ClientError
import os
import io


class DataHandler:

    @abc.abstractmethod
    def read_path(self, path: str) -> pd.DataFrame:
        """
        Reads the provided path and return the file as a DataFrame

        Parameters
        ----------
        inputs:
            path: str
                The path of the file to be read
        outputs:
            The file contests in a DataFrame
        """

    @abc.abstractmethod
    def save_prediction_df(self, df: pd.DataFrame, path: str) -> None:
        """
        Saves a DataFrame to the provided path

        Parameters
        ----------
        inputs:
            df: DataFrame
                DataFrame to be stored
            path: str
                The path where the DataFrame will be stored
        """


class S3_cls(DataHandler):

    def __init__(self) -> None:
        self.bucket_name = os.environ.get('DATASETS_BUCKET')
        self.raw_predictions_bucket = 'lwll-prediction-archive'
        self.log = get_module_logger(__name__)
        self.log.info(f"DATASETS_BUCKET set to {self.bucket_name}")

        try:
            # self.session = boto3.Session(profile_name='lwll_creds')
            # New deployment scheme, we only need S3 access
            self.log.info("SERVERTYPE set to %s" % os.environ.get('SERVERTYPE'))
            if os.environ.get('SERVERTYPE') == 'REMOTE':
                self.session = boto3.Session()
            else:
                self.session = boto3.Session(profile_name='saml-pub')

        except Exception as e:
            self.log.error(e)
            self.session = boto3.Session()
        self.s3 = self.session.client('s3')

    def read_path(self, path: str) -> pd.DataFrame:
        self.log.info(f"Reading label file from path: {path} in bucket {self.bucket_name}")
        try:
            obj = self.s3.get_object(Bucket=self.bucket_name, Key=path)
            body = obj['Body'].read()
            df = pd.read_feather(io.BytesIO(body))
            return df

        except FileNotFoundError as e:
            self.log.error(e)
            self.log.debug(f"{path} not found in bucket {self.bucket_name}")

        except TypeError as e:
            self.log.error(e)
            self.log.debug(f"bucket is of type {type(self.bucket_name)} and key is of type {type(path)}")
            self.log.debug(f"{path} not found in bucket {self.bucket_name}")

        except ClientError as e:
            self.log.error(e)
            self.log.debug(f"S3 Client error occurred when requesting {self.bucket_name}/{path}")
            raise Exception(str(e))

        return

    def save_prediction_df(self, df: pd.DataFrame, path: str) -> None:
        self.log.info(f"Saving DataFrame to path: {path}")
        buffer = io.BytesIO()
        df.reset_index(drop=True).to_feather(buffer)
        buffer.seek(0)
        self.s3.put_object(Body=buffer, Bucket=self.raw_predictions_bucket, Key=path)
        return


class LocalData(DataHandler):

    def __init__(self) -> None:
        self.log = get_module_logger(__name__)
        self.log.info(f"Local storage")

    def read_path(self, path: str) -> pd.DataFrame:
        self.log.info(f"Reading label file from path: {path} from local")
        try:
            df = pd.read_csv(path)
            return df

        except FileNotFoundError as e:
            self.log.error(e)
            self.log.debug(f"{path} not found")

        return

    def save_prediction_df(self, df: pd.DataFrame, path: str) -> None:
        self.log.info(f"Saving DataFrame to path: {path}")
        df_to_save = df.reset_index(drop=True)
        df_to_save.to_csv(path)
        return

# Default to LOCAL
storage_mode = os.environ.get('LWLL_STORAGE_MODE', 'S3')

if storage_mode == 'S3':
    s3_operator = S3_cls()
elif storage_mode == 'LOCAL':
    s3_operator = LocalData()
else:
    raise ValueError("Storage mode f{storage_mode} not supported")
