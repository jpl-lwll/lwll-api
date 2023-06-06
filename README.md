# LwLL API

REST API implementation that houses all business logic for sessions and querying capabilities. Route documentation is provided via FastAPI Swagger UI and ReDoc and can be accessed by running the API locally and going to `http://localhost:5000/docs` or `http://localhost:5000/redoc` respectively.

A walkthrough of the full API flow can be found in the `REST_API_Example_Walkthrough.ipynb` notebook at the root of this repository.

## Environments

This is being developed as a Python FastAPI application and being deployed to a serverless stack on AWS via a Cloudformation stack using Fargate and ALB. Some important points to note regarding the different environments.

### Local Testing

You can test the API locally by using a virtual environment, installing the appropriate dependencies, and launching via `uvicorn`.

***Note:** The local environment is mainly for use by developers of the API and relies on a GOVTEAM secret and temporary AWS credentials to fully work properly*

#### Setting up the API
```py
virtualenv -p python3.7 env
source env/bin/activate

pip install -r requirements.txt

LWLL_LOCAL_DB_FOLDER=<LWLL_LOCAL_DB_FOLDER> DATASETS_BUCKET=<DATASETS_BUCKET> LWLL_STORAGE_MODE=<LWLL_STORAGE_MODE> AWS_REGION_NAME=<AWS_REGION> GOVTEAM_SECRET=<GOVTEAM_SECRET> TEAM_SECRET=<TEAM_SECRET> uvicorn --reload --log-level debug main:app --port 5000
```

# Setup

## Environment Variables

In order to set up a working cloned copy of this repository, there are a series of environment variables you must have to authenticate to the appropriate services. You can source these environment variables however you see fit. They are used within the code as well as the Makefile.


| Environment Variable | Description                                                                                                                                                                                                                                                                                           |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| AWS_SSO_PROFILE_NAME | Name of temporary AWS SSO Profile                                                                                                                                                                                                                                                                     |
| AWS_REGION_NAME      | Region name                                                                                                                                                                                                                                                                                           |
| DATASETS_BUCKET      | Bucket name where the labels exist                                                                                                                                                                                                                                                                    |
| LWLL_SESSION_MODE    | Session mode --> Enum of one of `['DEV', 'SANDBOX', 'EVAL']`                                                                                                                                                                                                                                          |
| ECR_REPO             | ECR Repo name with whatever tag is desired. Ex. lwll_api_dev:latest (created prior to push)                                                                                                                                                                                                           |
| ECR_REPO_ARN         | ECR Repo ARN with whatever tag is desired. Ex. <account_number>.dkr.ecr.us-east-1.amazonaws.com/lwll_api_dev:latest (created prior to push)                                                                                                                                                           |
| LWLL_LOCAL_DB_FOLDER | Optional path to a folder containing a `DatasetMetadata.json`, `users.json`, and `dev_Task.json` file (see [test folder](/tests/test_data/local_db)). Session and Task json files will be created here if they do not exist. If this is not specified, a firebase creds file is required (see below). |
| LWLL_STORAGE_MODE    | Storage mode --> Enum of one of `['LOCAL', 'S3']`. If local is selected the `DATASETS_BUCKET` variable should point to the directory where the resources are located.                              |


These variables can either be prepended to the API command or stored in environment files named `.env_dev` or `.env_prod` in the root of the repo (those names are already ignored by git). Then run `source .env_dev` before starting the API locally or calling the commands specified in `DEPLOYMENT.MD`.


## Firebase

If a local database folder is not specified, a credentials file is required for authenticating with Firebase. Place this credentials file under the relative path:
```
lwll_api/classes/service_accounts/firebase_creds.json
```

## Initial Cloud Setup

In order to get the deployment up and running on the cloud here is a brief checklist of items that must be completed:

1. Setup up an account on AWS
    1. Create a VPC with two subnets
    2. Create a S3 Bucket to store development datasets
    3. Create a S3 Bucket to store evaluation datasets
    4. Create an ACM certificate and validate the hosting of the domain whereever it is registered
    5. Create a S3 Bucket for documentation hosting for dev and prod environments
    6. Create a ECR repo for both dev and prod container deployments

2. Setup an account on Firebase
    1. Generate a service account and rename in to `firebase_creds.json` to be placed in the appropriate place referenced above

3. Create your `.env_dev` and `.env_prod` files and fill them in with all of the appropriate env variables
4. Follow `DEPLOYMENT.md` to get the API deployed onto your environment

**Note that in order to interact with the API as demonstrated in the `REST_API_Example_Walkthrough.ipynb` file, you must also:**

1. Process datasets as demonstrated in the `dataset_prep` git repository
2. Create `Problem Definitions` in Firebase as seen in the `CreateTasks.ipynb` file in the `lwll_admin_scripts` git repository. (This functionality will eventually become part of the Frontend)
3. Create `Users` in Firebase for people to authenticate as when interacting with the api as seein in the `UserAccounts.ipynb` file in the `lwll_admin_scripts` git repository. (This functionality will eventually become part of the Frontend)
