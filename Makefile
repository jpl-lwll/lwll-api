# The AWS KEYS in this Makefile are restricted machine keys to READ ONLY access to our specific S3 bucket
# We will eventually remove these keys and further abstract them back and cycle these out

# Can't source via Makefile
# TODO: figure out progromattic way to do this
# source_envs_dev:
# 	source .envs_dev

# source_envs_staging:
# 	source .envs_staging

# source_envs_prod:
# 	source .envs_prod
# ifneq (,$(wildcard ./.envs_dev))
#     include .envs_dev
#     export
# endif


build_dev:
	docker build \
	--build-arg LWLL_SESSION_MODE=${LWLL_SESSION_MODE} \
	--build-arg AWS_REGION_NAME=${AWS_REGION_NAME}  \
	--build-arg DATASETS_BUCKET=${DATASETS_BUCKET} \
	-t ${ECR_REPO} .

build_staging:
	docker build \
	--build-arg LWLL_SESSION_MODE=${LWLL_SESSION_MODE} \
	--build-arg AWS_REGION_NAME=${AWS_REGION_NAME}  \
	--build-arg DATASETS_BUCKET=${DATASETS_BUCKET} \
	-t ${ECR_REPO} .

build_prod:
	docker build \
	--build-arg LWLL_SESSION_MODE=${LWLL_SESSION_MODE} \
	--build-arg AWS_REGION_NAME=${AWS_REGION_NAME}  \
	--build-arg DATASETS_BUCKET=${DATASETS_BUCKET} \
	-t ${ECR_REPO} .

tag_dev_to_ecr:
	docker tag ${ECR_REPO} ${ECR_REPO_ARN}

tag_staging_to_ecr:
	docker tag ${ECR_REPO} ${ECR_REPO_ARN}

tag_prod_to_ecr:
	docker tag ${ECR_REPO} ${ECR_REPO_ARN}

run_dev:
	docker run -p 5000:5000 ${ECR_REPO}

run_staging:
	docker run -p 5000:5000 ${ECR_REPO}

run_prod:
	docker run -p 5000:5000 ${ECR_REPO}

push_dev_to_ecr:
	docker push ${ECR_REPO_ARN}

push_staging_to_ecr:
	docker push ${ECR_REPO_ARN}

push_prod_to_ecr:
	docker push ${ECR_REPO_ARN}

build_docs:
	( \
	source env/bin/activate; \
	pushd docs; \
	make html; \
	popd; \
	)

push_docs_dev:
	aws s3 cp docs/_build/html/ s3://${S3_DOCS_BUCKET} --recursive --profile ${AWS_SSO_PROFILE_NAME}

push_docs_staging:
	aws s3 cp docs/_build/html/ s3://${S3_DOCS_BUCKET} --recursive --profile ${AWS_SSO_PROFILE_NAME}

push_docs_prod:
	aws s3 cp docs/_build/html/ s3://${S3_DOCS_BUCKET} --recursive --profile ${AWS_SSO_PROFILE_NAME}


# Build and run locally the docker image for local testing
build_run_dev: build_dev run_dev

# All steps needed in a deployment to dev environment after the cloudformation template is already deployed and S3 buckets for docs are created.
# Also note that only those with access to the account linked up to their sso (Only Mark Hoffmann) will be able to properly call these deployment commands
build_deploy_dev: build_dev tag_dev_to_ecr push_dev_to_ecr

build_deploy_staging: build_staging tag_staging_to_ecr push_staging_to_ecr

build_deploy_prod: build_prod tag_prod_to_ecr push_prod_to_ecr

access-key:
	aws-login -r us-east-1 -p $(AWS_SSO_PROFILE_NAME) -a arn:aws:iam::510230590071:role/power_user  

docker-login:
	aws ecr get-login-password --region $(AWS_REGION_NAME) --profile $(AWS_SSO_PROFILE_NAME) | \
    docker login --username AWS --password-stdin $(ECR_REPO_ARN)

login: access-key docker-login