# Deployment

## Post CI/CD Automation

We have 3 environments of `dev`, `staging`, and `prod`. The code on each one of the branches is hooked up to be run with a CD pipeline for the actual deployment. We trigger the appropriate merging of branches and versioning tags, then the CI will run on Gitlab for style checks and tests, then the repo gets mirrored onto AWS infrastructure and CodePipeline does the proper deployment commands authenticated within the ecosystem.

### Staging Release

**Note: TODO BEFORE PROMOTING TO STAGING**
Before promoting to staging, you must update the `HISTORY.md` file with the updates since the last version.

To promote from `dev` to `staging` we call our script with secondary arguments of `major_update`, `minor_update`, or `bug_fix`. These secondary options control the tag on the repo which gets applied on the staging branch code. For example, a `minor_update` argument would take the version `0.4.3` -> `0.5.0`. Where `0.4.3` is the theoretical last tag on the repository.

In addition, we can pass the secondary argument of `version_force` with the additional third argument of a version following the format `<major>.<minor>.<errata>`.

Example Staging Minor Release

```
./promote_release.sh staging minor_update
```

Example Staging Version Force Release

```
./promote_release.sh staging version_force 0.5.0
```

### Prod Release

To promote from `staging` to `prod` we call our script with simply the command

```
./promote_release.sh prod
```

## Before CI/CD Automation (These are the manual steps for deployment in the case where we want to change the automation)

In order to deploy the API on infrastructure, you have to do 4 steps. The fourth step will eventually go away once CodePipeline is fully configured.

1. Regenerate the proper credentials with the `access-key-generator` repo: https://github.jpl.nasa.gov/cloud/access-key-generator
2. Login in AWS ECR: `eval $(aws ecr get-login --no-include-email --profile jpl-sso --region us-east-1 | sed 's|https://||')`

You will see the following message on success

```
WARNING! Using --password via the CLI is insecure. Use --password-stdin.
Login Succeeded
```

3. Build and deploy the Docker image to ECR and update Docs: `make build_deploy_dev`
4. Go into ECS on the AWS Console and navigate to `Tasks`. You want to select all `Tasks` and Stop them. Wait approximately 30 seconds and the task will start to pull and provision the new Docker image.
