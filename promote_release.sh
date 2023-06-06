#!/bin/bash

#
# This script is to take care of release promotions i.e. `devel` -> `staging` and `staging` -> `master`
# When going from `devel` -> `staging` we also apply a tag on the push
#
# Valid primary arguments to calling this script are: `staging`, `prod`
#
# Valid secondary arguments when the primary argument is `staging` are: `bug_fix`, `minor_update`, `major_update`, 'version_force'
#
#
#
#    ****Example Staging Release****
#
#    ./promote_release.sh staging bug_fix
#
#    **Staging Release where we want to force a specific version**
#
#    ./promote_release.sh staging version_force 1.4.0
#
#
#    ****Example Production Release****
#
#    ./promote_release.sh prod
#
#

# Primary argument validation
if [[ $1 =~ ^(staging|prod)$ ]]
then
    echo 'Promoting lwll_api to -->' $1
    
else
    echo 'Bad input command, must be either `staging` or `prod`...'
    echo 'Exiting process...'
    exit 1
fi

if [ $1 = "staging" ]
then
    # Secondary argument validation
    if [[ $2 =~ ^(bug_fix|minor_update|major_update|version_force)$ ]]
    then
        echo 'Promoting lwll_api to -->' $2
        
    else
        echo 'Bad input command, must be either `bug_fix`, `minor_update`, `major_update`, or `version_force`...'
        echo 'Exiting process...'
        exit 1
    fi

    # Get the highest tag number
    git checkout master
    git pull
    if VERSION=$(git describe --tags $(git rev-list --tags --max-count=1))
    then
        echo ''
    else
        VERSION='0.0.0'
    fi
    VERSION=${VERSION:-'0.0.0'}

    # Get number parts
    MAJOR="${VERSION%%.*}"; VERSION="${VERSION#*.}"
    MINOR="${VERSION%%.*}"; VERSION="${VERSION#*.}"
    PATCH="${VERSION%%.*}"; VERSION="${VERSION#*.}"

    echo 'PREVIOUS TAG: ' "$MAJOR.$MINOR.$PATCH"

    # Bug Fix logic
    if [ $2 = "bug_fix" ]
    then
        echo "Bug Fixing..."
        # Increase version
        PATCH=$((PATCH+1))
    fi

    # Minor Update Logic
    if [ $2 = "minor_update" ]
    then
        echo "Minor Updating..."
        # Increase version
        MINOR=$((MINOR+1))
        PATCH=$((0))
    fi

    # Major Update Logic
    if [ $2 = "major_update" ]
    then
        echo "Major Updating..."
        # Increase version
        MAJOR=$((MAJOR+1))
        MINOR=$((0))
        PATCH=$((0))
    fi

    if [ $2 = "version_force" ]
    then
        echo 'Forcing Version... ' "$3"
    fi

    if [[ $2 =~ ^(bug_fix|minor_update|major_update)$ ]]
    then
        # Reassemble
        TAG="$MAJOR.$MINOR.$PATCH"
        echo 'NEW TAG: ' $TAG
    else
        TAG="$3"
        echo 'NEW TAG: ' $TAG
    fi
    

    # Making sure we have the latest changes in devel
    echo 'Step 1: Getting latest changes from `devel`' &&
    git checkout devel &&
    git pull &&
    echo '' &&

    # Making sure we have the latest changes in staging
    echo 'Step 2: Getting latest changes from `staging`' &&
    git checkout staging &&
    git pull &&
    echo '' &&

    # Merging devel into staging
    echo 'Step 3: Merging `devel` into `staging`' &&
    git merge devel -X theirs --no-edit &&

    # Tagging our Release
    echo 'Step 4: Tagging Release on `staging` and pushing' &&
    git tag -a $TAG -m $TAG &&
    git push origin $TAG &&
    git push &&

    # Finishing up
    echo '🚀🚀🚀 The latest changes are being deployed to staging via CI for Staging... 🚀🚀🚀' &&
    echo 'RELEASE VERSION:' $TAG
fi


if [ $1 = "prod" ]
then
    # Making sure we have the latest changes in staging
    echo 'Step 1: Getting latest changes from `staging`' &&
    git checkout staging &&
    git pull &&
    echo '' &&

    # Making sure we have the latest changes in master
    echo 'Step 2: Getting latest changes from `master`' &&
    git checkout master &&
    git pull &&
    echo '' &&

    # Merging staging into master
    echo 'Step 3: Merging `staging` into `master`' &&
    git merge staging -X theirs --no-edit &&

    # Pushing changes up to staging
    echo 'Step 4: Pushing `master`' &&
    git push &&
    echo '🚀🚀🚀 The latest changes are being deployed to prod via CI for PROD... 🚀🚀🚀'
fi
