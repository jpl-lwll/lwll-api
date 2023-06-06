## Next (Update this to the appropriate version when doing a release)

### Enhancements

- Add new metric Precision at K (50) and Average Precision at K (50)

### Bugfixes

-

### Other

-

## 2.0.2

### Enhancements

-

### Bugfixes

- Set probabilities metrics to None if predictions with probabilities are not provided.

### Other

-

## 2.0.1

### Enhancements

- Add support for UDA for base datasets.

### Bugfixes

- Add validation to not allow UDA request after labels has been requested.


## 2.0.0

### Enhancements

- Switches from Flask to FastAPI (inherent speed enhancements)
- More transparent error handling and logging improvements
- General code cleanup and refactoring
- Adds Sentry integration
- Adds a new endpoint called `submit_standard_zsl_predictions` that allows performers to submit predictions using a model trained to perform standard ZSL. This means that the model only outputs classes in the `unseen_classes` list. The backend will ignore any predictions for test images that belong to the seen classes. Note that this endpoint is optional, and needs to occur before the generalized ZSL task is submitted using the `submit_predictions` endpoint. This is because the submit_predictions endpoint will set the current session's status to done, which will prevent the performer from submitting predictions for the standard ZSL task after `submit_predictions` is called.
- `get_seen_labels`: Returns a map of filenames to class labels for training images that belong to the seen classes
- `get_unseen_ids`: Returns a list of filenames (but NO class labels) for training images that belong to the unseen classes.
- Not really an endpoint, but the dataset metadata for ZSL tasks come with a `zsl_description` field that contains natural language descriptions of _unseen_ classes.
- Recall score using scikit-learn's implementation (lwll_api/classes/metrics.py)
- Calculate accuracy (top-1 and top-5), average per-class recall and ROC_AUC for seen, unseen, and all classes (lwll_api/classes/models.py)

### Bugfixes

- Handles errors occuring from non-standard or incomplete datasets in dev causing seed labels or other fields to be invalid
- Fixed an issue where the backend depended on DatasetMetdata's `zsl_description` to get the set of unseen classes. This problem manifested after we added descriptions for the seen classes in `zsl_description` in addition to the unseen classes. The backend now uses DatasetMetadata.seen_classes and DatasetMetadata.unseen_classes to get the set of seen and unseen classes. Thus, each dataset that is intended to be used as a ZSL dataset should include: `zsl_description`, `seen_classes` and `unseen_classes`.

### Other

- Auto-generated documentation with examples now available at `{{API}}/docs` or `{{API}}/redoc`
- Test cases now use the FastAPI TestClient, which is a mock client and does not make external requests
- Raw prediction files name should contain checkpoint number [#135](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/135) and MR ![81](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/81)
- Added ZSL walkthrough to the example notebook. The lwll_api commit history also contains a sample script that Edwin wrote for ZSL: https://gitlab.lollllz.com/lwll/lwll_api/-/commit/9044763193d4e42997d28d9013d98f5c464bac3f

## 1.1.3

### Enhancements

-

### Bugfixes

- Fixes DynamoDB Label Querying Error. See ![69](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/69)

### Other

-

## 1.1.2

### Enhancements

-

### Bugfixes

- Fix id alignment for top_5_accuracy. See [!68](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/68)

### Other

-

## 1.1.0

### Enhancements

- Speedup mAP calculations. See [!65](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/65)
- Exposing metrics after the 8th checkpoint. See [#117](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/117) and MR [!64](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/64)

### Bugfixes

- Add fail-safes for all new metrics. See MR [!66](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/66)

### Other

-

## 0.4.2
### Enhancements

- Add endpoint for unsupervised domain adaptation predictions `submit_UDA_predictions`
- Added test data and tests for seed labels and UDA
- Added option to skip checkpoint. See [#115](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/115) and MR [!60](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/60)
- Added probabilistic predictions. See [#8](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/8) and MR [!59](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/59)
- Added accuracy weighted by class. See [#113](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/113) and MR [!59](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/59)

### Bugfixes

- Updated Task object to retrieve 'uda_base_to_adapt_overlap_ratio' and 'uda_adapt_to_base_overlap_ratio' fields from
  firebase. See issue [#109](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/109) and MR [!57 ](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/57)
- Return None from labels when the class is an empty string
- Decrement the budget by the number of labels returned instead of requested

### Other

-

## 0.4.1

### Enhancements

-   Added support for video classification tasks.

### Bugfixes


### Other

## 0.4.0

### Enhancements

-   The first four checkpoints are sized at 1, 2, 4, and 8 labels per class.
-   Removed `secondary_seed_labels` endpoint
-   `seed_labels` can now be called for the first 4 checkpoints to obtain seed labels deterministically
-   `seed_labels` are retrieved from FireBase Task
-   Internal variable `budget_used` resets when switching from base to adaptation.

### Bugfixes


### Other


## 0.3.0 / 0.3.1

### Enhancements

-   Increased Load Balancer timeout to 1000 seconds from 60 seconds [#87](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/87)
-   Validation on machine translation queries. Queries now have a max of 25k translations in any given single call or else we return an empty response with status code 400. [#87](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/87)
-   Fix for asyncio ensure futures loop bug [#89](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/89)
-   Validation around not allowing prediction submissions on Sessions that are not `In Progress` [#75](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/75) [!35](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/35)
-   Added extra `govteam_secret` authentication for evaluation prod endpoint
-   Added string representation of sacre bleu to metrics dict in sessions [#92](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/92) [!37](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/37)
-   Started saving out the prediction submissions to S3 for archiving purposes [#18](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/18)[!38](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/38)

### Bugfixes

-   Fixed MT sacre bleu calculation bug where dataframes were not getting aligned properly before calcuating score [#92](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/92) [!37](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/37)
-   Cannot request duplicate labels and enforcing budget for requesting labels based on example ids requested not labels returned
-   Adds additional images for object detection seed_labels and seconday_seeds labels if full budget not used

### Other

-

## 0.2.3

### Enhancements

-   Using sacrebleu score instead of nltk bleu score for machine translation metric [#43](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/43) [!33](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/33)

### Bugfixes

-   Safety around object detection mAP metric when a dataset might have a float as a bounding box coordinate. This will be better validated on dataset processing output, but this is a temporary fix that doesn't hurt for now [#86](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/86)

### Other

-   Added `CONTRIBUTING.md` guide [!32](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/32)

## 0.2.0

### Enhancements

-   More consistent naming for datasets in whitelists [#60](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/60)
-   Support for machine translation [#66](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/66) [!24](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/24)
-   Query for labels explicit return keys [#73](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/73) [!26](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/26) [#57](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/57)
-   Removed outdated `data_url` field from api docs [#68](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/68) [!28](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/28)
-   Route to see all of the classes of a particular dataset [#65](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/65) [!27](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/27)

### Bugfixes

-   Timeout errors fixed for listing tasks [#62](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/62)
-   Machine Translation timeout errors [#37](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/37)

### Other

-   Automation around environment promotion process [#70](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/70) [!25](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/25)
-   Doc generation is synced up automatically to mirror build deploy [#74](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/74) [!29](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/29)

## 0.1.0/0.1.1

### Enhancements

-   Adding a Staging environment for stable releases `https://api-staging.lollllz.com`
    [!23](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/23)

-   Budget Function to generate checkpoints based on dataset size (logarithmic)
    [#38](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/38)
    [!2](https://gitlab.lollllz.com/lwll/lwll_admin_scripts/-/merge_requests/2)
-   Get the secondary seed labels for the current dataset you are on
    using `/secondary_seed_labels`
    [!20](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/20)
-   Ability for performers to name sessions
    [#34](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/34)
    [!13](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/13)
-   List all sessions `In Progress`
    [#31](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/31)
    [!12](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/12)
-   Ability for performers to deactivate sessions. Sessions can either be
    `In Progress`, `Complete`, or `Deactivated`.
    [#32](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/32)
    [!16](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/16)

### Bugfixes

-   Submitting wrong labels returns an accuracy of 0.01
    [#28](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/28)
    [!11](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/11)
-   Consistency between image dataset keys in calls. Use the term `class`
    for all class/category labels and `bbox` for the bounding boxes
    [#25](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/25)
    [!11](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/11)
-   Image classification classes reported in API response is not correct
    [#22](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/22)
    [!5](https://gitlab.lollllz.com/lwll/dataset_prep/-/merge_requests/5)

### Other

-   New URL for api-dev `https://api-dev.lollllz.com/`
    [#51](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/51)
    [!16](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/16)
-   `/get_session_token` is depreciated in favor of `/auth_create_session`
    [!13](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/13)
-   Logic for EVAL added to API
    [#46](https://gitlab.lollllz.com/lwll/lwll_api/issues/46)
    [#39](https://gitlab.lollllz.com/lwll/lwll_api/issues/39)
    [!14](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/14)
-   Clearer reporting of scores of each checkpoint and metadata
    [#20](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/20)
    [#21](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/21)
    [#12](https://gitlab.lollllz.com/lwll/lwll_api/-/issues/12)
    [!9](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/9)
-   Updated metadata to include class names, number of channels, language from,
    and language to
    [!7](https://gitlab.lollllz.com/lwll/lwll_api/-/merge_requests/7)
