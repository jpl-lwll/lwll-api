# Contributing

This is a guide that lays out how to contribute additional datasets or fixes to this lwll_api repository.

1. Make sure you have the latest updates from the `devel` branch. This branch should never be in a broken state.

```bash
git checkout devel
git pull
```

2. Create a feature branch with a descriptive name off of `devel`. If you are referencing a specific issue, you can do an auto link of the issue by specifying `#<issue_number>`. You can also specify the issue in the name of your MR when you make it similarly.

```bash
git checkout -b <feature_name>
```

3. Do your work in your feature branch with occassionally pushing up using the `git add`, `git commit`, `git push` flow. When you make your first push to your branch, you should open up a merge request on the repository going from your branch to `devel`. As you open this up, prefix your branch with `WIP:`, which stands for "Work In Progress" in order for others to know that you are not ready for a finaly code review yet. Example: `WIP: <feature>`. As you push code here, feel free to tag people for thoughts and questions in the MR. This allows dialouge to happen closer to the code and you can give suggestions while tagging specific lines of code.

4. Once your code is ready and you've tested various components of it by either doing doing a direct call to the endpoint, testing with the Jupyter notebook walkthrough, or running a system test with the JPL TA1 repo, make sure you have all of the latest changes by doing a rebase from `devel` and solve any necessary merge conflicts at this step. You should also be doing this pretty regularly as you develop to make sure you always have the latest code.

```bash
git checkout devel
git pull
git checkout <feature_branch_name>
git rebase devel
```

5. If you made changes that affect the user workflow at all, you must also find the appropriate place within the `REST_API_Example_Walkthrough.ipynb` and update the functionality there as well.

6. Update the `HISTORY.md` file describing your functionality.

7. Now, remove the `WIP:` prefix from the MR name. and tag either `Mark Hoffmann` or `Alice Yepremyan` in the 'Assigned To'. This signifies that you code is tested to your knowledge, good for a final review, and is ready to be merged into `devel`. NOTE: Before doing this step and removing `WIP:`, the CI should be passing. If the CI is not passing and you need help, tag someone in the comments on the MR for help, but don't request a final review.

8. Your branch will be merged by the code reviewer at this point and the CI will automatically sync to the infrastructure in the `DEV` environment.

**Merging Details**

When merging, we:

1. Delete source branch on merge
2. DO NOT squash commits

If you are a code reviewer doing the merge, while we do not have automation around alerts if the deployment pipeline fails, it is your responsibility to log into AWS and verify that the deployment pipeline passes.
