# Code Review Guidelines

All changes committed to our code base must be reviewed by one or more people to ensure the continued quality of our code and product. Gimlet’s code review guidelines are inspired by Google’s [Code Review Guidelines](https://google.github.io/eng-practices/review/). The main takeaways are highlighted in this document.

## Who Should Review

The reviewer should be someone who has strong knowledge of the area of code being modified. In some cases, this may mean waiting for approvals from multiple people before merging in a change.
Smaller changes are usually faster to review and require fewer reviewers since they likely to touch a smaller part of the codebase. It's the responsibility of the author to make sure the appropriate reviewers have looked at the code.

## Being a Reviewer

Code reviewers should look for:

* Design: Is the code well-designed and appropriate for the system?
* Functionality: Does the code behave as the author likely intended? Is the way the code behaves good for its users?
* Complexity: Could the code be made simpler? Would another developer be able to easily understand and use this code when they come across it in the future?
* Tests: Does the code have correct and well-designed automated tests?
* Comments: Are the comments clear and useful?
* Style: Does the code follow our style guides?
* Documentation: Did the author also update relevant documentation?

Larger architectural design changes should be documented in a design doc and discussed with the relevant parties before any code changes are made. **You should never be debating large design decisions in a code review.**

In general, reviewers should favor approving a PR once it is in a state where it definitely improves the overall code health of the system being worked on, even if the PR isn’t perfect. We want to make the necessary trade-offs to ensure the quality of the codebase does not degrade while continuing to make forward progress.

### What to look for

#### Design

This is the most important part of the code review. While large architectural changes should have been discussed prior to writing the code, we can explore if the changes belong in here, as part of a shared library, etc.

#### Functionality

Does the PR do what the developer intended? Is what the developer intended good for the users of this code? The “users” are usually both end-users (when they are affected by the change) and developers (who will have to “use” this code in the future).

Another time when it’s particularly important to think about functionality during a code review is if there is some sort of parallel programming going on in the PR that could theoretically cause deadlocks or race conditions.
These sorts of issues are very hard to detect by just running the code and usually need somebody (both the developer and the reviewer) to think through them carefully to be sure that problems aren’t being introduced. (Note that this is also a good reason not to use concurrency models where race conditions or deadlocks are possible—it can make it very complex to do code reviews or understand the code.)

#### Complexity

Is the PR more complex than it should be? Check this at every level of the PR—are individual lines too complex? Are functions too complex? Are classes too complex? “Too complex” usually means “can’t be understood quickly by code readers.” It can also mean “developers are likely to introduce bugs when they try to call or modify this code.”

A particular type of complexity is over-engineering, where developers have made the code more generic than it needs to be, or added functionality that isn’t presently needed by the system. Reviewers should be especially vigilant about over-engineering.
Encourage developers to solve the problem they know needs to be solved now, not the problem that the developer speculates might need to be solved in the future. The future problem should be solved once it arrives and you can see its actual shape and requirements in the physical universe.

#### Tests

Please refer to the testing guide for best practices on writing tests. Remember, tests are also code and need to be maintained. Avoid introducing unnecessary complexity in tests, even though they are not part of the main binary.

## Speed of Code Reviews

We should aim to provide code reviews within 24 hours. We want to strike the right balance between allowing a developer to focus on authoring their own code versus unblocking other team members by providing reviews.

If you’re not in the middle of a focused task, aim to provide feedback shortly after your review is requested. For example, this can be in between writing your own PRs, in the morning, or after a break.

To give others time to focus on their own code, try not to ping them about PRs unless there have been no reviews in the last 24 hours. Standup is a good time to remind others when you have PRs in need of feedback.
