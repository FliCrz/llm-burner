This directory contains a simple, programmatically evaluable benchmark for your trained inference model.

How it works:
- Each task has a prompt and an expected answer.
- Your inference engine should run each prompt and return a text completion.
- The scorer compares the completion to the expected values and returns a percentage.

Suggested workflow:
1. Run each prompt through your model.
2. Capture the raw output text.
3. Feed outputs into the scorer.
4. Report the total score as a percentage.

The file `tasks.json` defines the evaluation set.
