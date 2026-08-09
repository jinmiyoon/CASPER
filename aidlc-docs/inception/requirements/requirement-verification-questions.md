# Requirements Verification Questions

Please answer all questions by filling in the letter choice after each [Answer]: tag.

## Question 1
What is the primary goal for this AI-DLC cycle?

A) Safe refactoring and maintainability improvement with no scientific output changes

B) Add new scientific features or analysis capabilities

C) Mixed goal: refactor some modules and add selected new features

X) Other (please describe after [Answer]: tag below)

[Answer]: C

## Question 2
For this cycle, which scientific behavior change policy should apply?

A) No intended output changes are allowed for existing workflows

B) Output changes are allowed only if explicitly approved in advance

C) Output changes are allowed if tests pass

X) Other (please describe after [Answer]: tag below)

[Answer]: B

## Question 3
When comparing baseline vs post-refactor numerical outputs, which tolerance policy should we use?

A) Strict equality for all values

B) Floating-point tolerance (relative and absolute tolerance) per output file type

C) Manual scientist review for all non-identical values

X) Other (please describe after [Answer]: tag below)

[Answer]: B

## Question 4
Which dataset should be the mandatory baseline regression dataset for this cycle?

A) The current repository test set in casper/inputs/spectra/test_spectra with the corresponding param file

B) A larger internal science dataset selected by you later

C) Both the repository test set and a larger internal dataset

X) Other (please describe after [Answer]: tag below)

[Answer]: A

## Question 5
Should security extension rules be enforced for this project?

A) Yes — enforce all SECURITY rules as blocking constraints

B) No — skip all SECURITY rules

X) Other (please describe after [Answer]: tag below)

[Answer]: A

## Question 6
Should property-based testing rules be enforced for this project?

A) Yes — enforce all property-based testing rules as blocking constraints

B) Partial — enforce only for pure functions and serialization round-trips

C) No — skip all property-based testing rules

X) Other (please describe after [Answer]: tag below)

[Answer]: A

## Question 7
Should the resiliency baseline be applied to this project?

A) Yes — apply as directional best practices and design-time guidance

B) No — skip the resiliency baseline

X) Other (please describe after [Answer]: tag below)

[Answer]: I am not sure about the resiliency baseline. Please provide a summary of the key points and trade-offs so I can make an informed decision.

## Question 8
How should we prioritize refactoring units in Construction?

A) Start with highest-risk core pipeline modules first (batch/interface_main/spectrum)

B) Start with most-tested modules first (lower risk) and move toward core modules

C) Start with modules where code readability can be improved fastest

X) Other (please describe after [Answer]: tag below)

[Answer]: A
