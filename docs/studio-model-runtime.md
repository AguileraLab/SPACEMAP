# Local assistant shutdown and timeouts

Classic and Studio share the same runtime state. **Stop** requests shutdown and
shows **Stopping…** while the process remains alive. The controls offer **Start**
after it exits. A pending stop must not switch to another model, delete its files,
or show a completion message. Classic reports failures inline instead of raising
an unhandled Streamlit exception.

Shutdown runs in a background monitor. It allows 25 seconds for graceful exit
(the parent guard allows its child 20 seconds), then terminates the recorded model
process/group if needed. Only a matching local model process is eligible, checked
by host, command, ownership and recorded process start time. A process sharing
SPACEMAP's group is signalled individually. An older monitor cannot remove a newer
process record.

The Transformers streaming service reports generation errors as stream errors,
including out-of-memory errors and 120 seconds without new tokens. It requests
cancellation when the stream times out or closes. The worker retains the generation
lock until the computation really exits, preventing a retry from overlapping an
abandoned computation. Cancellation is cooperative between generation steps;
shutdown can still be needed for a stuck GPU operation.

Connection heartbeats keep the client responsive during preparation and hidden
model output. They do not extend the user's configured total response budget.
The local client distinguishes a read timeout, a disconnected/truncated answer,
and an explicit generation error. The default 600-second budget explains messages
around 601 seconds; it does not identify the underlying reason generation was slow.

Verification uses synthetic model responses, disposable sleeping/stubborn model
processes, Classic Streamlit AppTest controls, and a Studio Firefox state test.
It does not stop, restart or benchmark the user's active model. To load changes,
restart SPACEMAP and start the local model again, then test a short question before
repeating a large cluster review.
