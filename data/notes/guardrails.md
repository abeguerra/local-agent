# Notes on guardrails

Autonomy buys flexibility at the cost of higher token spend and the
risk of compounding errors, since each step can build on a mistake
from the previous one. The standard mitigations: a maximum iteration
count so a confused loop cannot run forever, a tool allowlist so the
model can only reach the actions you intended to expose, a confirmation
step before destructive or irreversible actions, and a sandboxed
execution environment so a tool call cannot touch anything outside the
scope you've defined for it.

Testing an agent thoroughly in a sandbox, with the same guardrails it
will run with in production, is cheaper than discovering a gap in those
guardrails from a live incident.
