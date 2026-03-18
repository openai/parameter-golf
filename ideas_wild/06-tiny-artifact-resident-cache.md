# 6. Tiny Artifact-Resident Cache

## Core Thesis

A very small learned key-value table embedded in the artifact might act as a broad statistical shortcut that is unusually efficient for bpb on a fixed validation distribution.

## What It Changes

Add a compact memory table to the submission itself:

- trained end-to-end
- stored inside the artifact budget
- queried during evaluation as part of the model

This is not external retrieval and not online data access. It is part of the submitted model.

## Why It Might Improve `val_bpb`

The metric is compression on a fixed validation distribution. That means nonstandard memory mechanisms can be more attractive than they would be in a generic LM benchmark, as long as they are paid for in bytes and trained honestly.

## Why It Is Risky

This would attract scrutiny because it sits closer to the boundary of what people will consider “in spirit.” It is also easy to waste bytes on memory that does not generalize enough to help.

## First Useful Experiment

Try a tiny table with very few entries and a clean learned query mechanism. If it does not help quickly, drop it. This is not a good idea to sink large implementation time into blindly.
