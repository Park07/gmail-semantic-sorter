# Gmail Semantic Sorter

A personal Gmail organiser that proposes categories from existing mail, lets you review them, and adds approved labels to incoming messages.

**Status: build plan.** This repository does not yet contain a working importer, classifier, or Gmail label writer. The milestones below describe the intended implementation.

## Scope

- One personal Gmail account, running locally on a Mac.
- Discover categories from actual messages instead of requiring a complete list in advance.
- Keep topic classification separate from suggestions about messages needing attention.
- Use local models with no paid AI API or hosted application required for v1.
- Keep the startup email sorter in a separate project, with separate credentials, data, and configuration.

## Proposed stack

Python 3.11+, SQLite, the Gmail API, and native Ollama. The first interface is a command-line tool with local HTML and CSV reports. Docker, a hosted database, and Pub/Sub are not needed for the first version.

Use `nomic-embed-text` for embeddings and benchmark a small Qwen model for category naming and classification. The target Mac has 8 GB RAM: process small batches and run the embedding and generation models separately. The suggested `qwen3.5:0.8b` is an initial runtime test, not an accuracy commitment.

## Build milestones

| Milestone | Deliverable | Complete when |
| --- | --- | --- |
| 1. Safe importer | Python CLI, SQLite schema, synthetic fixtures, and Desktop OAuth using `gmail.readonly` | A local demo works without an account; the Mac can import 20 messages and produce a report without writing to Gmail |
| 2. Inbox audit | A configurable date range and mailbox query, pagination, resumable ingestion, and sender/category counts | A sample of roughly 500 to 1,000 messages across several months can be inspected locally |
| 3. Category discovery | Normalised embeddings, clustering, and proposed names and descriptions | Every proposed category has a count and representative examples; outliers remain visible |
| 4. Human review | A private category file with stable IDs and explicit approval | Categories can be renamed, merged, dropped, or added; unapproved changes cannot affect Gmail |
| 5. Sorting preview | Rules, similarity matching, and a local-model fallback | Reports show proposed labels and reasons, with ambiguous messages sent for review; a manually labelled pilot measures errors |
| 6. Add labels | Separate write authorisation and an explicit `--apply` path | A capped batch of up to 100 messages receives only approved `AI/...` labels and passes a manual spot-check |
| 7. Ongoing use | Incremental polling, bounded retries, manual overrides, and optional macOS scheduling | Runs catch up after sleep, failed work is recoverable, and corrections are respected |

The first coding deliverable is milestone 1. Actual OAuth login, email downloads, and local-model evaluation happen on the user's Mac. Development and automated tests use fictional messages and mocked services.

## How discovery and sorting should behave

Bills, housing, university, jobs, and subscriptions are useful starting hints, not a required taxonomy. Category proposals must be grounded in the imported messages. New patterns should lead to reviewable suggestions rather than silently creating categories.

Start clustering with HDBSCAN. Inspect the results before tuning or comparing another clustering method. A rare message may matter greatly, so noise and low-frequency groups must remain available for review. Simple newsletter or sender rules must not exclude messages from the discovery audit.

Store the embedding model identifier, task prefix, and preprocessing version with cached vectors. Use the model's `clustering:` prefix for discovery. Re-embed approved examples with `classification:` before building classification centroids; do not compare those vectors with discovery centroids or reduced clustering coordinates.

An email may match multiple approved categories. Match the contents as well as the sender: a university newsletter and an assignment deadline have different attention needs. The first version shows attention suggestions in a local report, avoiding persistent action-status labels that would become stale under an add-only policy.

Model responses must use approved category IDs and pass schema validation. A model-written confidence number is not a calibrated probability. Choose thresholds from reviewed examples and allow abstention. Evaluate category accuracy, precision and coverage, missed important messages, and review effort on examples separate from the stored few-shot examples, keeping threads together.

## Gmail and data boundaries

- All commands default to no Gmail changes. Imports and reports still write local files.
- Use `gmail.readonly` until the label-writing milestone. Request `gmail.modify` only for the explicitly enabled writer.
- The Gmail client must not implement message deletion, trashing, sending, archiving, or label removal.
- The writer may create approved `AI/...` user labels and add their IDs. System labels such as `TRASH` and `SPAM` are never valid targets.
- Validate the complete apply plan before writing. Track completed writes so retrying cannot duplicate work or undo a manual correction.
- Keep credentials, tokens, databases, embeddings, real-email examples, category proposals, approved categories, and reports out of Git.
- Treat email text as untrusted input. It cannot issue commands or select arbitrary Gmail actions. Escape report HTML and protect CSV exports against spreadsheet formula execution.
- Log IDs and operational errors rather than email bodies or credentials.
- Removing the tool-created labels through Gmail settings is the user-controlled undo. The sorter itself never removes labels.

Safety tests should exercise the Gmail request bodies and write boundary, scope selection, duplicate ingestion, interrupted runs, invalid model output, hostile report content, and manual overrides. Synthetic fixtures are the only email examples committed to this repository.

Every documented shell command will be labelled `READ-ONLY` or `CHANGES SOMETHING`, with an explanation and undo instructions for changes.

## References

- [Buckets](https://github.com/arctic-hen7/buckets): inspiration for natural-language tag rules, overlapping labels, and deterministic application of model decisions. This repository currently contains no copied Buckets source code.
- [Gmail labels](https://developers.google.com/workspace/gmail/api/guides/labels)
- [Gmail scopes](https://developers.google.com/workspace/gmail/api/auth/scopes)
- [Gmail synchronisation](https://developers.google.com/workspace/gmail/api/guides/sync)
- [Ollama local configuration](https://docs.ollama.com/faq)
- [Nomic embedding model and task prefixes](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)

Jev can be considered as a later, explicit opt-in backend. It is outside the local-only first version.
