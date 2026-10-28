# Gmail Semantic Sorter

Discover what arrives in your personal Gmail, approve useful categories, and add labels using local AI on your Mac.

A standalone Python application with SQLite and native Ollama. It imports **real Gmail metadata and snippets**, discovers categories, produces review reports, classifies messages, and optionally adds approved `AI/...` labels. The startup's sorter is a separate project.

**Version 0.1:** the complete workflow is implemented and tested with mocked Gmail/model services. Live OAuth, model accuracy, and performance still need validation on your Mac and inbox. No real email or credentials are included in this repository.

## Features

- Resumable, paginated, batched Gmail imports with rate limiting and a durable retry queue.
- Sender/Gmail-category counts and newsletter/notification hints. These hints are not importance judgements.
- Category discovery using normalised Nomic embeddings and HDBSCAN, with an explicit k-means comparison option.
- Local AI category naming, representative examples, visible outliers, and suggested category merges.
- Explicit category approval. Edits invalidate approval and older apply plans.
- Classification through manual overrides, sender/domain rules, classification-space centroids, and a structured local-model fallback. Categories can overlap.
- Local HTML/CSV previews. Ambiguous messages go to `AI/Review`; model errors are reported and never automatically labelled.
- Separate read/write credentials, add-only Gmail writes, account checks, fresh-message checks, and an interrupted-write journal.
- Learning from explicit corrections and observed changes to previously applied Gmail labels.
- Evaluation on unseen threads, with accuracy, coverage, per-category precision/recall/F1, and a confusion table.
- Optional macOS scheduling, defaulting to dry runs.

## 1. Install on your Mac

Requirements: Python 3.11+, Git, and native Ollama. Docker is not used. On an 8 GB Mac, start with the small model and default batches. Embedding and generation requests release their model memory after each request.

If Python 3.12 is not installed:

**CHANGES SOMETHING:** installs Python through Homebrew. Undo with `brew uninstall python@3.12` only if nothing else needs that installation.

```bash
brew install python@3.12
```

The branch below contains the initial implementation PR. After it is merged, `main` also works. If you already cloned the repository, check out this branch in the existing checkout instead of making another copy.

**CHANGES SOMETHING:** downloads the project, creates a virtualenv, installs dependencies, and initialises private local folders. Undo by stopping any configured schedule and removing this checkout. Gmail is untouched.

```bash
git clone --branch codex/personal-sorter-plan https://github.com/Park07/gmail-semantic-sorter.git
cd gmail-semantic-sorter
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
gmail-sorter init
```

Run subsequent commands here with the virtualenv active. `--root /absolute/project/path` can be placed before a subcommand. Configuration is in private `sorter.toml`; `sorter.example.toml` documents the defaults.

In another Terminal tab, start Ollama. Quit any existing Ollama server first if it already occupies port 11434.

**CHANGES SOMETHING:** starts a loopback-only model server with cloud features disabled. Undo with Ctrl+C in this tab. Leave it running while using the sorter.

```bash
OLLAMA_NO_CLOUD=1 OLLAMA_HOST=127.0.0.1:11434 OLLAMA_CONTEXT_LENGTH=4096 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 ollama serve
```

In the project tab:

**CHANGES SOMETHING:** downloads two local models. Undo with `ollama rm nomic-embed-text:latest` and `ollama rm qwen3.5:0.8b` if no other project needs them.

```bash
ollama pull nomic-embed-text:latest
ollama pull qwen3.5:0.8b
```

**READ-ONLY:** checks configuration and downloaded-model metadata. It does not log in to Gmail or send email to a model.

```bash
gmail-sorter doctor
```

The small Qwen model is a starting point, not a promised accuracy level. Select another downloaded local model or adjust thresholds in `sorter.toml`. The app rejects remote model endpoints, redirects, cloud-tagged models, and advertised cloud-backed models. Keep Ollama cloud features disabled as an additional boundary.

## 2. Connect personal Gmail

Do this on your Mac. Do not upload credentials or email exports to an assistant or commit them to Git.

1. Create a project in [Google Cloud Console](https://console.cloud.google.com/) and enable **Gmail API**.
2. Configure Google Auth Platform with **External** audience and **Testing** publishing status. Add your Gmail as a test user.
3. Create an OAuth client of type **Desktop app**.
4. Download its JSON and place it at `credentials.json` in this project root. This application does not require Pub/Sub, a service account, a webhook, or billing setup.

**CHANGES SOMETHING:** opens Google sign-in and stores a local read-only OAuth token. It does not change Gmail. Undo by removing `data/token-readonly.json` and revoking the app in your Google account's third-party connections page.

```bash
gmail-sorter auth
```

The project binds to the account returned by Google. Other accounts are rejected before importing or writing. External apps in Testing generally need reauthorisation after seven days. Rerun `auth` when prompted. Production publishing and OAuth verification are separate decisions, not prerequisites for this personal pilot.

## 3. Import real mail

**CHANGES SOMETHING:** imports metadata/snippets into local SQLite and writes audit reports. Gmail is unchanged. Undo local imports using the reset procedure below, which also erases checkpoints and learning history.

```bash
gmail-sorter ingest --limit 20
gmail-sorter ingest --limit 1000
```

Open `reports/audit.html`. Imports resume their listing cursor and deduplicate messages. The default query covers received mail from the last year, including archived mail, and excludes chats, sent mail, drafts, spam, and trash. `--query` overrides the backfill query; `--restart` restarts its listing. Check `complete` and `pending` before assuming an import is finished.

Ongoing sync processes all newly received messages, excluding sent/draft/spam/trash messages. Only requested headers and Gmail snippets are fetched. Full bodies, attachments, and quoted histories are not downloaded, limiting what the model can infer.

## 4. Discover and approve categories

**CHANGES SOMETHING:** caches embeddings and writes proposals and discovery reports. Gmail and existing approved categories are unchanged. Undo by removing the proposal file and discovery reports.

```bash
gmail-sorter discover
```

Open `reports/discovery.html` and edit `data/categories.proposed.yaml`. Rename categories, improve descriptions, merge `example_ids`, or remove groups. Keep stable `id` values for categories you retain. Optional `senders` and `domains` lists are explicit rules you approve; domain rules also match subdomains. `Review` is reserved.

Every sampled outlier remains visible. If HDBSCAN produces no useful groups, adjust `min_cluster_size`/`min_samples` or compare k-means:

**CHANGES SOMETHING:** replaces current proposals and discovery reports. Undo by restoring a saved proposal or rerunning your earlier settings. Approved categories remain unchanged.

```bash
gmail-sorter discover --method kmeans --clusters 8
```

**CHANGES SOMETHING:** copies reviewed categories to private `categories.yaml` and records approval. Undo by editing and reapproving a prior version from `data/taxonomies/`. Gmail is unchanged.

```bash
gmail-sorter approve --file data/categories.proposed.yaml
```

After editing an existing `categories.yaml`, approve that file explicitly. A fresh discovery is always a proposal. A rename affects future writes; existing Gmail labels are never renamed or removed by the sorter.

## 5. Preview and evaluate

**CHANGES SOMETHING:** saves local predictions, an immutable apply plan, and CSV/HTML reports. Gmail is unchanged. Undo by removing generated reports; plans remain inert unless applied. Local reset removes stored plans.

```bash
gmail-sorter classify --limit 1000
gmail-sorter eval-sample --count 100
```

Open the preview path printed by `classify`. Evaluation samples intentionally omit predictions. Fill `expected_category_ids` in `reports/evaluation-sample.csv` with approved IDs separated by semicolons, or `REVIEW` when no category fits. IDs appear in the evaluation HTML and the `categories` command.

**CHANGES SOMETHING:** predicts the labelled holdout and writes evaluation reports and a cached summary. Gmail is unchanged. Undo by removing generated evaluation reports; local reset removes the cached summary.

```bash
gmail-sorter evaluate --file reports/evaluation-sample.csv
```

Inspect missed important messages as well as accuracy. Centroid similarity and model self-confidence are different scores; neither is a calibrated probability. Tune settings using reviewed examples, then use a separate holdout. A small pilot does not establish accuracy for every rare category.

Attention suggestions are currently English phrase heuristics for possible payment issues, deadlines, replies, and account-security problems. They can miss important emails. They are not a task manager or due-date extractor, and do not make an inbox safe to ignore.

## 6. Add approved labels

Review the preview first. Category approval and an unchanged plan are required. Application defaults to a maximum of 100 messages; `--limit` deliberately changes that cap.

**CHANGES SOMETHING:** obtains a separate modification token, authorising later writes but making none itself. Undo by removing `data/token-modify.json` and revoking Google's grant. Revoking a shared OAuth client can also revoke its read access.

```bash
gmail-sorter auth --enable-labels
```

If Google combines grants and the read-only login no longer has only read permission, use separate Desktop OAuth clients: the read client in `credentials.json`, the write client in `credentials-write.json`, then reauthorise each mode.

**CHANGES SOMETHING:** validates the plan and opens local SQLite state, but makes no Gmail changes without `--apply`. Undo any local initialisation with the reset procedure.

```bash
gmail-sorter apply
```

**CHANGES SOMETHING:** creates approved `AI/...` labels and adds them to up to 100 messages. Undo labels manually through Gmail settings. Emails themselves are untouched; this tool never removes labels.

```bash
gmail-sorter apply --apply --limit 100
```

The writer checks account identity, approval, settings, corrections, current content, and existing labels. Already processed messages and manual `AI/...` labels are protected. Model errors are skipped. Requests use `addLabelIds` only and cannot target system labels such as `TRASH` or `SPAM`.

A manual edit concurrent with an apply request can race with the preflight check because there is no atomic Gmail compare-and-set here. The app never removes labels; subsequent runs protect processed messages.

Interrupted writes are not blindly retried. `recover` checks Gmail and marks missing or partial outcomes as held. Only use `apply --apply --retry-held` after inspecting them. Retrying still requires the same intended labels and a valid plan.

## 7. Corrections and ongoing use

Change a previously applied label in Gmail, then sync. Observed changes become local examples; the tool's own additions do not. Removing all sorter labels records an instruction to leave that message unlabelled. Existing messages are not repeatedly relabelled.

**CHANGES SOMETHING:** refreshes local mail and corrections, and generates a new preview. Gmail is unchanged. Undo learning by restoring a local backup or clearing a specific override; local reset clears all history.

```bash
gmail-sorter sync
gmail-sorter feedback
gmail-sorter run
```

`correct MESSAGE_ID --category CATEGORY_ID` records a local correction. Repeat `--category` for multiple categories or use `--leave-unlabelled`. It does not change existing Gmail labels. `correct MESSAGE_ID --clear` removes that override but retains write protection. Explicit corrections take precedence over implicit feedback until cleared.

**CHANGES SOMETHING:** syncs, previews, and adds approved labels to up to 100 eligible messages. Undo added labels through Gmail settings.

```bash
gmail-sorter run --apply --limit 100
```

Expired history triggers a rescan of the configured backfill scope and previously imported eligible messages. Work is queued durably before the history cursor advances. Failed fetches remain queued; missing messages are marked unavailable locally. Keep the database: its correction and write history prevents unwanted relabelling.

## 8. Optional macOS scheduling

Keep Ollama running for scheduled classification. The generated job does not launch it. The Mac must be awake; later runs catch up through Gmail history. Reauthorise manually when Google tokens expire.

**CHANGES SOMETHING:** generates a launchd plist without activating it. Undo by removing the generated file before installation. Add `--apply` only if future scheduled runs should add Gmail labels.

```bash
gmail-sorter schedule --interval 3600 --limit 100
```

Inspect `data/local.gmail-semantic-sorter.plist`. It uses absolute interpreter and project paths; keep the checkout and virtualenv at those locations.

**CHANGES SOMETHING:** registers the schedule for your macOS user. Undo with the `bootout` command below. It does not run immediately at installation.

```bash
launchctl bootstrap "gui/$(id -u)" "$PWD/data/local.gmail-semantic-sorter.plist"
```

**CHANGES SOMETHING:** stops and unregisters the schedule. Undo by running the preceding `bootstrap` command again.

```bash
launchctl bootout "gui/$(id -u)" "$PWD/data/local.gmail-semantic-sorter.plist"
```

Private logs live under `data/`. A lock prevents overlapping runs. Editing a loaded plist requires unregistering and registering it again.

## Status, recovery, and reset

**READ-ONLY:** displays local account binding, counts, pending work, and last evaluation without contacting Gmail.

```bash
gmail-sorter status
```

**CHANGES SOMETHING:** reads Gmail and reconciles interrupted writes in the local journal. It never changes Gmail. Restore a prior local backup to undo journal changes, with the caveat that this loses knowledge of successful writes.

```bash
gmail-sorter recover
```

To uninstall, stop the schedule and any Ollama server started only for this project, optionally revoke Google's access, and delete the checkout. Remove downloaded models only if no other app needs them. Gmail labels remain until you remove them in Gmail settings.

To reset local state while keeping code, stop automation first, back up or remove `data/`, `reports/`, and `categories.yaml`, then run `init`. This erases imported metadata, OAuth tokens, approvals, evaluations, corrections, and write history. Do not reset merely to retry a failed run: use the queues and `recover`. Keep any backups private.

## Development

**CHANGES SOMETHING:** installs development dependencies in the active virtualenv and creates test caches. Undo by removing the virtualenv and caches. Tests never contact a real mailbox or paid model service.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check sorter tests
```

Tests cover staged OAuth/PKCE, actual Google SDK request bodies, forbidden Gmail operations, stale plans, account isolation, pagination, expired history, interrupted writes, corrections, embedding-cache invalidation, outliers, invalid model output, report injection, held-out evaluation, process locking, and a complete mocked CLI workflow. GitHub Actions is configured for Linux and macOS.

The source under `sorter/` is original implementation code. [Buckets](https://github.com/arctic-hen7/buckets) inspired the natural-language tag rules, overlapping labels, and deterministic label application. Jev, hosted AI, sending mail, archiving, and the startup project are outside this version.

## API references

- [Google Desktop OAuth / Python quickstart](https://developers.google.com/workspace/gmail/api/quickstart/python)
- [Gmail scopes](https://developers.google.com/workspace/gmail/api/auth/scopes)
- [Gmail synchronisation](https://developers.google.com/workspace/gmail/api/guides/sync)
- [Gmail batch label additions](https://developers.google.com/workspace/gmail/api/reference/rest/v1/users.messages/batchModify)
- [Ollama local configuration](https://docs.ollama.com/faq)
- [Nomic embedding task prefixes](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
