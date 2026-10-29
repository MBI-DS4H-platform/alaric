# Preparation contract handoff

This records the W0 contract work and the feasibility findings available on
2026-10-29. The durable schema/fixture package is described in
[preparation-contracts.md](preparation-contracts.md). Its Python and TypeScript
checks validate representations and pure decision contracts; they do not
establish that the current Preparic application implements them.

The W0 package is complete. Parent review verified **261 shared cases** in
Python and compiled TypeScript with identical results, JSON round trips,
golden-reference metadata and active-inventory coverage. Rebuilding the
generated schema/fixture artifacts left their contents unchanged. No new
author decision was needed. This closes the schema/fixture work and its
completion/handoff record, while preserving the later acceptance obligations
below.

Run the shared checks from the Alaric repository root:

```sh
python tests/contracts/run_contract_checks.py
```

## Earlier feasibility findings

These findings come from the earlier isolated runtime and real-Chrome runs.
They were not rerun while implementing the schema/fixture package. The user
requested removal of the temporary `plans/w0-checks` directory; this document
retains the relevant conclusions and their limits.

| Area | Recorded finding | Limit |
|---|---|---|
| Local execution | Sixteen small tasks used eight distinct spawn-worker PIDs, with persistence disabled and no hashserver/database configured. | Future scientific steps still need their own sandbox acceptance. |
| Modules and folders | A stdlib module executed in a spawn worker. Folder pins received two checked-in A1 PDB byte strings; deepfolder pins received checksum members. | Representative transport, not actual A2/A4/A6 output acceptance. |
| Cancellation | Supersession delivered the newest result. Soft cancellation detached the promise while its body continued; the worker backend's hard-cancel operation stopped an instrumented body and allowed replacement work. | This does not select a production cancellation API. |
| Read-only mounts | Synthetic source/model/ring mounts changed within one live Context through automatic polling, without an explicit post-edit sync. | Actual Gaussian installation artifacts were not changed. |
| Equal-value writes | An HTTP PUT returned 200 with its marker unchanged and no downstream body re-execution. The real browser client acknowledged an equal write and cleared pending/writing state without a WebSocket update. | Browser acknowledgment and execution counting were established in separate runs. |
| Browser reads | Real Chrome tabs discovered shares and reread ordinary values on reconnect. An explicit CIF GET returned exactly the fixture bytes. | Dotted `cif.cif` was not automatically read; the tested CIF was not durably mounted. |
| Browser writes | The current client retried stale intents with refreshed markers and could overwrite another tab. Held stale-marker PUTs conflicted; an accepted write with a lost HTTP acknowledgment was recovered by rereading. | I3 rejects a separate adapter: W1 retains client retries and the author's corrective-write rule. |
| Prerequisite handoff | A7/A8 were accepted by explicit author ruling, with tests assumed passing. | No additional Gaussian test, cache hashing/decompression or generation was performed. |
| Scorer dependencies | All 26 entries of the full attract-jax list existed, including the four required NumPy scorer dependencies and `attract-par.npz`. | The reduced A5 dependency list and its sandbox remain delivery work. |

The browser run used Chrome 155.0.8059.39 and the real served Seamless client.
Its Mol* viewer could not initialize WebGL. Consequently it established share
transport and client behavior, not UI viewer restoration. Initial read-only
enforcement with held decision reads and sustained pending computed results
were not established by that run.

## Settled design inputs

- Decisions retain semantic bases, active flags and ordered orphan cleanup.
  Activation tokens and parent-token linkage were rejected (I1).
- Validation receipts were rejected (I4). Browser validation and backend
  validation/error reporting remain. Gate 4→5 also requires T10 not failed;
  gate 5→6 also requires T27 not failed. A nonfailed node does not prove
  freshness or satisfy a gate's other conditions.
- Extra model/ring identity fields across result envelopes were rejected
  (I5). Existing dependency and scientific-data identities remain.
- A separate write adapter was rejected (I3). Writes use the Seamless client
  and its retries; assume no withdrawal and always queue the correct state.
  The proposed valid-prefix fit gate was also rejected (I6): group B stays
  gated on S8's active flag, including the accepted stale-active restart case.
- Gaussian descriptors use shared backend Kabsch geometry. Gaussian scoring
  and computed classification are backend-only (O5): T28 publishes S25;
  T17/S16 are removed. The browser consumes results and supports saved class
  edits without calculating scores or computed classes.
- Energy computations supply the explicit Copy source. The editable S24
  table starts empty, is filled by Copy and/or typing, and retains values by
  protein-range row identity. It is not automatically filled from computation.
- Fragment numbers are ordinal; a gap-spanning window is omitted. For
  residue IDs `[101,102,104,105]`, valid dinucleotides have keys 1 and 3.
  The earlier `Reference` offset-numbering finding has a settled correction
  assigned to W8; it is not an open W0 decision.
- The concurrency guarantee excludes racing tab edits, while an idle second
  tab must follow another tab's Continue (D17/O7). The rejected atomic
  admission proposal and its model are not implementation prerequisites.
- Restart correctness applies when the browser restarts immediately after
  the backend, with mounted cells surviving (O6). Keeping old tabs alive
  through backend restart is undefined; those traces do not mandate a new
  generation fence.

## Acceptance owned by later implementation

| Delivery | Acceptance still required |
|---|---|
| W1 | Real application decision loading, durable CIF, freshness barriers, corrective write handling, T10/T27 failure vetoes, a nonediting follower tab and fresh-browser restoration after backend restart. |
| W2 | Actual A2/A4/A6 declared dependencies, spawn-compatible sandbox execution, output goldens, raw threshold-field preservation and reference/fit fragment-key and atom parity. |
| W4 | Reduced A5 scorer dependency completeness/minimality and NumPy scoring sandbox acceptance. |
| W7 | Actual T28 model self-tests, backend score/classification results, and 1b7f T28-versus-A9/T21 score agreement to 1e-6. No browser numerical parity test is required. |
| W8 | Ordinal `Reference` addressing and affected-tool corrections, including a non-1 first residue and a numbering gap. |

These are acceptance of the delivered integrations and scientific steps, not
prerequisites requiring those implementations to exist before W0 can finish.
The separately owned prefilter-thresholding issue remains excluded.
