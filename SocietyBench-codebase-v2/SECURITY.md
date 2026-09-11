# Security and anonymization policy

Two kinds of report matter for this repository. Please use the private channel for both.

## 1. Anonymization leaks

This is the one specific to SocietyBench, and we treat it as the higher severity of the two.

The benchmark's validity depends on a model being unable to identify which real event an
anonymized timeline describes. Report it privately if you find:

- a residual real name, handle, ticker, place, or organization in a released timeline,
  context, question bank, or ground-truth file
- a detail that makes an event identifiable even without a name — an unshifted date, an
  exact figure that only one real event matches, a quoted sentence that can be searched
- an entity replacement table, real-name variant, or date offset that has been published by
  accident, in this repository or the dataset repository

**Please do not open a public issue and do not post the identification itself.** A public
report that names the underlying event contaminates the benchmark for everyone. Say which
file and which line, and describe the category of leak — that is enough for us to act.

## 2. Ordinary software vulnerabilities

Credential handling, unsafe deserialization, command injection in the pipeline scripts, and
similar. Same private channel.

## How to report

Open a private advisory via **Security → Report a vulnerability** on this repository, or
email the corresponding author listed in [`CITATION.cff`](CITATION.cff).

Please include: the file and line, what you observed, and how you found it. If you have a
patch, attach it as a diff rather than as a public pull request.

## What to expect

- Acknowledgement within about a week
- For a confirmed anonymization leak: the affected material is re-anonymized and the dataset
  repository is updated; the fix is noted in [`CHANGELOG.md`](CHANGELOG.md) without repeating
  the leaked detail
- Credit in the changelog if you would like it

## Out of scope

- Findings that depend on the private replacement tables, which are not released
- Reports that an anonymized event *could in principle* be identified by a determined
  researcher with unlimited effort. That is a known and accepted limitation, discussed in the
  paper. We are interested in leaks that a model or a casual reader would actually hit.
