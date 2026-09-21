# Static site assets

The learned weight samples are extracted from
[Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca),
released by the Qwen team under the Apache License 2.0. A copy accompanies this
site in `Qwen3-LICENSE.txt`. These JSON files are derived samples, with axis
transpositions and head slices matching the educational implementation; they
are not the original checkpoint files. Sampled numerical values are unchanged.
The prepared-data manifest records the source revision, file hash, and tokenizer
provenance. No other model's learned weight values are distributed here.

Kalam, STIX Two Text, Source Serif 4, and Source Sans 3 are distributed with their
SIL Open Font License files in `fonts/`. See `fonts/README.md` for their use.

The displayed educational implementation files come from the source snapshot
identified by `implementations.lock.json` in the project. The static site's
`site-manifest.json` records this commit and the build inputs. Token examples
contain prepared token IDs and pieces; tokenizer tables are not distributed.
