# Manuscript reference (verbatim copy)

`validation/` holds the manuscript's independent NumPy/SciPy implementation of
the Section 5.3.1 smooth-channel benchmark and its saved results, copied
byte for byte from the manuscript repository on 2026-09-25 so that the
benchmark tests run from this repository alone.

| file | sha256 |
|---|---|
| `full_response.py` | `d54fbb8bf68c351c9712b65b07bae18e8f00626cdb714df2a5620f83a811d57a` |
| `full_response_product.py` | `4d69ef07a7b5a639be8e722aae4558de4174996a160a258a947107677e77804b` |
| `full_response_results.json` | `720820a1b55c26584e5cd758f40aa2d7a6a3298b9f1f0effd27044bd1a4da290` |

`full_response_results.json` records the hashes of the two scripts that
produced it; `tests/model/test_reference_provenance.py` checks that the copies
match them. Do not edit or reformat these files (ruff and black exclude this
directory). To update, regenerate the results in the manuscript repository
with `python -m validation.full_response`, copy all three files, and update
the table.
