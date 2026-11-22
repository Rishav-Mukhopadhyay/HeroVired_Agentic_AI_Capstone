# Scribe prompt comparison

3 ACI-Bench validation encounters (D2N068, D2N069, D2N070), scored against the expert-written reference notes.

| version       |   rougeL_f1 |   section_coverage |   unsupported_meds |   flags |   seconds |
|:--------------|------------:|-------------------:|-------------------:|--------:|----------:|
| v1_baseline   |       0.318 |                  1 |                  0 |   1     |    18.09  |
| v2_role_rules |       0.324 |                  1 |                  0 |   1.667 |    19.706 |
| v3_few_shot   |       0.366 |                  1 |                  0 |   0.333 |    14.437 |

- `rougeL_f1`, `section_coverage`, `flags`, `seconds`: mean per note
- `unsupported_meds`: total medication names not found in the conversation (lower is better)

## Per-encounter results

| encounter   | version       |   rougeL_f1 |   section_coverage |   unsupported_meds | unsupported_list   |   flags |   seconds |
|:------------|:--------------|------------:|-------------------:|-------------------:|:-------------------|--------:|----------:|
| D2N068      | v1_baseline   |       0.309 |                  1 |                  0 |                    |       1 |    14.791 |
| D2N068      | v2_role_rules |       0.278 |                  1 |                  0 |                    |       2 |    19.981 |
| D2N068      | v3_few_shot   |       0.316 |                  1 |                  0 |                    |       0 |    11.67  |
| D2N069      | v1_baseline   |       0.335 |                  1 |                  0 |                    |       1 |    20.828 |
| D2N069      | v2_role_rules |       0.333 |                  1 |                  0 |                    |       2 |    19.439 |
| D2N069      | v3_few_shot   |       0.395 |                  1 |                  0 |                    |       1 |    14.26  |
| D2N070      | v1_baseline   |       0.309 |                  1 |                  0 |                    |       1 |    18.652 |
| D2N070      | v2_role_rules |       0.362 |                  1 |                  0 |                    |       1 |    19.698 |
| D2N070      | v3_few_shot   |       0.386 |                  1 |                  0 |                    |       0 |    17.379 |

## Findings

1. **Few-shot wins.** `v3_few_shot` scored the highest ROUGE-L (0.366 vs 0.318 for the baseline,
   +15%) and was best on every encounter. It was also the fastest and raised the fewest flags:
   a worked example tells the model what "good" looks like better than rules do.
2. **Rules alone barely moved ROUGE-L** (0.324 vs 0.318) but made the model more cautious: it
   flagged more items for verification (1.7 per note vs 1.0). Rules change behaviour (safety),
   the example changes style (closeness to the expert note).
3. **Two metrics saturated.** Every version filled 100% of the expert's sections, because the
   `SOAPNote` schema already requires them: structured output did that work, not the prompt. No
   version invented medications on these clean dialogues, so that check could not separate them.
4. **Limits.** Three encounters is a small sample, and ROUGE-L measures wording overlap, not
   clinical correctness. Faithfulness is evaluated separately with an LLM judge.

**Decision:** `v3_few_shot` is the Scribe agent's default prompt.

## Case study: noisy speech-recognition input

Running `v3_few_shot` on the Whisper transcript of D2N081 (synthetic audio, 9.7% WER):

| Heard in the transcript | Written in the note | Flagged for verification |
|---|---|---|
| "the system as well as the up side" (chemotherapies) | cisplatin and etoposide | yes |
| "radiation at 45 grade" | 45 Gy | yes |
| "a little cane swish and swallow" | lidocaine swish and swallow | **no** (silent correction) |
| "non-smile cell lung cancer" | non-small cell lung cancer | yes |

Every plan item (prednisone 40 mg daily for 5 days, swish and swallow four times a day, repeat
imaging) was checked against the transcript and is grounded in what the doctor said. The model
repaired speech-recognition errors from clinical context and listed most corrections under
"Items to verify", as the prompt instructs. It missed one: "little cane" became lidocaine without a
flag. That single silent correction is exactly why physician review stays mandatory: corrections
are plausible, not guaranteed.

Side effect on metrics: the "unsupported medications" check would count *cisplatin*, *etoposide*
and *lidocaine* as unsupported here, since the exact words were never transcribed. String-matching
hallucination checks need care with noisy audio.
