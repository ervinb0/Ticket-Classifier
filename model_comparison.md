- Which model has the best macro-F1? Is the difference big or small?
  LogReg and SVC have the best macro-F1 score (0.66 & 0.65). In third place it's RF(0.627) a noticable difference.

- Does any model fix question recall, or do they all struggle? If they all struggle, what does that tell you about where the problem is: the model, or the data and features?
  LogReg has the highest question recall (0.57), better than SVC (0.42). RF struggled quite a lot (0.16).
  3 very different kinds of model all reach a question F1 of 0.42 or less, while bug gets about 0.8. When very different models hit the same ceiling, the limit is in the data and features, not the model. That's because the question signals are deleted before any model sees them.

- How much slower is the Random Forest, and is it worth it?
  RF is the slowest taking about 40s to train, LogReg takes 15s meanwhile SVC takes approx 7s. Currently RF is not worth it given its high training time and poor question recall.

- Why might a linear model do well on TF-IDF features?
  There are 10,000 features, but the median issue uses only 22 of them. 99.7% of all values are zero.
  A linear model learns one weight per word and adds up the evidence from every word in the issue. Each of the 22 words pushes a little toward bug, enhancement or question, and the sum decides. Many weak signals spread over many sparse features is exactly where linear models shine.
  A Random Forest looks at one word per decision, and at each split it only considers a random 100 of the 10,000 words. For any given word, almost every issue has a zero, so most splits separate only a few issues. Each tree needs many splits to collect evidence a linear model gets in one sum. With balanced weights, it still mostly votes for the majority class. That's why linear models are the standard baseline for text classification with TF-IDF.

Limitation:
Language filter falsely rejects some English issues containing logs/URLs (langdetect on raw text, ~8% rejected). Possible fix: detect on cleaned text. Deferred: expected gain small.

| Validation run            | logreg macro-F1 | logreg question F1 | svc macro-F1 | rf macro-F1 |
| ------------------------- | --------------: | -----------------: | -----------: | ----------: |
| Step 3 baseline           |           0.661 |              0.415 |        0.653 |       0.627 |
| + question signals (\_up) |           0.691 |              0.476 |        0.679 |       0.664 |
| + bigrams, 10k            |           0.699 |              0.489 |        0.696 |       0.684 |
| + bigrams, 50k            |           0.710 |              0.506 |        0.689 |       0.653 |
| + bigrams, 100k           |           0.716 |              0.517 |        0.698 |       0.637 |

Threshold tuning (5-fold OOF on train) chose t = 0.52. Val macro-F1 0.716 → 0.723, question precision 0.45 → 0.52 at recall 0.61 → 0.53. Balanced class weights had already done most of the work; tuning mainly trades question recall for precision.

Error analysis showed qmark firing on stack traces ([?:?], ?? ??:0). Counting only ? after a word: macro-F1 0.723 → 0.724 (noise), kept because it removes the artifact.

Sentence embeddings (all-MiniLM-L6-v2) not evaluated. Error analysis showed most remaining errors are label noise or undecidable support requests, so the expected gain is small, and shipping embeddings would add PyTorch (~1 GB image, ~500 MB RAM) to the service. Future work: evaluate on a GPU and, if it helps, distil into a lighter model.
