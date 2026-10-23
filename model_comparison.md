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
