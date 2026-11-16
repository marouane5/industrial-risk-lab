# Candidate review and extension guide

This initial implementation used AI assistance. Run it, explain the split invariants and add a documented experiment before presenting it in depth.

1. **Explain the target:** next-six-hour event risk is a classification problem. It is not remaining useful life, anomaly detection or a causal model of why a machine failed.
2. **Draw the timeline:** show feature window `[t-11,t]`, label interval `(t,t+6]` and why labels crossing train/validation boundaries must be purged.
3. **Read a leakage test:** changing future sensors must not alter a past feature vector. Changing every failure label must not alter any feature vector.
4. **Explain the optimization:** standardized inputs, logistic sigmoid, cross-entropy gradient, L2 weight penalty, deterministic batch updates. Discuss collinearity among rolling features.
5. **Discuss class imbalance:** AP is more informative than accuracy here. A classifier alarming on every row obtains recall 1 but low precision. Read the confusion counts, not just F1.
6. **Explain the artifact:** JSON stores the scaler and feature contract alongside weights. A model without its training normalization can silently return wrong scores.
7. **Explain PSI:** fixed train bins make comparisons meaningful; recomputing bins on the comparison set can conceal drift. Drift does not necessarily mean model performance degraded.
8. **State the evidence limit:** synthetic and correlated observations cannot substantiate real operational performance. Same-asset future testing does not establish unseen-machine generalization.

Suggested first extension: repeat the complete protocol over five independent simulation seeds and add an asset-level bootstrap interval. Suggested stronger extension: replace simulation with a documented public dataset, preserve its official evaluation split and compare the logistic model with a tree model using identical feature-availability constraints.
