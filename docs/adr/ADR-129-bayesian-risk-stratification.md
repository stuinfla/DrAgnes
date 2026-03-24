Updated: 2026-03-24 18:00:00 EST | Version 1.0.0
Created: 2026-03-24

# ADR-129: Bayesian Risk Stratification — Making Every Result Actionable

## Status: IMPLEMENTED | Last Updated: 2026-03-24 18:00 EST

## Context

Binary classification ("melanoma yes/no") produces a PPV of only 8.9% at real-world 2% melanoma prevalence. When Dr. Agnes says "melanoma detected," it's wrong 91% of the time. This destroys user trust and is clinically irresponsible.

But the model's AUROC is 0.960 — it discriminates melanoma brilliantly. The problem isn't the model; it's the threshold-based presentation.

## Decision

Replace binary classification with Bayesian post-test probability risk stratification.

### The Math

```
post_test_odds = pre_test_odds × likelihood_ratio
pre_test_odds = prevalence / (1 - prevalence)
likelihood_ratio = malignant_probability / (1 - malignant_probability)
post_test_probability = post_test_odds / (1 + post_test_odds)
```

### Risk Levels

| Level | Post-test Prob | Headline | Action | Timeframe |
|-------|---------------|----------|--------|-----------|
| Very High | >50% | "Urgent: see a dermatologist" | Schedule immediately | Within 1 week |
| High | 20-50% | "See a dermatologist" | Schedule appointment | Within 2 weeks |
| Moderate | 5-20% | "Worth monitoring" | Photograph monthly | Ongoing |
| Low | 1-5% | "Low concern" | Routine skin checks | Annual |
| Minimal | <1% | "No concerning features" | Normal schedule | As needed |

### Age-Adjusted Prevalence

Melanoma risk varies 8x between young and old adults:

| Age | Prevalence Multiplier | Effective Prevalence |
|-----|----------------------|---------------------|
| Under 30 | 0.3× | 0.6% |
| 30-50 | 0.7× | 1.4% |
| 50-70 | 1.5× | 3.0% |
| Over 70 | 2.5× | 5.0% |

### Calibration Correction

Temperature scaling (T=1.23) reduces Expected Calibration Error from 0.078 to 0.044. Applied to logits before softmax to ensure the model's probability outputs match reality.

## Consequences

### Positive
- Every result is actionable ("see a doctor in 2 weeks" vs "melanoma detected")
- Honest about uncertainty at real-world prevalence
- Age-appropriate risk assessment
- No more panic-inducing false "melanoma" alerts

### Negative
- More complex to explain than binary yes/no
- Prevalence assumptions may not match all populations
- Requires trust in Bayesian framework (validated by AUROC 0.960)

## Implementation

- `risk-stratification.ts`: Bayesian post-test probability calculation
- `consumer-translation.ts`: Updated to use risk assessment
- `calibrate-temperature.py`: Optimal T=1.23 computed
- `calibration-results.json`: ECE before/after evidence

## Author
Stuart Kerr + Claude Flow
