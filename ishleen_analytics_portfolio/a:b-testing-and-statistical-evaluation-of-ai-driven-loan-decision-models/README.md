# A/B Testing and Statistical Evaluation of AI-Driven Loan Decision Models

[📄 View Full Report](ai-loan-model-evaluation-report.pdf)

**Tools:** R, ggplot2, effectsize, FactoMineR  
**Techniques:** Experimental design, hypothesis testing, PCA, effect size analysis, Welch’s t-test  

## Overview
Evaluated the performance of a newly developed AI-driven loan approval model through a controlled A/B test.  
The goal was to determine whether the AI system improved loan officer decision accuracy, reduced bias, and increased confidence compared to traditional methods.

## Approach
- Conducted **A/B testing** between two groups — control (existing decision system) and treatment (AI-assisted system).  
- Performed **data cleaning and feature engineering** to compute recall, precision, conflict rate, and confidence improvements.  
- Used **Welch’s t-tests** to compare performance metrics and validate statistical significance.  
- Applied **Principal Component Analysis (PCA)** to construct an **Overall Evaluation Criterion (OEC)**, combining all key variables with appropriate weighting.  
- Calculated **effect size (Cohen’s d = 0.65)** and **power analysis** to confirm both statistical and practical significance.

## Results
- AI-assisted model achieved a **93.9% improvement** in decision quality.  
- Recall increased by **3596%**, precision by **249%**, and conflict rate dropped by **227%**.  
- Welch’s t-tests confirmed all major results were statistically significant (**p < 0.001**).  
- Power analysis indicated that a sample size of ~64 per group was sufficient for an 80% detection power at α = 0.05.

## Impact
Demonstrated ability to combine **machine learning evaluation** with rigorous **statistical experiment design**.  
The findings supported the AI model’s rollout by proving measurable accuracy and fairness improvements over manual decision-making.

