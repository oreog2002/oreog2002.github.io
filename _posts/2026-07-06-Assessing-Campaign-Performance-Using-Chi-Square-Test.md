---
layout: post
title: Assessing Campaign Performance Using Chi-Square Test For Independence
image: "/posts/chi_squared.jpg"
tags: [Python, Pandas, SciPy, Statistics, A/B Testing, ABC Grocery]
---

### This project is about assessing the campaign performance of two grocery mailers using the Chi Square Test for independence.
---

Add the necessary libraries. We will use Pandas for loading and manipulating our tabular data, as well as SciPy's stats module to compute our contingency table statistics and critical values.

```python
import pandas as pd
from scipy.stats import chi2_contingency, chi2
```

First, we'll load the marketing campaign dataset from our Excel sheet into a Pandas DataFrame.

```python
campaign_data = pd.read_excel("grocery_database.xlsx", sheet_name = "campaign_data")
```

Next, we filter our data. Because we want to directly compare the performance of our two active mailer types --- Mailer 1 and Mailer 2 --- against each other, we use the ".loc" indexer to remove the "Control" group records from the dataset.

```python
campaign_data = campaign_data.loc[campaign_data["mailer_type"] != "Control"]
```

Now, we summarize our data to get our observed frequencies. By using "pd.crosstab", we cross-tabulate the mailer types against the signup flags to see exactly how many people did or didn't sign up for each group, extracting just the raw matrix values using ".values".

```python
observed_values = pd.crosstab(campaign_data["mailer_type"], campaign_data["signup_flag"]).values
observed_values
```

To get a quick baseline understanding of our performance, we can manually calculate and print out the raw signup conversion rates for Mailer 1 and Mailer 2.

```python
mailer1_signup_rate = 123 / (252 + 123)
mailer2_signup_rate = 127 / (209 + 127)
print(mailer1_signup_rate, mailer2_signup_rate)
```

---
#### Setting up our framework for decision making:

Before running the mathematical test, we formally state our hypotheses and establish our acceptance criteria. We set our alpha level ("accpt_crit") to 0.05, meaning we require a 95% confidence level to declare a significant relationship.

```python
null_hyp = "No relationship between mailer type and signup rate. They're independent."
alt_hyp = "There IS a relationship between mailer type and signup rate. NOT independent."
accpt_crit = 0.05 # Acceptance criteria
```

---
#### Now, let's run the statistical calculations!

We pass our observed frequency matrix into "chi2_contingency". This automatically calculates our Chi-Square test statistic ("chi2_stat"), our p-value, the degrees of freedom ("dof"), and an array of what frequencies we'd expect to see if the null hypothesis were true ("exp_value").

```python
chi2_stat, p_value, dof, exp_value = chi2_contingency(observed_values, correction = False)
print(chi2_stat, p_value)
```

To determine whether our test statistic is high enough to be meaningful, we find the critical value boundary line for our specific test using the percent point function ("chi2.ppf") based on our degrees of freedom and chosen alpha.

```python
crit_value = chi2.ppf(1 - accpt_crit, dof)
print(crit_value)
```

---
#### Finally, let's interpret and print out our findings!

We evaluate our test using two identical logic tracks. First, we compare our calculated Chi-Square statistic against our critical value boundary.

```python
if chi2_stat >= crit_value:
    print(f"As our chi-square statistic of {chi2_stat} is higher than our critical value of {crit_value} - we reject the null hypothesis, and conclude that: {alt_hyp}")
else:
    print(f"As our chi-square statistic of {chi2_stat} is lower than our critical value of {crit_value} - we retain the null hypothesis, and conclude that: {null_hyp}")
```

Second, we verify the result by evaluating the p-value against our 0.05 alpha threshold. Both evaluation methods will yield the exact same decision on whether our mailers had a genuinely statistically significant impact on user behavior.

```python
if p_value <= accpt_crit:
    print(f"As our p-value of {p_value} is lower than our acceptance criteria of {accpt_crit} - we reject the null hypothesis, and conclude that: {alt_hyp}")
else:
    print(f"As our p-value of {p_value} is higher than our acceptance criteria of {accpt_crit} - we retain the null hypothesis, and conclude that: {null_hyp}")
```

And just like that, we've successfully processed an A/B test and evaluated data independence using Python and SciPy!
