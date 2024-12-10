
# Dark Patterns Survey Analysis

### Researchers
Holly Jordan (holly.jordan@nyu.edu)

Sahil Krishnani (snk9513@nyu.edu)

Ashish Tiwari (ajt9694@nyu.edu)


## Survey Questions

### Intro

#### Q1. Consent - This study aims to analyze user interactions with online privacy policies. You'll be asked about term definitions, hypothetical scenarios, and your opinions. Participation is voluntary, and you may stop at any time. Conducted for the Security and Human Behavior course (CS-GY9223/CS-UY3943) at NYU Tandon School of Engineering, Fall 2024. All data will be viewed only by researchers and Professor Rachel Greenstadt. Demographic information will be anonymized. Questions? Contact researchers: Holly Jordan (holly.jordan@nyu.edu), Sahil Krishnani (snk9513@nyu.edu), or Ashish Tiwari (ajt9694@nyu.edu).
- 1: Yes

#### Selecting 'yes' below indicates that I have read and agree to the full consent form [(click here to view)](https://drive.google.com/file/d/1H2oN9EDqN2IRyvhkDCIgPrOpotHRfHLH/view?usp=sharing) ave read the description of the study, I am currently located in the United States, and I agree to participate in the study.
- 1: Yes

### Demographics

#### Q2. Age
- 1: Under 18
- 2: 18-21
- 3: 22-25
- 4: 26-29
- 5: 30-34
- 6: 35-44
- 7: 45-54
- 8: 55-64
- 9: 65-74
- 10: 75+

#### Q3. Gender
- 1: Male
- 2: Female
- 3: Other

#### Q4. Highest Education Level
- 1: High School diploma or equivalent
- 2: Associate degree
- 3: Bachelor's degree
- 4: Master's degree
- 5: Doctoral degree (Ph.D.)
- 6: Other (please specify)

### Background

#### Q5. On average, how many hours per day do you spend using electronic devices (e.g., smartphones, computers, tablets, gaming consoles)?
- 1: 0-4 hours
- 2: 5-8 hours
- 3: 9-12 hours
- 4: More than 12 hours

#### Q6. How confident are you in your ability to protect your privacy online?
- 1: Strongly disagree
- 2: Somewhat disagree
- 3: Neither agree nor disagree 
- 4: Somewhat agree
- 5: Strongly agree

### Personal actions, knowledge & experiences

#### Q7. Do you typically read the terms and conditions when signing up for a new online service?
- 1: Never
- 2: Rarely
- 3: Sometimes
- 4: Often
- 5: Always
- 6: Not Applicable

#### Q8. How often do you review and adjust app permissions?
- 1: Never
- 2: Rarely
- 3: Sometimes
- 4: Often
- 5: Always
- 6: Not Applicable

#### Q9. When encountering cookie consent pop-ups, how often do you customize your preferences?
- 1: Never
- 2: Rarely
- 3: Sometimes
- 4: Often
- 5: Always
- 6: Not Applicable

#### Q10. Are you familiar with the term "dark patterns" in relation to website or app design?
- 1: Yes
- 2: No

#### IF YES Q10.1. Optional: How would you define "dark patterns"?

### Pressure & manipulation

#### Q11. Have you ever felt pressured to create an account when you didn't want to?
- 1: Yes
- 2: No

#### IF YES Q11.1. Optional: How did this make you feel?

### Difficulty & obscurity

#### Q12. Have you ever tried to delete an online account but found the process difficult or unclear?
- 1: Yes
- 2: No

#### IF YES Q12.1. Optional: Please describe your experience briefly.

#### Q13. Are there any specific online platforms or services that you feel are particularly aggressive or deceptive in their data collection practices? Why?

#### Q14. Have you ever used a data removal service (e.g., Incogni, Aura)?
- 1: Yes
- 2: No

#### IF YES Q14.1. Optional: What motivated you to use a data removal service?

#### IF YES Q14.2. Optional: Which data removal service did you use?
- 1: Aura
- 2: Incogni
- 3: DeleteMe
- 4: HelloPrivacy
- 5: Optery
- 6: Other (Please specify)

#### Q15. Are you aware of privacy laws such as the CCPA (California Consumer Privacy Act)?
- 1: Yes
- 2: No

#### IF YES Q15.1. Optional: Have you ever tried to use such laws to enhance your privacy? How was your experience?

## Run Locally

Clone the project

```bash
  git clone https://github.com/holly-nyu/dark-patterns.git
```

Go to the project directory

```bash
  cd dark-paterns
```

Install dependencies

```bash
  pip install pandas
  pip install nltk
  pip install wordcloud
  pip install matplotlib
  pip install seaborn
  pip install scikit-learn
  pip install scipy
```

Create the dataframe

```bash
  python create_df.py
```

Run the analysis

```bash
  python analysis.py
```

