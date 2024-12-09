
# Dark Patterns Survey Analysis

### Researchers
Holly Jordan (holly.jordan@nyu.edu)

Sahil Krishnani (snk9513@nyu.edu)

Ashish Tiwari (ajt9694@nyu.edu)


## Survey Questions

### Intro

#### Q3. This study aims to analyze user interactions with online privacy policies. You'll be asked about term definitions, hypothetical scenarios, and your opinions. Participation is voluntary, and you may stop at any time. Conducted for the Security and Human Behavior course (CS-GY9223/CS-UY3943) at NYU Tandon School of Engineering, Fall 2024. All data will be viewed only by researchers and Professor Rachel Greenstadt. Demographic information will be anonymized. Questions? Contact researchers: Holly Jordan (holly.jordan@nyu.edu), Sahil Krishnani (snk9513@nyu.edu), or Ashish Tiwari (ajt9694@nyu.edu).
- Yes(1)

#### Q26. Selecting 'yes' below indicates that I have read and agree to the full consent form [(click here to view)](https://drive.google.com/file/d/1H2oN9EDqN2IRyvhkDCIgPrOpotHRfHLH/view?usp=sharing) ave read the description of the study, I am currently located in the United States, and I agree to participate in the study.
- Yes (1)

### Demographics (Q25)

#### Q4. Age
- Under 18 (1)
- 18-21 (2)
- 22-25 (3)
- 26-29 (4)
- 30-34 (5)
- 35-44 (6)
- 45-54 (7)
- 55-64 (8)
- 65-74 (9)
- 75+ (10)

#### Q6. Gender
- Male (1)
- Female (2)
- Other (4)

#### Q7. Highest Education Level
- High School diploma or equivalent (1)
- Associate degree (2)
- Bachelor's degree (3)
- Master's degree (4)
- Doctoral degree (Ph.D.) (5)
- Other (please specify) (6)

### Background

#### Q8. On average, how many hours per day do you spend using electronic devices (e.g., smartphones, computers, tablets, gaming consoles)?
- 0-4 hours (1)
- 5-8 hours (2)
- 9-12 hours (3)
- More than 12 hours (4)

#### Q20. How confident are you in your ability to protect your privacy online? (6)
- 1: Strongly disagree
- 2: Somewhat disagree
- 3: Neither agree nor disagree 
- 4: Somewhat agree
- 5: Strongly agree

### Personal actions, knowledge & experiences

#### Q21.1. Do you typically read the terms and conditions when signing up for a new online service?
- 1: Never
- 2: Rarely
- 3: Sometimes
- 4: Often
- 5: Always
- 6: Not Applicable

#### Q21.2. How often do you review and adjust app permissions?
- 1: Never
- 2: Rarely
- 3: Sometimes
- 4: Often
- 5: Always
- 6: Not Applicable

#### Q21.3. When encountering cookie consent pop-ups, how often do you customize your preferences?
- 1: Never
- 2: Rarely
- 3: Sometimes
- 4: Often
- 5: Always
- 6: Not Applicable

#### Q22. Are you familiar with the term "dark patterns" in relation to website or app design?
- Yes (1)
    - #### Q24. Optional: How would you define "dark patterns"?
- No (2)

### Pressure & manipulation

#### Q23. Have you ever felt pressured to create an account when you didn't want to?
- Yes (1)
    - #### Q25. Optional: How did this make you feel?
- No (2)

### Difficulty & obscurity

#### Q16. Have you ever tried to delete an online account but found the process difficult or unclear?
- Yes (1)
    - #### Q17. Optional: Please describe your experience briefly.
- No (2)

#### Q18. Are there any specific online platforms or services that you feel are particularly aggressive or deceptive in their data collection practices? Why?

#### Q25. Have you ever used a data removal service (e.g., Incogni, Aura)?
- Yes (1)
    #### Q23. Optional: What motivated you to use a data removal service?

    #### Q19. Optional: Which data removal service did you use?
- Aura (11)
- Incogni (12)
- DeleteMe (13)
- HelloPrivacy (14)
- Optery (15)
- Other (Please specify) (16)
- No (2)

#### Q20. Are you aware of privacy laws such as the CCPA (California Consumer Privacy Act)?
- Yes (1)
    - #### Q21. Optional: Have you ever tried to use such laws to enhance your privacy? How was your experience?
- No (2)
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

