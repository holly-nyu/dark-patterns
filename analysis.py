import pandas as pd
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.sentiment import SentimentIntensityAnalyzer
import seaborn as sns
from scipy.stats import chi2_contingency

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
stop_words = set(stopwords.words('english'))

def preprocess_text(text, exclude_words=None):
    # Initialize lemmatizer here
    lemmatizer = WordNetLemmatizer()

    if isinstance(text, str):
        # Tokenize the text
        tokens = word_tokenize(text.lower())
        # Lemmatize the tokens
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
        # Remove stopwords
        tokens = [token for token in tokens if token not in stop_words]
        # Remove any words from the exclude list
        if exclude_words:
            tokens = [token for token in tokens if token not in exclude_words]
        return ' '.join(tokens)
    return ''


def analyze_open_ended_question(df, column_name, num_topics=5, exclude_words=None):
    # Preprocess the text
    preprocessed_texts = df[column_name].apply(preprocess_text, exclude_words=exclude_words)

    # Topic Modeling
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(preprocessed_texts)

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)

    # Get top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

    # Sentiment Analysis
    sia = SentimentIntensityAnalyzer()
    sentiments = df[column_name].apply(lambda x: sia.polarity_scores(str(x)) if isinstance(x, str) else None)

    avg_sentiment = sentiments.apply(lambda x: x['compound'] if x else None).mean()

    return {
        'topics': topics,
        'average_sentiment': avg_sentiment
    }

def create_combined_wordcloud(results, title, exclude_words=None):
    # Combine all words from all topics into a single string
    all_words = ' '.join([' '.join(topic.split(': ')[1].split(', ')) for topic in results['topics']])

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

    # Exclude specific words from the word cloud
    if exclude_words:
        wordcloud.words_ = {word: freq for word, freq in wordcloud.words_.items() if word not in exclude_words}

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)

    # Save the word cloud as a PNG file
    plt.savefig('results/'+title+'_wordcloud.png', dpi=300)

def age_gender_confidence(df):
    # Convert columns to numeric and map age and gender values
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Gender'] = pd.to_numeric(df['Gender'], errors='coerce')

    # Define mappings
    age_mapping = {
        1: "Under 18",
        2: "18-21",
        3: "22-25",
        4: "26-29",
        5: "30-34",
        6: "35-44",
        7: "45-54",
        8: "55-64",
        9: "65-74",
        10: "75+"
    }

    gender_mapping = {
        1: "Male",
        2: "Female",
        4: "Other"
    }

    # Apply mappings
    df['Age'] = df['Age'].map(age_mapping).fillna("Unknown")
    df['Gender'] = df['Gender'].map(gender_mapping).fillna("Unknown")

    # Filter the data by gender
    male_df = df[df['Gender'] == "Male"]
    female_df = df[df['Gender'] == "Female"]

    # Create cross-tabulations
    male_cross_tab = pd.crosstab(
        male_df['Age'],
        male_df['Confidence']
    )

    female_cross_tab = pd.crosstab(
        female_df['Age'],
        female_df['Confidence']
    )

    # Plot heatmap for males
    plt.figure(figsize=(12, 8))
    sns.heatmap(male_cross_tab, annot=True, cmap='Blues', cbar=False)
    plt.title('Confidence in Privacy Protection by Age (Male)')
    plt.tight_layout()
    plt.savefig('results/Male_Confidence_Heatmap.png', dpi=300)
    # plt.show()

    # Plot heatmap for females
    plt.figure(figsize=(12, 8))
    sns.heatmap(female_cross_tab, annot=True, cmap='Reds', cbar=False)
    plt.title('Confidence in Privacy Protection by Age (Female)')
    plt.tight_layout()
    plt.savefig('results/Female_Confidence_Heatmap.png', dpi=300)
    # plt.show()


def create_heatmap_education_vs_familiarity(df):
    """
    Creates a heatmap to visualize the relationship between Education Level ('Education')
    and Familiarity with "Dark Patterns" ('FamiliarDP').
    """
    # Filter and copy the DataFrame
    filtered_df = df[(df['Education'] != 0) & (df['FamiliarDP'] != 0)].copy()

    # Map FamiliarDP values to 'Yes' and 'No'
    familiarity_mapping = {'1': 'Yes', '2': 'No'}
    filtered_df['FamiliarDP'] = filtered_df['FamiliarDP'].map(familiarity_mapping)

    education_mapping = {'1': 'High School diploma or equivalent',
                         '2': 'Associate degree', 
                         '3': 'Bachelor\'s degree',
                         '4': 'Master\'s degree', 
                         '5': 'Doctoral degree (Ph.D.)',
                         '6': 'Other (please specify)',
                        }
    filtered_df['Education'] = filtered_df['Education'].map(education_mapping)
 

    # Check for emptiness
    if filtered_df.empty:
        print("Filtered DataFrame is empty. No data to plot.")
        return

    # Create a crosstab for the heatmap
    heatmap_data = pd.crosstab(filtered_df['FamiliarDP'], filtered_df['Education'])

    # Check if heatmap_data is empty
    if heatmap_data.empty:
        print("No data for the heatmap after filtering.")
        return

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', cbar=True, fmt='d')
    plt.title('Education Level vs. Familiarity with Dark Patterns')
    plt.xlabel('Familiarity with Dark Patterns')
    plt.xticks(rotation=45)
    plt.ylabel('Education Level')
    plt.tight_layout()

    # Save and/or show the heatmap
    plt.savefig('results/Education_vs_FamiliarDP_Heatmap.png', dpi=300)
    # plt.show()


def correlate_privacy_actions(df, x_axis_col, title, x_axis_title, file_name, privacy_cols=None):
    """
    Analyzes and plots the relationship between screen time and multiple privacy actions.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing survey data.
    - screen_time_col (str): Column name for daily device usage (Screen Time).
    - privacy_cols (list of str): List of column names representing privacy actions.

    Returns:
    - None: Saves the plot as a PNG file and displays it.
    """
    if privacy_cols is None:
        privacy_cols = ['Terms&Conditions', 'ReviewPermissions', 'Cookies']

    # Ensure columns exist in the DataFrame
    missing_cols = [col for col in [x_axis_col] + privacy_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

    # Convert the columns to numeric
    df[x_axis_col] = pd.to_numeric(df[x_axis_col], errors='coerce')
    for col in privacy_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Melt the DataFrame to have a single column representing privacy actions
    df_melted = df.melt(
        id_vars=[x_axis_col], 
        value_vars=privacy_cols, 
        var_name='PrivacyAction', 
        value_name='Frequency'
    )

    # Drop NaN values for analysis
    df_cleaned = df_melted.dropna(subset=[x_axis_col, 'Frequency'])

    # Generate the line plot
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df_cleaned,
        x=x_axis_col,
        y='Frequency',
        hue='PrivacyAction',
        ci='sd'
    )

    # Customize the plot
    plt.title(title, fontsize=14)
    plt.xlabel(x_axis_title, fontsize=12)
    plt.ylabel("Frequency of Privacy Actions", fontsize=12)
    plt.legend(title="Privacy Actions", loc='upper left')
    plt.grid(True)

    # Save the plot as a PNG file
    output_path = 'results/' + file_name + '.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

    # Optionally display the plot
    plt.show()





def main():
    # Load the cleaned DataFrame from the pickle file
    df = pd.read_pickle('survey_12-8.pkl')

    # 
    correlate_privacy_actions(df, x_axis_col='ScreenTime', 
                              title='Daily Device Usage vs. Privacy Actions',
                              x_axis_title='Daily Screen Time (hours)', 
                              file_name='ScreenTime_vs_PrivacyActions',
                              privacy_cols=['Terms&Conditions', 'ReviewPermissions', 'Cookies'])
    correlate_privacy_actions(df, x_axis_col='Confidence', 
                              title='Confidence in Ability to Protect Privacy vs. Actions',
                              x_axis_title='Confidence in ability to protect privacy online', 
                              file_name='PrivacyConfidence_vs_PrivacyActions',
                              privacy_cols=['Terms&Conditions', 'ReviewPermissions', 'Cookies'])


    # Create a heatmap for Education vs FamiliarDP
    create_heatmap_education_vs_familiarity(df)

    # Analyze correlation between Age, Gender, and
    # ranked confidence in ability to protect their privacy online
    # Creates heatmaps for Female and Male
    age_gender_confidence(df)

 
    # Analyze common sentiments reported by participants for question
    # create a wordmap of the most common themes
    exclude_words = ['dark', 'patterns', 'user'] # Words to exclude
    results = analyze_open_ended_question(df, 'DefineDP_TEXT', 5, exclude_words=exclude_words)
    create_combined_wordcloud(results, "How would you define \"dark patterns\"?")

if __name__ == "__main__":
    main()