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

def analyze_open_ended_question(df, column_name, num_topics=5):
    # Preprocess the text
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        if isinstance(text, str):
            tokens = word_tokenize(text.lower())
            tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
            tokens = [token for token in tokens if token not in stop_words]
            return ' '.join(tokens)
        return ''

    preprocessed_texts = df[column_name].apply(preprocess_text)

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

def create_combined_wordcloud(results,title):
    # Combine all words from all topics into a single string
    all_words = ' '.join([' '.join(topic.split(': ')[1].split(', ')) for topic in results['topics']])

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)

    # Save the word cloud as a PNG file
    plt.savefig('results/'+title+'_wordcloud.png', dpi=300)

def main():
    # Load the cleaned DataFrame from the pickle file
    df = pd.read_pickle('cleaned_survey_data.pkl')



    q24 = "How would you define \"dark patterns\"?"
    q24_results = analyze_open_ended_question(df, 'Q24')
    create_combined_wordcloud(q24_results,q24)

    # Create age groups
    df['Age_Group'] = pd.cut(df['Q4'].astype(int), bins=[0, 25, 35, 45, 55, 100], labels=['18-25', '26-35', '36-45', '46-55', '55+'])

    # Create a cross-tabulation
    cross_tab = pd.crosstab([df['Age_Group'], df['Gender - Selected Choice']],
                            df['Click to write the question text - How confident are you in your ability to protect your privacy online?'])

    # Perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(cross_tab)

    # Visualize the data
    plt.figure(figsize=(12, 8))
    sns.heatmap(cross_tab, annot=True, cmap='YlGnBu')
    plt.title('Confidence in Privacy Protection by Age and Gender')
    plt.tight_layout()
    plt.show()

    print(f"Chi-square statistic: {chi2}")
    print(f"p-value: {p_value}")



if __name__ == "__main__":
    main()