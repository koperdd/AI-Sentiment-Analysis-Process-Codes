import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
import nltk
import os

# Set NLTK data path to Disk D
nltk.data.path.append(r'D:/nltk_data')  # Ensure nltk_data is on Disk D
nltk.download('punkt', quiet=True)  # Download punkt tokenizer if missing
nltk.download('stopwords', quiet=True)  # Download stopwords if missing
nltk.download('wordnet', quiet=True)  # Download WordNet if missing

# Load the dataset
file_path = 'D:/RedditFilteredSubmissions/brunei_submissions_filtered.csv'  # Path to your dataset
data = pd.read_csv(file_path)

# Initialize tokenizer, lemmatizer, and stopwords
tokenizer = TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define spam keywords
spam_keywords = [
    'subscribe', 'click here', 'free', 'promotion', 'dm for details', 
    'buy now', 'link in bio', 'limited offer', 'win big', 'check out'
]

# Step 1: Filter Irrelevant Data with Regex
def filter_irrelevant(text):
    """Filter irrelevant or spammy content using keywords and regex patterns."""
    if isinstance(text, str):
        # Check for excessive links
        if len(re.findall(r'http[s]?://\S+', text)) > 2:  # More than 2 links
            return None

        # Check for repetitive characters
        if re.search(r'(.)\1{4,}', text):  # Any character repeated 4+ times
            return None

        # Check for unusually long words
        if re.search(r'\b\w{20,}\b', text):  # Words longer than 20 characters
            return None

        # Check for repetitive emojis or symbols
        if re.search(r'(\ud83c\udf1f|\ud83d\ude4c|\ud83c\udf89){4,}', text):  # Emojis repeated 4+ times
            return None

        # Check for unwanted phrases
        if re.search(r'(dm me for details|link in bio)', text, re.IGNORECASE):
            return None

        # Check for spam keywords
        for keyword in spam_keywords:
            if keyword in text.lower():
                return None  # Spam keyword found

    return text

data['Title'] = data['Title'].apply(filter_irrelevant)
data['Self Text'] = data['Self Text'].apply(filter_irrelevant)

# Drop rows where both Title and Self Text are irrelevant
data = data.dropna(subset=['Title', 'Self Text'], how='all')

# Step 2: Normalize Text Data
def normalize_text(text):
    """Convert text to lowercase, remove punctuation, and remove extra whitespace."""
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

data['Title'] = data['Title'].apply(normalize_text)
data['Self Text'] = data['Self Text'].apply(normalize_text)

# Step 3: Remove Stop Words and Lemmatize

def remove_stopwords_and_lemmatize(text):
    """Remove stop words and lemmatize text."""
    try:
        if isinstance(text, str):
            words = tokenizer.tokenize(text)  # Use TreebankWordTokenizer
            filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
            return ' '.join(filtered_words)
    except Exception as e:
        print(f"Error during tokenization: {e}")
        return text
    return text

data['Title'] = data['Title'].apply(remove_stopwords_and_lemmatize)
data['Self Text'] = data['Self Text'].apply(remove_stopwords_and_lemmatize)

# Step 4: Validate Dataset Structure
def validate_dataset(df):
    """Validate dataset structure and check for essential columns."""
    required_columns = ['Title', 'Self Text']
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")
validate_dataset(data)

# Save the cleaned data to Disk D
output_dir = 'D:/CleanedData/'
os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
cleaned_file_path = os.path.join(output_dir, 'cleaned_brunei_submissions.csv')
data.to_csv(cleaned_file_path, index=False)

print(f"Cleaned dataset saved at: {cleaned_file_path}")
