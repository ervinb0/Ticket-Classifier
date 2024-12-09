import re, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from langdetect import detect, DetectorFactory

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

DetectorFactory.seed = 0

#Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# List of English stopwords
STOP_WORDS = set(stopwords.words('english'))

def remove_noise(text):
    """Remove special characters, numbers, and extra spaces from text."""
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))  # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()      # Remove extra spaces
    return text

def remove_emojis_and_links(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    link_pattern = re.compile(r'http\S+|www\S+')
    text = emoji_pattern.sub(r'', text)
    text = link_pattern.sub(r'', text)
    return text

def remove_stopwords_lemmatizer(text):
    """Remove stopwords from the text and lemmatize the tokens."""
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in STOP_WORDS]
    lemmatized_tokens = [lemmatizer.lemmatize(word.lower()) for word in filtered_words]  # Lemmatize each token
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

def is_english(text):
    """
    Checks if the input text is in English.
    Returns True if English, otherwise False.
    """
    try:
        return detect(text) == 'en'
    except:
        return False  # Return False if detection fails


def preprocess_text(text):
    """
    Preprocess the text by applying the necessary steps.
    Args:
        text (str): The input text.
    Returns:
        str: The preprocessed text.
    """
    # Check if text is in English
    if not is_english(text):
        raise ValueError("The input text is not in English.")
    
    # Lowercase the text
    text = text.lower()

    # Remove noise (special characters and numbers)
    text = remove_noise(text)
        
    # Remove emoticons and links
    text = remove_emojis_and_links(text)

    # Remove stopwords        
    text = remove_stopwords_lemmatizer(text)

    return text
