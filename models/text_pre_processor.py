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
    try:
        # Split text into words
        words = text.split()
        # Store English words
        english_words = []
        
        for word in words:
            try:
                # Check if the word is in English
                if detect(word) == 'en':
                    english_words.append(word)
            except:
                # If language detection fails for a word, skip it
                continue
        
        # Join the English words back together
        filtered_text = ' '.join(english_words)
        
        # Return the filtered text if there are any English words, otherwise return None
        return filtered_text if english_words else None
        
    except Exception as e:
        print(f"Error in language detection: {e}")
        return None


def preprocess_text(text):
   
    # Lowercase the text
    text = text.lower()
    
    # Check if text is English
    filtered_text = is_english(text)
    if filtered_text is None:
        return None

    # Remove noise (special characters and numbers)
    text = remove_noise(filtered_text)
        
    # Remove emoticons and links
    text = remove_emojis_and_links(text)

    # Remove stopwords        
    text = remove_stopwords_lemmatizer(text)

    return text
