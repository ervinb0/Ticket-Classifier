import re

import nltk
from langdetect import DetectorFactory, LangDetectException, detect
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

DetectorFactory.seed = 0

# Language detection is unreliable on very short texts ("add dark mode" is
# detected as Swedish), so only texts with at least this many words are checked.
MIN_WORDS_FOR_LANG_CHECK = 5

NLTK_RESOURCES = {
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
}


def ensure_nltk_resources():
    """Download the NLTK resources we need, but only if they are missing."""
    for name, path in NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.data.find(path + ".zip")
            except LookupError:
                nltk.download(name, quiet=True)


ensure_nltk_resources()

lemmatizer = WordNetLemmatizer()
QUESTION_WORDS = {"what", "why", "how", "when", "where", "who", "whom", "which"}

NEGATION_WORDS = {
    "not",
    "no",
    "nor",
    "never",
    "isn",
    "aren",
    "wasn",
    "weren",
    "hasn",
    "haven",
    "hadn",
    "doesn",
    "don",
    "didn",
    "couldn",
    "wouldn",
    "shouldn",
    "won",
    "can",
    "cannot",
    "mustn",
    "needn",
    "shan",
    "t",
}

WORDS_TO_PRESERVE = QUESTION_WORDS | NEGATION_WORDS
STOP_WORDS = set(stopwords.words("english")) - WORDS_TO_PRESERVE

EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f1e0-\U0001f1ff"  # flags (iOS)
    "]+",
    flags=re.UNICODE,
)
LINK_PATTERN = re.compile(r"http\S+|www\S+")

QUESTION_MARK_PATTERN = re.compile(
    r"(?<=[a-z])\?(?=\s|$)"
)  # "?" ending a word, not "[?:?]" in stack traces


def remove_emojis_and_links(text):
    text = EMOJI_PATTERN.sub("", text)
    text = LINK_PATTERN.sub("", text)
    return text


def convert_question_marks_to_words(text):
    """Turn a '?' that ends a word into the token 'qmark'."""
    return QUESTION_MARK_PATTERN.sub(" qmark ", text)


def remove_noise(text):
    """Remove special characters, numbers, and extra spaces from text."""
    text = re.sub(r"[^a-zA-Z\s]", " ", str(text))  # Keep only letters and spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords_lemmatizer(text):
    """Remove stopwords from the text and lemmatize the tokens."""
    words = word_tokenize(text)
    return " ".join(lemmatizer.lemmatize(w) for w in words if w not in STOP_WORDS)


def is_english(text):
    """Return False only when we are confident the text is not English."""
    if len(text.split()) < MIN_WORDS_FOR_LANG_CHECK:
        return True
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def preprocess_text(text):
    """Clean, filter and lemmatize a ticket. Returns None for non-English or empty input."""
    text = str(text).lower()

    if not is_english(text):
        return None

    # Links must go before noise removal, otherwise "https://..." turns into words
    text = remove_emojis_and_links(text)
    text = convert_question_marks_to_words(text)
    text = remove_noise(text)
    text = remove_stopwords_lemmatizer(text)

    return text or None
