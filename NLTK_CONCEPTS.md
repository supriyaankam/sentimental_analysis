# NLTK Concepts Used in This Project

The NLP pipeline is implemented mainly in `sentiment_app/nlp_engine.py`. It uses NLTK for text preparation, sentence analysis, and VADER sentiment scoring. Other tools in the pipeline, such as TextBlob and NRCLex, are separate libraries and are identified below so their roles are clear.

## NLTK Concepts

### 1. Sentence tokenization

`nltk.tokenize.sent_tokenize()` splits an input into sentences. The resulting sentence list is used for sentence counts and for per-sentence VADER analysis. Sentence-level analysis is capped at 200 sentences to limit processing time.

### 2. Word tokenization

`nltk.tokenize.word_tokenize()` splits cleaned text into word tokens. The tokens are used to calculate word count and provide input to later preprocessing steps.

### 3. Stopword removal

The project loads English stopwords with `nltk.corpus.stopwords.words('english')`. During preprocessing, stopwords such as common function words are removed from the token list used for keyword extraction and POS tagging. The original text is still used for sentiment scoring, so stopword removal does not remove context from VADER or TextBlob analysis.

### 4. Lemmatization

`nltk.stem.WordNetLemmatizer` reduces filtered words to dictionary forms (for example, plural nouns to singular forms). The resulting lemmas are used to find the most frequent keywords. This step requires the WordNet corpus.

### 5. Part-of-speech (POS) tagging

`nltk.tag.pos_tag()` assigns grammatical tags to filtered tokens, such as noun, verb, adjective, or adverb. Up to 30 tags are passed to the result page for display. Since stopwords have already been removed, these tags describe the filtered words rather than the complete grammatical structure of the original sentence.

### 6. VADER sentiment analysis

`nltk.sentiment.vader.SentimentIntensityAnalyzer` scores the original text and each detected sentence. VADER returns positive, negative, neutral, and compound scores. The compound score is mapped to positive (at least `0.05`), negative (at most `-0.05`), or neutral. The application also checks separated clauses for mixed positive and negative sentiment.

## Other NLP Libraries in the Pipeline

- **TextBlob** calculates polarity and subjectivity. It is not an NLTK component. Its label is combined with the overall VADER label.
- **NRCLex** is an optional library that uses the NRC Emotion Lexicon to estimate eight emotion categories: joy, sadness, anger, fear, surprise, disgust, trust, and anticipation. It is not part of NLTK, and emotion analysis is skipped if the package is unavailable.
- Mixed-clause detection and some food-quality vocabulary cues are custom application logic, not built-in NLTK features.

## Imported but Currently Unused

`PorterStemmer` is instantiated in `nlp_engine.py`, but the current analysis pipeline does not call it. Stemming is therefore not currently applied to the input.

## Simplified Processing Flow

1. Limit the input size and split it into sentences.
2. Clean a copy of the text, tokenize it into words, remove English stopwords, and lemmatize the remaining words.
3. POS-tag filtered words and extract frequent keywords.
4. Analyze the original text and sentences with VADER; analyze overall polarity with TextBlob.
5. Check clauses for positive and negative signals to identify mixed sentiment, then optionally estimate emotions with NRCLex.

## NLTK Resources

The pipeline expects NLTK data resources to be available in the runtime environment. In particular, it uses the English stopwords corpus, the Punkt sentence tokenizer data, the averaged perceptron POS tagger data, and WordNet for lemmatization. The app does not download NLTK resources automatically at startup.
