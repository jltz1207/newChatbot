import spacy
from spellchecker import SpellChecker
from spacy.matcher import Matcher
def correct_spelling(text):
    spell = SpellChecker()
    words = text.split()
    corrected_words = []
    for word in words:
        if word.isalpha():  # Check if the word contains only alphabetic characters
            misspelled = spell.unknown([word])
            if misspelled:
                candidates = spell.candidates(word)
                if candidates:
                    # Select the candidate with the closest length to the original word
                    corrected_word = min(candidates, key=lambda x: abs(len(x) - len(word)))
                    # Preserve capitalization
                    if word.istitle():
                        corrected_word = corrected_word.capitalize()
                    corrected_words.append(corrected_word)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)
    corrected_text = ' '.join(corrected_words)
    return corrected_text

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Define patterns for multi-word phrases with priority
patterns = [
    [{"POS": "VERB", "TAG": "VBG"}, {"POS": "DET", "OP": "?"}, {"POS": "NOUN"}],  # Verb with -ing suffix followed by optional determiner and a noun
    [{"POS": "VERB", "TAG": "VBG"}],  # Single verb with -ing suffix
    [{"POS": "AD", "TAG": "VBG"}, {"POS": "DET", "OP": "?"}, {"POS": "NOUN"}],
    [{"POS": "ADJ", "OP": "?"},{"POS": "NOUN"}],  # One or more nouns
    
]

name_patterns = [ # noun related, taking the greedy
    [{"POS": "PROPN", "OP": "+"}],
    [{"POS": "NOUN", "OP": "+"}],
]

matcher.add("Name", name_patterns, greedy="LONGEST")

org_pattern = [{"ENT_TYPE": "ORG"}]
matcher.add("ORG", [org_pattern])

# Add patterns to the matcher
matcher.add("NOUN_PHRASES", patterns)

def extract_interest(text):
    corrected_text = correct_spelling(text)
    doc = nlp(corrected_text)
    matches = matcher(doc)
    phrases = []
    
    for match_id, start, end in matches:
        span = doc[start:end]
        normalized_span = ' '.join([token.lemma_ for token in span if token.pos_ != 'DET']) 
        
        # Check if the phrase starts with a verb ending in -ing
        if span[0].tag_ == "VBG":
            # Extract the noun part and add it as a separate phrase
            noun_span = ' '.join([token.lemma_ for token in span[1:] if token.pos_ != 'DET'])
            if noun_span:
                phrases.append(normalized_span.lower())  # Convert to lowercase
                phrases.append(noun_span.lower())  # Convert to lowercase
        else:
            phrases.append(normalized_span.lower())  # Convert to lowercase
    
    # Remove duplicates and sort
    unique_phrases = sorted(set(phrases), key=lambda x: text.find(x))
    print("Extracted keywords:", unique_phrases)
    
    return unique_phrases

with open('./test.txt', 'r') as file:
    user_response = file.read()

corrected_text = correct_spelling(user_response)
print(f"Corrected Keywords: {corrected_text}")

keywords = extract_interest(corrected_text)
print(f"Extracted Keywords: {keywords}")