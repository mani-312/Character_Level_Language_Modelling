import re


def preprocess(corpora):
    preprocessed = []
    for corpus in corpora:
        # Remove the footer of the format: Page [number] Harry Potter ... - J.K. Rowling
        lines = corpus.split('\n')
        lines = [line for line in lines if not 'rowling' in line.lower()]
        lines = [line for line in lines if not 'Page ' in line]
        corpus = '\n'.join(lines)
        # Remove the new-line characters if they are not consecutive.
        corpus = re.sub(r'(\n)(?!\n)', ' ', corpus)
        # Replace multiple new-line characters with a single new-line character.
        corpus = re.sub(r'\n+', '\n', corpus)
        # Remove all non-alphabetical characters except spaces, commas, periods,
        # semi-colons, apostrophes, quotes, question marks, and exclamation marks.
        corpus = re.sub(r'''[^.,'"a-zA-Z\s\;\?\!\n]''', '', corpus)
        # Replace all multiple spaces with single spaces.
        corpus = re.sub(r' +', ' ', corpus)
        # Replace apostrophes followed or preceded by space with a single apostrophe.
        corpus = re.sub(r"' ", "'", corpus)
        corpus = re.sub(r" '", "'", corpus)
        # Replace quotes followed or preceded by space with a single quotes.
        corpus = re.sub(r'" ', '"', corpus)
        corpus = re.sub(r' "', '"', corpus)
        # Replace spaces followed or preceded by new-line characters with a new-line character.
        corpus = re.sub(r' \n', '\n', corpus)
        corpus = re.sub(r'\n ', '\n', corpus)
        # Replace multiple spaces with single spaces.
        corpus = re.sub(r' +', ' ', corpus)
        
        # append to preprocessed list
        preprocessed.append(corpus)
    
    # join all preprocessed corpora into one string and return
    return ' '.join(preprocessed)
