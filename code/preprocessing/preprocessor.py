import re
from nepali_stemmer.stemmer import NepStemmer
stemmer = NepStemmer()

def remove_nepali_stopwords(text, stopwords):
    """Removes Nepali stopwords from text.

    Parameters
    ----------
    text : str
        Text from where stopwords are to be removed.
    stopwords : list
        List of Nepali stopwords.

    Returns
    -------
    str
        Text without stopwords.
    """
    text = stemmer.stem(text)
    text = ' '.join([word for word in stemmer.stem(text).split() if word not in stopwords])

    return text


def remove_hyphens_and_colons(text, colon_words):
    """Removes hyphens and colons which are not the part of Nepali words. Nepali
    consists of words like 'नि-शब्द' or 'दु:ख'.

    Parameters
    ----------
    text : str
        Text to clean.
    colon_words : list
        List of Nepali colon words.

    Returns
    -------
    str
        Text with hyphens and colons removed.
    """

    processed_text = ''
    for word in text.split():
        if '-' not in word and ':' not in word:
            processed_text += word + ' '

        # Remove hyphens
        elif '-' in word:
            # Remove hyphen if hyphen is suffixed to a word
            if len(word) > 1 and word[-2] != '-' and word[-1] == '-':
                # print("Replacing ", word)
                processed_text += word[:-1] + ' '

            elif word.count('-') != len(word):
                # if there are other non-dash characters after dash
                chr_after_dash = (len(word[word.index('-'):]) >= 1) and (word[word.index('-') + 1] != '-')

                # if there are other non-dash characters before dash
                chr_before_dash = (len(word[:word.index('-')]) >= 1) and (word[word.index('-') - 1] != '-')

                # Preserve word if hyphen is part of word else remove
                if chr_after_dash and chr_before_dash:
                    # print("Removing ", word)
                    processed_text += word + ' '

        # Remove colons
        if ':' in word:
            # if there is colon (:) at the last in a word and that word doesn't contain in the
            # nepali_colon_words.txt
            if word[-1] == ':' and word not in colon_words:
                processed_text += word.replace(':', ' ')
            elif word[0] == ':':
                processed_text += word.replace(':', '')
            else:
                processed_text += word + ' '

    return processed_text


def convert_english_digits_to_nepali(text):
    """Converts English digits into Nepali digits.

    Parameters
    ----------
    text : str
        Text with English digits.

    Returns
    -------
    str
        Text with English digits replaced with Nepali digits.
    """

    # EN - NP digits map (For replacing English digits with Nepali digits)
    en_digits = '0123456789'
    np_digits = '०१२३४५६७८९'
    digits_dict = {}
    for (en, np) in zip(en_digits, np_digits):
        digits_dict[en] = np

    # Replacing English digits with Nepali digits
    for item in digits_dict.items():
        if item[0] in text:
            text = text.replace(item[0], item[1])
    return text


def keep_only_devanagari_script(text):
    """Remove characters which are not the part of Devanagari unicode.
    Other punctuations like '?', '!', ':' and '-' are not removed as they
    are also part of Nepali.

    Parameters
    ----------
    text : str
        Text to clean.

    Returns
    -------
    str
        Clean text.
    """

    # Unicode character ranges that represent Nepali characters in Devanagari script
    devanagari_pattern = re.compile('[^\u0901-\u0903\u0905-\u090C\u090F\u0910\u0913-\u0928\u092A-\u0930\u0932\u0935-\u0939\u093E-\u0944\u0947\u0948\u094B-\u094D\u094F\u0950\u0960-\u096F\u200d?!:-]')
    
    text = devanagari_pattern.sub(' ', text)
    return text


def remove_nepali_digits(text, digit_replacement_character = ''):
    """Remove Nepali digits from text.

    Parameters
    ----------
    text : str
        Text to clean.
    digit_replacement_character : str, optional
        This character replaces any Nepali digit., by default ''

    Returns
    -------
    str
        Clean text.
    """

    text = re.sub('[०-९]', digit_replacement_character, text)
    return text


def wrap_line_terminators_with_spaces(text):
    """Wrap line terminator symbols ('।', '?', '!') with spaces in case these symbols
    are joined with letters.
    
    This aids in better tokenization.

    Parameters
    ----------
    text : str
        Text to clean.

    Returns
    -------
    str
        Clean text.
    """

    terminators_dict = {
        '।': ' । ',
        '?': ' ? ',
        '!': ' ! ',
    }
    
    for item in terminators_dict.items():
        if item[0] in text:
            text = text.replace(item[0], item[1])
    return text


def remove_repeating_line_terminators_or_punctuations(text):
    """Remove repeating line terminators or punctuations.
    
    Symbols supported are '?', '.', '!', '।' and '-'.

    Parameters
    ----------
    text : str
        Text to clean.

    Returns
    -------
    str
        Clean text.
    """

    text = re.sub(r'[\?\.\!\।\-]+(?=[\?\.\!\।\-])', '', text)
    return text


def remove_extra_whitespaces(text):
    """Remove whitespaces if more than one are together.

    Parameters
    ----------
    text : str
        Text to clean.

    Returns
    -------
    str
        Clean text.
    """

    text = re.sub(' +', ' ', text).strip()
    return text


def remove_publication_date_and_location(text):
    """Remove publication dates and location information in the news article.
    
    These can be repeated information in text, and are not of good use.

    Parameters
    ----------
    text : str
        Text to clean.

    Returns
    -------
    str
        Clean text.
    """
    sentences = text.split(' । ')
    if len(sentences[0].split()) <= 4 or ('गते' in sentences[0] and len(sentences[0].split()) <= 5):
        sentences = sentences[1:]
        
    return ' । '.join(sentences)


def clean_nepalipatra_news(text):
    """Clean unnecessary footer text from specific source: Nepalipatra.

    Parameters
    ----------
    text : str
        Text to clean.

    Returns
    -------
    str
        Clean text.
    """
    sentences = []
    for sentence in text.split(' । '):
        if 'info@nepalipatra.com' not in sentence:
            sentences.append(sentence)
        else:
            break
            
    clean = ' । '.join(sentences)
        
    return clean


def export_df(df, path):
    """Export dataframe to CSV.
    
    Additionally, it resets index and drops index column.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame
    path : str
        Destination file.
    """
    df.reset_index(inplace=True, drop=True)
    df.to_csv(path, index=False)
    print(f'Exported {len(df)} rows to path {path}.')