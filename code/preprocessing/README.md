# Preprocessing

## Cleaning Nepali text
Here we define various methods which are required to clean Nepali text. Some features of Nepali language are: 
* Nepali is written using Devanagari script.
* The words are separated by spaces.
* It does not have capitalization.

Some basic preprocessing steps that can be applied are:
1. Convert English digits to Nepali digits.
2. Remove repetition of line terminator symbols and punctuations.
3. Remove hyphens and colon symbols.
4. Removing non-Devanagari characters.
5. Separating line terminators attached to words.
6. Remove publication dates and location info (for news articles)
7. Remove extra white spaces
8. Remove Nepali stop words
9. Remove Nepali digits

Its example usage can be found in `preprocessing_example.ipynb` notebook.

## Dataset sampling (different sizes)

## Train Valid Test Split