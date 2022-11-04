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
We have a huge overall dataset, so, for experiment purpose, we will be using small subsets of original dataset.

We have 4 different sizes of datasets:

- **30K** - 1500 rows per class
- **15K** - 500 rows per class
- **10K** - 250 rows per class
- **1K** - 50 rows per class

Moreover, we are experimenting with datasets having 20 classes and 16 classes. So these both variants have different data size versions
as number of rows per classes as mentioned above.

_**Note:** 30K size dataset is sampled from the original dataset, and only content having word count more than 20 and less than 1000 is taken. Additionally, these dataset sizes are pre-processed with before-mentioned steps to reduce the overhead later._


## Train Valid Test Split
Each of the data size variants are splitted in the ratio of 80 : 10 : 10 using stratified sampling as train, validation and test splits and are suffixed with '_train', '_valid' and '_test' in the file names. Those files with stop words removed are suffixed with '_nosw'.