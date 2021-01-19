# Naive-Bayes Text Categorizer

## Description

​	This project utilizes the Naive-Bayes machine learning algorithm to train and test a text categorization system. To use the program, simply run the Python file, enter the name of a file that lists file paths to labeled training data and unlabeled testing data in your corpora, and wait for the program to complete. The program will generate an output file with predicted category labels for all articles in your test file.

## How to Run

1. Install NLTK and Python 3.8
   - http://www.nltk.org/install.html
   - https://www.python.org/downloads/
2. Open your Python interpreter
3. Go to your project folder
4. `python TextCategorizer.py` to run the Python script.
5. Following user prompts, enter the name of your training and test file.
6. `perl analyze.pl <outputFile> <labeledTestSetFile>` to view the accuracy of the program's results.

## Implementation

- This text categorization system uses NLTK's recommended word tokenizer to tokenize training and test files. No weighting scheme was applied to the resulting tokens, other than ignoring punctuation. 
- The Naive-Bayes machine learning algorithm is used to compute a max likelihood estimate of the category for a given article in the supplied test set.
- A variation of Laplace smoothing (add-α smoothing) was used to implement Naive-Bayes. The smoothing parameter α = 0.1 was used, because it provided the best results experimentally.
- The Snowball stemming algorithm was used to simplify the word-form of tokens during training and testing. Adding a stemmer was found to improve the resulting accuracy of the algorithm, and the Snowball stemmer is said to be an all-around improvement to the original Porter stemmer algorithm. Both are built-in to the NLTK library. When tested experimentally, the Snowball stemmer provided marginally better results.
- Stop lists were tested during development, but ultimately found to provide no significant improvement to the accuracy of predictions, and were removed from the final implementation for simplicity.
- Case sensitivity and POS tagging were not tested.

## Testing

​	In order to evaluate the categorization system's performance for the second and third data sets, the corpora files were portioned into a smaller training and testing set, then categorized manually in order to compare results using the analyze.pl Perl script. Only a small portion of the corpora was used as testing data (under 10%), in order to minimize the time-consuming labor of manual categorization. Because the datasets were sufficiently large, I assumed this would not pose any significant issues.