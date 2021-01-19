from math import log
import string
import time

from nltk.stem import SnowballStemmer     # NLTK library stemmer & tokenizer
from nltk.tokenize import word_tokenize


# Ask for user input for filenames, store in variables.
def query_user():
    training_filename = input("Enter training filename: ")
    testing_filename = input("Enter testing filename: ")

    return training_filename, testing_filename


# Create an output file with certain content
def generate_output(contents):
    filename = input("Name your output file: ")

    output_file = open(filename, 'w') # Create output file
    
    for entry in contents: 
        output_file.write(entry)   # Write each string stored in the content array.
    
    output_file.close()  # When finished writing, close the file.


def naive_bayes(training_filename, testing_filename):
       # Declare training variables
    single_word_counter, total_word_counter, article_counter = dict(), dict(), dict()    # Dictionaries to count how many words, tokens, and articles (corpus files) belong to each category.
    stemmer = SnowballStemmer("english")     # The Snowball stemmer algorithm will be used to reduce the word-forms of tokens.

    # Training
    print("Training...")
    start_training = time.time()

    with open(training_filename, 'r') as f:
        for line in f:  # Open training file for reading, and parse through each line.
            # Split each line into its own file and category.
            file_and_category = line.split()   
            cur_filename = file_and_category[0]
            cur_category = file_and_category[1]
            
            # Update counter tracking how many articles belong to each category.
            if cur_category not in article_counter:
                article_counter[cur_category] = 0;  # Create new categories.
            article_counter[cur_category] += 1  # Increment existing categories.

            # Open, read and tokenize current line's associated article (corpus file) 
            cur_article = open(cur_filename, 'r') 
            tokenized_article = word_tokenize(cur_article.read())   # Using NLTK word tokenizer

            # Loop through each token in the article, to update word count.
            for token in tokenized_article:
                token = stemmer.stem(token) # Apply stemming algorithm to current token.

                # Add new categories to  word counters.
                if (token, cur_category) not in single_word_counter:
                    single_word_counter[(token,cur_category)] = 0
            
                if cur_category not in total_word_counter:
                    total_word_counter[cur_category] = 0

                # Update counter tracking how many times the current token has appeared in this category.
                single_word_counter[(token, cur_category)] += 1
                # Update counter tracking how many total tokens (words) are in this category.
                total_word_counter[cur_category] += 1

    end_training = time.time()
    print("Training complete. Took " + str(round(end_training-start_training)) + " seconds.")


    # Declare testing variables
    category_predictions = []    # Array to store generated category predictions to be printed to output file.
    alpha = 0.1   # Laplace (add-alpha) smoothing parameter

    # Testing
    print("Testing...")
    start_testing = time.time()

    with open(testing_filename, 'r') as f:
        for line in f:      # Open unlabeled testing file for reading. Parse through each line.
            line = line.strip()
            cur_article = open(line, 'r')  # Open article associated with each line for reading.
            tokenized_article = word_tokenize(cur_article.read())   

            category_clps = dict() # Dictionary for storing conditional log probabilities per category
            word_counter = dict()  # Dictionary to track word count of each word in the article.

            for token in tokenized_article:    # Tokenize file into words. Loop through each word.
                token = stemmer.stem(token)    # Apply stemming algorithm to token.

                # Update word counter. 
                if token not in list(string.punctuation):   # Don't count tokens for punctuation marks.
                    if token not in word_counter:          
                        word_counter[token] = 0.            # Create entry for new tokens
                    word_counter[token] += 1.               # Increment a word's count when it is read in the article

            # Apply the Naive-Bayes algorithm using Laplace Smoothing with custom parameter.
            for category in total_word_counter.keys():  
                clp = 0. # Conditional log probability of article being in this category.
                prior = article_counter[category] / sum(article_counter.values()) # Prior probability for category, derived from training.
                
                # Calculate conditional log probabilities for each category
                for word, count in word_counter.items():   
                    if (word, category) in single_word_counter:
                        word_count = single_word_counter[(word, category)] + alpha
                    else: 
                        word_count = alpha

                    cur_clp = count * log(word_count / (total_word_counter[category] + alpha * len(word_counter)))
                    clp += cur_clp
                    
                    category_clps[category] = clp + log(prior)
            
            category_prediction = max(category_clps, key=category_clps.get)   # Max Likelihood estimation
            
            category_predictions.append(line + " " + category_prediction + "\n")    # Generate string to add to output file, and store in array.

    end_testing = time.time()
    print("Testing complete. Took " + str(round(end_testing-start_testing)) + " seconds.")

    return category_predictions


def categorize_text():   # Function to categorize text, given a training file, testing file, and output filename.
    # Get test and training file names.
    training_filename, testing_filename = query_user()  
    
    # Apply naive-bayes on the training and testing datasets
    predictions = naive_bayes(training_filename, testing_filename)  

    # Create output file
    generate_output(predictions)
    print("The program has completed. You can now view the categorization in the output file.")



categorize_text()