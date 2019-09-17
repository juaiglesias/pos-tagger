# -*- encoding: utf-8 -*-
# Multiclass Perceptron
from collections import defaultdict
import numpy as np
import itertools
import json
import copy
import random

# Constants
BIAS = 'bias'
LAST_3_CHARACTERS = 'last_3_characters_'
LAST_CHARACTER = 'last_character_'
FIRST_CHARACTER = 'first_character_'
WORD_ITSELF = 'word_itself_'
STARTS_WITH_CAPITAL = 'starts_with_capital'
ALL_CAPITALS = 'all_capitals'
WORD_AT_I_MINUS_1 = 'word_at_i_minus_1_'
WORD_AT_I_MINUS_2 = 'word_at_i_minus_2_'
WORD_AT_I_PLUS_1 = 'word_at_i_plus_1_'
WORD_AT_I_PLUS_2 = 'word_at_i_plus_2_'

# Perceptron
class Perceptron:

    def __init__(self, labels):
        # Labels from the data, i.e. PoS tags
        self.labels = labels

        # Dictionary where each label(class) is mapped to another 
        # dictionary representing the weight vector per feature
        self.weights = defaultdict()
        for label in self.labels:
            self.weights[label] = defaultdict()

        # The accumulated values of the weight vector at the t-th
        # iteration: sum_{i=1}^{n - 1} w_i
        #
        # The current value (w_t) is not yet added. The key of this
        # dictionary is a pair (label, feature)
        self._accum = defaultdict(int)

        # The last time the feature was changed, for the averaging.
        self._last_update = defaultdict(int)

        # Number of examples seen
        self.n_updates = 0

    def train(self, train_data):
        
        line_number = 0
        # Go through each line
        for line, line_labels in train_data:
            # Go through each word in the line
            for word_position, (word, word_label) in enumerate(zip(line, 
                                                                   line_labels)):
                # Build word feature vector
                word_feature_vector = self.create_word_feature_vector(word, 
                                                                      line, 
                                                                      word_position)

                # Add word to label weights if not there already
                for feature in word_feature_vector:
                    if feature not in self.weights[word_label]:
                        for label in self.labels:
                            self.weights[label][feature] = 0

                # Predict word's class
                predicted_word_label = self.predict(word_feature_vector)

                # Update weights (if is the case)
                self.update(word_label, predicted_word_label, word_feature_vector)
                
            line_number += 1

    def test(self, test_data):

        error = 0
        n_predictions = 0

        # Go through each line
        for line, line_labels in test_data:
            # Go through each word in the line
            for word_position, (word, word_label) in enumerate(zip(line, 
                                                                   line_labels)):
                # Build word feature vector
                word_feature_vector = self.create_word_feature_vector(word, 
                                                                      line, 
                                                                      word_position)

                # Predict word's class
                predicted_word_label = self.predict(word_feature_vector)
                
                # Add 1 if there is an error
                if (word_label != predicted_word_label): 
                    error += 1
                
                # Update counter of predictions
                n_predictions += 1

        error_rate = error / n_predictions

        return error_rate

    def test_more(self, test_data, ambiguous_words, oov_words):

        error = 0
        ambiguous = 0
        oov = 0
        n_predictions = 0
        n_ambiguous = 0
        n_oov = 0

        # Go through each line
        for line, line_labels in test_data:
            # Go through each word in the line
            for word_position, (word, word_label) in enumerate(zip(line, 
                                                                   line_labels)):
                # Build word feature vector
                word_feature_vector = self.create_word_feature_vector(word, 
                                                                      line, 
                                                                      word_position)

                # Predict word's class
                predicted_word_label = self.predict(word_feature_vector)
                
                # Add 1 if there is an error
                if (word_label != predicted_word_label): 
                    error += 1
                
                if word in ambiguous_words:
                    n_ambiguous += 1
                    if (word_label != predicted_word_label): 
                        ambiguous += 1

                if word in oov_words:
                    n_oov += 1
                    if (word_label != predicted_word_label): 
                        oov += 1
                
                # Update counter of predictions
                n_predictions += 1

        error_rate = error / n_predictions
        ambiguous_error = ambiguous / n_ambiguous
        oov_error = oov / n_oov

        return error_rate, ambiguous_error, oov_error

    def predict(self, word_feature_vector):
        '''Scores the word features and current weights and return
        the best class (label).'''
        max_score = 0
        best_label = ""
        for label in self.labels:
            current_score = self.score(label, word_feature_vector)
            if current_score >= max_score:
                best_label = label

        return best_label
    
    def score(self, label, word_feature_vector):
        """
        Dot-product the word's features and the current label's weights, 
        for each feature in the word feature vector.
        """
        dot = 0 
        for feature, feature_value in word_feature_vector.items():
            label_weight_vector = self.weights[label]
            if feature in label_weight_vector:
                feature_weight = label_weight_vector.get(feature, 0)
                dot += feature_value * feature_weight

        return dot

    def update(self, true_label, predicted_label, features):
        
        def upd_feat(label, feature, v):
            param = (label, feature)
            self._accum[param] += ( (self.n_updates - self._last_update[param]) * 
                                    (self.weights[label][feature]) )
            self._last_update[param] = self.n_updates
            self.weights[label][feature] += v
            
        self.n_updates += 1

        if true_label == predicted_label:
            return

        for f in features: 
            upd_feat(true_label, f, 1.0)
            upd_feat(predicted_label, f, -1.0)

    def average_weights(self):
        """
        Average weights of the perceptron
        """
        self.weights_backup = copy.deepcopy(self.weights)
        for label, weights in self.weights.items():
            new_feat_weights = defaultdict()
            for feat, w in weights.items():
                param = (label, feat)
                # Be careful not to add 1 to take into account the
                # last weight vector (without increasing the number of
                # iterations in the averaging)
                total = self._accum[param] + \
                    (self.n_updates + 1 - self._last_update[param]) * w
                averaged = round(total / self.n_updates, 3)
                if averaged:
                    new_feat_weights[feat] = averaged
            self.weights[label] = new_feat_weights

    def de_average_weights(self):
        """
        De-average weights of the perceptron
        """
        self.weights = copy.deepcopy(self.weights_backup)

    def __getstate__(self):
        """
        Serialization of a perceptron

        We are only serializing the weight vector as a dictionnary
        because defaultdict with lambda can not be serialized.
        """
        # should we also serialize the other attributes to allow
        # learning to continue?
        return {"weights": {k: v for k, v in self.weights.items()}}

    def __setstate__(self, data):
        """
        De-serialization of a perceptron
        """

        self.weights = defaultdict(lambda: defaultdict(float), data["weights"])
        # ensure we are no longer able to continue training
        self._accum = None
        self._last_update = None

    def print(self):
        for label, label_weight_vector in self.weights.items():
            print(label, len(label_weight_vector))

    def print_one_label(self):
        for label, label_weight_vector in self.weights.items():
            print(label, label_weight_vector)
            break

    def create_word_feature_vector(self, word, context, 
                                   word_position_in_context):
    
        """
        Given a word and its context builds a sparse representation 
        of its feature vector.
        """
        feature_vector = defaultdict()

        # bias always 1
        feature_vector[BIAS] = 1

        # - the 3 last characters of the word
        if len(word) >= 3:
            last_3c = word[-3:]
            last_3_key = LAST_3_CHARACTERS + last_3c.lower()
            feature_vector[last_3_key] = 1

        # - the last character of the word
        c = word if (len(word) == 1) else word[-1]
        last_c = LAST_CHARACTER + c
        feature_vector[last_c] = 1

        # - the first character of the word
        c = word if (len(word) == 1) else word[0]
        first_c = FIRST_CHARACTER + c.lower()
        feature_vector[first_c] = 1

        # - the word
        w = WORD_ITSELF + word.lower()
        feature_vector[w] = 1

        # - a binary feature indicating whether the word starts with a capital letter or not
        c = word if (len(word) == 1) else word[1]
        if c.isupper():
            feature_vector[STARTS_WITH_CAPITAL] = 1

        # - a binary feature indicating whether the word is made only of capital letters or not
        if word.isupper():
            feature_vector[ALL_CAPITALS] = 1

        # - the word at position i − 1
        if word_position_in_context >= 1:
            previous_word = context[word_position_in_context - 1]
            w = WORD_AT_I_MINUS_1 + previous_word.lower()
            feature_vector[w] = 1

        # - the word at position i − 2
        if word_position_in_context >= 2:
            previous_word = context[word_position_in_context - 2]
            w = WORD_AT_I_MINUS_2 + previous_word.lower()
            feature_vector[w] = 1

        # - the word at position i + 1
        context_length = len(context)
        if context_length - word_position_in_context - 1 >= 1:
            next_word = context[word_position_in_context + 1]
            w = WORD_AT_I_PLUS_1 + next_word.lower()
            feature_vector[w] = 1

        # - the word at position i + 2
        context_length = len(context)
        if context_length - word_position_in_context - 1 >= 2:
            next_word = context[word_position_in_context + 2]
            w = WORD_AT_I_PLUS_2 + next_word.lower()
            feature_vector[w] = 1

        # other possible features: 
        # - position of word in context
        # - first 3 characters (root of verbs)
        # - ...

        return feature_vector

def load_dataset(filenames):
    
    n_filenames = 0
    for fn in filenames:
        with open(fn) as f:
            part = np.array( json.load(f) )
            if (n_filenames) < 1:
                dataset = part
            else:
                dataset = np.append(dataset, part, axis=0)
        n_filenames += 1

    return dataset

def get_ambiguous_words(train_set):
    '''Words that appear with more than one label in the train set'''
    
    train_lines, train_lines_labels = train_set[:,0], train_set[:,1]
    train_words = set( itertools.chain.from_iterable(train_lines) )

    ambiguous_words = set()
    word_labels = defaultdict()

    for line, line_labels in train_set:
        for word, word_label in zip(line,line_labels):
            if word in word_labels:
                lab = word_labels[word]
                if lab != word_label:
                    ambiguous_words.add(word)
            else:
                word_labels[word] = word_label

    ambiguous = len(ambiguous_words) / len(train_words)
    print("Ambiguous proportion: ", ambiguous)

    return ambiguous_words

def get_oov_words(train_set, test_set):
    '''Words appearing in the test set that are not contained on the train set'''
    
    train_lines, train_lines_labels = train_set[:,0], train_set[:,1]
    train_words = set( itertools.chain.from_iterable(train_lines) )

    test_lines, tests_lines_labels = test_set[:,0], test_set[:,1]
    test_words = set( itertools.chain.from_iterable(test_lines) )

    oov_words = test_words.difference(train_words)
    oov = len(oov_words) / len(test_words)
    print("OOV proportion: ", oov)
    return oov_words

def write_file(fn, test_names, errors):
    with open(fn, 'w') as f:
        f.write( ",".join(test_names) )
        for error in errors:
            f.write("\n")
            f.write( ",".join([str(x) for x in error]) )

def pipeline(train_filenames, dev_filenames, test_filenames, n_epochs=10):

    # ---------------------------------------------------------------
    # Train & Dev Stage
    # ---------------------------------------------------------------

    # Load train data
    train_set = load_dataset(train_filenames)
    lines, lines_labels = train_set[:,0], train_set[:,1]

    # PoS labels (classes)
    labels = set( itertools.chain.from_iterable(lines_labels) )
    labels.add(BIAS)

    # Initialize the perceptron
    perceptron = Perceptron(labels)

    # Load dev data to test each epoch
    dev_set = load_dataset(dev_filenames)

    # Loop through epochs training and testing to find ideal epoch
    epoch_error_min = 1
    best_epoch = 0
    best_weights = None

    for epoch in range(n_epochs):
        
        # Shuffle rows
        np.random.shuffle(train_set)

        # Train
        perceptron.train(train_set)

        # Average weights
        perceptron.average_weights()

        # Test with dev to find ideal epoch
        epoch_error = perceptron.test(dev_set)
        
        # Optimal epoch to train to
        if epoch_error < epoch_error_min:
            epoch_error_min = epoch_error
            best_epoch = epoch
            best_weights = perceptron.__getstate__()

        # De-average to continue training 
        perceptron.de_average_weights()

    print("Best epoch: ", best_epoch)
    print("Best epoch error: ", epoch_error_min)

    # Set perceptron's best weights
    perceptron.__setstate__(best_weights)

    # ---------------------------------------------------------------
    # Test Stage
    # ---------------------------------------------------------------

    test_errors = []
    ambiguous_errors = []
    oov_errors = []
    for test_fn in test_filenames:
        with open(test_fn) as f:

            # Load test data
            print("\nTest case: ", test_fn)
            test_set = np.array( json.load(f) )
            
            ambiguous_words = get_ambiguous_words(train_set)
            oov_words = get_oov_words(train_set, test_set)

            # Test and get error rate
            test_error, ambiguous_error, oov_error = perceptron.test_more(test_set, 
                                                                          ambiguous_words, 
                                                                          oov_words)
            test_errors.append(test_error)
            ambiguous_errors.append(ambiguous_error)
            oov_errors.append(oov_error)

            print("Test error: ", test_error)
            print("Ambiguous error: ", ambiguous_error)
            print("OOV error: ", oov_error)

    return test_errors, ambiguous_errors, oov_errors

# Main
def main():

    # ---------------------------------------------------------------
    # Data 
    # ---------------------------------------------------------------

    # path, prefix and posfix
    path = "data/en/"
    prefix_ewt = path + "ewt/en.ewt."
    prefix_gum = path + "gum/en.gum."
    prefix_lines = path + "lines/en.lines."
    prefix_partut = path + "partut/en.partut."
    prefix_foot = path + "foot/en.foot."
    prefix_natdis = path + "natdis/en.natdis."
    prefix_pud = path + "pud/en.pud."
    posfix_train = "train.json"
    posfix_dev = "dev.json"
    posfix_test = "test.json"

    # ewt
    train_ewt_fn = prefix_ewt + posfix_train
    dev_ewt_fn = prefix_ewt + posfix_dev
    test_ewt_fn = prefix_ewt + posfix_test

    # gum
    train_gum_fn = prefix_gum + posfix_train
    dev_gum_fn = prefix_gum + posfix_dev
    test_gum_fn = prefix_gum + posfix_test

    # lines
    train_lines_fn = prefix_lines + posfix_train
    dev_lines_fn = prefix_lines + posfix_dev
    test_lines_fn = prefix_lines + posfix_test

    # partut
    train_partut_fn = prefix_partut + posfix_train
    dev_partut_fn = prefix_partut + posfix_dev
    test_partut_fn = prefix_partut + posfix_test

    # pud
    test_pud_fn = prefix_pud + posfix_test

    # natdis (TODO: EQUAL TO FOOT)
    test_natdis_fn = prefix_natdis + posfix_test

    # foot
    test_foot_fn = prefix_foot + posfix_test

    # random seed
    random.seed( 30 )

    # max number of epochs
    max_epochs = 20

    # ---------------------------------------------------------------
    # Experiments 
    # ---------------------------------------------------------------
    test_filenames = [test_ewt_fn, test_gum_fn, test_lines_fn, 
                      test_partut_fn, test_pud_fn, test_foot_fn]
    test_names = ["ewt", "gum", "lines", "partut", "pud", "foot"]

    # Experiment 1: 
    # - Train:  [ewt]
    # - Dev:    [ewt]
    # - Test:   [ALL]
    print("\nEXPERIMENT 1: ewt\n")
    errors_1, ambiguous_1, oov_1 = pipeline([train_ewt_fn],
                                            [dev_ewt_fn],
                                            test_filenames,
                                            n_epochs=max_epochs)

    # Experiment 2: 
    # - Train:  [gum]
    # - Dev:    [gum]
    # - Test:   [ALL]
    print("\nEXPERIMENT 2: gum\n")
    errors_2, ambiguous_2, oov_2 = pipeline([train_gum_fn],
                                            [dev_gum_fn],
                                            test_filenames,
                                            n_epochs=max_epochs)

    # Experiment 3: 
    # - Train:  [lines]
    # - Dev:    [lines]
    # - Test:   [ALL]
    print("\nEXPERIMENT 3: lines\n")
    errors_3, ambiguous_3, oov_3 = pipeline([train_lines_fn],
                                            [dev_lines_fn],
                                            test_filenames,
                                            n_epochs=max_epochs)

    # Experiment 4: 
    # - Train:  [partut]
    # - Dev:    [partut]
    # - Test:   [ALL]
    print("\nEXPERIMENT 4: partut\n")
    errors_4, ambiguous_4, oov_4 = pipeline([train_partut_fn],
                                            [dev_partut_fn],
                                            test_filenames,
                                            n_epochs=max_epochs)

    # Experiment 5: 
    # - Train:  [ewt, gum, lines, partut]
    # - Dev:    [ewt, gum, lines, partut]
    # - Test:   [ALL]
    print("\nEXPERIMENT 5: ewt, gum, lines, partut\n")
    errors_5, ambiguous_5, oov_5 = pipeline([train_ewt_fn, train_gum_fn, train_lines_fn,            train_partut_fn],
                                            [dev_ewt_fn, dev_gum_fn, dev_lines_fn, dev_partut_fn],
                                            test_filenames,
                                            n_epochs=max_epochs)

    # Experiments results
    write_file('test_errors.txt', test_names, [errors_1, errors_2, errors_3, errors_4, errors_5])
    write_file('ambiguous_errors.txt', test_names, [ambiguous_1, ambiguous_2, ambiguous_3, ambiguous_4, ambiguous_5])
    write_file('oov_errors.txt', test_names, [oov_1, oov_2, oov_3, oov_4, oov_5])


if __name__ == "__main__":
    main()
