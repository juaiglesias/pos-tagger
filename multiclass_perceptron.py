# -*- encoding: utf-8 -*-
# Multiclass Perceptron
from collections import defaultdict
import numpy as np
import itertools
import json

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
                #TODO: word_features or all weight vector features?
                #self.update(word_label, predicted_word_label, self.weights[word_label].keys())
                self.update(word_label, predicted_word_label, word_feature_vector)
                
                '''
                if line_number%4000 == 0:
                    print('Line #: ', line_number)
                    print(word, word_label, predicted_word_label)
                '''
            
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

        Training can no longer be resumed.
        """
        for label, weights in self.weights.items():
            new_feat_weights = {}
            for feat, w in weights.items():
                param = (label, feat)
                # Be careful not to add 1 to take into account the
                # last weight vector (without increasing the number of
                # iterations in the averaging)
                total = self._accum[param] + \
                    (self.n_updates + 1 - self._last_update[param]) * w
                averaged = round(total / self.n_updates, 3)
                if averaged:
                    new_feat_weights[label] = averaged
            self.weights[label] = new_feat_weights

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

    def create_word_feature_vector(self, word, context, word_position_in_context):
    
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


# Main

def main():

    # ---------------------------------------------------------------
    # Data 
    # ---------------------------------------------------------------

    # First 3: dataset collected by the Universal Depedencies project
    # Last 2: annotated tweets talking about Football and Minecraft
    train_fn = "data/fr.ud.train.json"
    dev_fn = "data/fr.ud.dev.json"
    test_fn = "data/fr.ud.test.json"
    test_football_fn = "data/foot.json"
    test_minecraft_fn = "data/minecraft.json"


    # ---------------------------------------------------------------
    # Train & Dev Stage
    # ---------------------------------------------------------------

    # Load train data
    train_set = np.array( json.load(open(train_fn)) )
    lines, lines_labels = train_set[:,0], train_set[:,1]
    #print(lines[0])
    #print(lines_labels[0])
    #print(len(lines))

    # PoS labels (classes)
    labels = set( itertools.chain.from_iterable(lines_labels) )
    labels.add(BIAS)
    #print(classes)
    #print(len(classes))

    # Initialize the perceptron
    perceptron = Perceptron(labels)

    # Load dev data to test each epoch
    dev_set = np.array( json.load(open(dev_fn)) )
    lines_dev, lines_classes_dev = dev_set[:,0], dev_set[:,1]

    # Loop through epochs training and testing to find ideal epoch
    n_epochs = 2 #TODO: change this (maybe 10?)
    epoch_error_min = 1
    best_epoch = 0
    best_weights = None

    for epoch in range(n_epochs):
        
        # Shuffle rows
        np.random.shuffle(train_set)

        # Train
        perceptron.train(train_set)

        # Test with dev to find ideal epoch
        epoch_error = perceptron.test(dev_set)
        print("Epoch ", epoch, "error: ", epoch_error)
        
        # Optimal epoch to train to
        if epoch_error < epoch_error_min:
            epoch_error_min = epoch_error
            best_epoch = epoch
            best_weights = perceptron.__getstate__() 

    print("Best epoch: ", best_epoch)
    print("Best epoch error: ", epoch_error_min)
    
    # Set perceptron's best weights
    perceptron.__setstate__(best_weights)
    print("\nBest weights: (label, len(features)) ")
    perceptron.print()

    # ---------------------------------------------------------------
    # Test Stage
    # ---------------------------------------------------------------

    # Load test data
    test_set = np.array( json.load(open(test_fn)) )

    # Use best weights vector
    perceptron.__setstate__(best_weights) #TODO like this? 

    # Test and get error rate
    test_error = perceptron.test(test_set)
    print("\nError: ", test_error)

    # Test with other datasets
    test_football_set = np.array( json.load(open(test_football_fn)) )
    test_football_error = perceptron.test(test_football_set)
    print("\nFootball Error: ", test_football_error)

    test_minecraft_set = np.array( json.load(open(test_minecraft_fn)) )
    test_minecraft_error = perceptron.test(test_minecraft_set)
    print("\nMinecraft Error: ", test_minecraft_error)


if __name__ == "__main__":
    main()
