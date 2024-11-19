import sys
import re
import math
from collections import defaultdict, Counter
from itertools import islice

def parse_dataset(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    
    instances = re.findall(
        r'<instance id="(.*?)".*?<answer.*?senseid="(.*?)".*?<context>(.*?)</context>',
        data,
        re.DOTALL
    )
    
    print(f"Number of instances: {len(instances)}")
    
    dataset = []
    for instance_id, senseid, context in instances:
        
        context_words = re.findall(r'\b\w+\b', context.lower())
        if "plant" in context_words:
            context_words.remove("plant")  
        dataset.append((instance_id, senseid, context_words))
    return dataset

def create_folds(dataset):
    fold_size = len(dataset) // 5
    folds = [dataset[i * fold_size:(i + 1) * fold_size] for i in range(4)]
    folds.append(dataset[4 * fold_size:])  
    return folds



def train_naive_bayes(train_data):
    word_counts = defaultdict(lambda: defaultdict(int))
    sense_counts = Counter()
    vocabulary = set()
    
    for _, senseid, context_words in train_data:
        sense_counts[senseid] += 1
        for word in context_words:
            word_counts[senseid][word] += 1
            vocabulary.add(word)
    
    total_senses = sum(sense_counts.values())
    return word_counts, sense_counts, vocabulary, total_senses


def calculate_probability(word_counts, total_words, vocabulary_size):
    probabilities = {}
    for word, count in word_counts.items():
        
        probabilities[word] = (count + 1) / (total_words + vocabulary_size)
    return probabilities

def calculate_log_probability(context_words, word_probabilities, total_words, vocabulary_size):
    log_prob = 0
    for word in context_words:
        if word in word_probabilities:
            log_prob += math.log(word_probabilities[word])
        else:
            log_prob += math.log(1 / (total_words + vocabulary_size))
    return log_prob

def cross_validate(dataset):
    num_folds = 5
    fold_size = len(dataset) // num_folds
    accuracies = []
    predictions = []

   
    folds = [dataset[i * fold_size: (i + 1) * fold_size] for i in range(num_folds)]
    
    for i in range(num_folds):
        
        test_data = folds[i]
        train_data = [item for j, fold in enumerate(folds) if j != i for item in fold]
        
        
        word_counts_by_sense = {}
        total_words_by_sense = {}
        vocabulary = set()

        for instance_id, senseid, context_words in train_data:
            if senseid not in word_counts_by_sense:
                word_counts_by_sense[senseid] = {}
                total_words_by_sense[senseid] = 0
            
            for word in context_words:
                vocabulary.add(word)
                word_counts_by_sense[senseid][word] = word_counts_by_sense[senseid].get(word, 0) + 1
                total_words_by_sense[senseid] += 1
        
        vocabulary_size = len(vocabulary)
        word_probabilities_by_sense = {
            senseid: calculate_probability(word_counts, total_words_by_sense[senseid], vocabulary_size)
            for senseid, word_counts in word_counts_by_sense.items()
        }

        
        correct = 0
        fold_predictions = []

        for instance_id, true_senseid, context_words in test_data:
            
            sense_log_probs = {}
            for senseid in word_probabilities_by_sense:
                sense_log_probs[senseid] = calculate_log_probability(
                    context_words,
                    word_probabilities_by_sense[senseid],
                    total_words_by_sense[senseid],
                    vocabulary_size
                )
            
            
            predicted_senseid = max(sense_log_probs, key=sense_log_probs.get)
            fold_predictions.append((instance_id, predicted_senseid))
            
            if predicted_senseid == true_senseid:
                correct += 1

        accuracy = (correct / len(test_data)) * 100
        accuracies.append(accuracy)
        predictions.extend(fold_predictions)
    
    return accuracies, predictions



def main():
    if len(sys.argv) < 2:
        print("Usage: python wsd.py <dataset_file>")
        return
    
    file_path = sys.argv[1]
    dataset_name = file_path.split('.')[0]  
    
    dataset = parse_dataset(file_path)
    
    
    accuracies, predictions = cross_validate(dataset)
    
    for i, accuracy in enumerate(accuracies):
        print(f"Fold {i + 1} Accuracy: {accuracy:.2f}%")
    
    average_accuracy = sum(accuracies) / len(accuracies)
    print(f"Average Accuracy: {average_accuracy:.2f}%")

    output_file_name = f"{dataset_name}.wsd.out"
    with open(output_file_name, "w") as output_file:
        for i in range(5):
            output_file.write(f"Fold {i + 1}\n")
            for prediction in predictions:
                output_file.write(f"{prediction[0]} {prediction[1]}\n")
    
    print(f"Results written to {output_file_name}")

if __name__ == "__main__":
    main()


