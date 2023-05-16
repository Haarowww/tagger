import os
import sys
import argparse
import numpy as np

## command format is python3 tagger.py --trainingfiles <training files> --testfile <test file> --outputfile <output file>

## define some global variables
debug = False
wordMap = {}
tagMap = {}
firstMap = {}
traceMap = {}
tags = []
tagList = ["AJ0", "AJC", "AJS", "AT0", "AV0", "AVP", "AVQ", "CJC", "CJS", "CJT", "CRD",
            "DPS", "DT0", "DTQ", "EX0", "ITJ", "NN0", "NN1", "NN2", "NP0", "ORD", "PNI",
            "PNP", "PNQ", "PNX", "POS", "PRF", "PRP", "PUL", "PUN", "PUQ", "PUR", "TO0",
            "UNC", 'VBB', 'VBD', 'VBG', 'VBI', 'VBN', 'VBZ', 'VDB', 'VDD', 'VDG', 'VDI',
            'VDN', 'VDZ', 'VHB', 'VHD', 'VHG', 'VHI', 'VHN', 'VHZ', 'VM0', 'VVB', 'VVD',
            'VVG', 'VVI', 'VVN', 'VVZ', 'XX0', 'ZZ0', 'AJ0-AV0', 'AJ0-VVN', 'AJ0-VVD',
            'AJ0-NN1', 'AJ0-VVG', 'AVP-PRP', 'AVQ-CJS', 'CJS-PRP', 'CJT-DT0', 'CRD-PNI', 'NN1-NP0', 'NN1-VVB',
            'NN1-VVG', 'NN2-VVZ', 'VVD-VVN', 'AV0-AJ0', 'VVN-AJ0', 'VVD-AJ0', 'NN1-AJ0', 'VVG-AJ0', 'PRP-AVP',
            'CJS-AVQ', 'PRP-CJS', 'DT0-CJT', 'PNI-CRD', 'NP0-NN1', 'VVB-NN1', 'VVG-NN1', 'VVZ-NN2', 'VVN-VVD']
total_sentence = 0
initial_prob = {}
transition_prob = {}
observation_prob = {}

def split_sentence(trainTestData):
    # test example: ['Detective : NP0\n', 'Chief : NP0\n']
    # sentence example: ['Detective : NP0', 'Chief : NP0', ..., '. : PUN']
    result_list = []
    sentence_list = []
    
    for line in trainTestData:
        curr_line = line.strip('\n')
        checkWord = curr_line.split(' : ')[0]
        # print(curr_line)
        # print(checkWord)

        sentence_list.append(curr_line)
        
        if checkWord in {'.', '!', '?', '"'}:
            result_list.append(sentence_list)
            sentence_list = []

    if sentence_list:  # the sentence may not be a completed sentence. 
        result_list.append(sentence_list)
        
    return result_list

def split_test_sentence(testData):
    # test example: ['or\n', 'what\n', 'to\n', 'say\n', '.\n']
    # we want the sentence to be [['I', 'like', 'banana', '.'], ['You', 'like', 'chocolate', '!']]
    result_list = []
    sentence_list = []

    for line in testData:
        word = line.strip('\n')
        # print(word)

        sentence_list.append(word)
        # this means that the sentence is over.
        if word in ['.', '?', '"', '!']:
            result_list.append(sentence_list)
            sentence_list = []

    if sentence_list:  # the sentence may not be a completed sentence. 
        result_list.append(sentence_list)

    return result_list
    
def readLine(trainingList):
    #############################################################################
    # Initialize the global variables: wordMap, tagMap, tags, traceMap, firstMap#
    #############################################################################
    global wordMap
    global tagMap
    global tags
    global traceMap
    global firstMap
    global total_sentence
    split_text = []
    for file in trainingList:
        f = open(file, 'r')
        trainTestData = f.readlines()
        # print(trainTestData) # data like this ['no : ITJ\n', ', : PUN\n', 'she : PNP\n', 'was : VBD\n']
        split_text = split_text + split_sentence(trainTestData)
    total_sentence = len(split_text)
    # print(split_text)
    beforetag = None
    for sen in split_text:
        lengthCount = len(sen) ## calculate sentence and store them. 
        for i in range(lengthCount):
            # split word
            word = sen[i].split(' : ')[0]
            tag = sen[i].split(' : ')[1]
            # print(word, tag)

            if tag in tagMap:
                tagMap[tag] += 1
                if word in wordMap[tag]:
                    wordMap[tag][word] += 1
                else:
                    wordMap[tag][word] = 1
            else: # tag not in tagMap and this means it also does not have this tag in wordMap and tags
                tagMap[tag] = 1
                wordMap[tag] = dict()
                wordMap[tag][word] = 1
                tags.append(tag)

            if i == 0:
                if not tag in firstMap:
                    firstMap[tag] = 1
                else:
                    firstMap[tag] += 1
            else:
                if tag in traceMap:
                    if not beforetag in traceMap[tag]:
                        traceMap[tag][beforetag] = 1
                    else:
                        traceMap[tag][beforetag] += 1
                else:
                    traceMap[tag] = dict()
                    traceMap[tag][beforetag] = 1
            beforetag = tag

    # print(tags)
    # print(wordMap)
    # print(traceMap)
    # print(tagMap)
    # print(firstMap)
    
def calculate_prob():
    global initial_prob
    global transition_prob
    global observation_prob
    global total_sentence

    ## calculate initial probabilities
    for tag in firstMap:
        initial_prob[tag] = firstMap[tag] / total_sentence

    ## calculate transition probabilities
    for tag in traceMap:
        transition_prob[tag] = dict()
        for before_tag in traceMap[tag]:
            transition_prob[tag][before_tag] = traceMap[tag][before_tag] / tagMap[before_tag]

    ## calculate observation probabilities
    for tag in wordMap:
        observation_prob[tag] = dict()
        for word_key in wordMap[tag]:
            observation_prob[tag][word_key] = wordMap[tag][word_key] / tagMap[tag]

    # print(initial_prob)
    # print(transition_prob)
    # print(observation_prob)


def Viterbi(sentenceList):
    # the global variables we need in this function.
    global initial_prob
    global transition_prob
    global observation_prob
    global tagMap
    global tags
    # print(len(sentenceList))
    # print(len(tags))
    prob_matrix = np.zeros((len(sentenceList), len(tags)))
    prev_matrix = np.zeros((len(sentenceList), len(tags)))

    # BASE CASE: Determine values for time step 0 (in this case is our beginning word of the sentence)
    # first check whether we have seen the first word of the sentence in training data.
    checkStamp = any(sentenceList[0] in observation_prob[tag] and tag in initial_prob for tag in tags)
    
    for i, tag in enumerate(tags):
        if not checkStamp and tag in initial_prob:
            prob_matrix[0, i] = initial_prob[tag]
        if (sentenceList[0] in observation_prob[tag] and tag in initial_prob):
            prob_matrix[0, i] = initial_prob[tag] * observation_prob[tag][sentenceList[0]]
        prev_matrix[0, i] = None

    # RECURSIVE STEP: 
    for t in range(1, len(sentenceList)):
        # check whether we seen the word before.
        checkWord = any(sentenceList[t] in observation_prob[k] and k in transition_prob for k in tags)
        # we only want the prob_matrix to be [t - 1, j]
        alt_prob_matrix = np.tile(prob_matrix[t - 1], (len(tags), 1))

        M = np.zeros((len(tags), 1))
        T = np.zeros((len(tags), len(tags)))

        for i, cur_tag in enumerate(tags):
            # assign value of M Matrix.
            if not checkWord:
                M[i] = 1
            elif sentenceList[t] in observation_prob[cur_tag]:
                M[i] = observation_prob[cur_tag][sentenceList[t]]
            else:
                M[i] = 0
            # assign value for T Matrix.
            for j, previous_tag in enumerate(tags):
                if cur_tag in transition_prob and previous_tag in transition_prob[cur_tag]:
                    T[i, j] = transition_prob[cur_tag][previous_tag]
                else:
                    T[i, j] = 1 / tagMap[previous_tag]
        
        # prob[t, i] = prob[t-1, x] * T[x, i] * M[i, E(t)]. Since we want the most probability, max it.
        temp_matrix = np.multiply(np.multiply(alt_prob_matrix, T), M)
        alter_temp = temp_matrix.max(axis=1)
        prob_matrix[t] = np.divide(alter_temp, alter_temp.sum()) # normalize probability.
        prev_matrix[t] = temp_matrix.argmax(axis=1) # X is argmax_j in (prob * T * M). (according to slides.)

    return prob_matrix, prev_matrix

def yield_output(prob_matrix, prev_matrix, sentenceList):
    latest_max_index = np.argmax(prob_matrix[-1]) # track sequence, find the most likely probability index.
    our_result = []
    for i in range(len(sentenceList) - 1, -1, -1): # write from the last of the sequence to the first.
        our_result.append(f"{sentenceList[i]} : {tags[latest_max_index]}")
        if i != 0:
            latest_max_index = int(prev_matrix[i, latest_max_index]) # continuously update the last sequence.
    
    our_result.reverse() # since we write from last to first. we need to reverse it.
    return our_result

def initialize_global():
    global wordMap
    global tagMap
    global firstMap
    global traceMap
    global tags
    global total_sentence
    global initial_prob
    global transition_prob
    global observation_prob
    wordMap = {}
    tagMap = {}
    firstMap = {}
    traceMap = {}
    tags = []
    total_sentence = 0
    initial_prob = {}
    transition_prob = {}
    observation_prob = {}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainingfiles",
        action="append",
        nargs="+",
        required=True,
        help="The training files."
    )
    parser.add_argument(
        "--testfile",
        type=str,
        required=True,
        help="One test file."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file."
    )
    args = parser.parse_args()

    training_list = args.trainingfiles[0]
    print("training files are {}".format(training_list))
    # debug = True
    # read_training_file(training_list)
    # readLine(training_list)
    # calculate_prob()

    print("test file is {}".format(args.testfile))

    print("output file is {}".format(args.outputfile))


    print("Starting the tagging process.")
    ## initialize our global variables.
    initialize_global()

    ## read the training files and then calculate the probabilities.
    readLine(training_list)
    calculate_prob()

    ## open the test file and then read it.
    file = open(args.testfile, 'r')
    texts = file.readlines()
    out_texts = split_test_sentence(texts)
    # print(out_texts)
    file.close()

    ## run viterbi algorithm and yield the results.
    result_list = []
    for text in out_texts:
        # print(text)
        prob_matrix, prev_matrix = Viterbi(text)
        result_list += yield_output(prob_matrix, prev_matrix, text)

    ## write the file to our output file
    write_file = open(args.outputfile, 'w')
    for things in result_list:
        write_file.write(things + '\n')
    write_file.close()

    print("The tagging process finishes.")