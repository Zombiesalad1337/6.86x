import project1 as p1
import numpy as np
import json
import utils

dictionary = json.load(open('bag_of_words.json'))
best_theta = np.load('best_theta.npy')
best_theta_0 = np.load('best_theta_0.npy')

while(True):
    print("Enter a review:")
    user_review = input()
    user_review_feature = p1.extract_bow_feature_vectors(user_review, dictionary)
    output = p1.classify(user_review_feature, best_theta, best_theta_0)[0]
    if output > 1e-7:
        print("Sentiment: Positive!\n")
    else:
        print("Sentiment: Negative :(\n")
    print("1. Test another review")
    print("2. Exit")
    user_input = int(input())
    if user_input == 1:
        continue
    else:
        break




