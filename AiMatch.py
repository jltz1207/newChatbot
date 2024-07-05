import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests

def extract_features_cosine(user):
    features = []
    features.extend(user['interestKeywords'])
    features.extend(user['personalityKeywords'])
    features.append(user['bio'] if user['bio'] else '')
    return ' '.join(features)



r = requests.get(url = "https://localhost:7053/api/Summary/GetAllUserDetails", verify=False)
user_data = r.json()
with open('userJson.txt', 'w', encoding='utf-8') as file:
    file.write( json.dumps(user_data, indent=4))



default_user_id = '076188cc-57ba-4574-83ed-228b8d7a4339'

default_age_range = (25, 35)  # Filter users between the ages of 25 and 35
default_sex = 'female'  # 'female' 'male', omit = 'all'

# Create feature vectors for cosine similarity
user_features_cosine = {}
for user_id, user in user_data.items():
    user_features_cosine[user_id] = extract_features_cosine(user)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform user features for cosine similarity
feature_matrix_cosine = vectorizer.fit_transform(user_features_cosine.values())


# Function to update user features and feature matrix
def update_user_features(user_id):
    user_features_cosine[user_id] = extract_features_cosine(user_data)
    global feature_matrix_cosine
    feature_matrix_cosine = vectorizer.fit_transform(user_features_cosine.values())

def update_user_keywords(user_id, categoryId, keywords ):
    if categoryId ==1:
        user_data[user_id]["interestKeywords"].extend(keywords)
    elif categoryId == 2 :
        user_data[user_id]["personalityKeywords"].extend(keywords)
    elif categoryId == 3 :
        user_data[user_id]["hatePersonalityKeywords"].extend(keywords)
    else: return
    print(f"Updated user_data for {user_id}: {user_data[user_id]}")
    user_features_cosine[user_id] = extract_features_cosine(user_data[user_id])
    global feature_matrix_cosine
    feature_matrix_cosine = vectorizer.fit_transform(user_features_cosine.values())

def update_user_details(user_id, details): # when update, create new profile
    user_data[user_id] = details
    user_features_cosine[user_id] = extract_features_cosine(user_data[user_id])
    global feature_matrix_cosine
    feature_matrix_cosine = vectorizer.fit_transform(user_features_cosine.values())



# Function to calculate the matching score between two users
def calculate_matching_score(user1, user2, weights):
    score = 0

    # Check for common interestTagId
    common_interest_tags = set(user1['interestTagId']) & set(user2['interestTagId'])
    score += len(common_interest_tags) * weights['common_interest_tags']

    # Check for common aboutMeTagId
    common_about_me_tags = set(user1['aboutMeTagId']) & set(user2['aboutMeTagId'])
    score += len(common_about_me_tags) * weights['common_about_me_tags']

    # Check for common valueTagId
    common_value_tags = set(user1['valueTagId']) & set(user2['valueTagId'])
    score += len(common_value_tags) * weights['common_value_tags']

    # Check if personalityKeywords match with other's hatePersonalityKeywords
    matching_hate_keywords = set(user1['personalityKeywords']) & set(user2['hatePersonalityKeywords'])
    score -= len(matching_hate_keywords) * weights['hate_keywords']

    return score

# ... (Previous code remains the same)

# Function to find the best matches for a given user with filters
def find_best_matches(user_id,other_user_ids, weights=None  , min_Age=None,max_Age=None):
    if weights is None:
        weights = {
            'cosine_similarity': 1.0,
            'common_interest_tags': 10.0,
            'common_about_me_tags': 10.0,
            'common_value_tags': 10.0,
            'hate_keywords': 10.0,
            'age_as_expected':3.0 # have to adjust
        }

    user_index = list(user_features_cosine.keys()).index(user_id)
    user_vector = feature_matrix_cosine[user_index]

    # Calculate cosine similarity with all other users
    similarity_scores = cosine_similarity(user_vector, feature_matrix_cosine)

    # Calculate matching scores based on tagId and hate personality conflict
    matching_scores = {}
    for other_user_id in other_user_ids:
            
            other_user = user_data[other_user_id]
            matching_score = calculate_matching_score(user_data[user_id], other_user, weights)
            # Calculate age difference penalty
            age_difference_penalty = 0
            if min_Age and max_Age:
                if other_user['age'] < min_Age:
                    age_difference_penalty = (min_Age - other_user['age']) * weights['age_as_expected']
                elif other_user['age'] > max_Age:
                    age_difference_penalty = (other_user['age'] - max_Age) * weights['age_as_expected']

            # Apply age difference penalty to the matching score
            matching_score -= age_difference_penalty
            matching_scores[other_user_id] = matching_score

    # Normalize the scores
    min_similarity_score = np.min(similarity_scores)
    max_similarity_score = np.max(similarity_scores)
    normalized_similarity_scores = (similarity_scores - min_similarity_score) / (max_similarity_score - min_similarity_score)

    min_matching_score = min(matching_scores.values()) if matching_scores else 0
    max_matching_score = max(matching_scores.values()) if matching_scores else 0
    normalized_matching_scores = {user_id: (score - min_matching_score) / (max_matching_score - min_matching_score)
                                  for user_id, score in matching_scores.items()}

    # Combine cosine similarity and matching scores with weights
    combined_scores = {}
    for other_user_id in matching_scores.keys():
        other_user_index = list(user_features_cosine.keys()).index(other_user_id)
        combined_score = weights['cosine_similarity'] * normalized_similarity_scores[0][other_user_index] + \
                         weights['common_interest_tags'] * normalized_matching_scores[other_user_id] + \
                         weights['common_about_me_tags'] * normalized_matching_scores[other_user_id] + \
                         weights['common_value_tags'] * normalized_matching_scores[other_user_id]
        combined_scores[other_user_id] = combined_score

    # Sort the user IDs based on the combined scores in descending order
    best_matches = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)

    return best_matches

# Example usage with filters

# best_matches = find_best_matches(default_user_id, default_sex )
# print(f"Best matches for user {user_id} with filters (age range: {default_age_range}, sex: {default_sex}):")
# print(best_matches)