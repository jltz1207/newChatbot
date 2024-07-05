import json
import spacy

# Load the spaCy English model
nlp = spacy.load('en_core_web_md')

# Load the user data from the JSON file
with open('userJson.txt', 'r') as file:
    user_data = json.load(file)

# Function to calculate the matching score between two users
def calculate_matching_score(user1, user2):
    score = 0

    # Check for common interestTagId
    common_interest_tags = set(user1['interestTagId']) & set(user2['interestTagId'])
    score += len(common_interest_tags) * 10

    # Check for common aboutMeTagId
    common_about_me_tags = set(user1['aboutMeTagId']) & set(user2['aboutMeTagId'])
    score += len(common_about_me_tags) * 10

    # Check for common valueTagId
    common_value_tags = set(user1['valueTagId']) & set(user2['valueTagId'])
    score += len(common_value_tags) * 10

    # Check for common interestKeywords
    common_interest_keywords = set(user1['interestKeywords']) & set(user2['interestKeywords'])
    score += len(common_interest_keywords) * 5

    # Check for common personalityKeywords
    common_personality_keywords = set(user1['personalityKeywords']) & set(user2['personalityKeywords'])
    score += len(common_personality_keywords) * 5

    # Check if personalityKeywords match with other's hatePersonalityKeywords
    matching_hate_keywords = set(user1['personalityKeywords']) & set(user2['hatePersonalityKeywords'])
    score -= len(matching_hate_keywords) * 10

    # Calculate similarity score based on user bio using spaCy
    if user1['bio'] and user2['bio']:
        doc1 = nlp(user1['bio'])
        doc2 = nlp(user2['bio'])
        similarity_score = doc1.similarity(doc2)
        score += similarity_score * 20

    return score

# Function to find the best matches for a given user
def find_best_matches(user_id, num_matches=5):
    user = user_data[user_id]
    matching_scores = []

    # Calculate matching scores with all other users
    for other_user_id, other_user in user_data.items():
        if other_user_id != user_id:
            score = calculate_matching_score(user, other_user)
            matching_scores.append((other_user_id, score))

    # Sort the matching scores in descending order
    matching_scores.sort(key=lambda x: x[1], reverse=True)

    # Return the top num_matches users
    best_matches = [match[0] for match in matching_scores[:num_matches]]
    return best_matches

# Example usage
user_id = 'jason@test.com'
num_matches = 3
best_matches = find_best_matches(user_id, num_matches)
print(f"Best matches for user {user_id}:")
for match_id in best_matches:
    print(match_id)