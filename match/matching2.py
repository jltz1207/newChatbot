import json
from collections import defaultdict

with open('userJson.txt', 'r', encoding='utf-8') as f:
    user_data = json.load(f)
    
# Helper function to find matches based on shared keywords
def find_matches(user_data):
    matches = defaultdict(list)

    for user1, data1 in user_data.items():
        for user2, data2 in user_data.items():
            if user1 == user2:
                continue

            interest_matches = set(data1['interestKeywords']) & set(data2['interestKeywords'])
            personality_matches = set(data1['personalityKeywords']) & set(data2['personalityKeywords'])
            hate_personality_conflicts = set(data1['hatePersonalityKeywords']) & set(data2['personalityKeywords'])

            score = len(interest_matches) + len(personality_matches) - len(hate_personality_conflicts)

            if score > 0:
                matches[user1].append((user2, score, interest_matches, personality_matches, hate_personality_conflicts))

    return matches

# Find matches and print them
matches = find_matches(user_data)

for user, user_matches in matches.items():
    print(f"Matches for {user}:")
    for match in user_matches:
        user2, score, interest_matches, personality_matches, hate_personality_conflicts = match
        print(f"  - Match with {user2} (Score: {score})")
        print(f"    Interests: {interest_matches}")
        print(f"    Personality: {personality_matches}")
        print(f"    Conflicts: {hate_personality_conflicts}")
    print()