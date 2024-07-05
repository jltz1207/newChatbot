import face_recognition
import base64
from PIL import Image
import io


def find_similar_face(new_image_base64, user_photos_database):
    # Load the new user's photo from Base64
    new_image_bytes = base64.b64decode(new_image_base64)
    new_image = face_recognition.load_image_file(io.BytesIO(new_image_bytes))
    
    # Detect face locations in the new image
    face_locations = face_recognition.face_locations(new_image)
    if len(face_locations) == 0:
        print("No faces detected in the new image.")
        return None  # No faces detected in the new image
    
    # Get the face encoding for the new image
    new_face_encodings = face_recognition.face_encodings(new_image)
    if len(new_face_encodings) == 0:
        print("No face encodings found in the new image.")
        return None
    new_face_encoding = new_face_encodings[0]

    best_match_user = None
    best_match_distance = float('inf')

    # Iterate over all users in the database
    for user_id, user_photos_base64 in user_photos_database.items():
        user_face_distances = []

        # Iterate over all photos of the current user
        for user_photo_base64 in user_photos_base64:
            # Decode the user's photo from Base64
            user_photo_bytes = base64.b64decode(user_photo_base64)
            user_photo = face_recognition.load_image_file(io.BytesIO(user_photo_bytes))
            
            # Detect face locations in the user's photo
            user_face_locations = face_recognition.face_locations(user_photo)
            if len(user_face_locations) == 0:
                print(f"No faces detected in the photo of user {user_id}.")
                continue  # No faces detected in this user's photo
            
            # Get the face encoding for the user's photo
            user_face_encodings = face_recognition.face_encodings(user_photo)
            if len(user_face_encodings) == 0:
                print(f"No face encodings found in the photo of user {user_id}.")
                continue
            user_face_encoding = user_face_encodings[0]

            # Compare the new face encoding with the user's face encoding
            face_distance = face_recognition.face_distance([new_face_encoding], user_face_encoding)[0]
            user_face_distances.append(face_distance)

        if not user_face_distances:
            continue  # No valid face encodings for this user

        # Calculate the average face distance for the current user
        avg_face_distance = sum(user_face_distances) / len(user_face_distances)
        print(f"{user_id}: {avg_face_distance}")

        # Check if the current user is more similar than the previous best match
        if avg_face_distance < best_match_distance:
            best_match_user = user_id
            best_match_distance = avg_face_distance

    return best_match_user