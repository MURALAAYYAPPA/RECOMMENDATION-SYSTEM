import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from surprise import Dataset, SVD, accuracy
from surprise.model_selection import train_test_split

# Load the MovieLens 100k dataset
data = Dataset.load_builtin('ml-100k')

# Split data into training (75%) and testing (25%) sets
train_set, test_set = train_test_split(data, test_size=0.25, random_state=42)

# Initialize and train the SVD model
model = SVD()
model.fit(train_set)

# Predict ratings on the test set
predictions = model.test(test_set)

# Evaluate model performance
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

print("\nModel Evaluation:")
print(f" - RMSE: {rmse:.4f}")
print(f" - MAE : {mae:.4f}")

# Function to get top-N recommendations for each user
def get_top_n(predictions, n=4):
    top_n = defaultdict(list)
    for pred in predictions:
        top_n[pred.uid].append((pred.iid, pred.est))

    # Keep only top-N predictions for each user
    for uid, user_ratings in top_n.items():
        top_n[uid] = sorted(user_ratings, key=lambda x: x[1], reverse=True)[:n]
    return top_n

# Get top 4 movie recommendations for each user
top_recommendations = get_top_n(predictions, n=4)

# Print top 4 recommendations for 3 users
print("\nTop 4 Recommendations for Sample Users:")
for user_id, recs in list(top_recommendations.items())[:3]:
    print(f"\nUser {user_id}:")
    for movie_id, est_rating in recs:
        print(f" - Movie ID: {movie_id}, Predicted Rating: {round(est_rating, 2)}")

# Visualize predicted rating distribution
pred_ratings = [pred.est for pred in predictions]
plt.figure(figsize=(8, 5))
sns.histplot(pred_ratings, bins=20, kde=True, color='skyblue')
plt.title("Predicted Ratings Distribution")
plt.xlabel("Predicted Rating")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Optional: Visualize predicted vs true ratings
true_ratings = [pred.r_ui for pred in predictions]
plt.figure(figsize=(8, 5))
sns.scatterplot(x=true_ratings, y=pred_ratings, alpha=0.5, color='purple')
plt.title("True vs Predicted Ratings")
plt.xlabel("True Rating")
plt.ylabel("Predicted Rating")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
