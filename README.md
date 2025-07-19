RECOMMENDATION-SYSTEM
COMPANY : CODTECH IT SOLUTIONS

NAME : MURALA AYYAPPA

INTERN ID : CT04DH168

DOMAIN : MACHINE LEARNING

DURATION : 4 WEEKS

MENTOR : NEELA SANTHOSH KUMAR

DESCRIPTION :
This task is a movie recommendation system that suggests movies to users based on what they and others have rated before. It uses a popular method in machine learning called collaborative filtering, specifically a matrix factorization technique called SVD (Singular Value Decomposition). The idea is simple: if two users have rated some movies similarly, they’re likely to enjoy similar movies in the future too. This model learns those patterns and helps predict which movies a user might like next.

Editor Platform :
This project was created and run using Jupyter Notebook, which provides an interactive environment for writing and testing Python code along with visualizations.

Tools & Libraries Used
scikit-surprise -the main library for building recommender systems

matplotlib and seaborn: Used for creating a visualization of the predicted ratings and easier to understand

pandas: A powerful python library for data manipulation, used here to help organize data

SVD: A matrix factorization algorithm used to discover hidden relationships between users and items (in this case, movies). It’s great for collaborative filtering problems.

steps include ;
The notebook walks through all the main steps required to build a functional movie recommendation system

Loading the Dataset:
The project uses the built-in ml-100k MovieLens dataset, which has 100,000 movie ratings from users.

Splitting the Data:
The data is split into 75% for training and 25% for testing

Training the Model:
The SVD algorithm trains on the training dataset, using the rating patterns of users.

Making Predictions:
After training, the model predicts ratings for movies that users haven’t rated yet.

Top-4 Movie Suggestions:
For each user, the system suggests the top 4 movies with the highest predicted ratings.

Model Evaluation:
To test the system performs, two standard evaluation metrics are used:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)
These metrics show how close the predicted ratings are to the actual ratings.

Visualization:
The distribution of predicted ratings is displayed in a histogram to provide a quick overview of the movie rating

use cases :
recommendation systems like this are widely used in the real world:

netflix: To recommend TV shows and movies based on what you and similar users have watched

amazon,flipkart and others: To suggest products that match your shopping patterns

spotify: To create playlists and suggest songs you might like

YouTube: To personalize video recommendations

Conclusion :
This task demonstrates how a simple recommendation system works on real-world data. It gives you an idea of how we can leverage machine learning techniques to predict user preferences. With minimal tools and some sophisticated math under the hood, this notebook produces a simple but efficient recommender that would be easily modifiable to other domains such as e-commerce, music, or learning websites.

OUTPUT :
