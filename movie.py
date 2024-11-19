import pandas as pd
from sklearn.model_selection import train_test_split 
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
user_input = input("Rate a movie (Movie ID, Rating): ")


# Load ratinngs and movies data
rating = pd.read_csv("C:/users/ISHITA/Desktop/movie/rating.csv")
print("rating Data:")
print(rating.head())
movie = pd.read_csv("C:/users/ISHITA/Desktop/movie/movie.csv")
print("\nmovie Data")
print(movie.head())

# Check for missing values in both datasets
print(rating.isnull().sum())
print(movie.isnull().sum())

# Drop rows with missing values, if necessary
rating.dropna(inplace=True)
movie.dropna(inplace=True)

# Merge the ratings and movies datasets on 'movieId'
data = pd.merge(rating, movie, on='movieId')

# Display the first few rows of the merged dataset
print(data.head())

# Create a user-item matrix where rows are users and columns are movies
user_item_matrix = data.pivot_table(index='userId', columns='movieId', values='rating')
print(user_item_matrix.head())


# Prepare data for the surprise library
reader = Reader(rating_scale=(1, 5))
data_surprise = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data_surprise, test_size=0.2)

# Build an item-based collaborative filtering model
model = KNNBasic(sim_options={'user_based': False})  # Item-based
model.fit(trainset)

# Make predictions on the testset
predictions = model.test(testset)

# Calculate RMSE
rmse = accuracy.rmse(predictions)
print(f'RMSE: {rmse}')

# Function to recommend movies based on user input
def recommend_movies(user_id, num_recommendations=5):
    # Get a list of all movies
    all_movies = movie['title'].tolist()
    
    # Get movies the user has already rated
    rated_movies = data[data['userId'] == user_id]['movieId'].tolist()
    
    # Generate predictions for all movies the user hasn't rated yet
    user_predictions = []
    for movie_id in movie['movieId']:
        if movie_id not in rated_movies:
            prediction = model.predict(user_id, movie_id)
            user_predictions.append((movie_id, prediction.est))
    
    # Sort predictions by predicted rating
    user_predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N recommendations
    top_recommendations = user_predictions[:num_recommendations]
    
    # Print movie titles for the top recommendations
    recommended_movie_titles = [movie[movie['movieId'] == movie_id].iloc[0]['title'] for movie_id, _ in top_recommendations]
    
    return recommended_movie_titles

# Example usage
recommended_movies = recommend_movies(user_id=1)
print("Recommended Movies for User 1:")
for movie in recommended_movies:
    print(movie)


def recommend_movies(user_id, num_recommendations=5):
    # Get a list of all movies
    all_movies = movie['title'].tolist()
    
    # Get movies the user has already rated
    rated_movies = data[data['userId'] == user_id]['movieId'].tolist()
    
    # Generate predictions for all movies the user hasn't rated yet
    user_predictions = []
    for movie_id in movie['movieId']:
        if movie_id not in rated_movies:
            prediction = model.predict(user_id, movie_id)
            user_predictions.append((movie_id, prediction.est))
    
    # Sort predictions by predicted rating
    user_predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N recommendations
    top_recommendations = user_predictions[:num_recommendations]
    
    # Print movie titles for the top recommendations
    recommended_movie_titles = [movie[movie['movieId'] == movie_id].iloc[0]['title'] for movie_id, _ in top_recommendations]
    
    return recommended_movie_titles

# Example usage
recommended_movies = recommend_movies(user_id=1)
print("Recommended Movies for User 1:")
for movie in recommended_movies:
    print(movie)

from surprise.model_selection import cross_validate

# Perform cross-validation
results = cross_validate(model, data_surprise, measures=['RMSE'], cv=3, verbose=True)

