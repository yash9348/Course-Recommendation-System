import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import pickle

# Load dataset
df = pd.read_csv("courses_dataset.csv")

# Preprocess dataset
df['Skills Covered'] = df['Skills Covered'].fillna('').str.lower()
df['Type'] = df['Type'].str.lower()
df['Level'] = df['Level'].str.lower()

# Encode categorical variables
type_encoder = LabelEncoder()
level_encoder = LabelEncoder()
df['Type_Encoded'] = type_encoder.fit_transform(df['Type'])
df['Level_Encoded'] = level_encoder.fit_transform(df['Level'])

# Vectorize skills
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
skills_matrix = vectorizer.fit_transform(df['Skills Covered'])

# Combine features
features = pd.concat([
    pd.DataFrame(skills_matrix.toarray()),
    df[['Type_Encoded', 'Level_Encoded']]
], axis=1)

# Ensure all column names are strings
features.columns = features.columns.astype(str)

# Train model
model = NearestNeighbors(n_neighbors=5, metric='cosine')
model.fit(features)

# Save model and encoders
with open("model.pkl", "wb") as f:
    pickle.dump((model, vectorizer, type_encoder, level_encoder, df), f)
