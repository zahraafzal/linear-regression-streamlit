# Import Libraries
import kagglehub
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors

# ==========================
# 1. Download Dataset
# ==========================
path = kagglehub.dataset_download("arashnic/book-recommendation-dataset")
print("Dataset Path:", path)
print(os.listdir(path))

# ==========================
# 2. Load Dataset
# ==========================
books = pd.read_csv(os.path.join(path, "Books.csv"), encoding="latin-1")
ratings = pd.read_csv(os.path.join(path, "Ratings.csv"), encoding="latin-1")
users = pd.read_csv(os.path.join(path, "Users.csv"), encoding="latin-1")

print(books.head())
print(ratings.head())

# ==========================
# 3. Merge Books + Ratings
# ==========================
book_rating = ratings.merge(books, on="ISBN")
print(book_rating.head())

# ==========================
# 4. Remove Zero Ratings
# ==========================
book_rating = book_rating[book_rating["Book-Rating"] > 0]
print(book_rating.shape)

# ==========================
# 5. Select Popular Books
# ==========================
book_counts = book_rating["Book-Title"].value_counts()
popular_books = book_counts[book_counts >= 50].index
final_data = book_rating[book_rating["Book-Title"].isin(popular_books)]
print(final_data.shape)

# ==========================
# 6. Create Book-User Matrix
# ==========================
book_matrix = final_data.pivot_table(
    index="Book-Title",
    columns="User-ID",
    values="Book-Rating"
).fillna(0)

print(book_matrix.shape)
print(book_matrix.head())

# ==========================
# 7. Train KNN Model
# ==========================
knn_model = NearestNeighbors(metric="cosine", algorithm="brute")
knn_model.fit(book_matrix)

# ==========================
# 8. Recommendation Function
# ==========================
def recommend_book(book_name, n=5):
    if book_name not in book_matrix.index:
        return "Book not found"
    
    book_index = book_matrix.index.get_loc(book_name)
    
    distances, indices = knn_model.kneighbors(
        book_matrix.iloc[book_index].values.reshape(1, -1),
        n_neighbors=n+1
    )
    
    recommendations = []
    for i in indices[0][1:]:
        recommendations.append(book_matrix.index[i])
    
    return recommendations

# ==========================
# 9. Test Recommendation
# ==========================
book = "1984"
result = recommend_book(book, 5)

print("Recommended Books:")
for r in result:
    print(r)