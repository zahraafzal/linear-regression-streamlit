import streamlit as st
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors

st.title("📚 Book Recommendation System")

# Load and train model
@st.cache_resource
def load_and_train_model():
    try:
        import kagglehub
        
        # Download dataset
        path = kagglehub.dataset_download("arashnic/book-recommendation-dataset")
        
        # Load datasets
        books = pd.read_csv(os.path.join(path, "Books.csv"), encoding="latin-1")
        ratings = pd.read_csv(os.path.join(path, "Ratings.csv"), encoding="latin-1")
        
        # Merge books + ratings
        book_rating = ratings.merge(books, on="ISBN")
        
        # Remove zero ratings
        book_rating = book_rating[book_rating["Book-Rating"] > 0]
        
        # Select popular books (rated by at least 50 users)
        book_counts = book_rating["Book-Title"].value_counts()
        popular_books = book_counts[book_counts >= 50].index
        final_data = book_rating[book_rating["Book-Title"].isin(popular_books)]
        
        # Create book-user matrix
        book_matrix = final_data.pivot_table(
            index="Book-Title",
            columns="User-ID",
            values="Book-Rating"
        ).fillna(0)
        
        # Train KNN model
        knn_model = NearestNeighbors(metric="cosine", algorithm="brute")
        knn_model.fit(book_matrix)
        
        return knn_model, book_matrix, books
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

with st.spinner("Loading book data..."):
    knn_model, book_matrix, books = load_and_train_model()

if knn_model is not None:
    
    
    # Show total books available
    st.info(f"📖 {len(book_matrix)} popular books available for recommendations")
    
    # Input section
    st.markdown("### Enter a Book Title:")
    
    # Get book list for selectbox
    book_list = sorted(book_matrix.index.tolist())
    
    # Search box
    search_term = st.text_input("Search for a book:", "")
    
    if search_term:
        # Filter books based on search
        filtered_books = [book for book in book_list if search_term.lower() in book.lower()]
        
        if filtered_books:
            selected_book = st.selectbox("Select a book:", filtered_books)
        else:
            st.warning("No books found matching your search")
            selected_book = None
    else:
        # Show popular books as default
        selected_book = st.selectbox("Or select from popular books:", book_list[:100])
    
    # Number of recommendations
    num_recommendations = st.slider("Number of recommendations:", 3, 10, 5)
    
    if st.button("Get Recommendations 🔍"):
        if selected_book:
            try:
                # Get book index
                book_index = book_matrix.index.get_loc(selected_book)
                
                # Get recommendations
                distances, indices = knn_model.kneighbors(
                    book_matrix.iloc[book_index].values.reshape(1, -1),
                    n_neighbors=num_recommendations + 1
                )
                
                st.markdown(f"### 📚 Books similar to **{selected_book}**:")
                
                # Display recommendations
                for i, idx in enumerate(indices[0][1:], 1):
                    recommended_book = book_matrix.index[idx]
                    similarity = 1 - distances[0][i]
                    
                    st.markdown(f"**{i}.** {recommended_book}")
                    st.progress(similarity)
                    st.markdown(f"*Similarity: {similarity*100:.1f}%*")
                    st.markdown("---")
                
          
                
            except Exception as e:
                st.error(f"Error getting recommendations: {e}")
        else:
            st.warning("Please select a book first")

