import os
import sys
import pickle
import streamlit as st
import numpy as np
from books_recommender.logger.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.pipeline.training_pipeline import TrainingPipeline
from books_recommender.exception.exception_handler import AppException


class Recommendation:
    def __init__(self,app_config = AppConfiguration()):
        try:
            self.recommendation_config= app_config.get_recommendation_config()
        except Exception as e:
            raise AppException(e, sys) from e


    def fetch_poster(self,suggestion):
        try:
            book_name = []
            ids_index = []
            poster_url = []
            book_pivot =  pickle.load(open(self.recommendation_config.book_pivot_serialized_objects,'rb'))
            final_rating =  pickle.load(open(self.recommendation_config.final_rating_serialized_objects,'rb'))

            for book_id in suggestion:
                book_name.append(book_pivot.index[book_id])

            for name in book_name[0]: 
                ids = np.where(final_rating['title'] == name)[0][0]
                ids_index.append(ids)

            for idx in ids_index:
                url = final_rating.iloc[idx]['image_url']
                poster_url.append(url)

            return poster_url
        
        except Exception as e:
            raise AppException(e, sys) from e
        


    def recommend_book(self,book_name):
        try:
            books_list = []
            model = pickle.load(open(self.recommendation_config.trained_model_path,'rb'))
            book_pivot =  pickle.load(open(self.recommendation_config.book_pivot_serialized_objects,'rb'))
            book_id = np.where(book_pivot.index == book_name)[0][0]
            distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6 )

            poster_url = self.fetch_poster(suggestion)
            
            for i in range(len(suggestion)):
                    books = book_pivot.index[suggestion[i]]
                    for j in books:
                        books_list.append(j)
            return books_list , poster_url   
        
        except Exception as e:
            raise AppException(e, sys) from e


    def train_engine(self):
        try:
            obj = TrainingPipeline()
            obj.start_training_pipeline()
            st.text("Training Completed!")
            logging.info(f"Recommended successfully!")
        except Exception as e:
            raise AppException(e, sys) from e

    
    def recommendations_engine(self,selected_books):
        try:
            recommended_books,poster_url = self.recommend_book(selected_books)
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.text(recommended_books[1])
                st.image(poster_url[1])
            with col2:
                st.text(recommended_books[2])
                st.image(poster_url[2])

            with col3:
                st.text(recommended_books[3])
                st.image(poster_url[3])
            with col4:
                st.text(recommended_books[4])
                st.image(poster_url[4])
            with col5:
                st.text(recommended_books[5])
                st.image(poster_url[5])
        except Exception as e:
            raise AppException(e, sys) from e

if __name__ == "__main__":
    # Custom CSS for frame and styling
    st.markdown("""
    <style>
    .main-frame {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background-color: #fafafa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        text-align: center;
        padding: 10px;
        background-color: #f1f1f1;
        border-top: 1px solid #e0e0e0;
        font-size: 12px;
        color: #666;
    }
    .header-section {
        text-align: center;
        margin-bottom: 30px;
    }
    .step-section {
        margin: 20px 0;
        padding: 15px;
        border-left: 4px solid;
        border-radius: 5px;
        background-color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    # Main frame container
    st.markdown('<div class="main-frame">', unsafe_allow_html=True)
    
    # Header section
    st.markdown('<div class="header-section">', unsafe_allow_html=True)
    st.header('📚 Book Recommendation System')
    st.text("This is a Collaborative Filtering based recommendation system!")
    st.markdown('</div>', unsafe_allow_html=True)

    obj = Recommendation()

    # Training section with blue styling
    st.markdown('<div class="step-section" style="border-color: green;">', unsafe_allow_html=True)
    st.markdown('<p style="color: green; font-weight: bold; margin: 0;">🔧 Step 1: Train the System</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: green; margin: 5px 0 15px 0;">Click below to train the recommendation model</p>', unsafe_allow_html=True)
    if st.button('🚀 Train Recommendation System'):
        obj.train_engine()
    st.markdown('</div>', unsafe_allow_html=True)

    try:
        # Load and convert book names safely
        book_names = pickle.load(open(os.path.join('templates','book_names.pkl'), 'rb'))
        book_names_list = book_names.tolist() if hasattr(book_names, 'tolist') else list(book_names)
        
    except Exception as e:
        st.error(f"Error loading books: {e}")
        book_names_list = []

    # Book selection section
    st.markdown('<div class="step-section" style="border-color: orange;">', unsafe_allow_html=True)
    st.markdown('<p style="color: orange; font-weight: bold; margin: 0;">📖 Step 2: Select a Book</p>', unsafe_allow_html=True)
    
    if book_names_list:
        # Display first book in red
        st.markdown(f'<p style="color: orange; font-weight: bold; background-color: #ffebee; padding: 8px; border-radius: 5px;">Sample book available: <em>{book_names_list[0]}</em></p>', unsafe_allow_html=True)
    else:
        st.warning("No books available. Please train the system first.")

    selected_books = st.selectbox(
        "Choose a book from the dropdown:",
        book_names_list if book_names_list else ["-- Please train system first --"])
    st.markdown('</div>', unsafe_allow_html=True)

    # Recommendation section with green styling
    st.markdown('<div class="step-section" style="border-color: red;">', unsafe_allow_html=True)
    st.markdown('<p style="color: red; font-weight: bold; margin: 0;">🎯 Step 3: Get Recommendations</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: red; margin: 5px 0 15px 0;">Click below to see personalized book suggestions</p>', unsafe_allow_html=True)
    
    if st.button('📕 Show Recommendation'):
        if book_names_list and selected_books in book_names_list:
            obj.recommendations_engine(selected_books)
        else:
            st.error("Please train the system first to get recommendations.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Close main frame
    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        © 2025 Book Recommendation System. All rights reserved. | <h4 style="color: blue;"> Developed by Shewan </h4>
    </div>
    """, unsafe_allow_html=True)