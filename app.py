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
    # Enhanced CSS for modern look
    st.markdown("""
    <style>
    .main-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 20px;
    }
    .app-frame {
        border: 1px solid #ddd;
        border-radius: 15px;
        padding: 30px;
        margin: 20px 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    .app-header {
        text-align: center;
        margin-bottom: 40px;
        padding: 20px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    .step-card {
        background: white;
        padding: 25px;
        margin: 25px 0;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border-left: 5px solid;
        transition: transform 0.3s ease;
    }
    .step-card:hover {
        transform: translateY(-2px);
    }
    .footer {
        text-align: center;
        padding: 20px;
        margin-top: 40px;
        color: #666;
        font-size: 14px;
        border-top: 1px solid #e0e0e0;
        background: white;
        border-radius: 0 0 10px 10px;
    }
    .highlight-book {
        background: linear-gradient(45deg, #ffebee, #ffcdd2);
        padding: 12px;
        border-radius: 8px;
        margin: 15px 0;
        border-left: 4px solid #f44336;
    }
    </style>
    """, unsafe_allow_html=True)

    # Main container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # App frame
    st.markdown('<div class="app-frame">', unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="app-header">', unsafe_allow_html=True)
    st.markdown('<h1 style="color: #2c3e50; margin-bottom: 10px;">📚 Book Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #7f8c8d; font-size: 16px;">Discover your next favorite read with AI-powered recommendations</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    obj = Recommendation()

    # Step 1: Training
    st.markdown('<div class="step-card" style="border-left-color: #3498db;">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #3498db; margin-bottom: 15px;">🔧 Step 1: System Training</h3>', unsafe_allow_html=True)
    st.markdown('<p style="color: #3498db; margin-bottom: 20px;">Initialize the recommendation engine by training the model</p>', unsafe_allow_html=True)
    
    if st.button('🚀 Train Recommendation System', key='train_btn'):
        with st.spinner('Training in progress... This may take a few minutes.'):
            obj.train_engine()
        st.success('✅ Training completed successfully!')
    st.markdown('</div>', unsafe_allow_html=True)

    try:
        # Load book data
        book_names = pickle.load(open(os.path.join('templates','book_names.pkl'), 'rb'))
        book_names_list = book_names.tolist() if hasattr(book_names, 'tolist') else list(book_names)
    except Exception as e:
        st.error(f"❌ Error loading book data: {e}")
        book_names_list = []

    # Step 2: Book Selection
    st.markdown('<div class="step-card" style="border-left-color: #e67e22;">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #e67e22; margin-bottom: 15px;">📖 Step 2: Book Selection</h3>', unsafe_allow_html=True)
    
    if book_names_list:
        st.markdown(f"""
        <div class="highlight-book">
            <p style="color: #c0392b; font-weight: bold; margin: 0;">
                📚 First available book: <em>{book_names_list[0]}</em>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No books available. Please train the system first.")

    selected_books = st.selectbox(
        "Choose your book:",
        book_names_list if book_names_list else ["-- Please train the system first --"],
        key='book_select'
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Step 3: Recommendations
    st.markdown('<div class="step-card" style="border-left-color: #27ae60;">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #27ae60; margin-bottom: 15px;">🎯 Step 3: Get Recommendations</h3>', unsafe_allow_html=True)
    st.markdown('<p style="color: #27ae60; margin-bottom: 20px;">Discover books similar to your selection</p>', unsafe_allow_html=True)
    
    if st.button('✨ Show Recommendations', key='recommend_btn'):
        if book_names_list and selected_books in book_names_list:
            with st.spinner('Finding the perfect recommendations for you...'):
                obj.recommendations_engine(selected_books)
        else:
            st.error("❌ Please train the system and select a valid book first.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Close app frame
    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <p style="margin: 0;">
            © 2024 Book Recommendation System | All Rights Reserved | Developed with ❤️ by Shewan
        </p>
        <p style="margin: 5px 0 0 0; font-size: 12px; color: #95a5a6;">
            Powered by Streamlit & Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Close main container
    st.markdown('</div>', unsafe_allow_html=True)