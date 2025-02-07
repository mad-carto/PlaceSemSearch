"""
App Name: Place-Based Semantic Search
Author: Madalina Gugulica
Date: 18.11.2024
Description:
    This application uses Streamlit to build an interactive dashboard for Place-Based Similarity Search. 
    It enables users to:
      - Select a target location (H3 hexbin) by interacting with a map.
      - Compute the top N most similar places to the selected location based on semantic embeddings.
      - Visualize the target and similar locations on a map 
      - Explore geospatial insights dynamically through interactive tooltips, bar charts and word clouds.

Dependencies:
    - Streamlit
    - Pandas
    - NumPy
    - SQLite3
    - Folium
    - WordCloud
    - SentenceTransformers
    - H3
    - Streamlit-Folium
    - Scikit-learn (for cosine similarity)

Features:
    - Interactive map to select the target H3 hexbin.
    - Semantic similarity computation to find the most similar locations.
    - Dynamic visualization of the target and similar places on a Folium map.
    - Color-coded similarity scores and word cloud popups for detailed exploration.

Note:
    - Ensure the SQLite database with precomputed embeddings is available.
    - This application is designed to analyze geospatial data with semantic textual content.
    - The input geospatial data (CSV) should include columns for 'h3_index_9', 'latitude', and 'longitude'.

"""
import streamlit as st
from streamlit_folium import st_folium

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import folium
import matplotlib.colors as mcolors
from folium import IFrame, Element
from folium.plugins import FloatImage
from folium import Tooltip
import base64
from io import BytesIO
from PIL import Image
from matplotlib.colors import ListedColormap, to_hex
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import zipfile
import pandas as pd
import numpy as np
import io

import torch
from collections import Counter
import sqlite3

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import h3


def load_document_embedding_from_sqlite(h3_index, table_name, db_path):
    """
    Loads a specific embedding for the given h3_index from the specified table in the SQLite database.

    Parameters:
    - h3_index (str): The h3_index to load.
    - table_name (str): The table name from which to load the embedding.
    - db_path (str): The path to the SQLite database file.

    Returns:
    - np.ndarray: The loaded embedding.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute(f"SELECT embedding FROM {
              table_name} WHERE h3_index=?", (h3_index,))
    row = c.fetchone()
    conn.close()

    if row is not None:
        embedding = np.frombuffer(row[0], dtype=np.float32)
        return embedding
    else:
        raise ValueError(f"Embedding for h3_index {
                         h3_index} not found in {table_name} table.")


def compute_similar_places(target_h3_index, table_name, db_path, cos_sim):
    """
    Computes all H3 index cells with a similarity above a given threshold.

    Parameters:
    - target_h3_index (str): The H3 index for which to find similar cells.
    - table_name (str): The table name from which to load the embeddings.
    - db_path (str): The path to the SQLite database file.
    - similarity_threshold (float): The minimum similarity score to include an H3 index.

    Returns:
    - pd.DataFrame: A DataFrame with columns ['h3_index', 'similarity'] of cells above the threshold.
    """
    # Load the target embedding
    try:
        target_embedding = load_document_embedding_from_sqlite(
            target_h3_index, table_name, db_path)
    except Exception as e:
        raise ValueError(f"Error loading embedding for target H3 index {
                         target_h3_index}: {e}")

    # Compute post counts for each H3 index from the posts DataFrame
    h3_post_counts = df['h3_index_9'].value_counts()

    # Filter H3 indices with at least 5 posts, excluding the target index
    valid_h3_indices = h3_post_counts[h3_post_counts >= 5].index.tolist()
    if target_h3_index in valid_h3_indices:
        valid_h3_indices.remove(target_h3_index)

    # Connect to the database and retrieve embeddings for valid H3 indices
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    placeholders = ','.join(['?'] * len(valid_h3_indices))
    c.execute(f"""
        SELECT h3_index, embedding
        FROM {table_name}
        WHERE h3_index IN ({placeholders})
    """, valid_h3_indices)
    rows = c.fetchall()
    conn.close()

    if len(rows) == 0:
        raise ValueError(
            "No embeddings found in the database for the specified H3 indices.")

    # Compute cosine similarities
    similarities = []
    for row in rows:
        h3_index = row[0]
        embedding = np.frombuffer(row[1], dtype=np.float32)
        similarity = cosine_similarity([target_embedding], [embedding])[0][0]

        # Only include cells above the threshold
        if similarity >= cos_sim:
            similarities.append(
                {'h3_index': h3_index, 'similarity': similarity})

    # Convert to DataFrame and sort by similarity
    similarities_df = pd.DataFrame(similarities)
    similarities_df = similarities_df.sort_values(
        by='similarity', ascending=False)

    # Print the number of identified places
    print(f"Number of similar places identified: {len(similarities_df)}")

    return similarities_df


def visualize_initial_map(df):
    # Calculate map center
    map_center = [51.0504, 13.7373]
    folium_map = folium.Map(location=map_center,
                            zoom_start=12, tiles="Cartodb Positron")

    h3_indices = df['h3_index_9'].unique()

    for idx, h3_index in enumerate(h3_indices):
        boundary = h3_to_boundary(h3_index)
        folium.Polygon(
            locations=boundary,
            color="gray",
            fill=True,
            fill_opacity=0.3,
            weight=0.5,
            tooltip=f"H3 Index: {h3_index}"
        ).add_to(folium_map)

    return folium_map


def image_to_base64(image):
    """
    Converts a PIL Image object to a base64 string.
    """
    # Save the image to a BytesIO object
    buffered = BytesIO()
    image.save(buffered, format="PNG")

    # Convert the image to base64
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return img_str


def get_posts_for_h3_index(df, h3_index, text_column):
    filtered_df = df[df['h3_index_9'] == h3_index][text_column]
    # Ensure all tokens are joined as strings
    concatenated_text = ' '.join(filtered_df.astype(str))
    return concatenated_text


def generate_wordclouds_for_map(df, results_df, text_column, top_n):
    """
    Generates word clouds for all H3 hex bins in results_df.

    Parameters:
    - results_df (pd.DataFrame): The DataFrame with search results containing 'h3_index_9' and 'score'.
    - text_column (str): The column in results_df containing the text lists.

    Returns:
    - dict: A dictionary where keys are H3 indices, and values are word cloud images (PIL.Image).
    """
    wordclouds_dict = {}

    for h3_index in results_df['h3_index'].head(top_n):

        concatenated_text = get_posts_for_h3_index(df, h3_index, text_column)

        # Count word frequencies using Counter
        word_counts = Counter(concatenated_text.split())

        # Generate the word cloud using the word frequencies
        wordcloud = WordCloud(
            width=400,
            height=200,
            background_color='white',
            contour_width=0.5,
            contour_color='black',
            relative_scaling='auto',
            color_func=lambda *args, **kwargs: (2, 33, 15),
            normalize_plurals=True,
            repeat=False,
            min_word_length=3
        ).generate_from_frequencies(word_counts)

        # Save the word cloud to the dictionary
        wordclouds_dict[h3_index] = wordcloud.to_image()

    return wordclouds_dict


def generate_bar_charts_for_map(df, results_df, top_n, topic_column, topic_labels_column):
    """
    Generates pie charts for the top 10 topics for each H3 hex bin in results_df.

    Parameters:
    - df (pd.DataFrame): The DataFrame with the full data containing topic columns and H3 indices.
    - results_df (pd.DataFrame): The DataFrame with search results containing 'h3_index' and 'score'.
    - topic_column (str): The column containing topic numbers (e.g., 'T#_GTE1').
    - topic_names_column (str): The column containing descriptive names of the topics.
    - top_n (int): The number of top results to process.

    Returns:
    - dict: A dictionary where keys are H3 indices, and values are barcharts images (PIL.Image).
    """
    bar_charts_dict = {}

    for h3_index in results_df['h3_index'].head(top_n):
        # Filter rows for the current H3 index
        cell_data = df[df['h3_index_9'] == h3_index]

        # Count topic frequencies
        topic_counts = Counter(cell_data[topic_column])

        # Sort and select the top 10 topics
        top_10_topics = topic_counts.most_common(10)

        # Prepare labels and frequencies for the pie chart
        topics, frequencies = zip(*top_10_topics)

        # Map topic numbers to names (limit to first 3 words)
        topic_labels = {topic: label for topic, label in zip(
            df[topic_column], df[topic_labels_column])}
        labels = [topic_labels[topic] for topic in topics]

        # Create the horizontal bar chart
        plt.figure(figsize=(8, 4), dpi=100)
        plt.barh(labels, frequencies, color=plt.cm.tab10(range(len(topics))))

        # Customize font sizes
        # X-axis label with larger font size
        plt.xlabel('Number of Posts', fontsize=14)
        # Y-axis label with larger font size
        plt.title(f'Top {len(labels)} Topics for H3 Index {
                  target_h3_index}', fontsize=16)

        # Customize tick labels font size
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=15)
        plt.gca().invert_yaxis()  # Invert y-axis for better readability

        # Add layout adjustments for better spacing
        plt.tight_layout()
        # Save the chart to an image
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        bar_chart_image = Image.open(buf)

        # Store the chart in the dictionary
        bar_charts_dict[h3_index] = bar_chart_image

    return bar_charts_dict


def h3_to_boundary(h3_index):

    boundary = h3.cell_to_boundary(h3_index)

    return [(lat, lon) for lat, lon in boundary]


def visualize_similar_places_barcharts(df, target_h3_index, top_similar_h3_df, barcharts, top_n=10):
    """
    Visualizes the H3 grid cells on a map using Folium. Highlights the target cell and the top N similar cells, 
    and displays pre-generated barcharts of top 10 topics as popups.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'h3_index_9', 'latitude', and 'longitude' columns.
    - target_h3_index (str): The target h3_index.
    - top_similar_h3_df (pd.DataFrame): DataFrame containing 'h3_index' and 'similarity' columns for the top N most similar cells.
    - wordclouds (dict): Dictionary of pre-generated word cloud images keyed by H3 index.
    - text_column (str): The column containing the sentences.
    - top_n (int): The number of top similar h3_index cells to visualize.

    Returns:
    - folium.Map: A Folium map visualizing the H3 grid.
    """
    # Calculate the center of the map
    boundary = h3_to_boundary(target_h3_index)
    centroid = [
        np.mean([point[0] for point in boundary]),  # Average latitude
        np.mean([point[1] for point in boundary])   # Average longitude
    ]

    # Initialize the Folium map
    folium_map = folium.Map(
        location=centroid, zoom_start=12, tiles='Cartodb Positron')

    # Filter data for the H3 index
    cell_data = df[df['h3_index_9'] == target_h3_index]

    # Count topic frequencies
    topic_counts = Counter(cell_data['T#_GTE1'])

    # Sort and select the top N topics
    top_topics = topic_counts.most_common(10)
    topics, frequencies = zip(*top_topics)

    # Map topics to labels
    topic_labels = {topic: label for topic,
                    label in zip(df['T#_GTE1'], df['Label'])}
    labels = [topic_labels.get(topic, f'Topic {topic}') for topic in topics]

    # Create the horizontal bar chart
    plt.figure(figsize=(8, 4), dpi=100)
    plt.barh(labels, frequencies, color=plt.cm.tab10(range(len(topics))))

    # Customize font sizes
    # X-axis label with larger font size
    plt.xlabel('Number of Posts', fontsize=14)
    # Y-axis label with larger font size
    plt.title(f"Top {len(labels)} Topics for H3 Index {
              target_h3_index}", fontsize=16)  # Title with larger font size

    # Customize tick labels font size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=15)

    plt.gca().invert_yaxis()  # Invert y-axis for better readability

    # Add layout adjustments for better spacing
    plt.tight_layout()
    # Save the chart to an image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    bar_chart_image = Image.open(buf)

    # Convert the image to base64 for embedding in Folium popup
    img_base64 = image_to_base64(bar_chart_image)
    image_html = f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{img_base64}" style="width: 350px; height:auto;">
        </div>
        """
    popup = folium.Popup(image_html)

    boundary = h3_to_boundary(target_h3_index)
    folium_polygon = folium.vector_layers.Polygon(
        locations=boundary,
        color='#5e4fa2',
        fill=True,
        fill_opacity=0.5,
        weight=2,
        tooltip=f'Target: {target_h3_index}',
    )
    folium_polygon.add_child(popup)
    folium_map.add_child(folium_polygon)

    # Add the top N similar cells
    min_score = top_similar_h3_df['similarity'].min()
    max_score = top_similar_h3_df['similarity'].max()

    for i, row in top_similar_h3_df.head(top_n).iterrows():
        h3_index = row['h3_index']
        score = row['similarity']

        if h3_index in barcharts:
            barchart_img = barcharts[h3_index]
            img_base64 = image_to_base64(barchart_img)
            image_html = f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{img_base64}" style="width: 350px; height: auto;">
        </div>
        """
            popup = folium.Popup(image_html, max_width=500)
            boundary = h3_to_boundary(h3_index)
            folium_polygon = folium.vector_layers.Polygon(
                locations=boundary,
                color='#008837',
                fill=True,
                fill_opacity=0.3,
                weight=2,
                tooltip=f'{h3_index}\n Score: {score:.4f}',
            )
            folium_polygon.add_child(popup)
            folium_map.add_child(folium_polygon)

    # Add all other cells in gray
    unique_h3_indices = df['h3_index_9'].unique()
    for h3_index in unique_h3_indices:
        if h3_index not in top_similar_h3_df['h3_index'].values and h3_index != target_h3_index:
            boundary = h3_to_boundary(h3_index)
            folium_polygon = folium.vector_layers.Polygon(
                locations=boundary,
                color='gray',
                fill=True,
                fill_opacity=0.0,
                weight=0.5,
                tooltip=f'H3: {h3_index}',
            )
            folium_map.add_child(folium_polygon)

    return folium_map


def visualize_similar_places_wordclouds(df, target_h3_index, top_similar_h3_df, wordclouds, top_n=10):
    """
    Visualizes the H3 grid cells on a map using Folium. Highlights the target cell and the top N similar cells, 
    and displays pre-generated word clouds as popups.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'h3_index_9', 'latitude', and 'longitude' columns.
    - target_h3_index (str): The target h3_index.
    - top_similar_h3_df (pd.DataFrame): DataFrame containing 'h3_index' and 'similarity' columns for the top N most similar cells.
    - wordclouds (dict): Dictionary of pre-generated word cloud images keyed by H3 index.
    - text_column (str): The column containing the sentences.
    - top_n (int): The number of top similar h3_index cells to visualize.

    Returns:
    - folium.Map: A Folium map visualizing the H3 grid.
    """
    # Calculate the center of the map
    boundary = h3_to_boundary(target_h3_index)
    centroid = [
        np.mean([point[0] for point in boundary]),  # Average latitude
        np.mean([point[1] for point in boundary])   # Average longitude
    ]

    # Initialize the Folium map
    folium_map = folium.Map(
        location=centroid, zoom_start=12, tiles='Cartodb Positron')

    concatenated_text = get_posts_for_h3_index(df, target_h3_index, 'tokens')
    # Count word frequencies using Counter
    word_counts = Counter(concatenated_text.split())

    # Generate the word cloud using the word frequencies
    wordcloud = WordCloud(
        width=400,
        height=200,
        background_color='white',
        contour_width=0.5,
        contour_color='black',
        relative_scaling='auto',
        color_func=lambda *args, **kwargs: (29, 19, 54),
        normalize_plurals=True,
        repeat=False,
        min_word_length=3
    ).generate_from_frequencies(word_counts)

    wordcloud_image = wordcloud.to_image()
    # Convert the image to base64 for embedding in Folium popup
    img_base64 = image_to_base64(wordcloud_image)
    image_html = f"""
    <div style="text-align: center;">
            <img src="data:image/png;base64,{img_base64}" style="width: 350px; height: auto;">
        </div>
        """
    popup = folium.Popup(image_html, max_width=500)

    boundary = h3_to_boundary(target_h3_index)
    folium_polygon = folium.vector_layers.Polygon(
        locations=boundary,
        color='#5e4fa2',
        fill=True,
        fill_opacity=0.5,
        weight=2,
        tooltip=f'Target: {target_h3_index}',
    )
    folium_polygon.add_child(popup)
    folium_map.add_child(folium_polygon)

    # Add the top N similar cells
    min_score = top_similar_h3_df['similarity'].min()
    max_score = top_similar_h3_df['similarity'].max()

    for i, row in top_similar_h3_df.head(top_n).iterrows():
        h3_index = row['h3_index']
        score = row['similarity']

        if h3_index in wordclouds:
            wordcloud_img = wordclouds[h3_index]
            img_base64 = image_to_base64(wordcloud_img)
            image_html = image_html = f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{img_base64}" style="width: 350px;">>
        </div>
        """
            popup = folium.Popup(image_html, max_width=400)
            boundary = h3_to_boundary(h3_index)
            folium_polygon = folium.vector_layers.Polygon(
                locations=boundary,
                color='#008837',
                fill=True,
                fill_opacity=0.3,
                weight=2,
                tooltip=f'{h3_index}\n Score: {score:.4f}',
            )
            folium_polygon.add_child(popup)
            folium_map.add_child(folium_polygon)

    # Add all other cells in gray
    unique_h3_indices = df['h3_index_9'].unique()
    for h3_index in unique_h3_indices:
        if h3_index not in top_similar_h3_df['h3_index'].values and h3_index != target_h3_index:
            boundary = h3_to_boundary(h3_index)
            folium_polygon = folium.vector_layers.Polygon(
                locations=boundary,
                color='gray',
                fill=True,
                fill_opacity=0.0,
                weight=0.5,
                tooltip=f'H3: {h3_index}',
            )
            folium_map.add_child(folium_polygon)

    return folium_map


# Streamlit App
st.set_page_config(page_title="Place-Based Semantic Search", layout="wide")
st.title("Place-Based Semantic Search")
st.subheader("Find similar places", divider='rainbow')

# Credits
st.sidebar.markdown(
    """
    <p style="font-size:12px;">
    <strong>© Gugulica & Burghardt (2025)</strong> | Licensed under 
    <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank">CC BY 4.0</a>
    </p>
    """,
    unsafe_allow_html=True
)

# Sidebar
# Instructions
st.sidebar.markdown(

    """
        ### Instructions

        ##### 🎛️ Adjust the parameters

        ##### 🎯 Select a hexbin on the map

        ##### 🔍 Click the **"Find Similar Places"** button
        
        ##### ✨ Click on the **highlighted hexbins** to explore place semantics

        #####
        """
)
st.sidebar.subheader("Parameters")
db_path = "01_Input/2024-12_DocumentEmbeddings_Topics.db"

table_name = "chi_weighted_mean_pooling"

text_column = "tokens"

top_n = st.sidebar.number_input(
    label="Number of Similar Places",
    min_value=1,
    max_value=50,
    value=5,
    step=1,
    help="Select the number of similar places to display."
)

cos_threshold = st.sidebar.slider(
    label="Cosine Similarity Threshold",
    min_value=0.8,  # Minimum value
    max_value=1.0,  # Maximum value
    value=0.9,      # Default value
    step=0.05,       # Step size
    help="Select the minimum cosine similarity threshold to consider a place as similar."
)
popup_viz = st.sidebar.selectbox("PlaceSem Viz", ["BarChart", "WordCloud"],
                                 help="Select WordCloud to visualize the gsm posts or BarChart for visualization of the top 10 topics.")


# Load the geospatial data
zip_file_path = f"01_Input/Dataset_Classified_GTE1_withOpenAILabels.zip"
try:
    # Check if DataFrame is already in session state
    if "df" not in st.session_state or st.session_state.df is None:
        with zipfile.ZipFile(zip_file_path, 'r') as z:
            # List the files inside the ZIP
            file_names = z.namelist()
            csv_file_name = [file for file in file_names if file.endswith('.csv')][0]  # Get the first CSV file
            
            # Open and read the CSV file
            with z.open(csv_file_name) as f:
                df = pd.read_csv(io.TextIOWrapper(f, 'utf-8'))
                
        # Store DataFrame in session state
        st.session_state.df = df

        # Initialize other session state variables after reading data
        if "selected_h3" not in st.session_state:
            st.session_state.selected_h3 = None
        if "map_mode" not in st.session_state:
            st.session_state.map_mode = "initial"
        st.success(f"Data loaded successfully!")
    else:
        df = st.session_state.df
except FileNotFoundError:
    st.error(f"File not found: {file_path}")
    st.stop()  # Stop the script if the file is not found
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()  # Stop the script if another error occurs

# Display button (always visible but disabled initially)
find_similar_button = st.sidebar.button(
    "Find Similar Places", disabled=(st.session_state.selected_h3 is None))


# Main Map Display
if "df" in st.session_state:
    df = st.session_state.df

    # Ensure session state variables exist
    if "selected_h3" not in st.session_state:
        st.session_state.selected_h3 = None
    if "map_mode" not in st.session_state:
        st.session_state.map_mode = "initial"
    if "last_clicked" not in st.session_state:
        st.session_state.last_clicked = None

    # Initial map display (user selects a hexbin)
    if st.session_state.map_mode == "initial":
        initial_map = visualize_initial_map(df)
        st_folium_map = st_folium(initial_map, width=1000, height=500)

        # Detect user's click
        if st_folium_map is not None and "last_clicked" in st_folium_map and st_folium_map["last_clicked"] is not None:
            clicked_lat = st_folium_map["last_clicked"]["lat"]
            clicked_lng = st_folium_map["last_clicked"]["lng"]
            clicked_h3 = h3.latlng_to_cell(clicked_lat, clicked_lng, 9)

            # Store the selected hexbin
            if clicked_h3 and clicked_h3 != st.session_state.selected_h3:
                st.session_state.selected_h3 = clicked_h3
                st.session_state.last_clicked = (clicked_lat, clicked_lng)

        # Display a message if a hexbin is selected
        if st.session_state.selected_h3:
            st.markdown(f"**Selected Hexbin:** {st.session_state.selected_h3}")

        # Enable the "Find Similar Places" button
        if st.session_state.selected_h3 and find_similar_button:
            st.session_state.map_mode = "similar_places"
            st.rerun()

    # Display similar places after clicking the button
    elif st.session_state.map_mode == "similar_places":
        target_h3_index = st.session_state.selected_h3

        try:
            # Compute similar places
            similar_places_df = compute_similar_places(
                target_h3_index, table_name, db_path, cos_threshold)

            # Ensure 'h3_index' column exists
            if 'h3_index' not in similar_places_df.columns:
                st.error(
                    "No valid similar places found. Try lowering the similarity threshold.")
                st.stop()

            similar_places_df = similar_places_df.head(top_n)

            # Handle visualization type
            if popup_viz == "WordCloud":
                wordclouds = generate_wordclouds_for_map(
                    df, similar_places_df, text_column, top_n)
                similar_map = visualize_similar_places_wordclouds(
                    df, target_h3_index, similar_places_df, wordclouds, top_n)
            else:
                barcharts = generate_bar_charts_for_map(
                    df, similar_places_df, top_n, topic_column='T#_GTE1', topic_labels_column='Label')
                similar_map = visualize_similar_places_barcharts(
                    df, target_h3_index, similar_places_df, barcharts, top_n)

            # Display the updated map
            st_folium(similar_map, width=1000, height=500)

            # Display the total number of similar places found
            st.markdown(
                f"**Total Similar Places Found: {len(similar_places_df)}**")

            # Add a legend
            st.markdown(
                """
                    🟣 **Target Place**  
                    🟢 **Similar Place**
                    """
            )

        except Exception as e:
            st.error(f"Error computing or visualizing similar places: {e}")

        # Reset search
        if st.sidebar.button("New Search"):
            st.session_state.map_mode = "initial"
            st.session_state.selected_h3 = None
            st.session_state.last_clicked = None
            st.rerun()
