# Personality Classifier Web App
# Features:
# - Streamlit web interface
# - Likert scale (1-5) questions
# - Random Forest classifier
# - Model saving/loading
# - Personality type explanations

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Set page config
st.set_page_config(
    page_title="Personality Type Predictor",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Load the personality dataset"""
    try:
        df = pd.read_csv("personality_dataset.csv")
        return df
    except FileNotFoundError:
        st.error("Error: Could not find 'personality_dataset.csv'. Please make sure the file exists in the same directory.")
        st.stop()

def train_model(X, y):
    """Train and return a RandomForest classifier"""
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X, y)
    return model

def predict_personality(model, input_vector, label_encoder):
    """Predict personality type and return the label and confidence"""
    prediction = model.predict([input_vector])[0]
    probas = model.predict_proba([input_vector])[0]
    conf = np.max(probas) * 100
    personality_label = label_encoder.inverse_transform([prediction])[0]
    return personality_label, conf

def show_personality_info(personality_type):
    """Display information about the predicted personality type"""
    info = {
        "Alpha": {
            "description": "Confident, assertive leaders who thrive in social hierarchies.",
            "strengths": "Natural leadership, decisiveness, charisma",
            "weaknesses": "Can be domineering, struggle with vulnerability",
            "famous_examples": "Steve Jobs, Gordon Ramsay"
        },
        "Beta": {
            "description": "Supportive, empathetic team players who value harmony.",
            "strengths": "Diplomacy, cooperation, emotional intelligence",
            "weaknesses": "May avoid conflict, can be overly accommodating",
            "famous_examples": "Fred Rogers, Ellen DeGeneres"
        },
        "Sigma": {
            "description": "Independent thinkers who operate outside social hierarchies.",
            "strengths": "Self-sufficiency, strategic thinking, calm under pressure",
            "weaknesses": "Can be overly detached, may struggle with teamwork",
            "famous_examples": "Keanu Reeves, Clint Eastwood"
        },
        "Omega": {
            "description": "Rebellious, creative outsiders who challenge norms.",
            "strengths": "Creativity, non-conformity, unique perspectives",
            "weaknesses": "May struggle with authority, can be unpredictable",
            "famous_examples": "Banksy, Lady Gaga"
        },
        "Gamma": {
            "description": "Wise, complex thinkers who seek deeper meaning.",
            "strengths": "Insightfulness, depth of thought, philosophical nature",
            "weaknesses": "Can overanalyze, may struggle with practicality",
            "famous_examples": "Carl Jung, Albert Einstein"
        },
        "Delta": {
            "description": "Reserved, hardworking individuals who value stability.",
            "strengths": "Reliability, work ethic, attention to detail",
            "weaknesses": "May resist change, can be overly cautious",
            "famous_examples": "Warren Buffett, Mark Zuckerberg"
        }
    }
    
    if personality_type in info:
        st.subheader(f"About {personality_type} Personality")
        st.markdown(f"**Description**: {info[personality_type]['description']}")
        st.markdown(f"**Strengths**: {info[personality_type]['strengths']}")
        st.markdown(f"**Weaknesses**: {info[personality_type]['weaknesses']}")
        st.markdown(f"**Famous Examples**: {info[personality_type]['famous_examples']}")
    else:
        st.warning("No additional information available for this personality type.")

# Main App
def main():
    st.title("🔮 Personality Type Predictor")
    st.markdown("""
    Answer the following questions on a scale of 1 (Strongly Disagree) to 5 (Strongly Agree) 
    to discover your personality archetype.
    """)
    
    # Load data
    df = load_data()
    
    # Prepare features and target
    X = df.drop("Personality", axis=1)
    y = df["Personality"]
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Load or train model
    model_file = "personality_model.pkl"
    if not os.path.exists(model_file):
        with st.spinner("Training model for the first time..."):
            model = train_model(X, y_encoded)
            joblib.dump(model, model_file)
            st.success("Model trained and saved!")
    else:
        model = joblib.load(model_file)
    
    # Display model info in sidebar
    st.sidebar.header("Model Information")
    st.sidebar.markdown(f"**Algorithm**: Random Forest Classifier")
    st.sidebar.markdown(f"**Training Samples**: {len(df)}")
    st.sidebar.markdown(f"**Features**: {X.shape[1]} questions")
    
    # Personality type legend in sidebar
    st.sidebar.header("Personality Types")
    st.sidebar.markdown("""
    - **Alpha**: Confident leader
    - **Beta**: Supportive and empathetic
    - **Sigma**: Lone wolf, strategic
    - **Omega**: Rebellious, creative outsider
    - **Gamma**: Wise and complex thinker
    - **Delta**: Reserved, hardworking
    """)
    
    # Question inputs
    st.header("Personality Questionnaire")
    questions = [
        "I enjoy being the center of attention.",
        "I prefer working alone rather than in a team.",
        "I am emotionally expressive and open.",
        "I like taking charge during group activities.",
        "Creativity is more important to me than logic.",
        "I often break social rules or expectations.",
        "I prefer planning over improvisation.",
        "Recognition and status motivate me.",
        "I value independence over group consensus.",
        "I usually keep my thoughts and feelings to myself."
    ]
    
    input_vector = []
    for i, question in enumerate(questions):
        st.subheader(f"Question {i+1}")
        st.write(question)
        value = st.slider(
            "Select your response:",
            min_value=1,
            max_value=5,
            value=3,
            key=f"q{i}",
            help="1 = Strongly Disagree, 5 = Strongly Agree"
        )
        input_vector.append(value)
    
    # Predict button
    if st.button("Predict My Personality"):
        if len(input_vector) != len(questions):
            st.error("Please answer all questions.")
        else:
            with st.spinner("Analyzing your responses..."):
                personality, confidence = predict_personality(model, input_vector, label_encoder)
                
                st.success(f"🎭 Your Predicted Personality: **{personality}**")
                st.metric("Confidence Level", f"{confidence:.1f}%")
                
                # Show personality info
                show_personality_info(personality)
                
                # Show some fun stats
                st.subheader("Your Response Profile")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average Response", f"{np.mean(input_vector):.1f}/5")
                with col2:
                    st.metric("Response Range", f"{min(input_vector)}-{max(input_vector)}")
                
                # Show radar chart of responses
                st.subheader("Your Response Pattern")
                chart_data = pd.DataFrame({
                    "Question": [f"Q{i+1}" for i in range(len(questions))],
                    "Score": input_vector
                })
                st.bar_chart(chart_data.set_index("Question"))

if __name__ == "__main__":
    main()