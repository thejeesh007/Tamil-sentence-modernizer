import streamlit as st
from modernizer_rules import modernize_text

# Configure page
st.set_page_config(
    page_title="Tamil Classical Text Modernizer",
    page_icon="🕉",
    layout="wide"
)

st.title(" Tamil Classical Text Modernizer")
st.markdown("Convert classical Tamil text to modern conversational Tamil with the help of AI assistance")

# Input text area (full width now)
input_text = st.text_area(
    "Enter classical Tamil sentence:",
    height=150,
    placeholder="உदாहரणம்: அவன் பள்ளிக்குச் செல்கிறான் மற்றும் பாடுகின்றான்"
)

# Set default values (previously from settings)
similarity_threshold = 0.3
use_word_level = True
use_patterns = True
show_details = False

# Main processing button
if st.button(" Modernize", type="primary"):
    if input_text.strip():
        # Show loading spinner
        with st.spinner('Processing...'):
            result = modernize_text(
                input_text,
                use_word_level=use_word_level,
                use_patterns=use_patterns,
                similarity_threshold=similarity_threshold
            )
        
        # Display results in columns
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            st.subheader(" Modernized Tamil:")
            st.markdown(f"<div style='background-color: #fffff; padding: 15px; border-radius: 10px; font-size: 18px; font-weight: bold;'>{result['modernized']}</div>", 
                       unsafe_allow_html=True)
        
        with result_col2:
            st.subheader(" English Translation:")
            st.markdown(f"<div style='background-color: #fffff; padding: 15px; border-radius: 10px; font-size: 16px;'>{result['english']}</div>", 
                       unsafe_allow_html=True)
        
        # Method and confidence information
        st.markdown("---")
        
        method_col1, method_col2, method_col3 = st.columns([1, 1, 1])
        
        with method_col1:
            # Method used with appropriate icons
            method_icons = {
                "word_level": "🔤", 
                "pattern_based": "🔄",
                "semantic_rules": "🧠",
                "semantic_extended": "🚀",
                "original": "❓"
            }
            
            method_descriptions = {
                "word_level": "Word-level Transformation",
                "pattern_based": "Pattern-based Transformation", 
                "semantic_rules": "Semantic Matching (Rules)",
                "semantic_extended": "AI Semantic Matching",
                "original": "No transformation applied"
            }
            
            icon = method_icons.get(result['method'], "")
            desc = method_descriptions.get(result['method'], "Unknown method")
            st.metric("Method Used", f"{icon} {desc}")
    
    else:
        st.warning(" Please enter some Tamil text to modernize")

# Sample text examples
st.markdown("---")
st.subheader(" Try these examples:")

# Create example buttons in columns
example_col1, example_col2, example_col3 = st.columns([1, 1, 1])

examples = [
    "அவன் பள்ளிக்குச் செல்கிறான் மற்றும் பாடுகின்றான்",
    "வாடையும் பிரிந்தினோர்க்கு அழலே",
    "நான் என் வாழ்நாளில் மேற்கொண்ட செயல்களைப்பற்றி சிந்திக்கிறேன்",
    "அவள் மிகவும் அழகாக விழாவிற்கு வந்தாள்",
    "மழை பெய்யும் சுழற்காற்றின் இசை இனிமையானது",
    "அந்த இடம் பசுமை நிறைந்தது மற்றும் அமைதியாக இருந்தது"
]

# Display examples in a more organized way
for i, example in enumerate(examples[:3]):
    with example_col1 if i == 0 else (example_col2 if i == 1 else example_col3):
        if st.button(f"Example {i+1}", key=f"example_{i+1}"):
            st.session_state.example_text = example

# Handle example selection
if 'example_text' in st.session_state:
    st.rerun()

# Additional examples
with st.expander(" More Examples"):
    for i, example in enumerate(examples[3:], 4):
        if st.button(f" {example[:30]}...", key=f"example_{i}"):
            st.session_state.example_text = example

# Footer with information
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px;'>
    <p> This tool uses rule-based matching, word-level transformations, pattern recognition, and AI semantic analysis</p>
    <p> Advanced processing with optimized settings for best results</p>
</div>
""", unsafe_allow_html=True)

# Add custom CSS for better styling
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        margin-bottom: 10px;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #007bff;
    }
    
    .stTextArea > div > div > textarea {
        font-family: 'Tamil', 'Noto Sans Tamil', sans-serif;
        font-size: 16px;
    }
    
    .title {
        text-align: center;
        padding: 20px 0;
    }
</style>
""", unsafe_allow_html=True)