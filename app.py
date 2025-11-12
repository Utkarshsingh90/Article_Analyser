import streamlit as st
import json
from datetime import datetime
from io import StringIO
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import spacy
from typing import Dict, List, Optional
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Police Recognition Analytics",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for appealing UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .info-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .success-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Cache models for performance
@st.cache_resource
def load_models():
    """Load all required ML models"""
    try:
        # Sentiment analysis model
        sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # NER model for entity extraction
        ner_model = pipeline(
            "ner", 
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Summarization model
        summarizer = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Translation model (multi-language to English)
        translator = pipeline(
            "translation", 
            model="Helsinki-NLP/opus-mt-mul-en",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Q&A model
        qa_model = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Load spaCy for additional NER
        nlp = spacy.load("en_core_web_sm")
        
        return sentiment_analyzer, ner_model, summarizer, translator, qa_model, nlp
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None, None

def detect_language(text: str) -> str:
    """Simple language detection"""
    # Basic heuristic - check for non-ASCII characters
    if any(ord(char) > 127 for char in text):
        return "non-english"
    return "english"

def translate_to_english(text: str, translator) -> str:
    """Translate non-English text to English"""
    try:
        if detect_language(text) == "non-english":
            result = translator(text, max_length=512)
            return result[0]['translation_text']
        return text
    except:
        return text

def extract_officer_info(text: str, nlp) -> Dict:
    """Extract officer names and departments using NER"""
    doc = nlp(text)
    
    officers = []
    departments = []
    locations = []
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # Check if it's likely an officer name
            if any(title in text.lower() for title in ['officer', 'sgt', 'sergeant', 'det', 'detective', 'chief', 'captain']):
                officers.append(ent.text)
        elif ent.label_ == "ORG":
            departments.append(ent.text)
        elif ent.label_ == "GPE":
            locations.append(ent.text)
    
    return {
        "officers": list(set(officers)),
        "departments": list(set(departments)),
        "locations": list(set(locations))
    }

def analyze_sentiment_detailed(text: str, sentiment_analyzer) -> Dict:
    """Perform sentiment analysis with detailed scores"""
    try:
        result = sentiment_analyzer(text[:512])[0]
        
        # Convert to normalized score between -1 and 1
        if result['label'] == 'POSITIVE':
            sentiment_score = result['score']
        else:
            sentiment_score = -result['score']
        
        return {
            "label": result['label'],
            "score": result['score'],
            "normalized_score": sentiment_score
        }
    except Exception as e:
        return {"label": "NEUTRAL", "score": 0.5, "normalized_score": 0.0}

def extract_competency_tags(text: str) -> List[str]:
    """Extract competency tags from text using keyword matching"""
    competencies = {
        "community_engagement": ["community", "engagement", "outreach", "relationship", "trust", "friendly"],
        "de-escalation": ["de-escalate", "calm", "peaceful", "resolved", "mediation", "conflict resolution"],
        "rapid_response": ["quick", "fast", "immediate", "prompt", "timely", "rapid"],
        "professionalism": ["professional", "courteous", "respectful", "polite", "manner"],
        "life_saving": ["saved", "rescue", "life-saving", "emergency", "critical"],
        "investigation": ["investigation", "solved", "detective", "evidence", "case"],
        "compassion": ["compassion", "care", "kindness", "empathy", "understanding", "helped"],
        "bravery": ["brave", "courage", "heroic", "danger", "risk"]
    }
    
    text_lower = text.lower()
    found_tags = []
    
    for tag, keywords in competencies.items():
        if any(keyword in text_lower for keyword in keywords):
            found_tags.append(tag)
    
    return found_tags if found_tags else ["general_commendation"]

def generate_summary(text: str, summarizer) -> str:
    """Generate summary of the text"""
    try:
        if len(text) < 100:
            return text
        
        summary = summarizer(text[:1024], max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return text[:200] + "..."

def calculate_recognition_score(sentiment_score: float, tags: List[str], text_length: int) -> float:
    """Calculate a recognition score based on multiple factors"""
    base_score = (sentiment_score + 1) / 2  # Normalize to 0-1
    
    # Boost for specific high-value tags
    high_value_tags = ["life_saving", "bravery", "de-escalation"]
    tag_boost = sum(0.1 for tag in tags if tag in high_value_tags)
    
    # Boost for detailed feedback
    length_boost = min(0.1, text_length / 1000 * 0.1)
    
    final_score = min(1.0, base_score + tag_boost + length_boost)
    return round(final_score, 3)

def process_text(text: str, models_tuple) -> Dict:
    """Process text through the entire pipeline"""
    sentiment_analyzer, ner_model, summarizer, translator, qa_model, nlp = models_tuple
    
    # Translate if needed
    original_text = text
    translated_text = translate_to_english(text, translator)
    
    # Extract entities
    entities = extract_officer_info(translated_text, nlp)
    
    # Analyze sentiment
    sentiment = analyze_sentiment_detailed(translated_text, sentiment_analyzer)
    
    # Extract tags
    tags = extract_competency_tags(translated_text)
    
    # Generate summary
    summary = generate_summary(translated_text, summarizer)
    
    # Calculate score
    score = calculate_recognition_score(
        sentiment['normalized_score'],
        tags,
        len(translated_text)
    )
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "original_text": original_text,
        "translated_text": translated_text if original_text != translated_text else None,
        "summary": summary,
        "extracted_officers": entities['officers'],
        "extracted_departments": entities['departments'],
        "extracted_locations": entities['locations'],
        "sentiment_label": sentiment['label'],
        "sentiment_score": sentiment['normalized_score'],
        "suggested_tags": tags,
        "recognition_score": score,
        "text_length": len(translated_text)
    }
    
    return result

def answer_question(question: str, context: str, qa_model) -> str:
    """Answer questions about the processed text"""
    try:
        result = qa_model(question=question, context=context)
        return result['answer']
    except Exception as e:
        return f"Unable to answer: {str(e)}"

# Main App
def main():
    st.markdown('<h1 class="main-header">🚔 Police Recognition Analytics Platform</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <b>Welcome!</b> This platform uses AI to analyze public feedback, news articles, and social media posts 
        to identify and recognize outstanding police work. Upload documents or paste text to get started.
    </div>
    """, unsafe_allow_html=True)
    
    # Load models with progress
    with st.spinner("🔄 Loading AI models... This may take a moment on first run."):
        models = load_models()
    
    if models[0] is None:
        st.error("Failed to load models. Please check your internet connection and try again.")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/police-badge.png", width=100)
        st.title("Navigation")
        
        st.markdown("---")
        st.subheader("📊 Statistics")
        st.metric("Total Processed", len(st.session_state.processed_data))
        
        if st.session_state.processed_data:
            avg_score = sum(d['recognition_score'] for d in st.session_state.processed_data) / len(st.session_state.processed_data)
            st.metric("Avg Recognition Score", f"{avg_score:.2f}")
        
        st.markdown("---")
        if st.button("🗑️ Clear All Data", type="secondary"):
            st.session_state.processed_data = []
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("💾 Export Data", type="primary"):
            if st.session_state.processed_data:
                df = pd.DataFrame(st.session_state.processed_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    "police_recognition_data.csv",
                    "text/csv"
                )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Process Feedback", "📊 Dashboard", "💬 Q&A Chat", "📈 Detailed View"])
    
    with tab1:
        st.header("Process New Feedback")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            input_method = st.radio("Input Method:", ["Text Input", "File Upload"], horizontal=True)
            
            text_to_process = ""
            
            if input_method == "Text Input":
                text_to_process = st.text_area(
                    "Paste feedback, news article, or social media post:",
                    height=200,
                    placeholder="Example: Officer Smith from the 14th Precinct showed incredible compassion when helping a lost child find their parents..."
                )
            else:
                uploaded_file = st.file_uploader(
                    "Upload file (TXT, PDF, or DOCX)",
                    type=['txt', 'pdf', 'docx']
                )
                
                if uploaded_file:
                    if uploaded_file.type == "text/plain":
                        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                        text_to_process = stringio.read()
                    elif uploaded_file.type == "application/pdf":
                        try:
                            import PyPDF2
                            pdf_reader = PyPDF2.PdfReader(uploaded_file)
                            text_to_process = ""
                            for page in pdf_reader.pages:
                                text_to_process += page.extract_text()
                        except:
                            st.error("Please install PyPDF2: pip install PyPDF2")
                    
                    if text_to_process:
                        st.text_area("Extracted Text Preview:", text_to_process[:500] + "...", height=150)
        
        with col2:
            st.info("**Supported Languages:**\n- English\n- Spanish\n- French\n- German\n- And more!")
            
            st.success("**Features:**\n✅ Sentiment Analysis\n✅ Entity Extraction\n✅ Auto-Summarization\n✅ Competency Tagging")
        
        if st.button("🚀 Process Feedback", type="primary", use_container_width=True):
            if text_to_process:
                with st.spinner("🔍 Analyzing feedback..."):
                    result = process_text(text_to_process, models)
                    st.session_state.processed_data.append(result)
                
                st.markdown('<div class="success-box">✅ <b>Processing Complete!</b></div>', unsafe_allow_html=True)
                
                # Display results in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{result['recognition_score']}</h3>
                        <p>Recognition Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    sentiment_emoji = "😊" if result['sentiment_label'] == 'POSITIVE' else "😐"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{sentiment_emoji}</h3>
                        <p>{result['sentiment_label']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{len(result['extracted_officers'])}</h3>
                        <p>Officers Found</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{len(result['suggested_tags'])}</h3>
                        <p>Competency Tags</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Detailed results
                with st.expander("📋 View Detailed Results", expanded=True):
                    if result['translated_text']:
                        st.warning("**Original text was translated to English**")
                        st.text_area("Original:", result['original_text'], height=100)
                        st.text_area("Translated:", result['translated_text'], height=100)
                    
                    st.subheader("📝 Summary")
                    st.info(result['summary'])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("👮 Extracted Officers")
                        if result['extracted_officers']:
                            for officer in result['extracted_officers']:
                                st.write(f"- {officer}")
                        else:
                            st.write("No specific officers mentioned")
                        
                        st.subheader("🏢 Departments")
                        if result['extracted_departments']:
                            for dept in result['extracted_departments']:
                                st.write(f"- {dept}")
                        else:
                            st.write("No departments identified")
                    
                    with col2:
                        st.subheader("🏷️ Competency Tags")
                        for tag in result['suggested_tags']:
                            st.write(f"- {tag.replace('_', ' ').title()}")
                        
                        st.subheader("📍 Locations")
                        if result['extracted_locations']:
                            for loc in result['extracted_locations']:
                                st.write(f"- {loc}")
                        else:
                            st.write("No locations identified")
                
                # Export single result
                st.download_button(
                    "📥 Export This Result (JSON)",
                    json.dumps(result, indent=2),
                    f"recognition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
            else:
                st.warning("Please enter or upload some text to process.")
    
    with tab2:
        st.header("Recognition Dashboard")
        
        if st.session_state.processed_data:
            df = pd.DataFrame(st.session_state.processed_data)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Submissions", len(df))
            with col2:
                st.metric("Avg Score", f"{df['recognition_score'].mean():.2f}")
            with col3:
                positive_pct = (df['sentiment_label'] == 'POSITIVE').sum() / len(df) * 100
                st.metric("Positive Feedback", f"{positive_pct:.1f}%")
            with col4:
                total_officers = sum(len(officers) for officers in df['extracted_officers'])
                st.metric("Officers Recognized", total_officers)
            
            st.markdown("---")
            
            # Top officers
            st.subheader("🏆 Top Recognized Officers")
            all_officers = []
            for officers in df['extracted_officers']:
                all_officers.extend(officers)
            
            if all_officers:
                officer_counts = pd.Series(all_officers).value_counts().head(10)
                st.bar_chart(officer_counts)
            else:
                st.info("No officers identified yet in processed feedback.")
            
            # Tag distribution
            st.subheader("📊 Competency Tag Distribution")
            all_tags = []
            for tags in df['suggested_tags']:
                all_tags.extend(tags)
            
            if all_tags:
                tag_counts = pd.Series(all_tags).value_counts()
                st.bar_chart(tag_counts)
            
            # Recent submissions
            st.subheader("📜 Recent Submissions")
            for idx, row in df.tail(5).iterrows():
                with st.expander(f"Submission {idx + 1} - Score: {row['recognition_score']}"):
                    st.write(f"**Summary:** {row['summary']}")
                    st.write(f"**Officers:** {', '.join(row['extracted_officers']) if row['extracted_officers'] else 'None identified'}")
                    st.write(f"**Tags:** {', '.join(row['suggested_tags'])}")
                    st.write(f"**Sentiment:** {row['sentiment_label']} ({row['sentiment_score']:.2f})")
        else:
            st.info("No data processed yet. Go to 'Process Feedback' tab to get started!")
    
    with tab3:
        st.header("💬 Q&A Chat")
        st.write("Ask questions about the processed feedback")
        
        if st.session_state.processed_data:
            # Combine all processed texts for context
            all_texts = " ".join([d['translated_text'] or d['original_text'] for d in st.session_state.processed_data])
            
            question = st.text_input("Ask a question:", placeholder="e.g., What acts of bravery were mentioned?")
            
            if st.button("Get Answer", type="primary"):
                if question:
                    with st.spinner("🤔 Thinking..."):
                        answer = answer_question(question, all_texts[:2000], models[4])
                        st.session_state.chat_history.append({"question": question, "answer": answer})
            
            # Display chat history
            if st.session_state.chat_history:
                st.markdown("---")
                st.subheader("Chat History")
                for i, chat in enumerate(reversed(st.session_state.chat_history)):
                    st.markdown(f"**Q{len(st.session_state.chat_history) - i}:** {chat['question']}")
                    st.info(f"**A:** {chat['answer']}")
                    st.markdown("---")
        else:
            st.warning("Please process some feedback first before using Q&A!")
    
    with tab4:
        st.header("📈 Detailed Data View")
        
        if st.session_state.processed_data:
            df = pd.DataFrame(st.session_state.processed_data)
            
            # Data table with selection
            st.subheader("Complete Data Table")
            
            # Select columns to display
            columns_to_show = st.multiselect(
                "Select columns to display:",
                df.columns.tolist(),
                default=['timestamp', 'recognition_score', 'sentiment_label', 'extracted_officers', 'suggested_tags']
            )
            
            if columns_to_show:
                st.dataframe(df[columns_to_show], use_container_width=True)
            
            # Export options
            st.markdown("---")
            st.subheader("📥 Export Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    "full_data.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                json_data = df.to_json(orient='records', indent=2)
                st.download_button(
                    "Download JSON",
                    json_data,
                    "full_data.json",
                    "application/json",
                    use_container_width=True
                )
            
            with col3:
                excel_buffer = StringIO()
                df.to_excel(excel_buffer, index=False, engine='openpyxl')
                st.download_button(
                    "Download Excel",
                    excel_buffer.getvalue(),
                    "full_data.xlsx",
                    "application/vnd.ms-excel",
                    use_container_width=True
                )
        else:
            st.info("No data available yet.")

if __name__ == "__main__":
    main()
