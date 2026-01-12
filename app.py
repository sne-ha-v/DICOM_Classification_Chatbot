import streamlit as st
from datetime import datetime
import requests

# Import modules (in real implementation, these would be separate files)
from config import Config
from utils.data_processor import DataProcessor
from components.chat_handler import ChatHandler
from components.ui_components import UIComponents

# Page configuration
st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon=Config.APP_ICON,
    layout=Config.PAGE_LAYOUT
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! I'm your Lung Nodule Analysis Assistant. Upload a CT scan image (NIfTI, DICOM, or TCIA format) using the sidebar, or ask me questions about lung nodules!",
        "timestamp": datetime.now()
    }]

if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None

# Render UI
UIComponents.render_header()
UIComponents.render_sidebar()

# Main content area with two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Chat Interface")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            st.caption(message["timestamp"].strftime("%I:%M %p"))
    
    # Chat input
    if prompt := st.chat_input("Ask me about lung nodules..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now()
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            # Create context with current analysis and conversation history
            context = {}
            if st.session_state.current_analysis:
                context['current_analysis'] = st.session_state.current_analysis

            # Add conversation history for ReAct memory system
            if st.session_state.messages:
                context['chat_history'] = st.session_state.messages[-6:]  # Last 6 messages for context

            response = ChatHandler.get_response(prompt, context)
            st.markdown(response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now()
            })

with col2:
    st.header("📤 Image Analysis")

    uploaded_file = st.file_uploader(
        "Upload Medical Scan",
        type=Config.ALLOWED_EXTENSIONS,
        help="Upload a NIfTI file (.nii or .nii.gz), DICOM file (.dcm or .dicom), or TCIA file (.tcia)"
    )

    if uploaded_file is not None:
        # Display file info
        file_ext = uploaded_file.name.split('.')[-1].lower()
        if file_ext in ['dcm', 'dicom', 'tcia']:
            file_type = "DICOM" if file_ext != 'tcia' else "DICOM (TCIA)"
        else:
            file_type = "NIfTI"
        
        st.success(f"✅ {file_type} file uploaded: {uploaded_file.name}")

        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            # Validate file
            is_valid, message = DataProcessor.validate_file(uploaded_file)

            if not is_valid:
                st.error(message)
            else:
                with st.spinner("Analyzing image..."):
                    try:
                        # Call the FastAPI prediction endpoint
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}
                        response = requests.post(f"{Config.API_BASE_URL}/predict", files=files)

                        if response.status_code == 200:
                            result = response.json()

                            # Store current analysis for context-aware chat
                            st.session_state.current_analysis = {
                                'classification': result["classification"],
                                'confidence': result["confidence"],
                                'probability_malignant': result["probability_malignant"],
                                'probability_benign': result["probability_benign"],
                                'size': result["features"]["nodule_size"],
                                'location': result["features"]["location"]
                            }

                            # Display results
                            UIComponents.render_classification_result(
                                result["classification"],
                                f"{result['confidence']:.1%}"
                            )

                            # Generate detailed report
                            report = ChatHandler.generate_classification_report(result, result["features"])

                            # Add to chat
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": report,
                                "timestamp": datetime.now()
                            })

                            st.success("✅ Analysis complete! Check the chat for detailed report.")
                            st.rerun()

                        else:
                            error_detail = response.json().get("detail", "Unknown error")
                            st.error(f"❌ API Error: {error_detail}")

                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Connection Error: Could not connect to API server. Please ensure the API is running on {Config.API_BASE_URL}")
                    except Exception as e:
                        st.error(f"❌ Unexpected Error: {str(e)}")
    
    st.divider()
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("ℹ️ What is a benign nodule?", use_container_width=True):
            context = {}
            if st.session_state.current_analysis:
                context['current_analysis'] = st.session_state.current_analysis
            response = ChatHandler.get_response("what is benign", context)
            st.session_state.messages.append({
                "role": "user",
                "content": "What is a benign nodule?",
                "timestamp": datetime.now()
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now()
            })
            st.rerun()
        
        if st.button("⚠️ What is a malignant nodule?", use_container_width=True):
            context = {}
            if st.session_state.current_analysis:
                context['current_analysis'] = st.session_state.current_analysis
            response = ChatHandler.get_response("what is malignant", context)
            st.session_state.messages.append({
                "role": "user",
                "content": "What is a malignant nodule?",
                "timestamp": datetime.now()
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now()
            })
            st.rerun()
    
    with col2:
        if st.button("🧹 Clear Analysis Context", use_container_width=True):
            st.session_state.current_analysis = None
            st.success("Analysis context cleared. Chat will now give general responses.")
        
        if st.session_state.current_analysis:
            st.info(f"💡 Current analysis: {st.session_state.current_analysis['classification']} ({st.session_state.current_analysis['confidence']:.1%})")

# Footer
st.divider()
st.caption("⚠️ **Medical Disclaimer:** This AI tool is for educational and screening purposes only. Always consult with qualified healthcare professionals for medical diagnosis and treatment decisions.")
