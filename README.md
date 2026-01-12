# Lung Nodule AI Assistant

A comprehensive AI-powered tool for lung nodule classification using deep learning, featuring both a FastAPI backend and Streamlit frontend with intelligent ReAct-based medical chat assistance.

## 🏗️ Architecture

- **Backend**: FastAPI server serving TensorFlow model predictions
- **Frontend**: Streamlit web application for user interaction
- **AI Chat**: ReAct framework-powered medical assistant using open-source LLM
- **Model**: ResNet-based classifier trained on LUNA16 dataset
- **Data Format**: NIfTI (.nii/.nii.gz), DICOM (.dcm/.dicom), and TCIA (.tcia) medical imaging files
- **Memory System**: Conversation history tracking for contextual responses

## 📋 Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- NIfTI (.nii/.nii.gz), DICOM (.dcm/.dicom), or TCIA (.tcia) format lung CT scans
- Hugging Face API key (optional, free tier available)

## 🚀 Quick Start

### Option 1: One-Click Run (Recommended)
```bash
# Make sure you're in the project directory
cd /path/to/lung-nodule-ai-assistant

# Run everything with one command
./run.sh
```
This will automatically start both the FastAPI backend and Streamlit frontend.

**What `run.sh` does:**
- ✅ **Port Management**: Automatically kills existing processes on ports 8000/8501
- ✅ **Environment Setup**: Activates virtual environment if available
- ✅ **Service Startup**: Starts FastAPI backend and Streamlit frontend
- ✅ **Health Checks**: Verifies services are running properly
- ✅ **Status Display**: Shows access URLs and service status
- ✅ **Graceful Shutdown**: Ctrl+C stops both services cleanly

### Option 2: Manual Setup

### Option 2: Manual Setup
#### 1. Setup Environment

```bash
# Clone or navigate to project directory
cd /path/to/lung-nodule-assistant

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
# Start FastAPI backend
python run_api.py
```

The API will be available at: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### 3. Start the Streamlit App

Open a new terminal and run:

```bash
# Start Streamlit frontend
streamlit run app.py
```

The app will be available at: http://localhost:8501

## ✨ Features

### 🫁 **AI-Powered Classification**
- **Deep Learning Model**: ResNet-based classifier trained on LUNA16 dataset
- **Binary Classification**: Benign vs Malignant nodule detection
- **Confidence Scoring**: Detailed probability assessments
- **Mock Predictions**: Fallback system when model loading fails

### 🖼️ **Medical Imaging Support**
- **NIfTI Format**: Standard .nii and .nii.gz files
- **DICOM Support**: Direct .dcm and .dicom file processing
- **TCIA Support**: .tcia files from The Cancer Imaging Archive (must be actual DICOM images, not manifest files)
- **Auto-Conversion**: DICOM and TCIA files automatically converted to NIfTI
- **File Validation**: Size and format checking

### 🤖 **Intelligent Chat Assistant**
- **ReAct Framework**: Reasoning + Acting prompting system
- **Medical Knowledge Base**: Comprehensive pulmonary medicine information
- **Conversation Memory**: Tracks chat history for contextual responses
- **Fallback Responses**: Intelligent answers when LLM unavailable
- **Open-Source LLM**: Hugging Face integration with DialoGPT

### 📊 **Clinical Analysis**
- **Detailed Reports**: Comprehensive classification results
- **Risk Assessment**: Clinical risk factor analysis
- **Next Steps**: Evidence-based follow-up recommendations
- **Medical Disclaimer**: Appropriate professional consultation guidance

### 🔧 **Technical Features**
- **RESTful API**: FastAPI backend with automatic documentation
- **Web Interface**: Modern Streamlit UI with real-time updates
- **Port Management**: Automatic conflict resolution
- **Environment Handling**: Virtual environment support
- **Error Recovery**: Graceful handling of API failures

## 📁 Project Structure

```
lung-nodule-assistant/
├── api.py                 # FastAPI backend application
├── app.py                 # Streamlit frontend application
├── run_api.py            # API server startup script
├── config.py             # Application configuration
├── requirements.txt      # Python dependencies
├── models/               # Trained models and parameters
│   ├── nodule_classifier_resnet_attention.h5
│   ├── preprocessing_params.pkl
│   └── model_metrics.pkl
├── utils/                # Utility modules
│   └── data_processor.py # NIfTI file processing
└── components/           # UI components
    ├── chat_handler.py   # Chat response logic
    └── ui_components.py  # Streamlit UI components
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:

```bash
# Hugging Face API Key (optional - free tier available)
HUGGINGFACE_API_KEY=hf_your_api_key_here
```

### Config File
Edit `config.py` to customize:

- Model paths and settings
- API endpoints and ports
- File upload size limits
- UI appearance and themes
- Chat system parameters

### Getting Hugging Face API Key
1. Visit https://huggingface.co/
2. Create account or sign in
3. Go to Settings → Access Tokens
4. Create new token with "Read" role
5. Add to `.env` file as shown above

**Note**: API key is optional. Free tier works for basic usage.

## 📊 Model Details

- **Architecture**: ResNet with attention mechanism
- **Training Data**: LUNA16 dataset (888 CT scans, 1186 nodules)
- **Input**: 3D NIfTI volumes (64x64x64 voxels)
- **Output**: Binary classification (Benign/Malignant)
- **Performance**: ~92% accuracy on validation set

## 🖥️ API Endpoints

### GET /health
Check API and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "models/nodule_classifier_resnet_attention.h5"
}
```

### POST /predict
Upload NIfTI, DICOM, or TCIA file for classification.

**Request:** Multipart form with `file` field containing .nii/.nii.gz, .dcm, .dicom, or .tcia file

**Response:**
```json
{
  "classification": "Benign",
  "confidence": 0.87,
  "probability_malignant": 0.13,
  "probability_benign": 0.87,
  "features": {
    "nodule_size": "64x64x64",
    "location": "Center (assuming nodule is centered)",
    "characteristics": ["3D volumetric analysis", "HU value normalization"],
    "risk_factors": ["Size and shape analysis", "Density patterns"]
  }
}
```

## 🔒 Security & Privacy

- Files are processed temporarily and not stored
- No patient data is retained
- Model runs locally (no external API calls)
- CORS enabled for local development

## ⚠️ Medical Disclaimer

**This is a screening tool only, not for clinical diagnosis.**

- Always consult qualified healthcare professionals
- AI predictions should be validated by radiologists
- False positives and negatives can occur
- Regular medical follow-up is essential

## 🐛 Troubleshooting

### Port Conflicts
```bash
# Kill processes on specific ports
lsof -ti:8000 | xargs kill -9  # Kill API process
lsof -ti:8501 | xargs kill -9  # Kill Streamlit process

# Or use the run.sh script which handles this automatically
./run.sh
```

### Model Loading Issues
- **Layer Count Mismatch**: The app uses mock predictions as fallback
- **Missing Model Files**: Ensure `models/nodule_classifier_resnet_attention.h5` exists
- **TensorFlow Version**: Compatible with TensorFlow 2.13+

### Chat System Issues
- **Hugging Face API**: Free tier works without API key
- **Network Issues**: Fallback to intelligent medical responses
- **Memory System**: Conversation history tracked automatically

### File Upload Problems
- **DICOM Support**: .dcm and .dicom files accepted
- **TCIA Support**: .tcia files accepted (must be actual DICOM images from Cancer Imaging Archive, not manifest files)
- **NIfTI Support**: .nii and .nii.gz files accepted
- **Size Limits**: Maximum 100MB per file
- **Format Errors**: Check file integrity

### Virtual Environment Issues
```bash
# Recreate virtual environment
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Import Errors
```bash
# Install missing packages
pip install pydicom nibabel python-dotenv requests

# For Mac with M1/M2 chips
pip install tensorflow-macos
```

### Performance Issues
- **Large Files**: Processing may take time for large CT scans
- **Memory Usage**: Ensure sufficient RAM (8GB+ recommended)
- **API Timeouts**: Chat responses may take 5-10 seconds

## 📞 Support

For technical issues:
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Ensure ports 8000 and 8501 are available
4. Try running `./run.sh` for automatic setup

For medical questions, consult qualified healthcare professionals.

## 📝 Recent Updates

### v1.0.0 - Complete AI Assistant
- ✅ **ReAct Chat System**: Intelligent medical conversations with reasoning framework
- ✅ **DICOM Support**: Direct processing of .dcm/.dicom files
- ✅ **Conversation Memory**: Chat history tracking for contextual responses
- ✅ **Port Management**: Automatic conflict resolution in run.sh
- ✅ **Enhanced Fallbacks**: Intelligent medical responses when LLM unavailable
- ✅ **Updated Dependencies**: Clean requirements.txt with proper versioning
- ✅ **Comprehensive Documentation**: Detailed README with troubleshooting

### Key Features Added:
- **Medical Knowledge Base**: Built-in pulmonary medicine expertise
- **Auto-Conversion**: DICOM to NIfTI processing pipeline
- **Error Recovery**: Graceful handling of API failures and model issues
- **One-Click Setup**: `./run.sh` for complete system startup
- **Professional UI**: Medical-grade interface with disclaimers

---

**Built with ❤️ for advancing AI-assisted medical diagnosis while maintaining clinical safety standards.**
- Verify preprocessing parameters file exists

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper testing
4. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please consult appropriate licenses for medical AI applications.

## 📞 Support

For technical issues or questions:
- Check the troubleshooting section
- Review API documentation at `/docs`
- Ensure all prerequisites are met