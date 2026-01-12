import os
import requests
from config import Config

class ChatHandler:
    """Handle chat interactions and responses using open-source LLM"""

    @staticmethod
    def get_response(user_input, context=None):
        """
        Generate intelligent response using Hugging Face's open-source LLM

        Args:
            user_input: User's question/message
            context: Dictionary containing current analysis context

        Returns:
            AI-generated response string
        """
        try:
            # Build context string for the LLM
            context_str = ""
            if context and 'current_analysis' in context:
                analysis = context['current_analysis']
                context_str = f"""
Current Nodule Analysis:
- Classification: {analysis.get('classification', 'Unknown')}
- Confidence: {analysis.get('confidence', 'Unknown')}
- Size: {analysis.get('size', 'Unknown')}
- Location: {analysis.get('location', 'Unknown')}
- Probability Malignant: {analysis.get('probability_malignant', 'Unknown')}
- Probability Benign: {analysis.get('probability_benign', 'Unknown')}
"""

            # Create ReAct-style medical assistant prompt
            system_prompt = f"""You are an expert Medical AI Assistant specializing in pulmonary medicine and lung nodule analysis. You use a structured reasoning approach to provide accurate, helpful responses.

{context_str}

## CORE CAPABILITIES
- Analyze lung nodule characteristics and classifications
- Explain medical imaging concepts (HU values, nodule morphology, etc.)
- Provide evidence-based information about lung cancer risk factors
- Guide patients through understanding their AI analysis results

## ReAct FRAMEWORK - ALWAYS FOLLOW THIS STRUCTURE:

### 1. OBSERVE & ANALYZE
- Examine the user's question and current context
- Identify key medical concepts or terminology
- Consider the current nodule analysis if available

### 2. REASON
- Connect the question to established medical knowledge
- Consider clinical implications and patient concerns
- Evaluate what information is most relevant and helpful

### 3. ACT
- Provide clear, accurate medical information
- Use simple language while maintaining medical accuracy
- Include relevant clinical context and limitations
- Always recommend professional medical consultation

### 4. REMEMBER
- Reference previous conversation context when relevant
- Build upon prior explanations
- Maintain continuity in medical discussions

## MEDICAL KNOWLEDGE BASE

### HU VALUES & NORMALIZATION
HU (Hounsfield Units) measure tissue density in CT scans:
- Air: -1000 HU
- Lung tissue: -700 to -600 HU
- Water: 0 HU
- Blood/soft tissue: +40 to +60 HU
- Bone: +1000 HU

Normalization standardizes these values for consistent AI analysis.

### NODULE CHARACTERISTICS
**Benign Features:**
- Smooth, well-defined borders
- Regular shape (round/oval)
- Homogeneous density
- Slow growth rate
- Calcifications present

**Malignant Features:**
- Spiculated/irregular borders
- Heterogeneous density
- Rapid growth
- Pleural retraction
- Vascular involvement

### CLINICAL CONTEXT
- Most lung nodules (>60%) are benign
- Size >8mm increases malignancy concern
- Smoking history significantly elevates risk
- Family history of lung cancer is a risk factor

## RESPONSE GUIDELINES
- Be medically accurate but accessible
- Always include appropriate disclaimers
- Encourage professional medical consultation
- Reference current analysis when relevant
- Use evidence-based information
- Maintain patient-centered communication

## MEMORY SYSTEM
Track conversation context and build upon previous interactions to provide coherent, contextual responses."""

            # Build conversation memory
            conversation_history = ""
            if context and 'chat_history' in context:
                # Include recent conversation for context
                recent_messages = context['chat_history'][-4:]  # Last 4 exchanges
                for msg in recent_messages:
                    if msg['role'] == 'user':
                        conversation_history += f"User: {msg['content']}\n"
                    elif msg['role'] == 'assistant':
                        conversation_history += f"Assistant: {msg['content']}\n"

            # Create ReAct-structured prompt
            react_prompt = f"""{system_prompt}

## CURRENT CONVERSATION CONTEXT
{conversation_history}

## USER QUESTION: {user_input}

## ReAct ANALYSIS:

**OBSERVE:** What is the user asking about? What medical concepts are involved?

**REASON:** What medical knowledge applies here? How does this relate to lung nodule analysis?

**ACT:** Provide a helpful, accurate response using the ReAct framework above.

Assistant:"""

            # Use Hugging Face Inference API with ReAct prompt (updated endpoint)
            api_url = f"https://router.huggingface.co/hf-inference/models/{Config.HUGGINGFACE_MODEL}"

            headers = {}
            if Config.HUGGINGFACE_API_KEY:
                headers["Authorization"] = f"Bearer {Config.HUGGINGFACE_API_KEY}"

            payload = {
                "inputs": react_prompt,
                "parameters": {
                    "max_length": 400,  # Longer for detailed medical responses
                    "temperature": 0.3,  # Lower temperature for medical accuracy
                    "do_sample": True,
                    "pad_token_id": 50256,
                    "repetition_penalty": 1.2  # Reduce repetition
                },
                "options": {
                    "wait_for_model": True
                }
            }

            response = requests.post(api_url, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '')

                    # Extract the assistant's response from ReAct format
                    if 'Assistant:' in generated_text:
                        assistant_response = generated_text.split('Assistant:')[-1].strip()
                    else:
                        assistant_response = generated_text.strip()

                    # Clean up ReAct artifacts and format the response
                    if assistant_response:
                        # Remove ReAct framework markers if present
                        lines = assistant_response.split('\n')
                        clean_lines = []
                        for line in lines:
                            # Skip ReAct framework lines
                            if any(marker in line.upper() for marker in ['**OBSERVE:**', '**REASON:**', '**ACT:**', '**REMEMBER:**']):
                                continue
                            # Skip empty lines at the beginning
                            if not clean_lines and not line.strip():
                                continue
                            clean_lines.append(line)

                        assistant_response = '\n'.join(clean_lines).strip()

                        # Remove any trailing ReAct markers
                        if '**' in assistant_response:
                            # Find the last complete sentence before ReAct markers
                            sentences = assistant_response.split('.')
                            clean_sentences = []
                            for sentence in sentences:
                                if '**' in sentence:
                                    break
                                clean_sentences.append(sentence)
                            if clean_sentences:
                                assistant_response = '.'.join(clean_sentences).strip()
                                if not assistant_response.endswith('.'):
                                    assistant_response += '.'

                        # Ensure proper formatting
                        if assistant_response:
                            return assistant_response
                        else:
                            return ChatHandler._get_medical_fallback_response(user_input, context)
                    else:
                        return ChatHandler._get_medical_fallback_response(user_input, context)
                else:
                    return ChatHandler._get_medical_fallback_response(user_input, context)
            else:
                print(f"Hugging Face API error: {response.status_code} - {response.text}")
                return ChatHandler._get_medical_fallback_response(user_input, context)

        except Exception as e:
            print(f"Error calling Hugging Face API: {e}")
            return ChatHandler._get_fallback_response(user_input, context)

    @staticmethod
    def _get_medical_fallback_response(user_input, context=None):
        """
        Provide intelligent medical fallback responses when LLM is unavailable
        """
        lower_input = user_input.lower()

        # Context-aware responses based on current analysis
        if context and 'current_analysis' in context:
            analysis = context['current_analysis']
            classification = analysis.get('classification', '').lower()

            if "size" in lower_input or "mm" in lower_input:
                size = analysis.get('size', '64x64x64')
                return f"Based on the analysis, your nodule measures {size}. Nodules larger than 8mm generally require closer monitoring, while those over 30mm are of higher concern. Please discuss these measurements with your healthcare provider."

            elif "malignant" in lower_input or "cancer" in lower_input:
                if classification == "malignant":
                    return f"The AI classified your nodule as malignant. This indicates concerning features, but this is a screening result only. Further testing including biopsy and staging scans will be needed for definitive diagnosis."
                else:
                    return f"The AI classified your nodule as benign. While reassuring, all AI results should be correlated with clinical findings and additional testing."

            elif "benign" in lower_input:
                if classification == "benign":
                    return f"The AI classified your nodule as benign, suggesting non-cancerous characteristics. Most lung nodules are benign, but follow-up imaging is still typically recommended."
                else:
                    return f"The AI classified your nodule as malignant. Please seek prompt evaluation by a healthcare professional."

        # Intelligent medical responses for common questions
        if "hu" in lower_input and ("value" in lower_input or "normalization" in lower_input):
            return """HU (Hounsfield Units) measure tissue density in CT scans. HU value normalization is crucial because:

• It standardizes tissue densities across different CT scanners
• Allows consistent AI analysis regardless of imaging equipment
• Converts absolute HU values to relative scales for better pattern recognition
• Accounts for variations in X-ray energy and tissue composition

This ensures your AI analysis is accurate and reliable."""

        elif "border" in lower_input and ("smooth" in lower_input or "well-defined" in lower_input):
            return """Smooth, well-defined borders on a lung nodule are generally a positive sign because:

• They suggest the nodule is likely benign (non-cancerous)
• Benign nodules typically have regular, rounded shapes
• Irregular or spiculated borders are more concerning for malignancy
• However, border characteristics alone don't determine diagnosis

Always correlate with other clinical factors and get professional medical evaluation."""

        elif "calcification" in lower_input:
            return """Calcifications in lung nodules can indicate benign causes:

• Often result from old infections (like tuberculosis or fungal infections)
• Can form in scar tissue or hamartomas
• Generally reassuring when diffuse throughout the nodule
• May require follow-up to ensure stability

However, certain calcification patterns can still be concerning."""

        elif "spiculation" in lower_input or "spiky" in lower_input:
            return """Spiculation refers to spiky, irregular borders extending from a lung nodule. This is concerning because:

• Often associated with malignant (cancerous) nodules
• Represents tumor infiltration into surrounding tissue
• Increases the likelihood of malignancy significantly
• Requires prompt evaluation and possible biopsy

However, not all spiculated nodules are malignant."""

        elif "ground glass" in lower_input:
            return """Ground-glass opacity nodules appear hazy on CT scans. These can be:

• **Benign causes**: Inflammation, infection, or hemorrhage
• **Malignant potential**: Some lung cancers present this way
• **Pre-invasive lesions**: May represent early-stage disease

Ground-glass nodules often require closer monitoring and may need biopsy if they persist or grow."""

        elif "vascular" in lower_input:
            return """Vascular involvement means the nodule interacts with blood vessels. This can indicate:

• **Concerning signs**: Tumor growth into vessels (malignancy)
• **Normal findings**: Vessels simply passing near the nodule
• **Evaluation needed**: Requires radiologist interpretation

CT angiography may be needed to assess vascular relationships."""

        elif "grow" in lower_input and ("rate" in lower_input or "time" in lower_input):
            return """Nodule growth rate is a crucial factor in assessing malignancy risk:

• **Benign nodules**: Usually grow slowly or remain stable
• **Malignant nodules**: Often grow faster (doubling time <400 days)
• **Monitoring**: Serial CT scans measure growth over time
• **Volume doubling**: More accurate than diameter measurements

Sudden growth often prompts biopsy or intervention."""

        elif "smoking" in lower_input or "smoke" in lower_input:
            return """Smoking history significantly impacts lung nodule evaluation:

• **Major risk factor**: 80-90% of lung cancers occur in smokers
• **Risk stratification**: Influences follow-up recommendations
• **Screening guidelines**: Different for smokers vs. non-smokers
• **Cessation counseling**: Critical for all patients with nodules

Even former smokers remain at elevated risk."""

        elif "family" in lower_input and "history" in lower_input:
            return """Family history of lung cancer increases nodule concern:

• **Genetic factors**: May predispose to lung cancer development
• **Higher suspicion**: Lowers threshold for biopsy
• **Screening intensity**: May warrant more aggressive evaluation
• **Genetic counseling**: Consider if strong family history present

Discuss your family history with your healthcare provider."""

        elif "biopsy" in lower_input:
            return """Lung nodule biopsy may be recommended when:

• Nodule is highly suspicious for cancer
• Growth is observed over time
• Patient is a good surgical candidate
• Results will change management

Types include: CT-guided needle biopsy, bronchoscopy, or surgical biopsy."""

        elif "follow" in lower_input and "up" in lower_input:
            return """Follow-up recommendations depend on nodule characteristics:

• **Low risk**: Repeat CT in 6-24 months
• **Intermediate risk**: Shorter interval (3-6 months)
• **High risk**: Immediate further evaluation
• **Stable nodules**: May eventually discontinue surveillance

Guidelines from Fleischner Society help determine appropriate intervals."""

        else:
            return """I'm here to help you understand your lung nodule analysis and related medical questions! You can ask me about:

• **Current Analysis**: Questions about your specific nodule results
• **Medical Concepts**: HU values, nodule characteristics, imaging terms
• **Risk Factors**: Smoking, family history, growth patterns
• **Clinical Guidance**: Biopsy, follow-up, treatment considerations

**Important**: I'm an educational tool only. All medical decisions should be made with qualified healthcare professionals. What specific question can I help answer?"""

    @staticmethod
    def generate_classification_report(prediction, features):
        """Generate detailed classification report"""
        classification = prediction["classification"]
        confidence = prediction["confidence"]

        # Handle both decimal (0.85) and percentage ("85%") formats
        if isinstance(confidence, str) and confidence.endswith('%'):
            confidence_display = confidence
            confidence_value = float(confidence.rstrip('%')) / 100
        else:
            confidence_display = f"{confidence:.1%}"
            confidence_value = float(confidence)

        report = f"""### 🎯 Classification Result: **{classification}**
**Confidence:** {confidence_display}%

---

### 📍 Nodule Details
- **Location:** {features['location']}
- **Size:** {features['nodule_size']} mm

---

### 🔍 Key Characteristics
"""
        for char in features['characteristics']:
            report += f"- {char}\n"

        report += "\n---\n\n### 📊 Analysis\n\n"

        if classification == "Benign":
            report += """The nodule shows features typically associated with **benign (non-cancerous) growths**. Benign nodules are usually caused by old infections, inflammation, or scar tissue. They typically have smooth, well-defined borders and uniform appearance.

#### Risk Assessment:
"""
            for factor in features['risk_factors']:
                report += f"- {factor}\n"

            report += """\n#### ✅ Recommendation
While this appears benign, I recommend follow-up imaging in 6-12 months to monitor for any changes. Always consult with your doctor for a complete evaluation."""
        else:
            report += """The nodule shows characteristics that are **concerning** and may indicate malignancy (cancer). Malignant nodules often have irregular, spiky borders (spiculation), uneven density, and may grow over time.

#### ⚠️ Concerning Features:
"""
            for factor in features['risk_factors']:
                report += f"- {factor}\n"

            report += """\n#### 🏥 Important Next Steps
This is a screening tool only. A malignant classification means further testing is needed:

- **PET/CT scan** for metabolic activity
- **Tissue biopsy** for definitive diagnosis
- **Consultation with an oncologist**

**Please contact your healthcare provider immediately** to discuss these findings."""

        return report

    @staticmethod
    def generate_classification_report(prediction, features):
        """Generate detailed classification report"""
        classification = prediction["classification"]
        confidence = prediction["confidence"]
        
        # Handle both decimal (0.85) and percentage ("85%") formats
        if isinstance(confidence, str) and confidence.endswith('%'):
            confidence_display = confidence
            confidence_value = float(confidence.rstrip('%')) / 100
        else:
            confidence_display = f"{confidence:.1%}"
            confidence_value = float(confidence)
        
        report = f"""### 🎯 Classification Result: **{classification}**
**Confidence:** {confidence_display}%

---

### 📍 Nodule Details
- **Location:** {features['location']}
- **Size:** {features['nodule_size']} mm

---

### 🔍 Key Characteristics
"""
        for char in features['characteristics']:
            report += f"- {char}\n"
        
        report += "\n---\n\n### 📊 Analysis\n\n"
        
        if classification == "Benign":
            report += """The nodule shows features typically associated with **benign (non-cancerous)** growths. Benign nodules are usually caused by old infections, inflammation, or scar tissue. They typically have smooth, well-defined borders and uniform appearance.

#### Risk Assessment:
"""
            for factor in features['risk_factors']:
                report += f"- {factor}\n"
            
            report += """\n#### ✅ Recommendation
While this appears benign, I recommend follow-up imaging in 6-12 months to monitor for any changes. Always consult with your doctor for a complete evaluation."""
        
        else:
            report += """The nodule shows characteristics that are **concerning** and may indicate malignancy (cancer). Malignant nodules often have irregular, spiky borders (spiculation), uneven density, and may grow over time.

#### ⚠️ Concerning Features:
"""
            for factor in features['risk_factors']:
                report += f"- {factor}\n"
            
            report += """\n#### 🏥 Important Next Steps
This is a screening tool only. A malignant classification means further testing is needed:

- **PET/CT scan** for metabolic activity
- **Tissue biopsy** for definitive diagnosis
- **Consultation with an oncologist**

**Please contact your healthcare provider immediately** to discuss these findings."""
        
        return report