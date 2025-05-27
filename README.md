# AI Skin Monitoring Tool

This application uses GPT-4 Vision API to analyze skin images and provide risk assessments for skin lesions. It generates PDF reports with the analysis results.

## Features
- Upload skin images for AI analysis
- Receive risk assessment (Low, Medium, or High risk)
- Get detailed explanations of the assessment
- Download PDF reports with analysis results

## Setup Instructions

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Run the application:
   ```bash
   python app.py
   ```

## Important Notes
- This tool is not for medical diagnosis
- Always consult a healthcare professional for medical concerns
- The application requires an OpenAI API key with access to GPT-4 Vision API
- Generated reports are saved in the `reports` directory

## Security
- Never commit your API key to version control
- Keep your `.env` file secure and local
- The application stores uploaded images temporarily for report generation 