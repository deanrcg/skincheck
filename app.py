import gradio as gr
from PIL import Image
from openai import OpenAI
from fpdf import FPDF
import os
import datetime
import base64
import io
from dotenv import load_dotenv
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

def assess_skin_image(image):
    try:
        logger.info("Starting image assessment...")
        logger.info(f"Initial image type: {type(image)}")
        
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            logger.info(f"Converting numpy array of shape {image.shape}")
            image = Image.fromarray(image)
            logger.info("Successfully converted to PIL Image")
        else:
            logger.info(f"Image is already PIL Image: {type(image)}")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            logger.info(f"Converting from {image.mode} to RGB")
            image = image.convert('RGB')
            logger.info("Successfully converted to RGB")
        
        # Resize if too large
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            logger.info(f"Resizing from {image.size} to {new_size}")
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.info("Successfully resized image")
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        img_bytes = buffered.getvalue()
        logger.info(f"Image saved to bytes, size: {len(img_bytes)} bytes")
        
        img_str = base64.b64encode(img_bytes).decode()
        logger.info(f"Image converted to base64, length: {len(img_str)}")

        # API request
        messages = [
            {
                "role": "system",
                "content": "You are a helpful medical assistant. This tool is not for diagnosis but helps users understand if a mole or skin lesion may need medical attention. Focus on the ABCDE criteria (Asymmetry, Border irregularity, Color variation, Diameter >6mm, Evolution) in your assessment."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please review this image and assess whether the mole or lesion shows any risk signs. Consider the ABCDE criteria. Categorize it as low, medium, or high risk, and explain why."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]

        logger.info("Preparing to send request to OpenAI API...")
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=500
            )
            logger.info("Successfully received API response")
            logger.info(f"Response type: {type(response)}")
            logger.info(f"Response content: {response.choices[0].message.content[:100]}...")
            
            reply = response.choices[0].message.content
            
            # Determine risk level
            risk_level = "Medium Risk - Monitor"  # Default
            reply_lower = reply.lower()
            if "low" in reply_lower and "high" not in reply_lower:
                risk_level = "Low Risk - Likely Benign"
            elif "high" in reply_lower:
                risk_level = "High Risk - Seek Medical Advice"

            logger.info(f"Determined risk level: {risk_level}")
            report_path = generate_pdf_report(image, risk_level, reply)
            logger.info(f"Generated PDF report at: {report_path}")
            
            return risk_level, reply, report_path

        except Exception as api_error:
            logger.error(f"API Error: {str(api_error)}", exc_info=True)
            raise

    except Exception as e:
        error_message = f"An error occurred during analysis: {str(e)}"
        logger.error(f"Error in assess_skin_image: {str(e)}", exc_info=True)
        return "Error", error_message, None

def generate_pdf_report(image, risk_level, explanation):
    try:
        now = datetime.datetime.now()
        filename = f"Skin_Report_{now.strftime('%Y%m%d_%H%M%S')}.pdf"
        folder = "reports"
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)

        # Create PDF with UTF-8 support
        pdf = FPDF()
        pdf.add_page()
        
        # Add title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="AI Skin Monitoring Report", ln=True, align='C')
        
        # Add content with UTF-8 encoding
        pdf.set_font("Arial", size=12)
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Date: {now.strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(5)
        
        # Risk level
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt=f"Risk Level: {risk_level}", ln=True)
        pdf.ln(5)
        
        # Explanation - handle Unicode characters
        pdf.set_font("Arial", size=12)
        # Split explanation into lines to handle long text
        lines = explanation.split('\n')
        for line in lines:
            # Replace any problematic characters
            safe_line = line.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 10, txt=safe_line)
        pdf.ln(10)

        # Save and add image
        temp_img_path = os.path.join(folder, "temp_image.jpg")
        image.save(temp_img_path)
        pdf.image(temp_img_path, x=10, w=100)
        
        # Add disclaimer
        pdf.ln(10)
        pdf.set_font("Arial", 'I', 10)
        disclaimer = "Disclaimer: This report is generated by AI and is not a medical diagnosis. Please consult a healthcare professional for medical advice."
        pdf.multi_cell(0, 10, txt=disclaimer)
        
        pdf.output(filepath)
        
        # Clean up temporary image
        os.remove(temp_img_path)
        
        return filepath

    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}", exc_info=True)
        return None

# Gradio interface
iface = gr.Interface(
    fn=assess_skin_image,
    inputs=gr.Image(type="numpy", label="Upload a photo of your skin lesion"),
    outputs=[
        gr.Text(label="Risk Assessment"),
        gr.Text(label="Explanation"),
        gr.File(label="Download PDF Report")
    ],
    title="DeanAI SkinCheck with GPT-4o",
    description="Upload a skin photo to receive an AI-generated risk assessment and PDF report. Not a diagnosis — consult a healthcare professional for medical concerns.",
    examples=[],
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(share=True) 