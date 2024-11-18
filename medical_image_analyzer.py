from PIL import Image
from base64 import b64encode
import google.generativeai as genai
import re
import cv2
import requests
from io import BytesIO

def encode_image(image):
    """
    Encode image bytes to base64 string.
    """
    return b64encode(image).decode('utf-8')

def extract_text_from_image(image, api_key="AIzaSyCQrYGVRTNivr4Dh_xhJLkVovy6kDEFhKY"):
    """
    Extract text from an image using Gemini 1.5 Flash model.
    """
    # Configure the Gemini API
    genai.configure(api_key=api_key)

    # Encode the image
    encoded_image = encode_image(image)
    
    # Create model instance
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Prepare the prompt
    prompt = """
    Analyze this image and provide a detailed report including:
                    1. Type of scan/image
                    2. Key findings and observations
                    3. Notable anatomical structures
                    4. Any abnormalities or pathological findings
                    5. Potential diagnoses based on imaging features
                    6. Areas requiring attention or follow-up
                    7. Technical quality of the image
                    8. Recommendations for additional imaging if needed.

                    all these must be atmost 4 sentences and keep good formating in output.

                    Then provide a summary using medical terms for disease and injuries name in 2-3 sentences.
    """
    
    # Create message parts
    message = [
        {
            "role": "user",
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": encoded_image
                    }
                }
            ]
        }
    ]
    
    # Generate response
    response = model.generate_content(message)
    return response.text

def clean_text(raw_text):
    """
    Clean the extracted text to remove unnecessary characters, format it, and print it clearly.
    """
    formatted_text = re.sub(r"(\d+\.\s)", r"\n\1", raw_text)

    # Remove unnecessary spaces or blank lines
    formatted_text = "\n".join(line.strip() for line in formatted_text.split("\n") if line.strip())

    # Print the formatted text clearly
    print("Formatted Text:\n")
    print(formatted_text)
    return formatted_text

def process_image_from_path(image_path):
    """
    Processes an image from the given path.
    - Resizes the image to 256x256.
    - Converts it to grayscale.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        processed_image: Processed image in grayscale.
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found or path is incorrect.")
        
        # Resize the image to 256x256
        resized_image = cv2.resize(image, (256, 256))
        
        # Convert the image to grayscale
        processed_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        
        return processed_image
    except Exception as e:
        print(f"Error processing the image: {e}")
        return None


def show_image_from_url(image_url):
    """
    Fetches an image from the web using the provided URL and displays it.
    
    Args:
        image_url (str): The URL of the image.
    
    Returns:
        None
    """
    try:
        # Send a GET request to fetch the image
        response = requests.get(image_url)
        
        # Ensure the request was successful
        if response.status_code == 200:
            # Open the image from the content of the response
            image = Image.open(BytesIO(response.content))
            
            # Show the image
            return image
        else:
            print(f"Error: Unable to fetch image. HTTP status code {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")
