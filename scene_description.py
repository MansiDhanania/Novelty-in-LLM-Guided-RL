# Import Necessary Libraries
import base64
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
def scene_description(new_image):
        image_data = encode_image(new_image)
             # Original Prompt
            #  "Describe the scene in front of me."
             # Improved Prompt
             # "You are helping a blind grocery shopper understand their environment. "
             #        "When asked to describe the scene, do not mention irrelevant things like lighting, floor, or ceiling. "
             #        "Instead, describe the grocery layout in terms of aisles, shelves, and products: "
             #        "- Say what is on the left, right, or in front of the user (e.g., 'The aisle to your right has cereal boxes'). "
             #        "- Estimate approximate distance in steps or meters ('the aisle is about 3 steps away or 3 meters away'). "
             #        "- Mention whether there are obstructions or people nearby. "
             #        "- If it’s an open shelf/section, describe its category (e.g., bakery, produce, frozen goods). "
             #        "- Keep the explanation simple, concise, and actionable for someone who cannot see. "
             #        "Do not add extra visual descriptions that are not useful for navigation or shopping."
        # Improved Prompt
        new_query = (
        "You are helping a blind grocery shopper understand their environment."
        "\n\nYour job is to describe the user’s immediate surroundings using only information that is clearly visible in the image."
        "Do not assume the user is in a grocery aisle unless there is clear evidence for it."
        "\n\nGuidelines:\n- Only describe aisles, shelves, and products or store areas if they are definitely visible."
        "\n- If the image does not show a grocery area, describe what is present using generic but helpful terms"
        "(e.g., 'There are shelves to your left with unknown items').\n- State the approximate position: left, right, front, behind."
        "\n- Estimate distance in steps or meters (e.g., 'The shelf is about 3 meters ahead').\n- Mention any visible obstructions or people nearby."
        "\n- If you are unsure about part of the scene, say so clearly (e.g., 'It is not clear what is on your right')."
        "\n- Keep explanations simple, concise, and actionable for someone who cannot see."
        "\n- Do not describe lighting, ceiling, or floor unless it is needed for navigation.")
        completion = client.chat.completions.create(model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[{"role": "user", "content": [{"type": "text", "text": new_query}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}]}],
            temperature=0, max_tokens=400, top_p=1, stream=False, stop=None,)
        response_text = completion.choices[0].message.content
        return response_text

print(scene_description(r"original_images/marinara_top.jpg"))