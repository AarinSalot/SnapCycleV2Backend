from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import replicate
import io
import openai 
import os

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv('OPENAIKEY')

@app.route("/process", methods=['POST'])
def process():
    picture_data = request.json['picture']
    picture_bytes = base64.b64decode(picture_data.split(',')[1])
    image_file = io.BytesIO(picture_bytes)
    output = replicate.run(
                "salesforce/blip:2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746",
                input={
                    "image": image_file, 
                    "task": "visual_question_answering",
                    "question": "What item is this?"
                    }
            )
    
    remove_caption = output.split(':')[1]
    prompt = create_prompt(remove_caption)
    answer = get_response(prompt)
    return jsonify({"answer": answer})

def create_prompt(message):
    return ("Please provide a a super short, succinct, useful explaination while avoiding generalizations and summaries. How do I recycle: " 
                + message
            )

def get_response(message):
    response = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        temperature = 1,
        messages = [
                {"role": "user", "content": message}
                ]
    )
    rv = response.choices[0]["message"]["content"]
    return rv


if __name__ == "__main__":
    app.run(debug=True)
