import os
import uuid
import base64
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

JOBS = {}


def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")


def analyze_body_part(image_bytes):
    try:
        b64 = encode_image(image_bytes)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64," + b64,
                                "detail": "high"
                            }
                        },
                        {
                            "type": "text",
                            "text": (
                                "Analyze this body part photo for tattoo placement. "
                                "Describe: 1) the exact body part (e.g. inner forearm, outer upper arm, shoulder blade), "
                                "2) skin tone (fair/medium/olive/dark), "
                                "3) approximate dimensions and shape of the area, "
                                "4) any existing tattoos or notable features. "
                                "Be concise and specific. This will be used to generate a realistic tattoo preview."
                            )
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("Vision analysis error: %s", str(e))
        return "A body part suitable for tattoo placement"


def generate_tattoo_image(body_analysis, tattoo_description):
    prompt = (
        "Ultra-realistic tattoo preview photograph. "
        "Body part: " + body_analysis + ". "
        "Tattoo design: " + tattoo_description + ". "
        "The tattoo is freshly applied, ink settled into skin naturally. "
        "Professional studio lighting. High detail. Photorealistic. "
        "The tattoo looks like a real photograph of actual skin with a real tattoo, not an illustration. "
        "No text overlays, no watermarks, no borders."
    )
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="hd",
        n=1
    )
    return response.data[0].url


@app.route("/")
def index():
    return send_file(str(Path(__file__).parent / "index.html"))


@app.route("/generate", methods=["POST"])
def generate():
    try:
        if "photo" not in request.files:
            return jsonify({"error": "No photo uploaded"}), 400

        photo = request.files["photo"]
        description = request.form.get("description", "").strip()
        plan = request.form.get("plan", "single")

        if not description:
            return jsonify({"error": "Please describe your tattoo"}), 400

        if len(description) > 500:
            return jsonify({"error": "Description too long (max 500 characters)"}), 400

        image_bytes = photo.read()
        if len(image_bytes) == 0:
            return jsonify({"error": "Empty image file"}), 400

        logger.info("Analyzing body part...")
        body_analysis = analyze_body_part(image_bytes)
        logger.info("Body analysis: %s", body_analysis)

        count = 5 if plan == "bundle" else 1
        image_urls = []

        for i in range(count):
            logger.info("Generating image %d of %d...", i + 1, count)
            url = generate_tattoo_image(body_analysis, description)
            image_urls.append(url)

        job_id = str(uuid.uuid4())
        JOBS[job_id] = {
            "urls": image_urls,
            "description": description,
            "body_analysis": body_analysis
        }

        return jsonify({
            "job_id": job_id,
            "images": image_urls,
            "count": count
        })

    except Exception as e:
        logger.error("Generation error: %s", str(e))
        return jsonify({"error": "Generation failed. Please try again."}), 500


@app.route("/job/<job_id>")
def get_job(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
