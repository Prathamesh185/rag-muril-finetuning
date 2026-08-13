import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai


load_dotenv()


# Gemini
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)


# Fine-tuned MuRIL V2
ENCODER_PATH = "models/fine_tuned_muril_v2"

encoder = SentenceTransformer(
    ENCODER_PATH
)

encoder.max_seq_length = 256

print("Fine-tuned MuRIL loaded.")
print(
    "Embedding dimension:",
    encoder.get_embedding_dimension()
)