from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from google import genai
from google.genai import types
import cv2
import time
import re
import os
import subprocess
import numpy as np
import whisper
import hashlib
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__, template_folder="templates")
CORS(app)

# ==========================================
# PATH CONFIGURATION
# ==========================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
FRAMES_DIR = os.path.join(BASE_DIR, "frames")

# ==========================================
# STATE & LOGGING MANAGEMENT
# ==========================================
STATE = {
    "vector_db": [],
    "is_processed": False,
    "full_transcript": "",
    "chat_history": []
}
PROCESSING_LOGS = []

def log_update(message):
    print(message)
    PROCESSING_LOGS.append(message)

# ==========================================
# CONFIGURATION
# ==========================================
VISION_MODEL = 'gemini-3.1-flash-lite'
QA_MODEL = 'gemini-3.1-flash-lite'
EMBEDDING_MODEL = 'gemini-embedding-001'
FLASH_LITE_DELAY = 4.5
EMBEDDING_DELAY = 0.7 

# ==========================================
# CORE RAG FUNCTIONS
# ==========================================
def get_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        chunk = f.read(8192)
        while chunk:
            hasher.update(chunk)
            chunk = f.read(8192)
    return hasher.hexdigest()

_whisper_model = None
def load_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        log_update("⚙️ Loading high-quality Whisper AI model into memory...")
        _whisper_model = whisper.load_model("small")
    return _whisper_model

def transcribe_audio(video_path):
    log_update("🎧 Extracting audio natively via FFmpeg...")
    audio_path = "temp_audio.wav"
    
    # Try to clean up the old audio file safely
    if os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except OSError:
            pass # Ignore if locked, ffmpeg will overwrite it
            
    command = ["ffmpeg", "-y", "-i", video_path, "-ab", "160k", "-ac", "2", "-ar", "44100", "-vn", audio_path]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # FIX: Check if the audio file is empty (prevents the 0-element tensor crash)
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
        log_update("⚠️ No valid audio extracted. Skipping transcription.")
        STATE['full_transcript'] = "No audio detected."
        return []

    log_update("📝 Transcribing audio to text (this may take a moment)...")
    model = load_whisper_model()
    result = model.transcribe(audio_path)
    
    STATE['full_transcript'] = result['text']
    log_update("✅ Audio transcription complete!")
    return result['segments']

def get_audio_context(timestamp_sec, audio_segments, window=10):
    spoken = ""
    for seg in audio_segments:
        if (seg['start'] <= timestamp_sec + window) and (seg['end'] >= timestamp_sec - 2):
            spoken += seg['text'] + " "
    return spoken.strip() if spoken else "No speech detected."

def caption_frame(client, image_path):
    import PIL.Image
    try:
        # Using "with" ensures the image file is closed immediately so it doesn't lock
        with PIL.Image.open(image_path) as img:
            response = client.models.generate_content(
                model=VISION_MODEL,
                contents=["Describe the screen exactly. Mention text, charts, or diagrams.", img]
            )
            time.sleep(FLASH_LITE_DELAY)
            return response.text
    except Exception as e:
        if "429" in str(e) or "exhausted" in str(e).lower():
            log_update("⚠️ Rate limit hit. Pausing for 15 seconds...")
            time.sleep(15)
            return caption_frame(client, image_path)
        return "Failed to caption frame."

def generate_embedding(client, text, task_type="RETRIEVAL_DOCUMENT"):
    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type)
        )
        time.sleep(EMBEDDING_DELAY)
        return response.embeddings[0].values
    except Exception as e:
        if "429" in str(e) or "exhausted" in str(e).lower():
            time.sleep(5)
            return generate_embedding(client, text, task_type)
        return [0] * 768

def process_video(client, video_path, extract_every_n_seconds=10):
    STATE['vector_db'] = [] 
    audio_segments = transcribe_audio(video_path)
    
    log_update(f"👁️ Starting Visual Processing (1 frame every {extract_every_n_seconds}s)...")
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    
    if not os.path.exists(FRAMES_DIR):
        os.makedirs(FRAMES_DIR)
        
    frame_interval = int(fps * extract_every_n_seconds) if fps > 0 else 10
    success, image = vidcap.read()
    count = 0
    frame_id = 0
    
    while success:
        if count % frame_interval == 0:
            timestamp_sec = count / fps
            minutes = int(timestamp_sec // 60)
            seconds = int(timestamp_sec % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"
            
            log_update(f"📸 Analyzing frame at {time_str}...")
            
            frame_filename = f"frame_{frame_id}.jpg"
            frame_abs_path = os.path.join(FRAMES_DIR, frame_filename)
            cv2.imwrite(frame_abs_path, image)
            frame_rel_path = f"frames/{frame_filename}"
            
            visuals = caption_frame(client, frame_abs_path)
            audio = get_audio_context(timestamp_sec, audio_segments)
            combined_context = f"At timestamp {time_str}. VISUALS ON SCREEN: {visuals} | SPOKEN AUDIO: {audio}"
            
            embedding = generate_embedding(client, combined_context, "RETRIEVAL_DOCUMENT")
            
            STATE['vector_db'].append({
                "time_str": time_str,
                "time_sec": timestamp_sec,
                "context": combined_context,
                "embedding": embedding,
                "frame_path": frame_rel_path
            })
            frame_id += 1
            
        success, image = vidcap.read()
        count += 1
        
    # FIX: Release the video file so Windows doesn't lock it!
    vidcap.release() 
    log_update("🎉 Multimodal processing completely finished!")

def ask_video(client, user_query):
    time_match = re.search(r'(\d{1,2}:\d{2})', user_query)
    combined_context = ""
    matched_frame_path = None
    top_k_results = []
    
    if time_match:
        target_time = time_match.group(1)
        if len(target_time) == 4: target_time = "0" + target_time 
        minutes, seconds = map(int, target_time.split(":"))
        target_sec = minutes * 60 + seconds
        
        closest_item = min(STATE['vector_db'], key=lambda x: abs(x["time_sec"] - target_sec))
        combined_context = closest_item["context"]
        matched_frame_path = closest_item["frame_path"]
        top_k_results.append({
            "time": closest_item['time_str'], "score": "Exact Time Match", "context": closest_item['context']
        })
    else:
        query_embedding = generate_embedding(client, user_query, "RETRIEVAL_QUERY")
        db_embeddings = [item['embedding'] for item in STATE['vector_db']]
        vector_similarities = cosine_similarity([query_embedding], db_embeddings)[0]
        
        db_contexts = [item['context'] for item in STATE['vector_db']]
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(db_contexts + [user_query])
            keyword_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
        except ValueError:
            keyword_similarities = np.zeros(len(db_contexts))

        ALPHA = 0.7 
        hybrid_scores = (ALPHA * vector_similarities) + ((1 - ALPHA) * keyword_similarities)
        
        K = min(3, len(hybrid_scores))
        top_k_indices = np.argsort(hybrid_scores)[-K:][::-1]
        
        retrieved_contexts = []
        for idx in top_k_indices:
            item = STATE['vector_db'][idx]
            retrieved_contexts.append(item['context'])
            top_k_results.append({
                "time": item['time_str'],
                "score": f"{hybrid_scores[idx]:.4f}",
                "context": item['context']
            })
            
        combined_context = "\n\n---\n\n".join(retrieved_contexts)
        matched_frame_path = STATE['vector_db'][top_k_indices[0]]["frame_path"]

    prompt = f"""
    You are an AI assistant answering questions about a video. 
    
    CONTEXT RETRIEVED FROM VIDEO (Top Chunks):
    {combined_context}
    
    USER QUESTION: {user_query}
    
    INSTRUCTIONS:
    1. If the user asks about a specific timestamp (e.g., 03:16), treat the provided context as the correct scene, even if the context's timestamp (e.g., 03:10 or 03:20) is a few seconds off due to our 10-second interval sampling.
    2. If the context contains the answer or describes the scene, explain it clearly using ONLY the provided text.
    3. If the context is completely irrelevant and DOES NOT contain the answer, you MUST begin your response exactly with: "The video does not mention this, but here is what I know:" and then answer using your own general knowledge.
    """
    
    final_response = client.models.generate_content(model=QA_MODEL, contents=prompt)
    return final_response.text, matched_frame_path, top_k_results

# ==========================================
# API ROUTES
# ==========================================
@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/frames/<path:filename>')
def serve_frames(filename):
    return send_from_directory(FRAMES_DIR, filename)

@app.route('/api/logs', methods=['GET'])
def get_logs():
    return jsonify(PROCESSING_LOGS)

@app.route('/api/process', methods=['POST'])
def handle_process():
    global PROCESSING_LOGS
    PROCESSING_LOGS.clear()
    log_update("🚀 Initializing processing pipeline...")
    
    api_key = request.form.get('api_key')
    video_file = request.files.get('video')
    force_reprocess = request.form.get('force_reprocess') == 'true'
    
    if not api_key or not video_file:
        return jsonify({"error": "Missing API key or video file"}), 400

    temp_video_path = "temp_video.mp4"
    video_file.save(temp_video_path)
    client = genai.Client(api_key=api_key)

    try:
        video_hash = get_file_hash(temp_video_path)
        cache_file = f"cache_{video_hash}.pkl"
        
        if os.path.exists(cache_file) and not force_reprocess:
            log_update("⚡ Cache found! Bypassing ingestion pipeline...")
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)
                for item in cache_data["vector_db"]:
                    if "frame_path" in item:
                        clean_path = item["frame_path"].replace("\\", "/")
                        item["frame_path"] = f"frames/{clean_path.split('/')[-1]}"
                STATE['vector_db'] = cache_data["vector_db"]
                STATE['full_transcript'] = cache_data["transcript"]
            STATE['is_processed'] = True
            log_update("✅ Loaded instantly from cache! Ready to chat.")
            return jsonify({"status": "success", "message": "Loaded instantly from cache!"})
        else:
            if force_reprocess:
                log_update("⚠️ Force Reprocess activated. Ignoring old cache...")
            else:
                log_update("🔍 No cache found. Beginning full deep-learning extraction...")
                
            process_video(client, temp_video_path, extract_every_n_seconds=10)
            
            with open(cache_file, "wb") as f:
                cache_data = {"vector_db": STATE['vector_db'], "transcript": STATE['full_transcript']}
                pickle.dump(cache_data, f)
            STATE['is_processed'] = True
            return jsonify({"status": "success", "message": "Video Ingestion Complete!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    if not STATE['is_processed']:
        return jsonify({"error": "Please process a video first."}), 400
        
    data = request.json
    api_key = data.get('api_key')
    query = data.get('query')
    
    if not api_key or not query:
        return jsonify({"error": "Missing API key or query."}), 400

    client = genai.Client(api_key=api_key)
    try:
        answer, image_path, top_k = ask_video(client, query)
        clean_image_path = image_path.replace("\\", "/") if image_path else None
        
        STATE['chat_history'].append({"role": "user", "text": query})
        STATE['chat_history'].append({"role": "assistant", "text": answer, "image": clean_image_path, "top_k": top_k})
        
        return jsonify({"answer": answer, "image": clean_image_path, "top_k": top_k})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/state', methods=['GET'])
def get_state():
    return jsonify({
        "is_processed": STATE["is_processed"],
        "transcript": STATE["full_transcript"],
        "vector_db": STATE["vector_db"]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)