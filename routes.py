from flask import Flask, render_template, request, Response, jsonify
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict
from LLMModule import LLMModule, ModelConfig, LLMException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize LLM model
try:
    model_path = "D:/PROJECT EXPO 1/Harshgup16/llama-3-8b-Instruct-bnb-4bit-laptop-recommendation/unsloth.Q4_K_M.gguf"  # Update with your model path
    model_config = ModelConfig(
        context_length=2048,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        repeat_penalty=1.1
    )
    llm = LLMModule(model_path, config=model_config, logger=logger)
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    raise

# Store chat histories
chat_histories = {}

@app.route('/')
def index():
    """Serve the main chat interface"""
    return render_template('index.html')

@app.route('/api/stream-chat', methods=['POST'])
def stream_chat():
    """Handle streaming chat requests"""
    try:
        data = request.json
        message = data.get('message')
        chat_id = data.get('chat_id', 'default')
        config = data.get('config', {})

        # Initialize or get chat history
        if chat_id not in chat_histories:
            chat_histories[chat_id] = [
                {"role": "system", "content": config.get('system_prompt', 'You are a helpful AI assistant that provides well-formatted responses using markdown.')}
            ]

        # Add user message to history
        chat_histories[chat_id].append({"role": "user", "content": message})

        # Update model configuration
        llm.config.temperature = float(config.get('temperature', 0.7))
        llm.config.top_p = float(config.get('top_p', 0.95))
        llm.config.top_k = int(config.get('top_k', 40))
        llm.config.repeat_penalty = float(config.get('repeat_penalty', 1.1))

        def generate():
            try:
                assistant_message = {"role": "assistant", "content": ""}
                
                # Stream the response
                for token in llm.stream_chat(
                    chat_histories[chat_id],
                    max_tokens=512 if config.get('limit_length') else 2048
                ):
                    assistant_message["content"] += token
                    yield f"data: {json.dumps({'token': token})}\n\n"
                
                # Add complete assistant message to history
                chat_histories[chat_id].append(assistant_message)
                
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"Streaming error: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(generate(), mimetype='text/event-stream')

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear-chat', methods=['POST'])
def clear_chat():
    """Clear chat history for a specific chat ID"""
    try:
        data = request.json
        chat_id = data.get('chat_id', 'default')
        system_prompt = data.get('system_prompt', 'You are a helpful AI assistant.')
        
        chat_histories[chat_id] = [
            {"role": "system", "content": system_prompt}
        ]
        
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Failed to clear chat: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/placeholder/<int:width>/<int:height>')
def placeholder_image(width: int, height: int):
    """Serve placeholder images for HTML content"""
    return Response(
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="{width}" height="{height}" fill="#ddd"/>'
        f'<text x="{width/2}" y="{height/2}" text-anchor="middle" alignment-baseline="middle">'
        f'{width}x{height}</text></svg>',
        mimetype='image/svg+xml'
    )

if __name__ == '__main__':
    app.run(debug=True)