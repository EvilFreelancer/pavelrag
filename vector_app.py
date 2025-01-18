import logging
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from manticore_store import ManticoreSearchStore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = Flask(__name__)
CORS(app)

MODELS_AVAILABLE = {
    'manticoresearch': ManticoreSearchStore()
}

MODELS_INSTANCES = {}


def get_model_instance(model_id: str):
    if model_id not in MODELS_INSTANCES:
        model_class = MODELS_AVAILABLE.get(model_id)
        if not model_class:
            return None
        MODELS_INSTANCES[model_id] = model_class
    return MODELS_INSTANCES[model_id]


@app.route('/v1/models', methods=['GET'])
def openai_list_models():
    models = [{
        "id": model.id,
        "object": "model",
        # "created": model.created,
        # "owned_by": model.owned_by
    } for model in MODELS_AVAILABLE.values()]
    return jsonify({"object": "list", "data": models})


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    data = request.json
    model_id = data.get('model')

    if not model_id:
        return jsonify({'error': 'No "model" provided'}), 400

    if model_id not in MODELS_AVAILABLE:
        available_models = ', '.join(MODELS_AVAILABLE.keys())
        return jsonify({'error': f'Model "{model_id}" not supported. Available models: {available_models}'}), 400

    model_instance = get_model_instance(model_id)
    if not model_instance:
        return jsonify({'error': f'Model "{model_id}" not found.'}), 500

    messages = data.get('messages', [])
    if not messages:
        return jsonify({'error': 'No "messages" provided'}), 400

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    try:
        response = model_instance.generate_response(messages)
        return Response(response, mimetype='text/event-stream')
    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
