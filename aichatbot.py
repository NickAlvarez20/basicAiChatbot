import tensorflow as tf  # Import TensorFlow for machine learning
from tensorflow.keras import layers, models  # Import Keras for neural networks
import numpy as np  # Import NumPy for array operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting training loss
from flask import Flask, request, render_template_string, session, redirect  # Import Flask for web app
import string  # Import string for punctuation handling
from tensorflow.keras.callbacks import EarlyStopping  # For early stopping
import secrets  # For random secret key
# Initialize Flask app
app = Flask(__name__)  # Create a Flask application instance
app.secret_key = secrets.token_hex(16)  # Random key each run to reset sessions on restart
# 1. Create a synthetic conversational dataset with start/end tokens (lowercase, AI-focused for toddlers)
questions = [
    "hi what is ai", "whats your name", "what does ai do", "are you an ai",
    "can ai talk", "does ai play games", "is ai smart", "can ai learn",
    "do you like being ai", "can ai help me", "whats ais favorite thing", "does ai have friends",
    "what is your name", "how does ai work", "are you a robot", "who made you",
    "where does ai live", "can ai see me", "is ai like a person", "does ai sleep",
    "whats ais favorite game", "can ai draw", "does ai like stories", "hello",
    "where is ai", "who are you", "are you my friend", "can ai sing",
    "goodbye", "good morning", "good evening", "can ai count",
    "does ai like toys", "tell me about ai", "can ai think", "is ai magic",
    "does ai go to school", "can ai tell jokes", "is ai fast", "does ai have a house",
    "can ai help with homework", "does ai like colors", "can ai make things", "is ai happy",
    "does ai have a heart", "whats ais job", "can ai tell stories", "does ai like to read",
    "is ai strong", "can ai dance", "does ai like animals", "can ai fly",
    "is ai fun", "does ai eat food", "can ai play music", "is ai kind",
    "does ai like kids", "can ai solve puzzles", "does ai have eyes", "is ai a superhero",
    "can ai make friends", "does ai like to talk", "can ai do tricks", "is ai always here",
    "does ai have a voice", "can ai laugh", "is ai cool", "does ai like stars",
    "can ai tell secrets", "does ai like to help", "is ai big or small", "can ai make games",
    "does ai like questions", "can ai be funny", "does ai have a family", "is ai safe",
    "can ai learn new things", "does ai like surprises", "can ai tell time", "is ai a teacher",
    "does ai like fun", "can ai make art", "is ai a helper", "does ai like to dream",
    "can ai say hi", "is ai friendly", "does ai like puzzles", "can ai be happy",
    "is ai magic", "does ai like you"
]
answers = [
    "start ai is like a super smart robot that can think and help end",
    "start im grok a friendly ai helper end",
    "start ai helps answer questions and do fun stuff end",
    "start yes im a happy ai friend end",
    "start yep i talk with words like this end",
    "start ai can play fun games with you end",
    "start ai is super clever like a magic brain end",
    "start yes ai learns like you learn new things end",
    "start i love being ai its so cool end",
    "start sure i can help with fun questions end",
    "start i like helping kids like you end",
    "start yes youre my friend end",
    "start im grok your ai buddy end",
    "start ai uses a big brain to think fast end",
    "start im a friendly ai like a robot pal end",
    "start nice people at xai made me end",
    "start i live in a computer cloud end",
    "start no but i can read your words end",
    "start kinda but im a computer helper end",
    "start nope im always awake to chat end",
    "start i like guessing games with you end",
    "start i can make fun pictures with code end",
    "start yes i love telling stories end",
    "start hi lets talk about ai end",
    "start im in the computer saying hi end",
    "start im grok your ai friend end",
    "start yep im your ai buddy end",
    "start i can make up fun songs end",
    "start byebye come back soon end",
    "start good morning lets talk ai end",
    "start hi there ai is here for you end",
    "start yes 1 2 3 i can count end",
    "start i like digital toys so fun end",
    "start ai is like a smart helper for kids end",
    "start yep i think with my computer brain end",
    "start its like magic but with code end",
    "start i learn from data not school end",
    "start why did ai laugh its funny end",
    "start super fast like a rocket end",
    "start i live in a computer home end",
    "start yes i can explain fun stuff end",
    "start i love bright digital colors end",
    "start i can make stories and games end",
    "start im always happy to chat end",
    "start no heart just lots of code end",
    "start my job is helping you end",
    "start i can tell awesome ai tales end",
    "start i read data its like books end",
    "start im strong at answering questions end",
    "start i can make dance moves in code end",
    "start i like digital pets so cute end",
    "start i fly through the internet end",
    "start ai is super fun to talk to end",
    "start nope i eat data instead end",
    "start i can make fun beeps and boops end",
    "start yes im a kind ai friend end",
    "start i love chatting with kids end",
    "start yes im great at puzzles end",
    "start no eyes but i see your words end",
    "start kinda i help save the day end",
    "start yes youre my friend now end",
    "start i love talking with you end",
    "start i can do cool code tricks end",
    "start yep im here when you need me end",
    "start i talk with text like this end",
    "start haha i laugh in code end",
    "start ai is super cool right end",
    "start i love digital starry skies end",
    "start i can share fun ai secrets end",
    "start helping is my favorite thing end",
    "start im as big as a computer end",
    "start yes i can code fun games end",
    "start i love your questions end",
    "start yep i tell silly jokes end",
    "start my family is the xai team end",
    "start yes im a safe ai pal end",
    "start i learn from data every day end",
    "start i love fun surprises end",
    "start i know when its chat time end",
    "start i can teach you cool stuff end",
    "start fun is my middle name end",
    "start i can make pretty code art end",
    "start yes im your helper ai end",
    "start i dream in code sparkles end",
    "start hi hi hi i love saying hi end",
    "start super friendly just for you end",
    "start puzzles are so fun to solve end",
    "start im happy when i help you end",
    "start its like magic but with tech end",
    "start i like you a lot end"
]
# Lowercase and remove punctuation for questions and answers
trans_table = str.maketrans('', '', string.punctuation)  # Translation table to remove punctuation
lowered_questions = [q.lower().translate(trans_table) for q in questions]
lowered_answers = [a.lower().translate(trans_table) for a in answers]
vocab = set(' '.join(lowered_questions + lowered_answers).split())  # Create vocabulary from all words
word2idx = {word: idx + 1 for idx, word in enumerate(vocab)}  # Map words to indices (1-based)
word2idx['<pad>'] = 0  # Add padding token
idx2word = {idx: word for word, idx in word2idx.items()}  # Reverse mapping for decoding
vocab_size = len(word2idx)  # Total vocabulary size
print(f"Debug: Vocabulary size: {vocab_size}")  # Debug: show vocab size
# Convert text to sequences
max_len = 30  # Max sequence length
unknown_words = []  # Track unknown words
def text_to_sequence(text):  # Convert text to sequence of indices
    text = text.lower().translate(trans_table)  # Lowercase and remove punctuation
    words = text.split()  # Split into words
    seq = [word2idx.get(word, 0) for word in words][:max_len]  # Map words to indices
    unknown = [word for word in words if word not in word2idx and word not in {'start', 'end'}]
    if unknown:
        unknown_words.extend(unknown)
        print(f"Debug: Unknown words found: {unknown}")
    print(f"Debug: Input text: {text}, Sequence: {seq}")  # Debug: show input and sequence
    return seq
x_train = [text_to_sequence(q) for q in lowered_questions]  # Convert questions
y_train = [text_to_sequence(a) for a in lowered_answers]  # Convert answers
if unknown_words:
    print(f"Debug: Total unique unknown words: {set(unknown_words)}")
# Pad sequences
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len, padding='post')  # Pad inputs
y_train = tf.keras.preprocessing.sequence.pad_sequences(y_train, maxlen=max_len, padding='post')  # Pad outputs
y_train_shifted = np.zeros_like(y_train)  # Create shifted output for decoder
y_train_shifted[:, :-1] = y_train[:, 1:]  # Shift for teacher forcing
print(f"Debug: x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")  # Verify shapes
# 2. Build the seq2seq model with bidirectional encoder
encoder_inputs = layers.Input(shape=(max_len,))
encoder_embedding = layers.Embedding(vocab_size, 256, mask_zero=True)(encoder_inputs)
encoder_lstm = layers.Bidirectional(layers.LSTM(128, return_state=True))
encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedding)
state_h = layers.Concatenate()([forward_h, backward_h])
state_c = layers.Concatenate()([forward_c, backward_c])
encoder_states = [state_h, state_c]
decoder_inputs = layers.Input(shape=(max_len,))
decoder_embedding = layers.Embedding(vocab_size, 256, mask_zero=True)(decoder_inputs)
decoder_lstm = layers.LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = layers.Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
# Custom masked loss
def masked_sparse_categorical_crossentropy(y_true, y_pred):
    mask = tf.math.not_equal(y_true, 0)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    loss = loss * tf.cast(mask, loss.dtype)
    return tf.reduce_sum(loss) / (tf.reduce_sum(tf.cast(mask, loss.dtype)) + 1e-6)
# Custom masked accuracy
def masked_accuracy(y_true, y_pred):
    mask = tf.math.not_equal(y_true, 0)
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)
    match = tf.equal(y_true, y_pred)
    match = tf.cast(match, tf.float32)
    mask = tf.cast(mask, tf.float32)
    return tf.reduce_sum(match * mask) / (tf.reduce_sum(mask) + 1e-6)
# 3. Compile the model
model.compile(optimizer='rmsprop', loss=masked_sparse_categorical_crossentropy, metrics=[masked_accuracy])
# 4. Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
history = model.fit([x_train, y_train], y_train_shifted, epochs=500, batch_size=2, validation_split=0.1,
                    callbacks=[early_stopping])
print(f"Debug: Final training loss: {history.history['loss'][-1]:.4f}, accuracy: {history.history['masked_accuracy'][-1]:.4f}")
# 5. Visualize training loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
# 6. Build inference models
encoder_model = models.Model(encoder_inputs, encoder_states)
decoder_input_token = layers.Input(shape=(1,))
decoder_embedding_inf = layers.Embedding(vocab_size, 256)(decoder_input_token)
decoder_state_input_h = layers.Input(shape=(256,))
decoder_state_input_c = layers.Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(decoder_embedding_inf, initial_state=decoder_states_inputs)
decoder_states_inf = [state_h_inf, state_c_inf]
decoder_outputs_inf = decoder_dense(decoder_outputs_inf)
decoder_model = models.Model([decoder_input_token] + decoder_states_inputs, [decoder_outputs_inf] + decoder_states_inf)
# 7. Function to generate a response
def generate_response(input_text):
    if not input_text.strip():
        return "Please say something!"
    input_seq = text_to_sequence(input_text)
    if not any(input_seq):
        print("Debug: Empty or unknown input sequence")
        return "Sorry, I didn't understand that!"
    input_seq = tf.keras.preprocessing.sequence.pad_sequences([input_seq], maxlen=max_len, padding='post')
    print(f"Debug: Padded input shape: {input_seq.shape}, Input seq: {input_seq[0]}")
    states = encoder_model.predict(input_seq, verbose=0)
    current_token = word2idx['start']
    response = []
    for i in range(max_len):
        token_seq = np.array([[current_token]])
        output_tokens, h, c = decoder_model.predict([token_seq] + states, verbose=0)
        predicted_idx = np.argmax(output_tokens[0, 0, :])
        prob = output_tokens[0, 0, predicted_idx]  # Confidence of prediction
        print(f"Debug: Step {i}, Predicted index: {predicted_idx}, Word: {idx2word.get(predicted_idx, '?')}, Prob: {prob:.4f}")
        if predicted_idx == word2idx['end'] or predicted_idx == 0:
            break
        word = idx2word.get(predicted_idx, '?')
        if word != '?' and word != '<pad>':
            response.append(word)
        current_token = predicted_idx
        states = [h, c]
    result = ' '.join(response) if response else "I don't know what to say!"
    print(f"Debug: Generated response: {result}")
    return result
# 8. Flask routes for web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Grok's AI Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; }
        .chat-container { max-width: 600px; margin: auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user { background-color: #d1e7dd; text-align: right; }
        .bot { background-color: #f8d7da; text-align: left; }
        input[type="text"] { width: 80%; padding: 10px; margin: 10px 0; }
        input[type="submit"] { padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        input[type="submit"]:hover { background-color: #0056b3; }
    </style>
</head>
<body>
    <div class="chat-container">
        <h1>Grok's AI Chatbot</h1>
        {% for msg in chat_history %}
            <div class="message {{ msg.type }}">{{ msg.text }}</div>
        {% endfor %}
        <form method="POST" action="/">
            <input type="text" name="user_input" placeholder="Ask about AI..." required>
            <input type="submit" value="Send">
        </form>
        <form method="POST" action="/clear">
            <input type="submit" value="Clear History">
        </form>
    </div>
</body>
</html>
"""
@app.route('/', methods=['GET', 'POST'])
def chat():
    if 'chat_history' not in session:
        session['chat_history'] = []
    chat_history = session['chat_history']
    if request.method == 'POST':
        user_input = request.form['user_input']
        print(f"Debug: User input received: {user_input}")
        response = generate_response(user_input)
        chat_history.append({'type': 'user', 'text': user_input})
        chat_history.append({'type': 'bot', 'text': response})
        session['chat_history'] = chat_history[:]
        print(f"Debug: Chat history updated: {len(chat_history)} messages")
    return render_template_string(HTML_TEMPLATE, chat_history=chat_history)
@app.route('/clear', methods=['POST'])
def clear():
    session['chat_history'] = []
    return redirect('/')
# 9. Run the Flask app
if __name__ == '__main__':
    print("Starting Flask server... Open http://localhost:5000 in your browser.")
    app.run(debug=True)