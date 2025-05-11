
import torch
import shap
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load Pretrained Sentiment Model (XLM-RoBERTa)
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 🔹 Force CPU execution
device = torch.device("cpu")
print(f"Using device: {device}")
model.to(device)

# Create a pipeline for sentiment classification (CPU Mode)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)

# SHAP Explainer Setup - Runs on CPU
explainer = shap.Explainer(classifier, algorithm="partition", output_names=["negative", "neutral", "positive"])

# List of common stop words to ignore in SHAP output
STOP_WORDS = set(["the", "to", "in", "for", "of", "and", "on", "at", "a", "is", "it", "this", "that", "with", "as", "by"])

# Function for Safe Tokenization Before Classification
def safe_tokenize(text, tokenizer, max_length=512):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"][0]
    return tokenizer.decode(tokens, skip_special_tokens=True)

# Function for Sliding Window Tokenization
def sliding_window_tokenize(text, tokenizer, max_length=512, stride=256):
    tokens = tokenizer(text, return_tensors="pt", padding=False, truncation=False)["input_ids"][0]
    token_count = len(tokens)

    if token_count <= max_length:
        return [text]  # No need for a window if within limit

    windows = []
    for i in range(0, token_count, stride):
        window_tokens = tokens[i : min(i + max_length, token_count)]
        if len(window_tokens) == 0:  
            continue
        windows.append(tokenizer.decode(window_tokens, skip_special_tokens=True))

    return windows

# Function to Reconstruct Words from Subword Tokens
def merge_tokens(tokens, shap_scores):
    """
    Merges subword tokens into full words and filters out stop words.
    """
    full_words = []
    current_word = ""
    current_score = 0

    for token, score in zip(tokens, shap_scores):
        token = token.replace("▁", "").strip()  # Remove SentencePiece artifact
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue  # Ignore special tokens

        if token.startswith("##"):  
            current_word += token[2:]  # Merge subword
            current_score += score  # Aggregate score
        else:
            if current_word and current_word.lower() not in STOP_WORDS:
                full_words.append((current_word, current_score))
            current_word = token  # Start new word
            current_score = score

    if current_word and current_word.lower() not in STOP_WORDS:
        full_words.append((current_word, current_score))

    return sorted(full_words, key=lambda x: abs(x[1]), reverse=True)

# Function for SHAP-based Word Importance (Fixed)
def extract_shap_top_words(text, predicted_label, top_n=10):
    """
    Uses SHAP to extract the most important words influencing the predicted sentiment.
    """
    if not isinstance(text, str) or text.strip() == "":
        return "empty text"

    if len(text.split()) < 3:
        return "too short"

    try:
        # ✅ Tokenize text before SHAP using the same settings as the model
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        # ✅ Run SHAP on properly tokenized input
        shap_values = explainer([text], max_evals=100, batch_size=5)

        # ✅ Extract tokens from properly tokenized input
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # ✅ Convert SHAP scores to NumPy array and select only the predicted class
        label_index_map = {"negative": 0, "neutral": 1, "positive": 2}
        predicted_index = label_index_map[predicted_label]
        shap_scores = np.array(shap_values.values[0])[:, predicted_index]  # Select scores for predicted sentiment

        # ✅ Ensure SHAP values match token count
        if len(tokens) != len(shap_scores):
            print(f"⚠️ Warning: Token count ({len(tokens)}) does not match SHAP values count ({len(shap_scores)})")
            return "shap-token mismatch"

        # ✅ Merge subword tokens into full words
        merged_words = merge_tokens(tokens, shap_scores)

        # ✅ Return top N words
        return ", ".join([f"{word} ({score:.3f})" for word, score in merged_words[:top_n]])

    except Exception as e:
        print(f"Error extracting SHAP values: {e}")
        return f"error: {str(e)}"

# Function to Classify Sentiment (With Sliding Window and SHAP per Chunk)
def classify_text_sliding(text):
    if not isinstance(text, str) or text.strip() == "":
        return "neutral", 0.0, []

    windows = sliding_window_tokenize(text, tokenizer)
    if len(windows) == 0:
        print("⚠️ Sliding Window Error: No valid windows created!")
        return "error", 0.0, []

    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    weighted_score = 0.0
    chunk_results = []

    try:
        print(f"\n🔍 **Sliding Window Processing Started...**")
        print(f"📌 Total Windows Created: {len(windows)}")

        for i, window in enumerate(windows):
            clean_window = safe_tokenize(window, tokenizer)
            result = classifier(clean_window)
            if not result:
                continue
            label = result[0]["label"]
            score = result[0]["score"]

            # Get SHAP explanation for this chunk using the predicted sentiment
            shap_words = extract_shap_top_words(clean_window, label)

            sentiment_counts[label] += 1
            if label == "neutral":
                # Scale neutral scores between -0.3 and 0.3
                neutral_score = 0.3 * (2 * score - 1)  
                weighted_score += neutral_score
            else:
                # Standard sentiment scaling
                score_mapping = {"positive": 1, "negative": -1}
                weighted_score += score_mapping[label] * score


            print(f"\n  🟢 **Chunk {i+1}: {label} (Score: {score:.3f})**")
            print(f"     📜 **Text:** {clean_window}")  # ✅ Print the chunk's actual text
            print(f"     🔹 **Top Words Influencing Sentiment:** {shap_words}")

            chunk_results.append((f"Chunk {i+1}", label, score, shap_words, clean_window))

        total_classifications = sum(sentiment_counts.values())
        if total_classifications == 0:
            return "neutral", 0.0, chunk_results  

        # Final sentiment based on majority vote
        final_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        final_score = weighted_score / total_classifications

        return final_sentiment, final_score, chunk_results
    except Exception as e:
        print(f"Error classifying text: {e}")
        return "error", 0.0, []

# 🔹 Test Input
text_input = """


"""
sentiment_with_sliding, score_with_sliding, chunk_results = classify_text_sliding(text_input)

# 🔹 Print Results
print("\n🔍 **Final Sentiment Analysis Results:**\n")
print(f"📌 **Final Sentiment:** {sentiment_with_sliding} (Score: {score_with_sliding:.3f})")
print("\n🔍 **Chunk-wise Sentiment Breakdown:**")
for chunk in chunk_results:
    print(f"  - {chunk[0]}: {chunk[1]} (Score: {chunk[2]:.3f})")
    print(f"    📜 **Text:** {chunk[4]}")
    print(f"    🔹 **Top Words Influencing Sentiment:** {chunk[3]}")
