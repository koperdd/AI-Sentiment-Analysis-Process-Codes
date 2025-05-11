import pandas as pd
import torch
import shap
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load the cleaned CSV file (SUBMISSION ONLY)
submissions_file = "drive/MyDrive/2019-2024/CLEANED/SUBMISSIONS/PHILIPPINES_SUB19-24_cleaned.csv"
try:
    submissions_df = pd.read_csv(submissions_file, encoding="ISO-8859-1")
except UnicodeDecodeError:
    print("Unicode error encountered! Retrying with 'utf-8' encoding...")
    submissions_df = pd.read_csv(submissions_file, encoding="utf-8", errors="replace")

# Standardize column names (strip spaces, remove special characters)
submissions_df.columns = submissions_df.columns.str.strip().str.replace(r"[^\w\s]", "", regex=True).str.lower()

# Detect the correct 'cleaned' text column dynamically
sub_cleaned_col = [col for col in submissions_df.columns if "cleaned" in col.lower()]
if len(sub_cleaned_col) != 1:
    raise KeyError(f"Could not find a unique 'cleaned' text column: {sub_cleaned_col}")
sub_cleaned_col = sub_cleaned_col[0]

# Load Pretrained Sentiment Model (XLM-RoBERTa)
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)  # ✅ Use fast tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Enable GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create a pipeline for sentiment classification
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# SHAP Explainer Setup (Corrected)
explainer = shap.Explainer(classifier)

# Function to Classify Sentiment with Dynamic Neutral Scaling
def classify_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return "neutral", 0.0  # Handle empty text as strictly neutral

    try:
        result = classifier(text)
        label = result[0]["label"]
        score = result[0]["score"]

        if label == "neutral":
            # Dynamic scaling: Neutral sentiment leans toward positive or negative
            neutral_score = 0.3 * (2 * score - 1)  # Scales between -0.3 to 0.3
            return label, neutral_score
        else:
            # Standard sentiment scaling
            score_mapping = {"positive": 1, "negative": -1}
            return label, score_mapping[label] * score

    except Exception:
        return "error", 0.0

# Apply Sentiment Classification
submissions_df["Submission Sentiment"], submissions_df["Submission Score"] = zip(
    *submissions_df[sub_cleaned_col].apply(classify_text)
)

# Function to Extract SHAP-Based Top Impactful Words with Scores
def extract_shap_top_words(text, top_n=10):
    if not isinstance(text, str) or text.strip() == "":
        return "empty text"

    if len(text.split()) < 3:  # Skip texts shorter than 3 words
        return "too short"

    try:
        # Generate SHAP values with reduced computations
        shap_values = explainer([text], max_evals=100, batch_size=10)

        # Tokenize text input for word extraction
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])  # ✅ Fix: Ensure token alignment

        # Extract SHAP scores as a list
        shap_scores = shap_values.values[0].tolist()  # ✅ Fix: Convert NumPy array to list

        # Ensure SHAP values match token count
        if len(shap_scores) != len(tokens):
            return "shap-token mismatch"

        # Extract tokens and their SHAP importance scores
        token_importance = []
        for token, score in zip(tokens, shap_scores):
            if isinstance(score, list):  # ✅ Fix: Extract first score if it's a list
                score = score[0]

            # ✅ Clean up SentencePiece artifacts (e.g., "â–poor" → "poor")
            token_str = token.replace("â–", "").replace("▁", "").strip()

            if token_str not in ["[CLS]", "[SEP]", "[PAD]"]:  # Ignore special tokens
                token_importance.append((token_str, abs(score)))

        # Merge subwords into full words
        words = []
        for token, score in token_importance:
            if token.startswith("##"):  # Merge subwords
                words[-1] = (words[-1][0] + token.replace("##", ""), words[-1][1])
            else:
                words.append((token, score))

        # Remove duplicates and sort words by SHAP value
        sorted_words = sorted(list(set(words)), key=lambda x: x[1], reverse=True)[:top_n]

        # ✅ Return top impactful words with their SHAP scores
        return ", ".join([f"{word} ({score:.3f})" for word, score in sorted_words])
    except Exception as e:
        return f"error: {str(e)}"  # ✅ Debugging information

# Apply SHAP-Based Key Term Extraction
submissions_df["Top Impactful Words"] = submissions_df[sub_cleaned_col].apply(extract_shap_top_words)

# Save Final Results
output_file = "PHILIPPINES_SUB19-24_analysis_results_shap_fixed5002.csv"
submissions_df.to_csv(output_file, index=False)

print(f"✅ Sentiment analysis completed with SHAP! Results saved to: {output_file}")
