# Customer Feedback Intelligence

An interactive Streamlit app for customer review sentiment analysis and feedback exploration.

## Features
- Upload a CSV file or use the included demo dataset.
- Predict Positive, Neutral, or Negative sentiment using TF-IDF + Logistic Regression.
- View sentiment distribution, product comparisons, monthly trends (when a date column exists), average ratings, and common terms in negative feedback.
- Filter analyzed feedback and download predictions as CSV.

## Requirements
- Python 3.10 or newer recommended.

## Run locally (Windows PowerShell)
```powershell
cd Customer_Feedback_Intelligence
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run the commands with `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\python.exe -m streamlit run app.py`.

## Input CSV format
A feedback text column is required. The app recognizes one of:
`feedback`, `review`, `text`, `comment`, `customer_feedback`, `review_text`.

Optional columns:
- `rating` (or `stars`, `score`, `customer_rating`)
- `product` (or `product_name`, `item`, `category`)
- `date` (or `created_at`, `review_date`, `timestamp`)

Example:
```csv
feedback,product,rating,date
"Delivery was fast and product quality is great",Wireless Earbuds,5,2026-01-05
"Package arrived broken and support was unhelpful",Coffee Maker,1,2026-01-07
```

## Important model note
The bundled labeled dataset is small and illustrative, intended to make the app runnable as a demo. The model is not production-grade and may misclassify feedback. For a credible ML evaluation, train and test on a larger, representative, human-labeled dataset, split data into train/test sets, and report metrics such as precision, recall, F1-score, and confusion matrix. Do not present demo predictions as verified customer sentiment.

## Suggested extensions
- Add a separate training workflow for uploaded labeled data.
- Add aspect-based sentiment (delivery, quality, price, support).
- Add FastAPI prediction endpoint and Docker deployment.
