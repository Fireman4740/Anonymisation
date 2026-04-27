# Anonymization Error Analysis Tool

This Streamlit app allows you to visualize and analyze the errors made by the anonymization pipeline.

## Setup

1. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

cd /mnt/f/IA/Anonymisation/eval/streamlit_app && conda run -n ano --no-capture-output streamlit run app.py --server.headless true --server.port 8501 
2. Open your browser at the URL shown (usually http://localhost:8501).

3. Select a report from the sidebar.
   - The tool looks for JSON reports in `../../evaluation/reports/`.

4. Use the filters to find problematic documents (e.g., low recall, leaks).

5. Click on a document ID to see the text with highlighted errors:
   - **Green**: True Positive (Correctly anonymized)
   - **Red**: False Negative (Missed entity - Critical!)
   - **Yellow**: False Positive (Unnecessary redaction)

## Note on Full Text

The original evaluation pipeline only saved truncated text snippets. 
The pipeline has been updated to save the full text. 
**For the best experience, please re-run your evaluation pipeline** to generate new reports containing the full text of the documents.

```bash
cd ../..
python eval/evaluate_pipeline.py
```
