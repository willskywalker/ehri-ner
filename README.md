# EHRI-NER: Date Extractor

This Streamlit application queries the EHRI (European Holocaust Research Infrastructure) Knowledge Graph, extracts text from records, and uses Named Entity Recognition (NER) to identify dates. It then leverages an LLM (via LiteLLM) to normalize these dates into a standard format.

## Features

*   **Flexible SPARQL Querying**: Customize the SPARQL query to target specific data in the EHRI Knowledge Graph.
*   **LLM Agnostic**: Supports various LLM providers (OpenAI, Anthropic, Ollama, etc.) via [LiteLLM](https://docs.litellm.ai/docs/).
*   **Interactive Visualization**: View extracted entities and their context within the original text.
*   **Export**: Download extracted data as CSV.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd EHRI-NER
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download spaCy model**:
    ```bash
    python -m spacy download en_core_web_sm
    ```

## Usage

1.  **Run the Streamlit app**:
    ```bash
    streamlit run dashboard.py
    ```

2.  **Configure API Key**:
    *   The app will prompt you for an OpenAI API key by default if `OPENAI_API_KEY` is not set in your environment.
    *   If using other providers (e.g., Anthropic, Ollama), ensure the necessary environment variables are set or configured.

3.  **Interact with the App**:
    *   **LLM Model Name**: Enter the model name you wish to use (e.g., `gpt-4o-mini`, `ollama/llama2`).
    *   **SPARQL Query**: Edit the query to fetch different data. The query must return `?record`, `?recordLabel`, and `?text`.
    *   **Run Analysis**: Click the button to start the extraction process.

## Example SPARQL Query

```sparql
PREFIX rico: <https://www.ica.org/standards/RiC/ontology#>
PREFIX ehri:  <http://lod.ehri-project-test.eu/ontology#>

SELECT ?record ?recordLabel ?text
WHERE {
    ?record a ehri:RecordSet ;
    rico:title ?recordLabel;
    rico:scopeAndContent ?text;
    FILTER(lang(?text) = "en")
}
LIMIT 25
```
