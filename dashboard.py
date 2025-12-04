import streamlit as st
import pandas as pd
import spacy
from spacy import displacy
from SPARQLWrapper import SPARQLWrapper, JSON
from litellm import completion, acompletion
import asyncio
import os
import time
import getpass
import yaml

SPARQL_ENDPOINT = "https://lod.ehri-project-test.eu/sparql"
SPACY_MODEL = "en_core_web_sm"

# Default SPARQL Query
DEFAULT_QUERY = """
PREFIX db: <http://dbpedia.org/>
PREFIX dbp: <http://dbpedia.org/property/>
PREFIX rico: <https://www.ica.org/standards/RiC/ontology#>
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX schema: <http://schema.org/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ehri:  <http://lod.ehri-project-test.eu/ontology#>
PREFIX ehri_institutions: <http://lod.ehri-project-test.eu/institutions/>
PREFIX ehri_units: <http://lod.ehri-project-test.eu/units/>
PREFIX ehri_countries: <http://lod.ehri-project-test.eu/countries/>
PREFIX ehri_terms: <http://lod.ehri-project-test.eu/vocabularies/ehri-terms/>

SELECT ?record ?recordLabel ?text
WHERE {
    ?record a ehri:RecordSet ;
    rico:title ?recordLabel;
    rico:scopeAndContent ?text;
    FILTER(lang(?text) = "en")
}
LIMIT 25
"""

def load_config():
    """Loads configuration from config.yaml if it exists."""
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}

config = load_config()

if "OPENAI_API_KEY" not in os.environ:
    # Check if API key is in config
    if config.get("api_key"):
        os.environ["OPENAI_API_KEY"] = config["api_key"]
    else:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


@st.cache_resource
def load_spacy_model(model_name):
    """Loads the spaCy NLP model and caches it."""
    try:
        return spacy.load(model_name)
    except OSError:
        st.error(
            f"""spaCy model '{model_name}' not found. 
            Please run 'python -m spacy download {model_name}'"""
            )
        return None


@st.cache_data(ttl=3600)
def run_sparql_query(endpoint, query):
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
        records = []
        for result in results["results"]["bindings"]:
            # Safely get values, handling potential missing keys if query changes
            title = result.get("recordLabel", {}).get("value", "No Title")
            uri = result.get("record", {}).get("value", "No URI")
            text = result.get("text", {}).get("value", "")
            
            if text: # Only add if there is text to analyze
                records.append({
                    "title": title,
                    "uri": uri,
                    "text": text
                })
        return records

    except Exception as e:
        st.error(f"Error querying SPARQL endpoint: {e}")
        return []


@st.cache_data
def process_records(records, _nlp_model, model_name):
    """
    Processes a list of text records with the NLP model to find DATEs.
    """
    nlp = load_spacy_model(SPACY_MODEL)
    if nlp is None:
        return [], {}

    base_prompt = """
        You are an expert in converting dates into formatted strings.

        ##TASK##
        You will be provided with a short piece of text, which may or may not contains a particular date.
        Your task is to identify any DATE entities in the text and return them in a YYYY-MM-DD format.
        Only return the 'YYYY-MM-DD' strings.
        If no DATE entities are found, return "Not Applicable".
        Now here is the text:
    """

    all_entities = []
    record_docs = {}
    
    # Prepare tasks for async execution
    async def process_entity(ent, record_uri, record_title):
        try:
            response = await acompletion(
                model=model_name,
                messages=[
                    {"role": "system", "content": base_prompt},
                    {"role": "user", "content": ent.text}
                ],
                max_tokens=1000
            )
            formatted_date = response.choices[0].message.content.strip()
            return {
                "Record URI": record_uri,
                "Entity Title": record_title,
                "Entity Text": ent.text,
                "Entity Type": ent.label_,
                "Formatted Date": formatted_date
            }
        except Exception as e:
            # We can log the error but maybe return None to filter it out later
            # st.error(f"Error calling LLM: {e}") # Cannot call st.error from async easily without loop context issues sometimes, better to return error info
            print(f"Error calling LLM for {ent.text}: {e}")
            return None

    async def run_all_tasks(tasks):
        return await asyncio.gather(*tasks)

    tasks = []
    for record in records:
        doc = nlp(record["text"])
        record_docs[record["uri"]] = doc

        for ent in doc.ents:
            if ent.label_ in ["DATE"]:
                tasks.append(process_entity(ent, record["uri"], record["title"]))
    
    if tasks:
        # Run async tasks
        results = asyncio.run(run_all_tasks(tasks))
        # Filter out None results (errors)
        all_entities = [r for r in results if r is not None]

    return all_entities, record_docs


def main():
    st.set_page_config(page_title="EHRI Date Extractor", layout="wide")
    st.title("🗓️ EHRI Knowledge Graph Date Extractor")
    st.markdown(f"This tool queries the [EHRI SPARQL endpoint]({SPARQL_ENDPOINT}), extracts free-text descriptions, and uses SpaCy to find **DATE** entities.")

    # Load NLP model once
    nlp = load_spacy_model(SPACY_MODEL)
    if nlp is None:
        st.stop()

    # --- SIDEBAR CONTROLS ---
    st.sidebar.title("Controls")
    
    # LLM Selection
    default_model = config.get("llm_name", "gpt-5-nano")
    model_name = st.sidebar.text_input("LLM Model Name", value=default_model, help="Enter any model name supported by LiteLLM (e.g., gpt-4o, ollama/llama2, claude-3-opus)")
    
    # SPARQL Query Input
    st.sidebar.subheader("SPARQL Query")
    sparql_query = st.sidebar.text_area("Edit Query", value=DEFAULT_QUERY, height=300)

    run_button = st.sidebar.button("Run Analysis", type="primary")

    # --- TABS ---
    tab_main, tab_doc = st.tabs(["🚀 Analysis", "📚 Documentation"])

    with tab_main:
        if run_button:
            start_time = time.time()

            # 1. Fetch data
            with st.spinner(f"Executing SPARQL query..."):
                records = run_sparql_query(SPARQL_ENDPOINT, sparql_query)

            if not records:
                st.warning("No data found. The SPARQL endpoint might be down or the query returned no results.")
                st.stop()

            st.info(f"Fetched {len(records)} records.")

            # 2. Process data
            with st.spinner(f"Analyzing text with spaCy ({SPACY_MODEL}) and {model_name}..."):
                all_entities, processed_docs = process_records(records, SPACY_MODEL, model_name)

            total_time = time.time() - start_time

            if not all_entities:
                st.warning("Analysis complete, but no DATE or TIME entities were found in the processed records.")
                st.stop()

            st.success(f"Found {len(all_entities)} entities in {len(records)} records (in {total_time:.2f}s).")

            df = pd.DataFrame(all_entities)

            # 3. Display Results in Sub-Tabs
            subtab1, subtab2, subtab3 = st.tabs(["📊 Summary & Export", "📈 Entity Frequency", "📄 Detailed View"])
            
            with subtab1:
                st.subheader("Extracted Entities")
                st.dataframe(df, width='stretch')
                
                # Export button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name="ehri_ner_entities.csv",
                    mime="text/csv",
                )
                
            with subtab2:
                st.subheader("Most Common Entities")
                
                # Check if DataFrame is not empty
                if not df.empty:
                    # Top 20 most common entity texts
                    entity_counts = df["Entity Text"].value_counts().head(20)
                    st.bar_chart(entity_counts)
                else:
                    st.info("No entities to display.")
                
            with subtab3:
                st.subheader("Detailed Entity View per Record")
                st.markdown("This view highlights the `DATE` and `TIME` entities found in each record's text.")
                
                # Use the pre-processed docs for visualization
                for record in records:
                    doc = processed_docs.get(record["uri"])
                    if doc:
                        # Filter for DATE/TIME entities
                        entities_to_show = [ent for ent in doc.ents if ent.label_ in ["DATE", "TIME"]]
                        
                        if entities_to_show:
                            with st.expander(f"Record: {record['uri']}"):
                                st.markdown(f"**Original Text:**")
                                st.caption(record['text'])
                                
                                st.markdown(f"**Highlighted Entities:**")
                                # Create a new Doc with the same tokens and correct boolean spaces
                                words = [t.text for t in doc]
                                spaces = [True if t.whitespace_ else False for t in doc]
                                filtered_doc = spacy.tokens.Doc(doc.vocab, words=words, spaces=spaces)
                                # Recreate Span objects for the new Doc (can't reuse spans from the old Doc)
                                spans = [spacy.tokens.Span(filtered_doc, ent.start, ent.end, label=ent.label_) for ent in entities_to_show]
                                filtered_doc.ents = spans
                                
                                html = displacy.render(filtered_doc, style="ent")
                                st.write(html, unsafe_allow_html=True)

    with tab_doc:
        st.markdown("""
        ## 📚 Documentation

        ### Overview
        This application allows you to query the EHRI Knowledge Graph using SPARQL, extract text from the results, and use Named Entity Recognition (NER) to identify dates. It then uses an LLM to normalize these dates into a standard format.

        ### Features
        1.  **Flexible SPARQL Querying**: You can edit the SPARQL query directly in the sidebar to target specific data.
        2.  **LLM Agnostic**: The app uses [LiteLLM](https://docs.litellm.ai/docs/) to support various LLM providers (OpenAI, Anthropic, Ollama, etc.).
        3.  **Interactive Visualization**: Explore extracted entities and view them highlighted in the original text.

        ### Usage
        1.  **Select LLM**: Enter the model name in the sidebar. Default is `gpt-4o-mini`.
            *   For OpenAI: `gpt-4o`, `gpt-3.5-turbo`
            *   For Ollama (local): `ollama/llama2`
        2.  **Edit Query**: Modify the SPARQL query in the sidebar. Ensure your query returns `?record`, `?recordLabel`, and `?text` variables.
        3.  **Run Analysis**: Click the "Run Analysis" button.

        ### Example Queries

        **Default Query (Records with English scope and content):**
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

        **Query for a specific subject (e.g., "Ghettos"):**
        ```sparql
        PREFIX rico: <https://www.ica.org/standards/RiC/ontology#>
        PREFIX ehri:  <http://lod.ehri-project-test.eu/ontology#>
        PREFIX ehri_terms: <http://lod.ehri-project-test.eu/vocabularies/ehri-terms/>

        SELECT ?record ?recordLabel ?text
        WHERE {
            ?record a ehri:RecordSet ;
            rico:title ?recordLabel;
            rico:scopeAndContent ?text;
            rico:hasOrHadSubject ehri_terms:267 ; 
            FILTER(lang(?text) = "en")
        }
        LIMIT 20
        ```
        *(Note: `ehri_terms:267` corresponds to "Ghettos")*
        """)

if __name__ == "__main__":
    main()