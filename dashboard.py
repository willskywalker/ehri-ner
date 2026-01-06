import streamlit as st
import pandas as pd
import time

import os
import yaml
from spacy import displacy
import spacy

# Import refactored modules
from sparql_client import SPARQL_ENDPOINT, run_sparql_query, get_countries
from nlp_processor import SPACY_MODEL, load_spacy_model, process_records

# Default SPARQL Query (keep in dashboard or move? It's UI-related default value, fine here)
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

def main():
    st.set_page_config(page_title="EHRI Date Extractor", layout="wide")
    st.title("🗓️ EHRI Knowledge Graph Date Extractor")

    # API Key Handling in Dashboard
    if "OPENAI_API_KEY" not in os.environ:
        if config.get("api_key"):
            os.environ["OPENAI_API_KEY"] = config["api_key"]
        else:
            api_key_input = st.text_input("Enter OpenAI API Key", type="password", help="The app needs an OpenAI API key to function. Please enter it here.")
            if api_key_input:
                os.environ["OPENAI_API_KEY"] = api_key_input
                st.rerun() # Rerun to pick up the key
            else:
                st.warning("Please enter your OpenAI API Key to proceed.")
                st.stop()

    st.markdown(f"This tool queries the [EHRI SPARQL endpoint]({SPARQL_ENDPOINT}), extracts free-text descriptions, and uses SpaCy to find **DATE** entities.")

    # Load NLP model once
    nlp = load_spacy_model(SPACY_MODEL)
    if nlp is None:
        st.stop()

    # --- SIDEBAR CONTROLS ---
    st.sidebar.title("Controls")
    
    # LLM Selection
    default_model = config.get("llm_name", "gpt-4o-mini")
    model_name = st.sidebar.text_input("LLM Model Name", value=default_model, help="Enter any model name supported by LiteLLM (e.g., gpt-4o, ollama/llama2, claude-3-opus)")

    # --- QUERY INTERFACE ---
    st.sidebar.subheader("Query Configuration")
    
    # Fetch options
    available_countries = get_countries()

    query_mode = st.sidebar.radio("Query Mode", ["Structured Builder", "Raw SPARQL"], horizontal=True)

    final_query = DEFAULT_QUERY # Default fallback

    if query_mode == "Structured Builder":
        with st.sidebar.expander("Filters", expanded=True):
            selected_countries = st.multiselect("Select Countries", options=list(available_countries.keys()))
            keywords = st.text_input("Keywords (in Title or Text)")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                limit = st.slider("Max Results (LIMIT)", min_value=1, max_value=500, value=25)
            with col2:
                limit_input = st.number_input("Count", min_value=1, max_value=1000, value=limit, label_visibility="hidden")
                if limit_input != limit:
                    limit = limit_input

        # Construct Query Logic (Always updates final_query)
        where_clauses = [
            "?record a ehri:RecordSet ;",
            "rico:title ?recordLabel;",
            "rico:scopeAndContent ?text."
        ]
        
        filters = ['FILTER(lang(?text) = "en")']
        
        # Country Filter
        if selected_countries:
            country_uris = [f"<{available_countries[c]}>" for c in selected_countries]
            # Path: Record -> Holder -> Location -> (contained by)* -> Country
            where_clauses.append("?record rico:hasOrHadHolder ?holder .")
            where_clauses.append("?holder rico:agentHasOrHadLocation ?loc .")
            where_clauses.append("?loc rico:isOrWasContainedBy* ?country .")
            filters.append(f"FILTER(?country IN ({', '.join(country_uris)}))")

        # Keyword Filter
        if keywords:
            # Sanitize input (basic)
            safe_kw = keywords.replace('"', '\\"')
            filters.append(f'FILTER(REGEX(?text, "{safe_kw}", "i") || REGEX(?recordLabel, "{safe_kw}", "i"))')

        # Assemble
        final_query = f"""
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

        SELECT DISTINCT ?record ?recordLabel ?text
        WHERE {{
            {' '.join(where_clauses)}
            {' '.join(filters)}
        }}
        LIMIT {limit}
        """

    else: # Raw SPARQL
        sparql_query = st.sidebar.text_area("Edit Query", value=DEFAULT_QUERY, height=300)
        final_query = sparql_query

    # Run Button in Sidebar
    if st.sidebar.button("Run Analysis", type="primary"):
        st.session_state['generated_query'] = final_query
        st.session_state['should_run'] = True

    # --- TABS ---
    tab_main, tab_doc = st.tabs(["Analysis", "Documentation"])

    with tab_main:
        if st.session_state.get('should_run', False):
            # Reset run TRIGGER
            st.session_state['should_run'] = False
            
            # Use query from session state to ensure stability
            query_to_run = st.session_state.get('generated_query', final_query)

            start_time = time.time()

            # 1. Fetch data
            with st.spinner(f"Executing SPARQL query..."):
                # Show query for debug/transparency
                with st.expander("View Executed SPARQL Query"):
                    st.code(query_to_run, language="sparql")
                records = run_sparql_query(SPARQL_ENDPOINT, query_to_run)

            if not records:
                st.warning("No data found. The SPARQL endpoint might be down or the query returned no results.")
                # Don't stop here completely so they can try again, just return
            else:
                st.info(f"Fetched {len(records)} records.")

                # 2. Process data
                with st.spinner(f"Analyzing text with spaCy ({SPACY_MODEL}) and {model_name}..."):
                    all_entities, processed_docs = process_records(records, SPACY_MODEL, model_name)

                total_time = time.time() - start_time

                if not all_entities:
                    st.warning("Analysis complete, but no DATE or TIME entities were found in the processed records.")
                else:
                    st.success(f"Found {len(all_entities)} entities in {len(records)} records (in {total_time:.2f}s).")

                    df = pd.DataFrame(all_entities)

                    # 3. Display Results in Sub-Tabs
                    subtab1, subtab2, subtab3 = st.tabs(["📊 Summary & Export", "📈 Entity Frequency", "📄 Detailed View"])
                    
                    with subtab1:
                        st.subheader("Extracted Entities")
                        st.dataframe(df, width=1000)
                        
                        # Export button
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download data as CSV",
                            data=csv,
                            file_name="ehri_ner_entities.csv",
                            mime="text/csv",
                        )
                        
                    with subtab2:
                        st.subheader("Date Analysis")
                        
                        if not df.empty:
                            valid_dates = df[df["Formatted Date"].str.match(r'^\d{4}-\d{2}-\d{2}$', na=False)].copy()
                            
                            if not valid_dates.empty:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("### Most Common Dates")
                                    date_counts = valid_dates["Formatted Date"].value_counts().head(20)
                                    st.bar_chart(date_counts)
                                
                                with col2:
                                    st.markdown("### Most Common Years")
                                    valid_dates["Year"] = pd.to_datetime(valid_dates["Formatted Date"], errors='coerce').dt.year
                                    year_counts = valid_dates["Year"].dropna().astype(int).value_counts().head(20)
                                    st.bar_chart(year_counts)
                            else:
                                st.info("No valid dates found.")
                        else:
                            st.info("No entities to display.")
                        
                    with subtab3:
                        st.subheader("Detailed Entity View per Record")
                        st.markdown("This view highlights the `DATE` and `TIME` entities found in each record's text.")
                        
                        for record in records:
                            doc = processed_docs.get(record["uri"])
                            if doc:
                                entities_to_show = [ent for ent in doc.ents if ent.label_ in ["DATE", "TIME"]]
                                
                                if entities_to_show:
                                    with st.expander(f"Record: {record['uri']}"):
                                        st.markdown(f"**Original Text:**")
                                        st.caption(record['text'])
                                        
                                        st.markdown(f"**Highlighted Entities:**")
                                        words = [t.text for t in doc]
                                        spaces = [True if t.whitespace_ else False for t in doc]
                                        filtered_doc = spacy.tokens.Doc(doc.vocab, words=words, spaces=spaces)
                                        spans = [spacy.tokens.Span(filtered_doc, ent.start, ent.end, label=ent.label_) for ent in entities_to_show]
                                        filtered_doc.ents = spans
                                        
                                        html = displacy.render(filtered_doc, style="ent")
                                        st.write(html, unsafe_allow_html=True)

    with tab_doc:
        st.markdown("""
        ## Documentation
        
        ### Query Modes
        - **Structured Builder**: Easily filter by Country and Keywords without writing SPARQL.
        - **Raw SPARQL**: Full control for advanced queries.

        ### Overview
        This application allows you to query the EHRI Knowledge Graph using SPARQL, extract text from the results, and use Named Entity Recognition (NER) to identify dates. It then uses an LLM to normalize these dates into a standard format.
        """)

if __name__ == "__main__":
    main()