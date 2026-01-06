from SPARQLWrapper import SPARQLWrapper, JSON
import streamlit as st

SPARQL_ENDPOINT = "https://lod.ehri-project-test.eu/sparql"

@st.cache_data(ttl=3600)
def fetch_options(query):
    """Helper to fetch options for dropdowns."""
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        options = {}
        for result in results["results"]["bindings"]:
            uri = result["s"]["value"]
            name = result.get("name", {}).get("value", uri.split("/")[-1])
            options[name] = uri
        return options
    except Exception as e:
        return {}

def get_countries():
    query = """
    PREFIX rico: <https://www.ica.org/standards/RiC/ontology#>
    SELECT DISTINCT ?s ?name
    WHERE {
      ?s rico:name ?name .
      FILTER(STRSTARTS(STR(?s), "http://lod.ehri-project-test.eu/countries/"))
    }
    ORDER BY ?name
    """
    return fetch_options(query)

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
