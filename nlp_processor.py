import spacy
import asyncio
import streamlit as st
import litellm
from litellm import completion, acompletion

# Monkey-patch LoggingWorker to effectively kill the background loop that causes Streamlit issues
try:
    from litellm.litellm_core_utils import logging_worker
    async def no_op_worker_loop(self):
        pass
    logging_worker.LoggingWorker._worker_loop = no_op_worker_loop
except ImportError:
    pass

# aggressive disablement of logging to prevent event loop issues in streamlit
litellm.telemetry = False
litellm.suppress_instrumentation = True
litellm.logging = False
litellm.success_callback = []
litellm.failure_callback = []

SPACY_MODEL = "en_core_web_sm"

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
