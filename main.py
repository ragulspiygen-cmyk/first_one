from dotenv import load_dotenv
import os
import pandas as pd


from note_engine import note_engine
from prompts import new_prompt, instruction_str, context


from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.groq import Groq
from llama_index.experimental.query_engine.pandas import PandasQueryEngine
from llama_index.readers.file import PDFReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


load_dotenv()


Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


GROQ_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_KEY:
    raise RuntimeError("❌ GROQ_API_KEY missing in .env")

Settings.llm = Groq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_KEY,
)

# ======================================================
# POPULATION CSV (STRUCTURED DATA)
# ======================================================
population_path = os.path.join("data", "population.csv")
if not os.path.exists(population_path):
    raise FileNotFoundError("❌ population.csv not found")

population_df = pd.read_csv(population_path)

population_query_engine = PandasQueryEngine(
    df=population_df,
    verbose=True,
    instruction_str=instruction_str,
)

population_query_engine.update_prompts({
    "pandas_prompt": new_prompt
})


pdf_path = os.path.join("data", "India.pdf")
if not os.path.exists(pdf_path):
    raise FileNotFoundError("❌ India.pdf not found")

reader = PDFReader()
pdf_documents = reader.load_data(pdf_path)

pdf_index = VectorStoreIndex.from_documents(pdf_documents)

pdf_query_engine = pdf_index.as_query_engine(
    similarity_top_k=6
)


def dispatch(prompt: str):
    lp = prompt.lower().strip()

    # ---------- NOTES ----------
    if "note" in lp or "take a note" in lp:
        import re
        m = re.search(r'["\'](.+?)["\']', prompt)
        note_text = m.group(1) if m else prompt
        return f" {note_engine.fn(note_text)}"

    if "population" in lp:
        return population_query_engine.query(prompt)

   
    return pdf_query_engine.query(
        f"Answer strictly using India.pdf content:\n{prompt}"
    )


print("\n🇮🇳 India Knowledge Assistant (PDF-powered)")
print("Ask ANY question from India.pdf")
print("Examples:")
print("  What is India's average annual GDP?")
print("  Explain Indian automotive industry growth")
print("  What influences Indian climate?")
print("  What is India's coastline length?")
print("  Population of Tamil Nadu")
print("  take a note \"India became independent in 1947\"")

while True:
    prompt = input("\nenter a prompt (q to quit): ").strip()
    if prompt.lower() == "q":
        break
    try:
        print(dispatch(prompt))
    except Exception as e:
        print("❌ Error:", e)

