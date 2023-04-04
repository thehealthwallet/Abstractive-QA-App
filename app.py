
import pinecone
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import BartTokenizer, BartForConditionalGeneration


class BartGenerator:
    def __init__(self, model_name):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.generator = BartForConditionalGeneration.from_pretrained(model_name)

    def tokenize(self, query, max_length=1024):
        inputs = self.tokenizer([query], max_length=max_length, return_tensors="pt")
        return inputs

    def generate(self, query, min_length=20, max_length=40):
        inputs = self.tokenize(query)
        ids = self.generator.generate(inputs["input_ids"], num_beams=1, min_length=int(min_length), max_length=int(max_length))
        answer = self.tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return answer
    
@st.cache_resource
def init_models():
    retriever = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base")  
    generator = BartGenerator("vblagoje/bart_lfqa")
    return retriever, generator

PINECONE_KEY = "b9447d59-d2e5-466c-a04d-e30d91ff6db2"

@st.cache_resource
def init_pinecone():
    pinecone.init(api_key=PINECONE_KEY, environment="eu-west4-gcp")  # get a free api key from app.pinecone.io
    return pinecone.Index("meq-quick-start-guide")

retriever, generator = init_models()
index = init_pinecone()

def display_answer(answer):
    return st.markdown(f"""
    <div class="container-fluid">
        <div class="row align-items-start">
             <div  class="col-md-12 col-sm-12">
                 <span style="color: #808080;">
                    {answer}
                 </span>
             </div>
        </div>
     </div>
        """, unsafe_allow_html=True)

def display_context(title, context, url):
    return st.markdown(f"""
    <div class="container-fluid">
        <div class="row align-items-start">
             <div  class="col-md-12 col-sm-12">
                 <a href={url}>{title}</a>
                 <br>
                 <span style="color: #808080;">
                     <small>{context}</small>
                 </span>
             </div>
        </div>
     </div>
        """, unsafe_allow_html=True)

st.write("""
# Abstractive QA Bot
* Features - Top K, genarator parameters
Limited capabilities, genuinely abstarctive from wikipedia history fetching from pinecone
Exclusively for thehealthwallet.com
""")

st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
""", unsafe_allow_html=True)

def format_query(query, context):
    context = [f"<P> {m['metadata']['passage_text']}" for m in context]
    context = " ".join(context)
    query = f"question: {query} context: {context}"
    return query
    
st.sidebar.subheader("Retriever parameters:")
top_k = st.sidebar.slider("Top K", min_value=1, max_value=10, value=10)

st.sidebar.subheader("Generator parameters:")
min_length = st.sidebar.slider("Minimum Length", min_value=1, max_value=50, value=20)
max_length = st.sidebar.slider("Maximum Length", min_value=1, max_value=100, value=50)

query = st.text_input("Search!", "")

if query != "":
    with st.spinner(text="Fetching context passages 🚀🚀🚀"):
        xq = retriever.encode([query]).tolist()
        xc = index.query(xq, top_k=int(top_k), include_metadata=True)
        query = format_query(query, xc["matches"])

    with st.spinner(text="Generating answer ✍️✍️✍️"):
        answer = generator.generate(query, min_length=min_length, max_length=max_length)

    st.write("#### Generated answer:")
    display_answer(answer)
    st.write("#### Answer was generated based on the following passages:")

    for m in xc["matches"]:
        title = m["metadata"]["article_title"]
        url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
        context = m["metadata"]["passage_text"]
        display_context(title, context, url)
