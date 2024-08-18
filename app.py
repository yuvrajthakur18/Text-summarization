import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

## Streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")

# Custom CSS for styling the title
st.markdown(
    """
    <style>
    .custom-title {
        border: 2px solid pink;
        background-color: blackfade;
        color: white;
        padding: 10px;
        text-align: center;
        transition: all 0.3s ease-in-out;
    }
    .custom-title:hover {
        transform: scale(1.1);
        color: red;
        border-color: red;
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title with custom styling
st.markdown('<h1 class="custom-title">🦜 Summarize Text From YT or Website</h1>', unsafe_allow_html=True)

st.subheader('Enter URL of YouTube or Website.')

## Get the Groq API Key and url(YT or website) to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

## Gemma Model Using Groq API
llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YT video URL or a website URL")
    else:
        try:
            with st.spinner("Waiting..."):
                ## loading the website or yt video data
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url], ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                docs = loader.load()

                ## Chain For Summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")



# import validators,streamlit as st
# from langchain.prompts import PromptTemplate
# from langchain_groq import ChatGroq
# from langchain.chains.summarize import load_summarize_chain
# from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader


# ## sstreamlit APP
# st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
# st.title("🦜 Summarize Text From YT or Website")
# st.subheader('Enter URL of YouTube or Website.')



# ## Get the Groq API Key and url(YT or website)to be summarized
# with st.sidebar:
#     groq_api_key=st.text_input("Groq API Key",value="",type="password")

# generic_url=st.text_input("URL",label_visibility="collapsed")

# ## Gemma Model USsing Groq API
# llm =ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

# prompt_template="""
# Provide a summary of the following content in 300 words:
# Content:{text}

# """
# prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

# if st.button("Summarize the Content from YT or Website"):
#     ## Validate all the inputs
#     if not groq_api_key.strip() or not generic_url.strip():
#         st.error("Please provide the information to get started")
#     elif not validators.url(generic_url):
#         st.error("Please enter a valid Url. It can may be a YT video utl or website url")

#     else:
#         try:
#             with st.spinner("Waiting..."):
#                 ## loading the website or yt video data
#                 if "youtube.com" in generic_url:
#                     loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
#                 else:
#                     loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
#                                                  headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
#                 docs=loader.load()

#                 ## Chain For Summarization
#                 chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
#                 output_summary=chain.run(docs)

#                 st.success(output_summary)
#         except Exception as e:
#             st.exception(f"Exception:{e}")
                    