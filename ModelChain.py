from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from DocumentsFaissDB import DocumentsFaissDB


class ModelChain:
    def __init__(self, model_file_path, template, k, documentsFaissDB: DocumentsFaissDB):
        self.model_file_path = model_file_path
        self.template = template
        self.k = k
        self.documentsFaissDB = documentsFaissDB
        
    def load_model(self):
        llm = CTransformers(
            model = self.model_file_path,
            model_type = 'llama',
            # max_new_tokens = 1024,
            temperature = 0.01,
        )

        return llm

    def create_prompt(self, template):
        prompt = PromptTemplate(
            template=template,
            input_variables=['context', 'question'],
        )

        return prompt

    def create_chain(self, llm, prompt, db):
        chain = RetrievalQA.from_chain_type(
            llm = llm,
            chain_type = 'stuff',
            retriever = db.as_retriever( search_kwargs={"k": self.k}),
            return_source_documents = True,
            chain_type_kwargs = {'prompt': prompt},
        )

        return chain
    
    def run(self, question, db_is_exist):
        if(db_is_exist):
            db = self.documentsFaissDB.load_db()
        else:
            db = self.documentsFaissDB.create_db()
        llm = self.load_model()
        prompt = self.create_prompt(self.template)
        llm_chain = self.create_chain(llm, prompt, db)
        response = llm_chain.invoke({"query": question})
        
        return response