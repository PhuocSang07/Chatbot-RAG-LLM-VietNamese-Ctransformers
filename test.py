from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from DocumentsFaissDB import DocumentsFaissDB
from ModelChain import ModelChain

vectorDB_path = './db'
documents_path = './documents'
model_name = 'all-MiniLM-L6-v2.gguf2.f16.gguf'
# model_file_path = './models/ggml-vistral-7B-chat-q5_1.gguf'
model_file_path = './models/vinallama-7b-chat_q5_0.gguf'

# template ="""<s>[INST] <<SYS>>
# Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.
# Câu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực.Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch.
# Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời
# {context}
# <</SYS>>

# {question} [/INST]
# """

template = """<|im_start|>system
Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời \n
{context}<|im_end|>\n
<|im_start|>user\n
{question}!<|im_end|>\n
<|im_start|>assistant
"""

documentsFaissDB = DocumentsFaissDB(
    db_path=vectorDB_path, 
    model_name=model_name,
    documents_path=documents_path,
    chunk_size=1024,
    chunk_overlap=128
)

model_chain = ModelChain(
    model_file_path=model_file_path,
    template=template,
    k=5,
    documentsFaissDB=documentsFaissDB
)

res = model_chain.run("2/9 Việt Nam là gì?", db_is_exist=True)
print(res)


# print()
# query = "Sự kiện ngày 2/9 ở Việt Nam"
# db = documentsFaissDB.create_db()
# docs = db.similarity_search_with_score(query)

# for i in docs:
#     print(i)
#     print()