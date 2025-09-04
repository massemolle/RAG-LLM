from embedding import *
import torch
from transformers import pipeline
from huggingface_hub import repo_exists
import time
class RAG():

    def __init__(self, method = None, k=5, path = None, pipeline_model = "Felladrin/Smol-Llama-101M-Chat-v1", device="cpu"):
        self.k=k
        self.path = path
        self.pipe_model = pipeline_model
        self.pipe = get_pipeline(self.pipe_model)

        if method == 'BM25':
            self.model = BM25(k=self.k, path=self.path)
        elif method == 'BERT':
            self.model = BERT(k=self.k, device = "cuda:0" if torch.cuda.is_available() else "cpu", path=self.path)
        else : 
            raise NameError(f"Unknow '{method}' embedding method provided. Correct values are 'BERT' and 'BM25'")
        
    
    def answer(self,query, doc = None):
        """
        From an initial query, uses the LLM to get the answer to the query, with the context from the documents
        """
        context_list = self.model.retrieve(query, path = self.path, doc = doc)['doc']
        

        pipe_input = get_message(self.pipe_model, query, context_list)
        print("Sending message")
        print(pipe_input)
        try:
            
            return self.pipe(pipe_input)[0]['generated_text'][1]['content'].split('\n')[0]
        except : 
            return self.pipe(pipe_input)[0]['generated_text'][1]['content']


def get_pipeline(p_model, device = 'cuda:0'):  
    """
    This load the LLM model through the HuggingFace pipeline, and saves it locally. If already saved, the model is loaded locally.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if p_model in os.listdir('./model_w'):
            p= pipeline("text-generation", model="./model_w/"+p_model,device=device, trust_remote_code = True, use_fast=False)
    else :
        p = pipeline("text-generation", model=p_model,device=device, trust_remote_code = True, use_fast=False)
        p.save_pretrained('./model_w/'+p_model)
    return p

def get_message(p_model, query, context_list):
    """
    This function takes query and context, and formats it into an element to plug into pipe
    """
    
    try : 
        
        if len(context_list) != 0 :
            context = '\nUse these informations to answer the question : '
            for e in context_list:
                context = context+'\n'+e

            context=context[0]
            message = [
                {"role": "user", "content": str(query)+str(context)},
                ]
        else : 
        
            message = [
                    {"role": "user", "content": str(query)},
                    ]
        return message
    except TypeError:
        return [
                    {"role": "user", "content": str(query)},
                    ]
    except ValueError:
        return str(query)+str(context)
        
    
def get_llm(llm_path):
    """
    Function used to check if the model from the 'Other' choice in the front exists
    """
    try : 
        if repo_exists(llm_path):   
            return llm_path
        else :
            return "Felladrin/Smol-Llama-101M-Chat-v1"
    except :
        return "Felladrin/Smol-Llama-101M-Chat-v1"
    
def get_model_list():
    """
    Lists all hugging face models present locally on the server, in the 'model_w' folder
    """
    l = []
    for team in os.listdir('./model_w'):
        for model in os.listdir(os.path.join("./model_w", team)):
            l.append(team+'/'+model)
    l.append('Other')
    return l

def list_devices():
    """
    Checks for devices present, and formats the answer as a list
    """
    if torch.cuda.is_available():
        return ['CPU'] + [f'GPU:{i} ({torch.cuda.get_device_name(i)})' for i in range(torch.cuda.device_count())]
    else:
        return ['CPU']