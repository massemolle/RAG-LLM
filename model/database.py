from pypdf import PdfReader
import os
import docx
import re
import pdfplumber
import fitz  # PyMuPDF
import nltk
nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize

def cleanDocx(filename):
    """
    Used to load docx documennts, and split the pages into phrases
    """
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        if len(para.text) != 0 : 
            if para.text.count('.')>1 :
                fullText = fullText+para.text.split('.')
            else :
                fullText.append(para.text) 
    
    return fullText

def doc2Text(path):
    """
    Main file loading function, if you want to add other file types it's here
    """
    docTextList = []
    for file in os.listdir(path) : 

        f_type = file.split(".")[-1]

        if f_type == "pdf" : 
            p = pdf_to_phrases(os.path.join(path, file))
                    
            
            docTextList.append([p,file])

        elif f_type =='docx' : 
            T = cleanDocx(os.path.join(path, file))

            docTextList.append([T, file])
    return docTextList

def pdf_to_phrases(pdf_path):
    """
    Reads a PDF file and returns a list of phrases (sentences).
    
    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        List[str]: List of extracted phrases.
    """
    # Step 1: Read text from the PDF
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    # Step 2: Split the text into phrases using sentence tokenization
    phrases = sent_tokenize(full_text)

    clean_phrases = [phrase.replace('\n', ' ').strip() for phrase in phrases]

    return clean_phrases


# After this, all functions are deprecated and can (I think) be removed

def loadWiki(): 
    """
    Was used to load a wikipedia dataset for training purposes. Currently unused
    """
    #ds = load_dataset("wikimedia/wikipedia","20231101.fr")['train']
    ds=[]
    corpus = []
    i =0
    for page in ds :    #2564646 pages
        text = page['text'] 
        corpus = corpus + text.split('\n')
        i+=1
        if i==1000 : 
            break
    return corpus


def cleanPdf(text) : 
    phrases = text.split(".")
    
    for i, p in enumerate(phrases) :
        p = p.lower()
        p=p.replace('-\n', '')
        p=p.replace('\n','')
        phrases[i] = p

    print(phrases)

def saveEmbd(vector : list,line : list, doc : list, path : str) -> None :   
    F = []
    for i in range(len(vector)):
        F.append({"vec":vector[i], "line":line[i],"doc":doc[i]} )
    
    with open(path,'w') as f : 
        f.write('\n'.join(F))   

def loadEmbd(path : list) -> list : 
    vecList = []

    with open(path, 'r') as f :
        for line in f : 
            x = line[:-1]   # Remove the \n
            vecList.append(x)
    return vecList  # Format : list(dict)

def removeN(doc):
    """
    doc : single file as a single text or list of text
    return -> list of phrases
    """

    Text = doc.split('\n')
    if len(Text) ==1 :  # Initial case where we don't have \n and thus is a 1 element list
        return Text
    else : 
        L=[]
        for e in Text :
            clean_e = removeN(e)
            L = L + clean_e
        return L

def docToCorpus(doc) :
    spl = removeN(doc)
    spl = [e.lower() for e in spl]
    L=[]
    for e in spl :  
        x = e.split('.')

        for phrase in x :
            phrase = re.sub(r'[^\w\s]','',phrase)
            if phrase not in ['', ' '] and '\t' not in phrase:
                L.append(phrase)
    
    return L

def extract_data(feed):
    data = []
    with pdfplumber.open(feed) as pdf:
        for page in pdf.pages:
            t = page.extract_text(x_tolerance=1).split('\n')
            for para in t : 
                para = para.split('. ')
                if len(para)==0:
                    data.append(para)
                else : 
                    data = data + para
    return data

 