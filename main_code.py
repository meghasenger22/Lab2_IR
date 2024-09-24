# Create 'corpus' folder
import os

if not os.path.exists('corpus'):
    os.makedirs('corpus')

# Create doc1.txt
with open('corpus/doc1.txt', 'w') as f:
    f.write("Developing your Zomato business account and profile is a great way to boost your restaurant’s online reputation. Customers can easily access your menu, photos, and reviews.")

# Create doc2.txt
with open('corpus/doc2.txt', 'w') as f:
    f.write("Instagram is a popular social media platform for sharing photos and videos. Businesses use it to connect with customers and promote their brand through images and stories.")

# Create doc3.txt
with open('corpus/doc3.txt', 'w') as f:
    f.write("Swiggy is an online food delivery platform that partners with local restaurants. It provides customers with a seamless food ordering experience and quick delivery services.")

# Create doc4.txt
with open('corpus/doc4.txt', 'w') as f:
    f.write("Messenger is a communication app owned by Facebook. It allows users to send messages, make voice calls, and share media like photos and videos with friends.")

# Create doc5.txt
with open('corpus/doc5.txt', 'w') as f:
    f.write("WhatsApp is a cross-platform messaging app that enables users to send text messages, voice notes, and make video calls. It is widely used for personal and business communication.")

print("Corpus created successfully.")

#________________________________________________________________________________________________________________

# List the files in the 'corpus' directory
import os

print("Files in 'corpus' directory:")
print(os.listdir('corpus'))
#________________________________________________________________________________________________________________

import re
from collections import defaultdict
import math
import os

# List of stopwords (can be extended)
STOPWORDS = {"the", "is", "in", "and", "to", "with"}

class VectorSpaceModel:
    def __init__(self):
        self.dictionary = defaultdict(list)  # Term → [(doc_id, tf), ...]
        self.doc_lengths = {}  # Store document lengths for normalization
        self.N = 0  # Total number of documents
    
    # Preprocess each document (tokenization, stopword removal)
    def preprocess(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove non-alphabetic characters and tokenize
        tokens = re.findall(r'\b[a-z]+\b', text)
        # Remove stopwords
        tokens = [token for token in tokens if token not in STOPWORDS]
        return tokens
    
    # Add a document to the model and build postings list
    def add_document(self, doc_id, content):
        tokens = self.preprocess(content)
        term_frequencies = defaultdict(int)
        
        # Calculate term frequencies
        for token in tokens:
            term_frequencies[token] += 1
        
        # Build the dictionary and postings list
        for term, tf in term_frequencies.items():
            self.dictionary[term].append((doc_id, tf))
        
        # Calculate document length for normalization (Euclidean norm)
        doc_length = sum((1 + math.log10(tf))**2 for tf in term_frequencies.values())
        self.doc_lengths[doc_id] = math.sqrt(doc_length)
        
        self.N += 1
    
    # Calculate tf-idf for queries
    def calculate_tf_idf(self, term, tf):
        df = len(self.dictionary[term])  # Document frequency
        idf = math.log10(self.N / df)  # Inverse document frequency
        return (1 + math.log10(tf)) * idf
    
    # Rank documents based on cosine similarity
    def rank_documents(self, query):
        query_tokens = self.preprocess(query)
        query_vector = defaultdict(float)
        
        # Calculate query tf-idf (ltc)
        for token in query_tokens:
            if token in self.dictionary:
                df = len(self.dictionary[token])
                tf = query_tokens.count(token)
                query_vector[token] = (1 + math.log10(tf)) * math.log10(self.N / df)
        
        # Length of the query vector for normalization
        query_length = math.sqrt(sum(weight**2 for weight in query_vector.values()))
        
        # Score documents based on cosine similarity
        scores = defaultdict(float)
        for token, query_weight in query_vector.items():
            for doc_id, tf in self.dictionary[token]:
                doc_weight = 1 + math.log10(tf)  # lnc: log(tf) and normalize
                scores[doc_id] += query_weight * doc_weight
        
        # Normalize by document length
        for doc_id in scores:
            scores[doc_id] /= self.doc_lengths[doc_id]
        
        # Return the top 10 documents sorted by relevance
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Load all documents from the 'corpus' folder
    def load_corpus(self, folder_path):
        for idx, filename in enumerate(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                self.add_document(idx, content)
        print(f"Loaded {self.N} documents into the model.")

# Main function to run the VSM system
def main():
    # Create a VSM instance
    vsm = VectorSpaceModel()
    
    # Load the corpus
    corpus_path = 'corpus'  # Ensure this folder contains your text documents
    vsm.load_corpus(corpus_path)
    
    # Example query
    query = "Developing your Zomato business account"
    
    # Get ranked results
    results = vsm.rank_documents(query)
    
    # Output top relevant documents
    print("Top relevant documents for the query:")
    for doc_id, score in results:
        print(f"Document {doc_id}: Score = {score}")

if __name__ == "__main__":
    main()