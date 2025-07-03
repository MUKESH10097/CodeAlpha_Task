import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

nltk.download('stopwords')

# Sample FAQs and answers 
faqs = [
    "How do I place an order?",
    "What payment methods do you accept?",
    "How can I track my order?",
    "Can I cancel my order after placing it?",
    "What is your return policy?",
    "When will I receive my refund?",
    "Do you offer free shipping?",
    "How do I change my delivery address?",
    "What if I receive a damaged or wrong product?",
    "How do I contact customer support?"
]


answers = [
    "To place an order, browse our catalog, add items to your cart, and proceed to checkout to complete payment.",
    "We accept credit/debit cards, UPI, net banking, and digital wallets like Paytm, PhonePe, and Google Pay.",
    "After placing an order, go to 'My Orders' in your account to view the live tracking status.",
    "Yes, orders can be canceled within 30 minutes of placing them, or before they are shipped.",
    "You can return most products within 7 days of delivery. Items must be unused and in original packaging.",
    "Refunds are usually processed within 5-7 business days after the return is received and approved.",
    "Yes, we offer free shipping on all orders above ₹499. Charges apply for lower-value orders.",
    "You can update your delivery address from your account before the order is shipped.",
    "If you receive a damaged or incorrect product, raise a return request from 'My Orders' within 3 days.",
    "For help, go to the Help Center in the app or website, or call our 24x7 customer service line."
]


stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = ''.join(c for c in text if c not in string.punctuation)
    tokens = wordpunct_tokenize(text)
    return ' '.join([word for word in tokens if word not in stop_words])

processed_faqs = [preprocess(q) for q in faqs]
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(processed_faqs)

def get_response(user_question):
    user_q = preprocess(user_question)
    user_vec = vectorizer.transform([user_q])
    similarity = cosine_similarity(user_vec, faq_vectors)
    idx = similarity.argmax()
    return answers[idx]

# Chat UI with Gradio
def chatbot_interface(user_input, history):
    return get_response(user_input)

gr.ChatInterface(chatbot_interface, title="FAQ Chatbot").launch()

