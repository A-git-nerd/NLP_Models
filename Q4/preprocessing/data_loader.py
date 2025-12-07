import torch
from torch.utils.data import Dataset
import re

class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab, max_len=50):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
    
    def text_to_indices(self, text, vocab):
        words = text.split()
        indices = [vocab.get('<SOS>', 2)]
        
        for word in words:
            indices.append(vocab.get(word, vocab.get('<UNK>', 1)))
        
        indices.append(vocab.get('<EOS>', 3))
        
        if len(indices) < self.max_len:
            indices += [vocab.get('<PAD>', 0)] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        
        return indices
    
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        
        src_indices = self.text_to_indices(src_text, self.src_vocab)
        tgt_indices = self.text_to_indices(tgt_text, self.tgt_vocab)
        
        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(tgt_indices, dtype=torch.long)

def build_vocab(texts, min_freq=1):
    word_freq = {}
    for text in texts:
        for word in text.split():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def create_sample_dataset():
    english_sentences = [
        "hello how are you",
        "what is your name",
        "i am a student",
        "this is a book",
        "i like to read",
        "the weather is nice today",
        "where are you going",
        "i want to learn",
        "can you help me",
        "thank you very much",
        "good morning",
        "good night",
        "how old are you",
        "what time is it",
        "i am fine",
        "see you later",
        "nice to meet you",
        "have a good day",
        "i love pakistan",
        "this is my friend",
        "we are going to school",
        "he is a teacher",
        "she is a doctor",
        "they are students",
        "the food is delicious",
        "i am hungry",
        "i am thirsty",
        "please sit down",
        "how much is this",
        "where is the market",
        "i need help",
        "what are you doing",
        "i am working",
        "do you speak english",
        "yes i do",
        "no i do not",
        "maybe tomorrow",
        "i am sorry",
        "excuse me",
        "you are welcome",
        "it is very hot",
        "it is very cold",
        "i like tea",
        "i like coffee",
        "what is this",
        "this is a pen",
        "that is a car",
        "how many books",
        "i have two brothers",
        "she has one sister",
        "we live in lahore",
        "they work in karachi",
        "i study at university",
        "the book is on the table",
        "the cat is sleeping",
        "the dog is running",
        "birds are flying",
        "sun is shining",
        "moon is bright",
        "stars are beautiful",
        "water is essential",
        "food is ready",
        "door is open",
        "window is closed",
        "i am happy",
        "he is sad",
        "she is angry",
        "we are excited",
        "they are tired",
        "i can swim",
        "he can drive",
        "she can sing",
        "we can dance",
        "i will go tomorrow",
        "he went yesterday",
        "she is coming now",
        "we are waiting here",
        "they live there",
        "come here please",
        "go there now",
        "stay here tonight",
        "leave tomorrow morning",
        "return next week",
        "i eat rice",
        "he drinks milk",
        "she cooks food",
        "we play cricket",
        "they watch movies",
        "the sky is blue",
        "grass is green",
        "snow is white",
        "coal is black",
        "blood is red",
        "i am learning urdu",
        "he is teaching english",
        "she is reading a book",
        "we are writing letters",
        "they are making tea",
        "my father is kind",
        "my mother is caring",
        "my brother is smart",
        "my sister is beautiful",
        "my friend is loyal"
    ]
    
    urdu_sentences = [
        "سلام کیسے ہو",
        "تمہارا نام کیا ہے",
        "میں ایک طالب علم ہوں",
        "یہ ایک کتاب ہے",
        "مجھے پڑھنا پسند ہے",
        "آج موسم اچھا ہے",
        "تم کہاں جا رہے ہو",
        "میں سیکھنا چاہتا ہوں",
        "کیا تم میری مدد کر سکتے ہو",
        "بہت شکریہ",
        "صبح بخیر",
        "شب بخیر",
        "تمہاری عمر کتنی ہے",
        "کتنے بجے ہیں",
        "میں ٹھیک ہوں",
        "بعد میں ملتے ہیں",
        "آپ سے مل کر خوشی ہوئی",
        "آپ کا دن اچھا گزرے",
        "مجھے پاکستان سے محبت ہے",
        "یہ میرا دوست ہے",
        "ہم سکول جا رہے ہیں",
        "وہ ایک استاد ہے",
        "وہ ایک ڈاکٹر ہے",
        "وہ طالب علم ہیں",
        "کھانا لذیذ ہے",
        "مجھے بھوک لگی ہے",
        "مجھے پیاس لگی ہے",
        "براہ کرم بیٹھ جائیں",
        "یہ کتنے کا ہے",
        "بازار کہاں ہے",
        "مجھے مدد چاہیے",
        "تم کیا کر رہے ہو",
        "میں کام کر رہا ہوں",
        "کیا تم انگریزی بولتے ہو",
        "ہاں میں بولتا ہوں",
        "نہیں میں نہیں بولتا",
        "شاید کل",
        "مجھے افسوس ہے",
        "معاف کیجیے",
        "خوش آمدید",
        "بہت گرم ہے",
        "بہت ٹھنڈا ہے",
        "مجھے چائے پسند ہے",
        "مجھے کافی پسند ہے",
        "یہ کیا ہے",
        "یہ ایک قلم ہے",
        "وہ ایک گاڑی ہے",
        "کتنی کتابیں ہیں",
        "میرے دو بھائی ہیں",
        "اس کی ایک بہن ہے",
        "ہم لاہور میں رہتے ہیں",
        "وہ کراچی میں کام کرتے ہیں",
        "میں یونیورسٹی میں پڑھتا ہوں",
        "کتاب میز پر ہے",
        "بلی سو رہی ہے",
        "کتا دوڑ رہا ہے",
        "پرندے اڑ رہے ہیں",
        "سورج چمک رہا ہے",
        "چاند روشن ہے",
        "ستارے خوبصورت ہیں",
        "پانی ضروری ہے",
        "کھانا تیار ہے",
        "دروازہ کھلا ہے",
        "کھڑکی بند ہے",
        "میں خوش ہوں",
        "وہ اداس ہے",
        "وہ ناراض ہے",
        "ہم پرجوش ہیں",
        "وہ تھکے ہوئے ہیں",
        "میں تیر سکتا ہوں",
        "وہ گاڑی چلا سکتا ہے",
        "وہ گا سکتی ہے",
        "ہم ناچ سکتے ہیں",
        "میں کل جاؤں گا",
        "وہ کل گیا تھا",
        "وہ ابھی آ رہی ہے",
        "ہم یہاں انتظار کر رہے ہیں",
        "وہ وہاں رہتے ہیں",
        "براہ کرم یہاں آئیں",
        "ابھی وہاں جاؤ",
        "آج رات یہاں رہو",
        "کل صبح جاؤ",
        "اگلے ہفتے واپس آؤ",
        "میں چاول کھاتا ہوں",
        "وہ دودھ پیتا ہے",
        "وہ کھانا پکاتی ہے",
        "ہم کرکٹ کھیلتے ہیں",
        "وہ فلمیں دیکھتے ہیں",
        "آسمان نیلا ہے",
        "گھاس ہری ہے",
        "برف سفید ہے",
        "کوئلہ کالا ہے",
        "خون سرخ ہے",
        "میں اردو سیکھ رہا ہوں",
        "وہ انگریزی پڑھا رہا ہے",
        "وہ کتاب پڑھ رہی ہے",
        "ہم خط لکھ رہے ہیں",
        "وہ چائے بنا رہے ہیں",
        "میرے والد مہربان ہیں",
        "میری والدہ محبت کرنے والی ہیں",
        "میرا بھائی ہوشیار ہے",
        "میری بہن خوبصورت ہے",
        "میرا دوست وفادار ہے"
    ]
    
    return english_sentences, urdu_sentences

def load_translation_data():
    english_sentences, urdu_sentences = create_sample_dataset()
    
    english_sentences = [clean_text(sent) for sent in english_sentences]
    
    split_idx = int(0.8 * len(english_sentences))
    
    train_en = english_sentences[:split_idx]
    train_ur = urdu_sentences[:split_idx]
    test_en = english_sentences[split_idx:]
    test_ur = urdu_sentences[split_idx:]
    
    return train_en, train_ur, test_en, test_ur
