# bot_logic.py
import spacy
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import wikipedia

class BotLogic:
    def __init__(self, model_path='./gpt2-finetuned-science'):
        self.fine_tuned_model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path, pad_token='<pad>')
        self.nlp = spacy.load("en_core_web_sm")

    def extract_keywords(self, question):
        doc = self.nlp(question)
        keywords = [token.lemma_.lower() for token in doc if token.ent_type_ or token.pos_ in ["NOUN", "ADJ"]]
        return list(set(keywords))

    def generate_response(self, prompt, max_length=80):
        keywords = self.extract_keywords(prompt)
        core_query = ' '.join(keywords)        
        optimal_prompt = f"What is {core_query}?"

        input_ids = self.tokenizer.encode(optimal_prompt, return_tensors='pt', max_length=max_length, truncation=True)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        output = self.fine_tuned_model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, num_return_sequences=1)
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        try:
            splitted = generated_text.split()
            pos = splitted.index("Support:")
            ans = ' '.join(splitted[pos+1:]).split(".")[0]
            out = ans.split(".")
            final_output = out[0]+"."

        except:
            final_output = generated_text

        for keyword in keywords:
            try:
                if keyword.lower() not in generated_text.lower().split(" ") :
                    top_result = wikipedia.summary(core_query).split(".")[0] + wikipedia.summary(core_query).split(".")[1]
                    return f"Sorry, I don't have information on that. Here's what I found on Wikipedia:\n{top_result}."
                elif len(keyword) == 0:
                    return "Please reframe your question."
            except:
                return "Sorry looks like I don't know the answer :("
                
            
        return final_output

