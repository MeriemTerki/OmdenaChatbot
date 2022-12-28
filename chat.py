import random
import json
import string
import trainOmdena





intents = json.loads(open('intents .json').read())





bot_name = "Omdena Doctor"

def get_response(msg):
    while True:
        text_p=[]
        texts = input("You:")
        texts = [letters.lower() for letters in texts if letters not in string.punctuation]
        texts = ''.join(texts)
        text_p.append(texts)


        x_val = trainOmdena.tokenizer(
        text=text_p,
        add_special_tokens=True,
        max_length=20,
        truncation=True,
        padding='max_length', 
        return_tensors='tf',
        return_token_type_ids = False,
        return_attention_mask = True,
        verbose = True) 
        validation = trainOmdena.model.predict({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']})*100
        output=validation.argmax()
        response_tag = trainOmdena.le.inverse_transform([output])[0]
        print("doctor Omdena: ",random.choice(trainOmdena.responses[response_tag]))
        if response_tag == "goodbye":
           break
        return "I do not understand..."

    

    
    


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)

