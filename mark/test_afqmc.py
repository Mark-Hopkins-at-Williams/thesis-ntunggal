from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

def test_model(model_path, tokenizer_path, sentence1, sentence2):
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(model_path)
    model.eval()

    # Tokenize the input sentences
    inputs = tokenizer(sentence1, sentence2, return_tensors='pt', padding=True, truncation=True, max_length=512)
    #print(inputs)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
       
    # Get the logits (raw predictions)
    logits = outputs.logits

    # Convert logits to predicted class
    predicted_class = torch.argmax(logits, dim=-1).item()

    # Get probabilities from logits
    probs = F.softmax(logits, dim=-1)

    # Print results
    print(f"Sentence 1: {sentence1}")
    print(f"Sentence 2: {sentence2}")
    print(f"Predicted class: {predicted_class}")
    print(f"Probabilities: {probs}")
    print("-------")


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1]
    tokenizer_path = sys.argv[2]

    # From afqmc
    test_model(model_path, tokenizer_path, "支付宝系统点我的里面没有花呗这一项", "我下载支付宝怎么没有花呗的") # 1
    test_model(model_path, tokenizer_path, "如升级为网商贷用户而借呗的欠款怎么还", "我借呗已还清，为什么升级不了网商贷") # 0
    test_model(model_path, tokenizer_path, '可以用其他的银行卡还蚂蚁借呗的钱吗', '蚂蚁借呗昨天借的钱今天就到还款日') # 0

    # Created myself
    test_model(model_path, tokenizer_path, "你好吗？", "你好吗？") # 1
    test_model(model_path, tokenizer_path, '这是我的狗', '美国何以沦为全球最大失败国？') # 0
    test_model(model_path, tokenizer_path, '为什么要怕这顶帽子？', '这顶帽子有什么可怕的？') # 1
    
