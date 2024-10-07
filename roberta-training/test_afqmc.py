from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

#model_path = "/mnt/storage/ntunggal/baidu-model-small-finetuned/checkpoint-1611"

def test_model(model_path, sentence1, sentence2):
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    # Tokenize the input sentences
    inputs = tokenizer(sentence1, sentence2, return_tensors='pt', padding=True, truncation=True, max_length=512)
    print(inputs)

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

    test_model(model_path, "你好吗？", "你好吗？")
    test_model(model_path, "一顶帽子有什么可怕的？", "为什么要怕这顶帽子？")
    test_model(model_path, "我喜欢你", "我喜欢你的狗")
    test_model(model_path, '商家支持花呗，我的账户却不能用', '商家有花呗交易，我还是不能用花呗付款')
    
