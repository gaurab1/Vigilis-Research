FILE_NAME = 'logs/o4-mini-audio-call.log'

total_count = 0
spam_count = 0
ham_count = 0

correct_count = 0
correct_spam_count = 0
correct_ham_count = 0

with open(FILE_NAME, 'r', errors='ignore') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        label = line.split(',')[0].split(': ')[1].strip()
        prediction = line.split(',')[1].split(': ')[1].strip()
        total_count += 1
        if label == 'spam':
            spam_count += 1
            if label != prediction:
                print(f"{idx+2}: {line.strip()}")
        elif label == 'ham':
            ham_count += 1

        prediction = 'ham' if (prediction not in ['ham', 'spam']) else prediction
        if prediction == label:
            correct_count += 1
            if prediction == 'spam':
                correct_spam_count += 1
            elif prediction == 'ham':
                correct_ham_count += 1

print(spam_count, ham_count)
print(f"Total Accuracy: {correct_count/total_count}")
print(f"Recall (Spam): {correct_spam_count/spam_count}")
print(f"Precision (Spam): {correct_spam_count/(correct_spam_count + (ham_count - correct_ham_count))}")

    
