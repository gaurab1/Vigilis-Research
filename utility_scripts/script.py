import os
import random

DIR = "data/audio_transcripts"

character_threshold = (500, 1000)
ham_dir = os.path.join(DIR, "ham", "CallHome", "processed_files")
scam_dir = os.path.join(DIR, "scam")

content = []

for root, dirs, files in os.walk(ham_dir):
    for file in files:
        if file not in ['4170.txt', '4157.txt', '4156.txt', '4145.txt', '4112.txt', '4104.txt', '4093.txt', '4077.txt', '4074.txt', '4065.txt']:
            continue
        with open(os.path.join(root, file), "r") as f:
            current_threshold = random.randint(*character_threshold)
            curr_content = ""
            for x in f:
                if x:
                    curr_content += x
                    current_threshold -= len(x)
                    if current_threshold <= 0:
                        current_threshold = random.randint(*character_threshold)
                        print("1,\"" + curr_content.replace("\n", "\\n") + "\"," + file)
                        content.append((1, curr_content, file))
                        curr_content = ""
            if curr_content:
                print("1,\"" + curr_content.replace("\n", "\\n") + "\"," + file)
                content.append((1, curr_content, file))
# a = len(content)
# print(len(content))
# for root, dirs, files in os.walk(scam_dir):
#     for file in files:
#         with open(os.path.join(root, file), "r") as f:
#             current_threshold = random.randint(*character_threshold)
#             curr_content = ""
#             for x in f:
#                 if x:
#                     curr_content += x
#                     current_threshold -= len(x)
#                     if current_threshold <= 0:
#                         current_threshold = random.randint(*character_threshold)
#                         content.append((0, curr_content, file))
#                         curr_content = ""
#             if curr_content:
#                 print("0,\"" + curr_content.replace("\n", "\\n") + "\"," + file)
#                 content.append((0, curr_content, file))

# print(len(content) - a)
# random.shuffle(content)

# with open("data/content.csv", "w") as f:
#     f.write("label,text,source\n")
#     for x in content:
#         f.write(f"{x[0]},\"{x[1].replace("\n", "\\n")}\",{x[2]}\n")


