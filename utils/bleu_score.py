import os

def calculate_bleu_score():
    tt = open("./logs/test_targets_100K.txt").readlines()
    tp = open("./logs/test_predicted_100K.txt").readlines()
    _tt = open("./logs/final_targets.txt", "w")
    _tp = open("./logs/final_preds.txt", "w")

    for i, j in zip(tt, tp):
        eos_i = i.find("<eos>")
        _tt.write(i[6:eos_i] + "\n")

        eos_j = j.find("<eos>")
        _tp.write(j[6:eos_j] + "\n")

if __name__ == "__main__":
    print(" calculating Bleu Score...  ")
    print(os.path.exists("./logs"))
    
    cmd = "perl ./utils/mulit-bleu.perl logs/final_targets.txt < logs/final_preds.txt"
    os.system(cmd)