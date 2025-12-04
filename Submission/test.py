from submission import Submission
import torch

SFREQ = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sub = Submission(SFREQ, DEVICE)
model_1 = sub.get_model_challenge_1()
model_1.eval()

model_2 = sub.get_model_challenge_2()
model_2.eval()

print("success")

