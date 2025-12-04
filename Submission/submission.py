# submission.py
# from braindecode.models import EEGNeX
from braindecode.models import EEGSimpleConv
from braindecode.models import EEGNeX
import torch


class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def get_model_challenge_1(self):
        model_challenge1 = EEGSimpleConv(
            n_chans=129, n_outputs=1, sfreq=self.sfreq, n_times=int(2 * self.sfreq)
        ).to(self.device)
        # load from the current directory (/app/input/res/ is where the file resides on Codabench)
        model_challenge1.load_state_dict(
            torch.load(
                "/app/input/res/weights_challenge_1.pth", map_location=self.device
            )
        )
        return model_challenge1

    def get_model_challenge_2(self):
        model_challenge2 = EEGNeX(
            n_chans=129, n_outputs=1, sfreq=self.sfreq, n_times=int(2 * self.sfreq)
        ).to(self.device)

        return model_challenge2
