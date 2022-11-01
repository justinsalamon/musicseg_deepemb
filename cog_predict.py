# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import cog

import musicsections


class Predictor(cog.Predictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model_deepsim = musicsections.load_deepsim_model("models/deepsim")
        self.model_fewshot = musicsections.load_fewshot_model("models/fewshot")

    @cog.input("audio", type=cog.Path, help="Input audio file")
    def predict(self, audio):
        """Run a single prediction on the model"""
        segmentations, _ = musicsections.segment_file(
            str(audio),
            deepsim_model=self.model_deepsim,
            fewshot_model=self.model_fewshot,
            min_duration=8,
            mu=0.5,
            gamma=0.5,
            beats_alg="madmom",
            beats_file=None,
        )

        musicsections.plot_segmentation(
            segmentations, figsize=(10, 3), display_seconds=True
        )
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        plt.savefig(out_path)
        return out_path
