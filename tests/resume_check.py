"""Run by tests/test_acs.py in a subprocess (needs TABGEN_DATASET=acs before ladder is imported).
Trains a tiny step-5 GAN for 4 epochs straight, and for 2 + 2 epochs through the resume file,
and prints whether the final weights are identical."""
import os
import sys
import tempfile
from pathlib import Path

os.environ["TABGEN_DATASET"] = "acs"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

torch.set_num_threads(1)
import ladder  # noqa: E402
from evaluator import schema_from_frame  # noqa: E402
from test_acs import fake_people  # noqa: E402

frame = fake_people(300).drop(columns=["SERIALNO"])
fit, select = frame.iloc[:800].reset_index(drop=True), frame.iloc[800:1000].reset_index(drop=True)
schema = schema_from_frame(fit, ladder.NUMERIC, ladder.CATEGORICAL)
base = dict(ladder.RUNGS["5_rules"], epochs=4, score_every=2, batch_size=100, pac=10)
quiet = lambda m: None
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    ladder.train_rung("5_rules", base, 1, fit, select, None, schema, quiet, model_dir=tmp / "a")
    straight = torch.load(tmp / "a" / "5_rules_seed1.pt", weights_only=False)
    resume = tmp / "resume.pt"
    ladder.train_rung("5_rules", dict(base, epochs=2), 1, fit, select, None, schema, quiet,
                      model_dir=tmp / "b", resume_path=resume)
    ladder.train_rung("5_rules", base, 1, fit, select, None, schema, quiet, model_dir=tmp / "c", resume_path=resume)
    resumed = torch.load(tmp / "c" / "5_rules_seed1.pt", weights_only=False)
    same = all(torch.equal(straight["final"]["generator_state"][k], resumed["final"]["generator_state"][k])
               for k in straight["final"]["generator_state"])
    same &= all(torch.equal(straight["final"]["critic_state"][k], resumed["final"]["critic_state"][k])
                for k in straight["final"]["critic_state"])
    same &= straight["selected_epoch"] == resumed["selected_epoch"]
    print("IDENTICAL" if same else "DIFFERENT")
