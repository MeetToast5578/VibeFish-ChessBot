# VibeFish: notes for agents

Searchless chess bot. A small transformer picks moves with **one forward pass**. Do not add
search (alpha-beta, MCTS, k-ply lookahead) to the play path. That is the point of the project.
Strength must come from data, features, losses and training.

## Layout

| file | role |
|---|---|
| `model.py` | board encoding (`encode`, feature planes), move indexing, `ChessGPT` net, `pick()` |
| `teacher.py` | Stockfish: grading every legal move, score → grade conversion, game workers |
| `data.py` | Lichess eval DB parsing, SF grading jobs, DAgger games, encoding to `.npz` shards |
| `train.py` | trainer over shards (mixed sources, cosine LR), checkpoints |
| `evaluate.py` | puzzles, held-out wp-lost, matches vs Stockfish `UCI_Elo` |
| `play.py` | terminal play and `--uci` engine mode |
| `kaggle_run.py` | runs commands on a Kaggle GPU (or CPU) notebook with datasets attached, downloads outputs |
| `bench/` | benchmark sets (never trained on) + `results.jsonl` of every evaluation |
| `tests/` | `python -m pytest -q tests` |
| `HANDOFF.md` | where the work stopped last and what to do next (read first) |
| `PLAN.md` | goals, experiment protocol, data/compute inventory, lessons, roadmap. Keep it updated as steps finish. |

## Conventions

- Board is always encoded from the **side to move's view** (black positions mirrored). Move index
  = `from*64 + to` in that frame. Under-promotions fold into the queen promotion.
- Grades: one float per move, side-to-move view. `wp = sigmoid(0.00368 * cp)`. Mates sit outside
  [0, 1] (`1 + 0.2/n` win, `-0.2/n` loss) so shorter mates rank higher. Clip to [0, 1] for value/BCE targets.
- Raw data keeps **scores, not grades** (`teacher.MATE` encoding), so the grade curve can change
  without regenerating data.
- Changing `encode()` invalidates encoded shards (`data/*.npz`) and checkpoints. Bump
  `model.ENCODING_VERSION`.

## Environment

- Python venv at `/home/user/venv` (system pip can't build `chess`). `. /home/user/venv/bin/activate`
- Stockfish: `export STOCKFISH=/path/to/stockfish` (any modern NNUE build. v2/v3 data used SF 17.1)
- CPU only. bf16 autocast is *slower* here, and torch.compile needs a working g++ (cc1plus).
  Training ~320 pos/s for 0.8M params, ~130–220 pos/s for the 2.9M v2 model on 4 cores.
- Never run Stockfish/data jobs next to a training run: one busy core halves 4-thread PyTorch.
- Every input feature must be exercised by training data. Lichess/puzzle positions have no move
  history, so history features are only learned from DAgger games (see PLAN.md log).
- Big files (`data/`, Lichess dumps) are gitignored. **Every trained model is kept in `models/`** (owner's rule), shipped or not, named
  `vibefish_<run>.pt`, so it can be recalled later. GitHub's limit is 100 MB per file.
- Kaggle (user `yunus0901`): the API token is injected by the session's network proxy, never stored
  locally (`~/.kaggle/access_token` is a placeholder). Kaggle CLI 2.x checks the token by sending it
  in a request body, which the proxy can't fill, so the venv's `kaggle` entry point patches
  `KaggleApi._introspect_token` to skip that check. Never print or commit tokens.
- Kaggle hides a running kernel's logs and outputs; poll `kaggle kernels status`, then
  `kaggle kernels output`. Sessions stop at 12 h, so keep CPU grading jobs to ≤ 4 h or split them.
- Shipped model: `models/vibefish_v4_1.pt` (v4 fine-tuned on depth-12 self-play; 13.6M params, d=256, 16 layers + attention bias, played with `--mix 2`). Compare new candidates
  against it with `evaluate.py --mix 2 --vs models/vibefish_v4_1.pt@2` and the Stockfish matches.
