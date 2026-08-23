#!/usr/bin/env bash
# Download and unpack APTOS 2019 into data/raw/aptos2019/.
#
# Two prerequisites, both need a human with a browser:
#
#   1. Accept the data-use agreement. Open
#        https://www.kaggle.com/competitions/aptos2019-blindness-detection
#      and click "Late Submission" (Kaggle renames "Join Competition" once a
#      competition closes), then accept the rules. This enters you into nothing,
#      the competition ended in 2019; it is how Kaggle records that you accepted
#      the licence on patient medical imagery. The API returns 403 until you do.
#
#   2. Authenticate the CLI. Easiest is the browser flow:
#        .venv/bin/kaggle auth login
#      Alternatively set KAGGLE_API_TOKEN, or save a token generated at
#      https://www.kaggle.com/settings/api to ~/.kaggle/access_token
#
# Needs about 10 GB free while downloading and unzipping, 9 GB after.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST="data/raw/aptos2019"
COMP="aptos2019-blindness-detection"
KAGGLE=".venv/bin/kaggle"

[ -x "$KAGGLE" ] || { echo "ERROR: $KAGGLE missing. Run ./scripts/setup_env.sh first." >&2; exit 1; }

# Preflight: listing the files is a metadata-only call, so it surfaces an auth
# problem or an unaccepted licence in a couple of seconds rather than partway
# through a 9 GB transfer.
echo "Checking access to $COMP ..."
if ! "$KAGGLE" competitions files -c "$COMP" >/tmp/kaggle_preflight.$$ 2>&1; then
  echo >&2
  echo "Cannot reach the competition data. What the two failures look like:" >&2
  echo >&2
  sed 's/^/  | /' /tmp/kaggle_preflight.$$ >&2
  echo >&2
  echo "  403 / forbidden / rules  -> the data-use agreement is not accepted." >&2
  echo "     Open https://www.kaggle.com/competitions/$COMP and click" >&2
  echo "     \"Late Submission\" (Kaggle renames \"Join Competition\" once a" >&2
  echo "     competition closes). Accept the rules in the dialog." >&2
  echo >&2
  echo "  401 / auth / credentials -> run: $KAGGLE auth login" >&2
  rm -f /tmp/kaggle_preflight.$$
  exit 1
fi
echo "Access confirmed. Files available:"
head -12 /tmp/kaggle_preflight.$$ | sed 's/^/  /'
rm -f /tmp/kaggle_preflight.$$

mkdir -p "$DEST"
echo
echo "Downloading $COMP (about 9 GB), this will take a while ..."

# Let the CLI report its own auth/403 errors rather than second-guessing which
# of the three auth mechanisms is in use, then translate the common ones.
if ! "$KAGGLE" competitions download -c "$COMP" -p "$DEST"; then
  echo >&2
  echo "Download failed. The two usual causes:" >&2
  echo "  403 Forbidden  -> you have not clicked \"Join Competition\" yet:" >&2
  echo "                    https://www.kaggle.com/competitions/$COMP" >&2
  echo "  auth error     -> run: $KAGGLE auth login" >&2
  exit 1
fi

echo "Unzipping ..."
unzip -q -o "$DEST/${COMP}.zip" -d "$DEST"
rm -f "$DEST/${COMP}.zip"

echo
echo "Contents of $DEST:"
ls -1 "$DEST"

echo
echo "Checking the class distribution quoted in the synopsis:"
.venv/bin/python - <<'PY'
import pandas as pd
df = pd.read_csv("data/raw/aptos2019/train.csv")
counts = df.diagnosis.value_counts().sort_index()
print(f"  images in train.csv: {len(df)}   (synopsis says 3,662)")
print(f"  grade counts:        {list(counts)}")
print( "  synopsis says:       [1805, 370, 999, 193, 295]")
if list(counts) != [1805, 370, 999, 193, 295] or len(df) != 3662:
    print("  MISMATCH -> update the figures in the synopsis before submitting")
else:
    print("  match, synopsis figures are correct")
PY

echo
echo "Next: .venv/bin/python -m src.preprocess --raw $DEST --out data/processed --size 320"
