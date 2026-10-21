# Supplementary dataset species cross-check with Pl@ntNet

The Pl@ntNet review tool creates a separate CSV with a suggested orchid genus
for manual review. It does not edit image files, folder labels, the group
manifest, health labels, or training metrics. Pl@ntNet is a supplementary
identification service, not ground truth. Its top-match score is not a
calibrated probability.

Pl@ntNet is not designed to assess orchid health. This tool does not validate
the `healthy` / `not_healthy` labels and does not diagnose disease.

## Privacy and credentials

Running the tool sends a resized copy of each selected image to Pl@ntNet.
Review Pl@ntNet's current API terms and your dataset privacy requirements
before enabling external uploads. The first run is limited to five images.
Do not include personal or sensitive images unless you are authorized to share
them with this provider.

Create an API key through the Pl@ntNet API portal, then set it in the current
PowerShell session without putting it in source code or command-line arguments:

```powershell
$secureKey = Read-Host "Pl@ntNet API key" -AsSecureString
$env:PLANTNET_API_KEY = [System.Net.NetworkCredential]::new("", $secureKey).Password
Remove-Variable secureKey
```

The key is not saved by the tool. Do not paste it into chat, commit it, or place
it in the report.

## Run a small review

From the project root, explicitly confirm that image uploads are allowed:

```powershell
.\venv\Scripts\python.exe .\tools\validate_dataset_with_plantnet.py --limit 5 --confirm-external-upload
```

Review `dataset_validation_plantnet.csv`. Rows marked `review_recommended=true`
need a human decision. Agreement with a folder label is only a supplementary
signal; it is not proof the label is correct.

After reviewing the small sample and choosing to send the full dataset:

```powershell
.\venv\Scripts\python.exe .\tools\validate_dataset_with_plantnet.py --limit 0 --resume --confirm-external-upload
```

`--resume` skips image paths already in the CSV. The report is kept outside the
dataset directory and never changes dataset labels. API errors stop the run
with a concise message and omit response bodies and credentials.
