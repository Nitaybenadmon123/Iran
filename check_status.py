import json
from pathlib import Path
import pandas as pd

# Check timeline files
timeline_dir = Path('Data/Users_Timelines')
timeline_files = list(timeline_dir.glob('*_timeline.csv'))
print(f"📊 קבצי Timeline נוכחיים: {len(timeline_files)}")

# Check checkpoint
checkpoint_path = timeline_dir / 'collection_checkpoint.json'
if checkpoint_path.exists():
    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        checkpoint = json.load(f)
    print(f"✅ Completed: {len(checkpoint['completed'])}")
    print(f"❌ Failed: {len(checkpoint['failed'])}")
    print(f"📊 Total processed: {len(checkpoint['completed']) + len(checkpoint['failed'])}")
    
    # Check who was collected
    base_data = pd.read_csv('POIs/Classification/base_data_locals_vs_diaspora.csv')
    all_usernames = set(base_data['username'].str.lower())
    files_usernames = {f.stem.replace('_timeline', '') for f in timeline_files}
    
    missing = all_usernames - files_usernames
    print(f"\n🔍 עדיין חסרים: {len(missing)} משתמשים")
    print(f"דוגמאות של חסרים: {list(missing)[:10]}")
