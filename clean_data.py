import json
import re
from bs4 import BeautifulSoup
import sys

# 🛠️ Setup
INPUT_FILE = "outbox.json" # The big pile of raw data
OUTPUT_FILE = "akkoma_posts.jsonl" # The groomed kibble for the AI

def clean_text(html_content):
    """
    Grooms the messy HTML fur into shiny, clean text.
    """
    if not html_content:
        return ""
    
    # 1. HTML Stripping (Detangling the mats)
    # Using BeautifulSoup to lick away the tags like <p> and <br>
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator=" ") 
    
    # 2. Sanitize Mentions (Muzzling the pings)
    # We want to keep the names so the bot knows who it's talking to,
    # but we break the '@' so we don't spam poor users during training/testing.
    # @user -> @ user
    mention_pattern = r'(\B@)([\w\-\.]+(?:@[\w\-\.]+)?\b)'
    
    # Lambda function effectively puts a paw between the @ and the name
    text = re.sub(mention_pattern, r'\1 \2', text)
    
    # 3. Cleanup whitespace (The final brush)
    # Collapse multiple spaces into one smooth coat
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def main():
    print("🦊 *Ears perk up* Sniffing for data files...")
    
    # Try to load the file with gentle paws
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"😿 *Whimpers* I can't find {INPUT_FILE}! Did you bury it in the yard?")
        return

    # Akkoma/Pleroma exports are tricky treats. 
    # We need to dig through the structure to find the tasty 'items'.
    items = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        # Check for standard ActivityPub 'orderedItems' or 'items'
        if "orderedItems" in data:
            items = data["orderedItems"]
        elif "items" in data:
            items = data["items"]
        # Sometimes the export is an archive.json with 'statuses'
        elif "statuses" in data:
            items = data["statuses"]
        else:
            print("🙀 *Hisses* This JSON smells wrong! I can't find the post list!")
            return

    print(f"🐾 Found {len(items)} items in the pile! Starting the grooming session... :3")

    valid_count = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        for item in items:
            content = ""
            
            # Case A: It's a full ActivityPub object (The whole bone)
            if isinstance(item, dict) and "object" in item and isinstance(item["object"], dict):
                content = item["object"].get("content", "")
            # Case B: It's a flattened status object (Just the marrow)
            elif isinstance(item, dict) and "content" in item:
                content = item["content"]
            
            # Scrub scrub scrub!
            cleaned = clean_text(content)
            
            # Only save if there's actual text left (don't save empty shedding)
            if cleaned and len(cleaned) > 1:
                # Write to JSONL format for the training script
                json_record = json.dumps({"content": cleaned}, ensure_ascii=False)
                out.write(json_record + "\n")
                valid_count += 1

    print(f"✨ *Wagging tail* All done! Groomed {valid_count} posts.")
    print(f"💾 Saved to {OUTPUT_FILE}. Ready to feed the AI! Awoo! 🐺")

if __name__ == "__main__":
    main()