import json
import os

def verify_and_fix_dataset(file_path, output_path=None):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    modified = False
    errors = []

    for example in data.get("examples", []):
        id = example.get("id", "unknown")
        text = example.get("original_text", "")
        
        for i, ann in enumerate(example.get("annotations", [])):
            start = ann.get("start")
            end = ann.get("end")
            expected_text = ann.get("text")
            
            if start is None or end is None or expected_text is None:
                continue
                
            actual_text = text[start:end]
            
            if actual_text != expected_text:
                # Try to find the correct offset
                # First, check if it's just a slight shift
                found_at = -1
                search_range = 10
                for shift in range(-search_range, search_range + 1):
                    if text[start+shift:end+shift] == expected_text:
                        found_at = start + shift
                        break
                
                if found_at != -1:
                    new_start = found_at
                else:
                    new_start = text.find(expected_text)
                    if new_start != -1:
                        # Look for the occurrence closest to the original start
                        occurrences = []
                        curr = text.find(expected_text)
                        while curr != -1:
                            occurrences.append(curr)
                            curr = text.find(expected_text, curr + 1)
                        
                        new_start = min(occurrences, key=lambda x: abs(x - start))

                if new_start != -1:
                    new_end = new_start + len(expected_text)
                    errors.append(f"Fixed {id} (annotation {i}): '{expected_text}' move from [{start}:{end}] to [{new_start}:{new_end}]")
                    ann["start"] = new_start
                    ann["end"] = new_end
                    modified = True
                else:
                    errors.append(f"CRITICAL Error {id} (annotation {i}): '{expected_text}' not found in text at all!")

    if errors:
        for err in errors:
            print(err)
    else:
        print("All offsets are correct!")

    if modified and output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Saved fixed dataset to {output_path}")

if __name__ == "__main__":
    dataset_path = "eval/datasets/data/anonymization_dataset.json"
    verify_and_fix_dataset(dataset_path, dataset_path)
