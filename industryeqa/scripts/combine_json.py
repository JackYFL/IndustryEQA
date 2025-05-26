import argparse
import json
import os
from pathlib import Path
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default="data/open-eqa-v0.json", help="path to EQA dataset (default: data/open-eqa-v0.json)")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help=" model")
    parser.add_argument("--output-directory", type=Path, default="data/results", help="output directory (default: data/results)")
    parser.add_argument("--force", action="store_true", help="continue running on API errors (default: false)")
    parser.add_argument("--dry-run", action="store_true", help="only process the first 5 questions")
    args = parser.parse_args(); 
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.output_path = args.output_directory / (args.dataset.stem + "-{}.json".format(args.model))
    return args


    
def main(args: argparse.Namespace):
    assert "GOOGLE_API_KEY" in os.environ

    video_dir = Path("./data/large/human")
    video_exts = {'.mp4'}
    paths = [str(p) for p in video_dir.rglob("*") if p.suffix.lower() in video_exts]
    print(f"Found {len(paths)} video files")
    
    # Combine all results into one JSON file
    combine_json_results(paths)

def combine_json_results(paths, output_file="data/results/large_human.json"):
    """Combine all individual JSON results into a single JSON file."""
    all_qa_pairs = []

    for path in paths:
        json_path = Path(path).with_suffix(".json")
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    qa_pairs = json.load(f)

                # Add POSIX-style path to each QA pair
                for qa in qa_pairs:
                    qa['path'] = str(Path(path).as_posix())

                all_qa_pairs.extend(qa_pairs)
            except json.JSONDecodeError:
                print(f"Error reading {json_path}")

    # Reorder IDs
    for idx, qa in enumerate(all_qa_pairs, start=1):
        qa['id'] = idx
        
    # Save combined results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_qa_pairs, f, indent=4, ensure_ascii=False)

    print(f"Combined {len(all_qa_pairs)} QA pairs from {len(paths)} videos to {output_file}")
    return all_qa_pairs

if __name__ == "__main__":
    main(parse_args())

