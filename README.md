# IndustryEQA

### Installation

The code requires a `python>=3.9` environment. We recommend using conda:

```bash
conda create -n industryeqa python=3.9
conda activate industryeqa
pip install -r requirements.txt
```

### Running baselines

```bash
python industryeqa/baselines/<baseline>.py --dry-run  
```

### Running evaluations

```bash
python .\industryeqa\evaluation\direct_match.py --evaluator openai --model gpt-4o-mini --generated-answers <json>
```

