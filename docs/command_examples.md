### Translation + Evaluation + Report Generation
```bash
uv run src/pipeline/run_pipeline.py -i data/validation_set_tmp -l fr it de -s single dual -o data/validation_set_tmp/result
```

### Translation
```bash
uv run src/pipeline/translation.py --input data/validation_set --target-language it --strategy dual --output data/translations/validation_set

uv run src/pipeline/translation.py --input data/validation_set/carte_id_sr.png --target-language it --strategy dual --output data/translations/validation_set

uv run src/pipeline/translation.py --input data/pdfs/calculo_antecipado_da_pensao.pdf --target-language de --strategy dual --output data/translations
```

### Evaluation
```bash
uv run src/pipeline/evaluation.py --original data/validation_set --translated data/translations/validation_set --result-dir data/evaluation_results/validation_set

uv run src/pipeline/evaluation.py --original data/pdfs/calculo_antecipado_da_pensao.pdf --translated data/translations/calculo_antecipado_da_pensao_de_dual_20260714_160520.md --result-dir data/evaluation_results
```

### Report
```bash
uv run src/pipeline/report.py data/evaluation_results --output-dir data/reports

uv run src/pipeline/report.py data/evaluation_results/calculo_antecipado_da_pensao_de_dual_20260714_160520_20260714_160917.json --output-dir data/reports
```